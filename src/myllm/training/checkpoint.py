"""Orbax-backed sharded checkpoint manager.

Behaviour:
    - Atomic per-step directories: ``<root>/step-<N>/``
    - Per-step ``manifest.json`` written last and used by readers as the
      "this checkpoint is complete" marker — corrupt/partial checkpoints
      are detected by missing manifest.
    - Retention: keep the last ``keep_last_n``, plus every ``keep_every_n``
      thereafter.
    - Object storage: use a fuse mount or push step directories with rclone /
      boto3 from the loop after each save (separate concern, not Orbax's job).

This wrapper is intentionally thin — Orbax already handles sharded
PyTree saves correctly across multi-host. We add the retention policy and
the manifest-based completion marker.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from myllm.utils import get_logger
from myllm.utils.exceptions import CheckpointError
from myllm.utils.io import read_json, write_json_atomic

log = get_logger(__name__)


@dataclass(frozen=True)
class CheckpointConfig:
    root: str
    keep_last_n: int = 3
    keep_every_n: int = 5000
    r2_prefix: str | None = None  # if set, mirror each step dir to s3://<bucket>/<r2_prefix>/step-N/

    def __post_init__(self) -> None:
        if self.keep_last_n < 1:
            raise ValueError("keep_last_n must be >= 1")
        if self.keep_every_n < 1:
            raise ValueError("keep_every_n must be >= 1")


class CheckpointManager:
    """Sharded checkpoint manager around Orbax."""

    def __init__(self, config: CheckpointConfig) -> None:
        self.config = config
        # Orbax requires absolute paths for its tensorstore kvstore.
        self.root = Path(config.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._orbax = self._lazy_orbax()

    @staticmethod
    def _lazy_orbax() -> Any:
        try:
            import orbax.checkpoint as ocp

            return ocp.PyTreeCheckpointer()
        except ImportError as e:
            raise ImportError(
                "orbax-checkpoint not installed; install with "
                "`pip install orbax-checkpoint`"
            ) from e

    def step_dir(self, step: int) -> Path:
        return self.root / f"step-{step:09d}"

    def save(self, step: int, state: dict[str, Any], extra: dict[str, Any] | None = None) -> Path:
        """Save state for ``step``. Writes manifest last for atomicity."""
        # Coerce step to a Python int — caller may pass a JAX/numpy scalar.
        step = int(step)
        target = self.step_dir(step)
        if (target / "manifest.json").exists():
            log.warning("checkpoint_already_exists", step=step, path=str(target))
            return target
        target.mkdir(parents=True, exist_ok=True)
        try:
            self._orbax.save(target / "state", state)
        except Exception as e:
            raise CheckpointError(f"orbax save failed at step {step}: {e}") from e
        manifest = {"step": step, "extra": extra or {}}
        write_json_atomic(target / "manifest.json", manifest)
        log.info("checkpoint_saved", step=step, path=str(target))
        if self.config.r2_prefix:
            self._mirror_to_r2(target, step)
        self._apply_retention()
        return target

    def _mirror_to_r2(self, local_dir: Path, step: int) -> None:
        """Mirror this step's directory to R2 under ``<r2_prefix>/step-NNNN/``."""
        try:
            from myllm.utils.storage import upload_directory
        except ImportError:
            log.warning("checkpoint_r2_mirror_skipped_storage_unavailable", step=step)
            return
        remote = f"{self.config.r2_prefix.rstrip('/')}/step-{step:09d}"
        try:
            n = upload_directory(local_dir, remote)
            log.info(
                "checkpoint_mirrored_to_r2", step=step, remote_prefix=remote, files=n
            )
        except Exception as e:  # noqa: BLE001
            log.error(
                "checkpoint_r2_mirror_failed",
                step=step,
                remote_prefix=remote,
                error=str(e),
            )

    def restore(self, step: int, template: dict[str, Any] | None = None) -> dict[str, Any]:
        """Restore the state saved at ``step``.

        Args:
            step: the checkpoint step to load.
            template: optional pytree template with the EXPECTED structure
                of the restored state. When provided, Orbax matches the
                template's structure on restore — preserving namedtuple
                types like ``optax.MultiTransformState`` (B1 fix, audit
                2026-05-12). Without it, Orbax returns a plain dict and
                ``state["opt_state"].inner_states`` would fail at the
                next ``optimizer.update()`` call.

        B1 fix (2026-05-12 audit):
            muP uses ``optax.multi_transform`` whose state is a
            ``MultiTransformState`` namedtuple. Orbax 0.x serializes the
            namedtuple's leaves correctly but loses the type tag on save.
            On restore without a template, the result is a flat dict;
            ``state.inner_states`` (a namedtuple field) becomes a dict
            key access — silently breaking downstream optimizer updates.
            Passing the live ``optimizer.init(...)`` state as a template
            tells Orbax to rebuild the namedtuple structure.
        """
        target = self.step_dir(step)
        if not (target / "manifest.json").exists():
            raise CheckpointError(f"no complete checkpoint at step {step}")
        try:
            if template is not None:
                state = self._orbax.restore(target / "state", item=template)
            else:
                state = self._orbax.restore(target / "state")
        except Exception as e:
            raise CheckpointError(f"orbax restore failed at step {step}: {e}") from e
        log.info("checkpoint_restored", step=step, path=str(target),
                 used_template=template is not None)
        return state

    def latest_complete_step(self) -> int | None:
        candidates: list[int] = []
        for d in self.root.glob("step-*"):
            if (d / "manifest.json").exists():
                try:
                    candidates.append(int(d.name.split("-")[1]))
                except ValueError:
                    continue
        return max(candidates) if candidates else None

    def list_complete_steps(self) -> list[int]:
        steps = []
        for d in self.root.glob("step-*"):
            if (d / "manifest.json").exists():
                try:
                    steps.append(int(d.name.split("-")[1]))
                except ValueError:
                    continue
        return sorted(steps)

    def _apply_retention(self) -> None:
        steps = self.list_complete_steps()
        keep: set[int] = set()
        # Keep the last N.
        keep.update(steps[-self.config.keep_last_n :])
        # Keep every Nth.
        keep.update(s for s in steps if s % self.config.keep_every_n == 0)
        for s in steps:
            if s not in keep:
                self._delete_step(s)

    def _delete_step(self, step: int) -> None:
        import shutil

        path = self.step_dir(step)
        if path.exists():
            shutil.rmtree(path)
            log.info("checkpoint_pruned", step=step)

    # --------------------------------------------------------------------- #
    # R5 from 2026-05-11 dossier: WSM (Warmup-Stable-Merge).
    #
    # Tian et al. arXiv:2507.17634 show that averaging the last N checkpoints
    # from the WSD stable phase outperforms the WSD-decayed final by:
    #   +5.5% MMLU-Pro, +3.5% MATH, +2.9% HumanEval on Pythia-class smalls.
    # Cost: essentially zero (we already keep the last N checkpoints; merge
    # is one weighted-mean pass).
    # --------------------------------------------------------------------- #
    def merge_checkpoints(
        self,
        step_ids: list[int],
        output_step: int,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Element-wise average the model weights of multiple checkpoints.

        Only ``trainable_variables`` and ``non_trainable_variables`` PyTrees
        are averaged. Other state keys (``step``, ``opt_state``,
        ``lr_recovery_multiplier``) take the value from the most recent
        source checkpoint — averaging an optimizer state is undefined.
        """
        if len(step_ids) < 2:
            raise ValueError(f"WSM needs >= 2 checkpoints to merge; got {len(step_ids)}")
        log.info("wsm_merge_start", source_steps=step_ids, output_step=output_step)

        states = [self.restore(s) for s in step_ids]
        merged = self._average_state_trees(states)

        manifest_extra = {
            "wsm_merged": True,
            "source_steps": list(step_ids),
            "source_count": len(step_ids),
            **(extra or {}),
        }
        target = self.save(output_step, merged, extra=manifest_extra)
        log.info("wsm_merge_done", output_step=output_step, target=str(target))
        return target

    @staticmethod
    def _average_state_trees(states: list[dict[str, Any]]) -> dict[str, Any]:
        """Mean across the weight PyTrees of a list of states.

        Non-weight keys (``step``, ``opt_state``, etc.) are inherited from
        the LAST state, since averaging optimizer momentum or step counters
        is meaningless.
        """
        try:
            from jax import tree_util
        except ImportError as e:
            raise ImportError(
                "jax required for WSM merge (tree_util used to traverse "
                "the saved PyTree); install with `pip install jax`"
            ) from e

        last = states[-1]
        result = dict(last)
        for key in ("trainable_variables", "non_trainable_variables"):
            if key not in last:
                continue
            per_state_leaves = [tree_util.tree_leaves(s[key]) for s in states]
            n = len(states)
            avg_leaves = [
                sum(per_state_leaves[i][j] for i in range(n)) / n
                for j in range(len(per_state_leaves[0]))
            ]
            treedef = tree_util.tree_structure(last[key])
            result[key] = tree_util.tree_unflatten(treedef, avg_leaves)
        return result

    def merge_recent(self, n: int, output_step: int) -> Path:
        """Merge the last ``n`` non-merged checkpoints, save as ``output_step``.

        Excludes any prior WSM-merged checkpoints from the source list so
        that re-running WSM doesn't produce a meta-merge.
        """
        all_steps = self.list_complete_steps()
        plain_steps = [s for s in all_steps if not self._is_merged(s)]
        if len(plain_steps) < n:
            raise ValueError(
                f"WSM merge_recent({n}) needs {n} plain checkpoints; "
                f"only {len(plain_steps)} available."
            )
        return self.merge_checkpoints(plain_steps[-n:], output_step)

    def _is_merged(self, step: int) -> bool:
        manifest_path = self.step_dir(step) / "manifest.json"
        if not manifest_path.exists():
            return False
        try:
            m = read_json(manifest_path)
            return bool(m.get("extra", {}).get("wsm_merged", False))
        except (ValueError, KeyError):
            return False


def find_resume_step(checkpoint_root: str) -> int | None:
    """Read the latest complete checkpoint step, if any. Lightweight; no Orbax import."""
    root = Path(checkpoint_root)
    if not root.exists():
        return None
    candidates: list[int] = []
    for d in root.glob("step-*"):
        manifest = d / "manifest.json"
        if manifest.exists():
            try:
                m = read_json(manifest)
                candidates.append(int(m["step"]))
            except (ValueError, KeyError):
                continue
    return max(candidates) if candidates else None
