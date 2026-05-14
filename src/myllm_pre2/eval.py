"""Pre-2 checkpoint prediction and tiny eval bridge."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from myllm_pre2.checkpoint import load_checkpoint, restore_checkpoint_payload
from myllm_pre2.config import DensePre2Config
from myllm_pre2.model import DenseTransformerLM, build_dense_lm


@dataclass(frozen=True)
class NextTokenEvalExample:
    prompt_ids: list[int]
    target_id: int


@dataclass(frozen=True)
class NextTokenPrediction:
    prompt_ids: list[int]
    target_id: int
    predicted_id: int
    correct: bool
    nll: float


@dataclass(frozen=True)
class NextTokenEvalResult:
    checkpoint_step: int
    num_examples: int
    accuracy: float
    average_nll: float
    predictions: list[NextTokenPrediction]


def select_eval_device(device: str):
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected_device == "auto":
        selected_device = "cpu"
    return torch.device(selected_device)


def load_checkpointed_model_for_eval(
    cfg: DensePre2Config,
    checkpoint_dir: str | Path,
    *,
    device: str = "auto",
) -> tuple[DenseTransformerLM, int]:
    """Load a pre-2 smoke checkpoint into an eval-mode PyTorch model."""
    torch_device = select_eval_device(device)
    model = build_dense_lm(cfg).to(torch_device)
    payload = load_checkpoint(checkpoint_dir, map_location=torch_device)
    restore_checkpoint_payload(
        payload,
        cfg=cfg,
        model=model,
        optimizer=None,
        restore_rng=False,
    )
    model.eval()
    return model, payload.step


def validate_prompt_ids(prompt_ids: list[int], *, vocab_size: int) -> None:
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    for token_id in prompt_ids:
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(f"prompt token id out of vocab: {token_id}")


def predict_next_token_id(model: DenseTransformerLM, prompt_ids: list[int]) -> int:
    """Greedy next-token prediction for token-id prompts."""
    vocab_size = model.cfg.model.tokenizer.planning_vocab_size()
    validate_prompt_ids(prompt_ids, vocab_size=vocab_size)
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids).logits[:, -1, :]
    return int(torch.argmax(logits, dim=-1).item())


def build_next_token_predict_fn(
    cfg: DensePre2Config,
    checkpoint_dir: str | Path,
    *,
    device: str = "auto",
) -> Callable[[list[int]], int]:
    """Build a reusable ``prompt_ids -> next_token_id`` predict function."""
    model, _ = load_checkpointed_model_for_eval(cfg, checkpoint_dir, device=device)

    def predict(prompt_ids: list[int]) -> int:
        return predict_next_token_id(model, prompt_ids)

    return predict


def _example_nll(logits: Tensor, target_id: int) -> float:
    target = torch.tensor([target_id], dtype=torch.long, device=logits.device)
    return float(F.cross_entropy(logits, target).detach().cpu())


def evaluate_next_token_ids(
    model: DenseTransformerLM,
    examples: Iterable[NextTokenEvalExample],
    *,
    checkpoint_step: int = 0,
) -> NextTokenEvalResult:
    """Run a tiny real next-token eval over token-id examples."""
    vocab_size = model.cfg.model.tokenizer.planning_vocab_size()
    device = next(model.parameters()).device
    predictions: list[NextTokenPrediction] = []

    with torch.inference_mode():
        for example in examples:
            validate_prompt_ids(example.prompt_ids, vocab_size=vocab_size)
            if example.target_id < 0 or example.target_id >= vocab_size:
                raise ValueError(f"target token id out of vocab: {example.target_id}")
            input_ids = torch.tensor([example.prompt_ids], dtype=torch.long, device=device)
            logits = model(input_ids).logits[:, -1, :]
            predicted_id = int(torch.argmax(logits, dim=-1).item())
            nll = _example_nll(logits, example.target_id)
            predictions.append(
                NextTokenPrediction(
                    prompt_ids=list(example.prompt_ids),
                    target_id=example.target_id,
                    predicted_id=predicted_id,
                    correct=predicted_id == example.target_id,
                    nll=nll,
                )
            )

    if not predictions:
        raise ValueError("at least one eval example is required")

    correct = sum(1 for prediction in predictions if prediction.correct)
    return NextTokenEvalResult(
        checkpoint_step=checkpoint_step,
        num_examples=len(predictions),
        accuracy=correct / len(predictions),
        average_nll=sum(prediction.nll for prediction in predictions) / len(predictions),
        predictions=predictions,
    )


def run_checkpoint_next_token_eval(
    cfg: DensePre2Config,
    checkpoint_dir: str | Path,
    examples: Iterable[NextTokenEvalExample],
    *,
    device: str = "auto",
) -> NextTokenEvalResult:
    """Load a checkpoint and evaluate next-token examples."""
    model, step = load_checkpointed_model_for_eval(cfg, checkpoint_dir, device=device)
    return evaluate_next_token_ids(model, examples, checkpoint_step=step)
