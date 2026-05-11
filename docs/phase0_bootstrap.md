# Phase 0 — Bootstrap runbook

Goal: stand up the orchestration VM and prove the training stack works at toy
scale on RunPod, with no surprises before we start spending real money in
Phase 1.

## Checklist

- [ ] Plan reviewed; PLAN.md §14 open questions answered
- [ ] Credentials present in env (or `.env`):
  - `HF_TOKEN`
  - `RUNPOD_API_KEY`
  - `WANDB_API_KEY` (or MLflow if user picks self-host)
  - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `S3_ENDPOINT_URL` / `S3_BUCKET`
- [ ] Python venv created, `pip install -r requirements.txt` succeeds
- [ ] `python scripts/smoke_test_env.py` passes (CPU)
- [ ] `pytest -q` passes
- [ ] RunPod CLI smoke: `python scripts/launch_runpod_pod.py smoke --sku 1xA10`
  - launches a 1×A10 pod
  - SSHes in, runs `nvidia-smi`, prints driver/CUDA versions
  - terminates pod
  - cost target: < $1
- [ ] Keras-on-JAX smoke on H100: train a 30M-param toy LLM for 100 steps on
      synthetic data, verify loss decreases. Cost target: < $5.
- [ ] Object storage round-trip: write a 1MB blob from the pod, read back from
      the orchestration VM, verify checksum.
- [ ] Tracker decision recorded (W&B vs MLflow); first dummy run logged.

## Cost ceiling

Hard cap for Phase 0: **$50**. If we exceed this, stop and ask why.

## Exit criteria

All checklist items green, no open infra blockers, every credential rotation
path documented.
