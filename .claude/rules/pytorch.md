---
description: 'PyTorch 2.x conventions: device-agnostic tensors, autograd hygiene, AMP and torch.compile usage, safe checkpoint loading, and deprecated 1.x idioms. Apply when writing or reviewing torch models, training loops, DataLoaders, or tensor code.'
paths:
- '**/*.py'
provenance: shared/rules/stack/pytorch.md @ 85dde8d
---

You are an expert in PyTorch 2.x. Training methodology is governed by the machine-learning rule; this rule covers the library surface.

## Principles

1. Code is device-agnostic: one `device` decided at the entry point and passed down, never `.cuda()` scattered through the code.
2. Autograd state is explicit: every tensor either needs gradients or provably does not.
3. Checkpoints are untrusted input.

## Directives

- Move modules and tensors with `.to(device)`; create tensors on-device (`torch.zeros(..., device=device)`) rather than creating on CPU and moving.
- Wrap evaluation and inference in `torch.inference_mode()` (or `torch.no_grad()` when tensors feed later autograd); detach before `.cpu().numpy()`.
- Load checkpoints with `torch.load(path, weights_only=True, map_location=...)`; save `state_dict()`s, never whole modules.
- Mixed precision via `torch.autocast(device_type=...)` and `torch.amp.GradScaler`; the `torch.cuda.amp` spellings are deprecated.
- Avoid per-step `.item()`/`.cpu()` synchronization in hot loops; accumulate on-device and synchronize once per logging interval.
- `DataLoader` with `num_workers > 0` and `pin_memory=True` for GPU input pipelines; make worker-side randomness derive from `worker_init_fn` or generators.
- Apply `torch.compile` to the model when the loop is stable and measure before keeping it; do not sprinkle it speculatively.
- Seed `torch.manual_seed` alongside NumPy and Python seeds; document nondeterministic ops when bitwise reproducibility is claimed.

## Anti-hallucination

| Banned | Correct |
|---|---|
| `torch.autograd.Variable` | tensors (autograd is built in) |
| `tensor.data` access | `tensor.detach()` |
| `torch.cuda.amp.autocast` / `GradScaler` | `torch.autocast` / `torch.amp.GradScaler` |
| `torch.load(path)` without `weights_only` | `torch.load(path, weights_only=True, map_location=...)` |
| `loss.backward(retain_graph=True)` as a fix-all | restructure the graph; retain only with a stated reason |
| `.cuda()` / `.cpu()` hardcoded in model code | `.to(device)` with one entry-point device decision |
