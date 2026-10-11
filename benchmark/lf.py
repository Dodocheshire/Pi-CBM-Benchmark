"""Benchmark adapters using the frozen LF-Pi training and inference code.

No model implementation is duplicated here. Inputs are aligned sample caches;
CLIP cosine teachers are distinct from semantic annotation targets.
"""
from __future__ import annotations

import copy
import json
import io
import contextlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def upstream(root=None):
    root = Path(root or Path(__file__).resolve().parents[1] / "methods/LF-Pi")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def default_nec_grid(concepts, classes):
    """Dataset-sensitive per-class budgets; never exceed the concept bank."""
    candidates = ([1, 2, 5, 10, 15, 20, 30, 50, 75, 100]
                  if classes <= 20 else [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200])
    return sorted(set(min(concepts, x) for x in candidates))


def head_nec(weight, threshold=1e-5):
    return float((weight.abs() > threshold).sum()) / weight.shape[0]


def exact_budget(weight, target, threshold=1e-5):
    """VLG-style global smallest-weight pruning, without adding nonzero weights."""
    budget = int(round(weight.shape[0] * target))
    available = int((weight.abs() > threshold).sum())
    used = min(budget, available)
    result = torch.zeros_like(weight)
    if used:
        # Stable ordering makes tied weights deterministic.
        order = torch.argsort(weight.abs().flatten(), descending=True, stable=True)[:used]
        result.flatten()[order] = weight.flatten()[order]
    return result, {"requested_nonzero": budget, "achieved_nonzero": used,
                    "exact_budget_reached": used == budget,
                    "achieved_nec": head_nec(result, threshold)}


@torch.no_grad()
def proximal_residual(x, y, weight, bias, lam, alpha, lr=.1, batch_size=2048):
    """Full training-gradient proximal mapping norm for convergence auditing."""
    gw, gb = torch.zeros_like(weight), torch.zeros_like(bias)
    for xb, yb in zip(x.split(batch_size), y.split(batch_size)):
        errors = torch.nn.functional.linear(xb, weight, bias).softmax(-1)
        errors[torch.arange(len(yb)), yb] -= 1
        gw += errors.T @ xb
        gb += errors.sum(0)
    gw /= len(y); gb /= len(y)
    updated = weight - lr * gw
    updated = (updated.sign() * (updated.abs() - lr * lam * alpha).clamp_min(0)) / (1 + lr * lam * (1-alpha))
    delta = (updated - weight) / lr
    return {'proximal_gradient_l2': float(torch.sqrt(delta.square().sum() + gb.square().sum())),
            'proximal_gradient_max_abs': float(torch.maximum(delta.abs().max(), gb.abs().max())),
            'gradient_audit': 'full train split; unregularized bias; elastic-net proximal mapping'}


def fit_nec_path(train_c, train_y, val_c, val_y, classes, *, device="cpu",
                 seed=0, steps=300, batch_size=256, lr=0.1, alpha=0.99,
                 points=24, epsilon=0.0001, targets=None, root=None,
                 saga_gradient_kernel='upstream'):
    """Train a warm-start upstream glm_saga path using train and val only.

    The path starts at the empirical maximum regularization. Validation NLL
    (without the training penalty) chooses the unconstrained head. For each
    NEC budget, eligible paths with enough active weights are globally pruned;
    among the nearest raw-NEC candidates, val NLL resolves ties. Test inputs are
    intentionally absent from this API.
    """
    upstream(root)
    from pi_cbm.vendor.lfcbm.elasticnet import IndexedTensorDataset, glm_saga
    from pi_cbm.vendor.lfcbm import elasticnet
    from .saga_kernel import gradient_kernel
    torch.manual_seed(seed)
    train_c, val_c = train_c.detach().float().cpu(), val_c.detach().float().cpu()
    train_y, val_y = train_y.long().cpu(), val_y.long().cpu()
    model = nn.Linear(train_c.shape[1], classes).to(device)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    loader = DataLoader(IndexedTensorDataset(train_c, train_y), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    validation = DataLoader(TensorDataset(val_c, val_y), batch_size=batch_size)
    class SolverLog(io.StringIO):
        def write(self, value):
            sys.__stdout__.write(value)
            sys.__stdout__.flush()
            return super().write(value)
    log = SolverLog()
    with contextlib.redirect_stdout(log), gradient_kernel(elasticnet, saga_gradient_kernel) as kernel_metadata:
        output = glm_saga(model, loader, lr, steps, alpha, epsilon=epsilon, k=points,
                          do_zero=False, val_loader=validation, n_ex=len(train_y),
                          n_classes=classes, lookbehind=5)
    records, heads = [], []
    for p in output["path"]:
        w, b = p["weight"], p["bias"]
        logits = torch.nn.functional.linear(val_c, w, b)
        nll = torch.nn.functional.cross_entropy(logits, val_y).item()
        records.append({"lambda": float(p["lam"]), "raw_nec": head_nec(w),
                        "val_nll": nll, "val_accuracy": float((logits.argmax(1) == val_y).float().mean()),
                        "time_s": p["time"],
                        **proximal_residual(train_c, train_y, w, b, float(p['lam']), alpha, lr)})
        heads.append({"weight": w, "bias": b})
    best = min(range(len(records)), key=lambda i: records[i]["val_nll"])
    targets = targets or default_nec_grid(train_c.shape[1], classes)
    budgets, budget_heads = [], {}
    for target in targets:
        candidates = [i for i, r in enumerate(records) if r["raw_nec"] >= target]
        if not candidates:
            candidates = [max(range(len(records)), key=lambda i: records[i]["raw_nec"])]
        # Choose the closest raw NEC above target; val NLL breaks numerical ties.
        selected = min(candidates, key=lambda i: (abs(records[i]["raw_nec"] - target), records[i]["val_nll"]))
        w, detail = exact_budget(heads[selected]["weight"], target)
        b = heads[selected]["bias"]
        logits = torch.nn.functional.linear(val_c, w, b)
        budgets.append({"target_nec": target, "path_index": selected,
                        "lambda": records[selected]["lambda"],
                        "raw_nec": records[selected]["raw_nec"], **detail,
                        "val_nll": torch.nn.functional.cross_entropy(logits, val_y).item(),
                        "val_accuracy": float((logits.argmax(1) == val_y).float().mean())})
        budget_heads[str(target)] = {"weight": w, "bias": b}
    return {"solver": "frozen_upstream_glm_saga", "selection_split": "val",
            "saga_gradient_kernel": kernel_metadata,
            "test_selection": False, "seed": seed, "steps_limit": steps,
            "path_points": points, "epsilon": epsilon, "alpha": alpha,
            "solver_tolerance": 1e-4, "lookbehind": 5,
            "unconverged_path_points": log.getvalue().count("did not converge"),
            "convergence_evidence": log.getvalue().splitlines(),
            "train_examples": len(train_y), "val_examples": len(val_y),
            "concept_count": train_c.shape[1], "classes": classes,
            "path": records, "best_val_index": best, "budgets": budgets}, {
                "path": heads, "best": heads[best], "budgets": budget_heads}


def fit_common(data, config, *, device="cpu", root=None, adapt_pi=False):
    """Fit LF-CBM and optionally LF-Pi through the original public functions.

    data contains splits with features, teacher, labels and indices plus
    classes, concepts and CLIP text embeddings. A held-out val split must exist.
    """
    upstream(root)
    from pi_cbm.training.lfcbm import fit_lfcbm
    from pi_cbm.pipeline import build_model
    from pi_cbm.training.adapt import adapt
    from pi_cbm.randomness import seed_all
    config = copy.deepcopy(config)
    config["baseline"]["include_adapter_val"] = False
    config["baseline"]["validation_split"] = "val"
    seed_all(config["experiment"]["seed"])
    cbm, selected, metrics = fit_lfcbm(data, config, device)
    if not adapt_pi:
        config["injection"]["site"] = "baseline"
    model = build_model(cbm, data, selected, config, device)
    if adapt_pi:
        with torch.no_grad():
            clean = {name: torch.cat([cbm.concepts(b.to(device)).cpu()
                     for b in split["features"].split(256)])
                     for name, split in data["splits"].items()}
        history, refit = adapt(model, data, clean, selected, config, device)
        metrics.update(adapter_history=history, adapter_refit=refit)
    return model.eval(), selected, metrics


@torch.no_grad()
def predict_features(model, features, *, draws=1, seed=0, batch_size=256):
    """Standard concepts/logits export via original inference; supports global sites."""
    from pi_cbm.evaluation.inference import predict
    from pi_cbm.randomness import seeded
    device = next(model.parameters()).device
    all_p, all_c = [], []
    with seeded(seed):
        for batch in features.split(batch_size):
            predictions = [predict(model, batch.to(device)) for _ in range(draws)]
            all_p.append(torch.stack([p.cpu() for p, c in predictions]).mean(0))
            all_c.append(torch.stack([c.cpu() for p, c in predictions]).mean(0))
    p, c = torch.cat(all_p), torch.cat(all_c)
    # Log probabilities are valid classification logits and preserve MC averaging.
    return {"probabilities": p, "logits": p.clamp_min(1e-12).log(), "concepts": c}


def export_npz(path, prediction, split, concept_names, *, targets=None, known=None):
    values = {k: v.detach().cpu().numpy() for k, v in prediction.items()}
    values.update(labels=split["labels"].numpy(), features=split["features"].numpy(),
                  ids=split["indices"].numpy(), concept_names=np.asarray(concept_names))
    if targets is not None:
        values["targets"] = np.asarray(targets)
        values["known"] = np.asarray(known if known is not None else np.isfinite(targets))
    else:
        # No semantic annotations are available; unknown entries cannot be scored.
        values["targets"] = np.zeros_like(values["concepts"])
        values["known"] = np.zeros_like(values["concepts"], dtype=bool)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)
