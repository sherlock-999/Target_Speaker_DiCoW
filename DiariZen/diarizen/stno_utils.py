import math
from typing import Dict

import torch


STNO_CLASS_NAMES = ("S", "T", "N", "O")


def derive_stno_from_binary(target_active: torch.Tensor, other_active: torch.Tensor) -> torch.Tensor:
    """Derive one-hot STNO states from binary target/other activity tensors."""
    target_active = target_active.float()
    other_active = other_active.float()

    s = (1.0 - target_active) * (1.0 - other_active)
    t = target_active * (1.0 - other_active)
    n = (1.0 - target_active) * other_active
    o = target_active * other_active
    return torch.stack([s, t, n, o], dim=-1)


def derive_stno_from_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    binary = (probs >= threshold).float()
    return derive_stno_from_binary(binary[..., 0], binary[..., 1])


def binary_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    pred = pred.bool()
    target = target.bool()

    tp = (pred & target).sum().float()
    fp = (pred & ~target).sum().float()
    fn = (~pred & target).sum().float()
    union = (pred | target).sum().float()

    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / torch.clamp(tp + fn, min=1.0)
    f1 = 2.0 * precision * recall / torch.clamp(precision + recall, min=1e-8)
    iou = tp / torch.clamp(union, min=1.0)
    support = target.sum().float()

    if support == 0 and pred.sum() == 0:
        precision = torch.tensor(1.0, device=target.device)
        recall = torch.tensor(1.0, device=target.device)
        f1 = torch.tensor(1.0, device=target.device)
        iou = torch.tensor(1.0, device=target.device)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def summarize_stno_metrics_from_binary(pred_binary: torch.Tensor, target_binary: torch.Tensor) -> Dict[str, torch.Tensor]:
    pred_binary = pred_binary.float()
    target_binary = target_binary.float()

    target_metrics = binary_metrics(pred_binary[..., 0], target_binary[..., 0])
    other_metrics = binary_metrics(pred_binary[..., 1], target_binary[..., 1])

    pred_stno = derive_stno_from_binary(pred_binary[..., 0], pred_binary[..., 1])
    target_stno = derive_stno_from_binary(target_binary[..., 0], target_binary[..., 1])

    out: Dict[str, torch.Tensor] = {
        "target_precision": target_metrics["precision"],
        "target_recall": target_metrics["recall"],
        "target_f1": target_metrics["f1"],
        "other_precision": other_metrics["precision"],
        "other_recall": other_metrics["recall"],
        "other_f1": other_metrics["f1"],
    }

    stno_f1 = []
    stno_iou = []
    for idx, name in enumerate(STNO_CLASS_NAMES):
        cls_metrics = binary_metrics(pred_stno[..., idx], target_stno[..., idx])
        out[f"stno_{name}_precision"] = cls_metrics["precision"]
        out[f"stno_{name}_recall"] = cls_metrics["recall"]
        out[f"stno_{name}_f1"] = cls_metrics["f1"]
        out[f"stno_{name}_iou"] = cls_metrics["iou"]
        stno_f1.append(cls_metrics["f1"])
        stno_iou.append(cls_metrics["iou"])

    out["stno_macro_f1"] = torch.stack(stno_f1).mean()
    out["stno_macro_f1_tno"] = torch.stack(stno_f1[1:]).mean()
    out["target_iou"] = out["stno_T_iou"]
    out["overlap_iou"] = out["stno_O_iou"]
    out["transition_error"] = transition_error_rate(pred_binary[..., 0], target_binary[..., 0])
    return out


def summarize_stno_metrics(logits: torch.Tensor, target_binary: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
    probs = torch.sigmoid(logits)
    pred_binary = (probs >= threshold).float()
    return summarize_stno_metrics_from_binary(pred_binary=pred_binary, target_binary=target_binary)


def transition_error_rate(pred_target: torch.Tensor, ref_target: torch.Tensor) -> torch.Tensor:
    pred_target = pred_target.bool()
    ref_target = ref_target.bool()
    if pred_target.ndim == 1:
        pred_target = pred_target.unsqueeze(0)
        ref_target = ref_target.unsqueeze(0)

    pred_edges = pred_target[:, 1:] ^ pred_target[:, :-1]
    ref_edges = ref_target[:, 1:] ^ ref_target[:, :-1]
    mismatched = (pred_edges ^ ref_edges).sum().float()
    total = torch.clamp(ref_edges.sum().float(), min=1.0)

    if ref_edges.sum() == 0 and pred_edges.sum() == 0:
        return torch.tensor(0.0, device=pred_target.device)

    return mismatched / total


def estimate_pos_weight(target_sum: int, total_count: int) -> float:
    positive = max(int(target_sum), 1)
    negative = max(int(total_count) - int(target_sum), 1)
    return float(negative) / float(positive)


def round_up_even(value: int) -> int:
    return int(math.ceil(value / 2.0) * 2)
