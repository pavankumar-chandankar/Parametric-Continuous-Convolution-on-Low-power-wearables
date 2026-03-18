from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import json
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE:
# This repository uses implicit namespace packages (no __init__.py).
# Therefore, intra-package imports must be explicit.
from models.DeepConvLSTM_final import DeepConvLSTM


FREQ_FOR_CLASS: List[str] = [
    "6hz",   # 0: null
    "25hz",  # 1: benchdips
    "25hz",  # 2: burpees
    "6hz",   # 3: jogging
    "25hz",  # 4: jogging (buttkicks)
    "25hz",  # 5: jogging (rotating arms)
    "25hz",  # 6: jogging (sidesteps)
    "25hz",  # 7: joggging (skipping)
    "12hz",  # 8: lunges
    "12hz",  # 9: lunges (complex)
    "25hz",  # 10: pushups
    "25hz",  # 11: pushups(complex)
    "12hz",  # 12: situps
    "12hz",  # 13: situps (complx)
    "6hz",   # 14: stretching (hamstrings)
    "6hz",   # 15: stretching (lumbar rotation)
    "6hz",   # 16: stretching (lunging)
    "6hz",   # 17: stretching (shoulders)
    "6hz",   # 18: stretching (triceps)
]


def load_freq_mapping(path: str) -> List[str]:
    """
    Load a class->frequency mapping.

    Supported formats:
      - JSON:
          * list of strings, where index = class_id
          * dict mapping class_id (int/str) -> freq_tag
      - CSV:
          must contain columns: class_id and freq_tag (or best_freq_tag / best_frequency)

    Returns:
      list[str] of length (max_class_id+1)
    """
    if path is None or str(path).strip() == "":
        raise ValueError("path must be a non-empty string")
    path = str(path)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Frequency mapping file not found: {path}")

    if path.lower().endswith(".json"):
        with open(path, "r") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict):
            pairs = []
            for k, v in obj.items():
                try:
                    ki = int(k)
                except Exception:
                    continue
                pairs.append((ki, str(v)))
            if not pairs:
                raise ValueError(f"No valid class_id entries found in JSON mapping: {path}")
            max_k = max(k for k, _ in pairs)
            out = ["25hz"] * (max_k + 1)
            for k, v in pairs:
                out[k] = v
            return out
        raise ValueError(f"Unsupported JSON mapping structure in {path}: {type(obj)}")

    df = pd.read_csv(path)
    if "class_id" not in df.columns:
        raise ValueError(f"CSV mapping missing 'class_id' column: {path}")

    tag_col = None
    for cand in ("freq_tag", "best_freq_tag", "best_frequency", "best_freq"):
        if cand in df.columns:
            tag_col = cand
            break
    if tag_col is None:
        raise ValueError(
            f"CSV mapping must contain one of: freq_tag, best_freq_tag, best_frequency, best_freq. Got: {list(df.columns)}"
        )

    df = df[["class_id", tag_col]].copy()
    df["class_id"] = df["class_id"].astype(int)
    df[tag_col] = df[tag_col].astype(str)

    max_id = int(df["class_id"].max())
    out = ["25hz"] * (max_id + 1)
    for _, r in df.iterrows():
        out[int(r["class_id"])] = str(r[tag_col])
    return out


class MultiFreqDeepConvLSTM(nn.Module):
    def __init__(
        self,
        model_6hz: nn.Module,
        model_12hz: nn.Module,
        model_25hz: nn.Module,
        model_50hz: Optional[nn.Module] = None,
        freq_for_class: Optional[Sequence[str]] = None,
        freq_mapping_path: Optional[str] = None,
        history_len: int = 5,
        stability_threshold: float = 0.8,
        min_confidence: float = 0.6,
        default_freq: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.model_6hz = model_6hz
        self.model_12hz = model_12hz
        self.model_25hz = model_25hz
        self.model_50hz = model_50hz

        if freq_mapping_path is not None and str(freq_mapping_path).strip():
            self.freq_for_class = load_freq_mapping(freq_mapping_path)
        else:
            self.freq_for_class = list(freq_for_class) if freq_for_class is not None else list(FREQ_FOR_CLASS)

        self.history_len = int(history_len)
        self.stability_threshold = float(stability_threshold)
        self.min_confidence = float(min_confidence)

        self.available_models: Dict[str, Optional[nn.Module]] = {
            "6hz": self.model_6hz,
            "12hz": self.model_12hz,
            "25hz": self.model_25hz,
            "50hz": self.model_50hz,
        }

        if default_freq is not None:
            self.default_freq = str(default_freq)
        else:
            self.default_freq = "50hz" if self.model_50hz is not None else "25hz"

        self.reset_history()

    def reset_history(self) -> None:
        self.history_classes: List[int] = []
        self.history_confidences: List[float] = []

    def _update_history(self, classes, confidences) -> None:
        for cls, conf in zip(classes, confidences):
            self.history_classes.append(int(cls))
            self.history_confidences.append(float(conf))

            if len(self.history_classes) > self.history_len:
                self.history_classes.pop(0)
            if len(self.history_confidences) > self.history_len:
                self.history_confidences.pop(0)

    @staticmethod
    def _softmax_confidence(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits, dim=1)
        conf, _ = probs.max(dim=1)
        return probs, conf

    def _choose_frequency_from_history(self) -> str:
        if len(self.history_classes) == 0:
            return self.default_freq

        classes = torch.tensor(self.history_classes, dtype=torch.long)
        majority_class = int(torch.mode(classes).values.item())
        stability = (classes == majority_class).float().mean().item()

        if self.history_confidences:
            avg_conf = float(sum(self.history_confidences) / len(self.history_confidences))
        else:
            avg_conf = 0.0

        if stability >= self.stability_threshold and avg_conf >= self.min_confidence:
            if 0 <= majority_class < len(self.freq_for_class):
                cand = self.freq_for_class[majority_class]
                if cand in self.available_models and self.available_models[cand] is not None:
                    return cand
                return self.default_freq
            return self.default_freq

        return self.default_freq

    def step(
        self,
        x_25hz: torch.Tensor,
        x_12hz: torch.Tensor,
        x_6hz: torch.Tensor,
        x_50hz: Optional[torch.Tensor] = None,
    ):
        assert x_25hz.size(0) == x_12hz.size(0) == x_6hz.size(0), "Batch sizes must match."
        if self.model_50hz is not None:
            if x_50hz is None:
                raise ValueError("x_50hz must be provided when model_50hz is not None")
            assert x_50hz.size(0) == x_25hz.size(0), "Batch sizes must match (50hz)."

        device = next(self.parameters()).device
        x_25hz = x_25hz.to(device)
        x_12hz = x_12hz.to(device)
        x_6hz = x_6hz.to(device)
        if x_50hz is not None:
            x_50hz = x_50hz.to(device)

        batch_size = x_25hz.size(0)
        all_logits = []
        chosen_freqs: List[str] = []

        for i in range(batch_size):
            freq = self._choose_frequency_from_history()

            if freq == "6hz":
                logits_i = self.model_6hz(x_6hz[i : i + 1])
            elif freq == "12hz":
                logits_i = self.model_12hz(x_12hz[i : i + 1])
            elif freq == "50hz":
                assert self.model_50hz is not None
                assert x_50hz is not None
                logits_i = self.model_50hz(x_50hz[i : i + 1])
            else:
                logits_i = self.model_25hz(x_25hz[i : i + 1])

            _, conf_i = self._softmax_confidence(logits_i)
            pred_i = torch.argmax(logits_i, dim=1)

            self._update_history(pred_i.cpu().tolist(), conf_i.cpu().tolist())

            all_logits.append(logits_i)
            chosen_freqs.append(freq)

        return torch.cat(all_logits, dim=0), chosen_freqs
