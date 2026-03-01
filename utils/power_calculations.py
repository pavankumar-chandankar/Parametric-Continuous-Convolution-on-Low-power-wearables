from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict


DEFAULT_USAGE_COUNTS = {
    "12hz": 2327,
    "25hz": 9176,
    "6hz": 7233,
}


def _parse_kv_csv(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    s = (s or "").strip()
    if not s:
        return out

    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            raise ValueError(f"Bad item '{p}'. Expected format key:value (e.g., 25hz:3.2).")
        k, v = p.split(":", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Bad item '{p}': empty key.")
        out[k] = float(v)
    return out


def _counts_to_fracs(counts: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(counts.values()))
    if total <= 0:
        raise ValueError("Total usage count must be > 0.")
    return {k: float(v) / total for k, v in counts.items()}


@dataclass(frozen=True)
class EnergyReport:
    expected_energy_mj_per_window: float
    baseline_energy_mj_per_window: float
    relative_savings_vs_baseline: float 
    expected_power_w: float | None
    baseline_power_w: float | None


def estimate_gated_energy(
    usage_fracs: Dict[str, float],
    energy_mj: Dict[str, float],
    baseline_freq: str = "25hz",
    window_rate_hz: float | None = None,
) -> EnergyReport:
    p_sum = sum(usage_fracs.values())
    if abs(p_sum - 1.0) > 1e-3:
        raise ValueError(f"usage_fracs must sum to 1.0 (got {p_sum:.6f}).")

    missing = [f for f, p in usage_fracs.items() if p > 0 and f not in energy_mj]
    if missing:
        raise ValueError(
            "Missing energy_mj entries for: " + ", ".join(missing) +
            ". Provide --energy_mj like '25hz:...,12hz:...,6hz:...'."
        )

    if baseline_freq not in energy_mj:
        raise ValueError(f"baseline_freq '{baseline_freq}' missing in energy_mj.")

    e_avg = 0.0
    for f, p in usage_fracs.items():
        e_avg += p * float(energy_mj[f])

    e_base = float(energy_mj[baseline_freq])

    savings = 1.0 - (e_avg / e_base)

    p_avg_w = None
    p_base_w = None
    if window_rate_hz is not None:
        if window_rate_hz <= 0:
            raise ValueError("--window_rate_hz must be > 0 if provided.")
        p_avg_w = (e_avg * window_rate_hz) / 1000.0
        p_base_w = (e_base * window_rate_hz) / 1000.0

    return EnergyReport(
        expected_energy_mj_per_window=e_avg,
        baseline_energy_mj_per_window=e_base,
        relative_savings_vs_baseline=savings,
        expected_power_w=p_avg_w,
        baseline_power_w=p_base_w,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--usage_counts",
        type=str,
        default="",
        help="Optional. Usage counts like '25hz:9176,12hz:2327,6hz:7233'. If omitted, uses built-in defaults.",
    )
    ap.add_argument(
        "--usage_fracs",
        type=str,
        default="",
        help="Optional alternative to counts. Fractions like '25hz:0.49,12hz:0.124,6hz:0.386'. Must sum to 1.0.",
    )
    ap.add_argument(
        "--energy_mj",
        type=str,
        required=True,
        help="Energy per inference (per window) in mJ. Example: '25hz:3.2,12hz:1.9,6hz:1.1'",
    )
    ap.add_argument(
        "--baseline_freq",
        type=str,
        default="25hz",
        help="Baseline frequency for savings comparison (default: 25hz).",
    )
    ap.add_argument(
        "--window_rate_hz",
        type=float,
        default=None,
        help="Optional windows/sec to convert energy -> average power. Example: if stride=0.5s, rate=2.",
    )
    args = ap.parse_args()

    energy_mj = _parse_kv_csv(args.energy_mj)

    if args.usage_fracs.strip():
        usage_fracs = _parse_kv_csv(args.usage_fracs)
    else:
        if args.usage_counts.strip():
            counts = _parse_kv_csv(args.usage_counts)
        else:
            counts = {k: float(v) for k, v in DEFAULT_USAGE_COUNTS.items()}
        usage_fracs = _counts_to_fracs(counts)

    print("Usage fractions (gated, one model per window):")
    for k in sorted(usage_fracs.keys(), key=lambda x: ("25hz" not in x, x)):
        print(f"  {k}: {usage_fracs[k]*100.0:6.2f}%")

    rep = estimate_gated_energy(
        usage_fracs=usage_fracs,
        energy_mj=energy_mj,
        baseline_freq=args.baseline_freq,
        window_rate_hz=args.window_rate_hz,
    )

    print("\nEnergy per window:")
    print(f"  Expected (gated): {rep.expected_energy_mj_per_window:.6f} mJ/window")
    print(f"  Baseline ({args.baseline_freq} only): {rep.baseline_energy_mj_per_window:.6f} mJ/window")

    s_pct = rep.relative_savings_vs_baseline * 100.0

    if rep.expected_power_w is not None and rep.baseline_power_w is not None:
        print("\nAverage power (requires --window_rate_hz):")
        print(f"  Expected (gated): {rep.expected_power_w:.6f} W")
        print(f"  Baseline:         {rep.baseline_power_w:.6f} W")
        if rep.baseline_power_w > 0:
            print(f"  Relative savings: {(1.0 - rep.expected_power_w/rep.baseline_power_w)*100.0:+.2f}%")

if __name__ == "__main__":
    main()