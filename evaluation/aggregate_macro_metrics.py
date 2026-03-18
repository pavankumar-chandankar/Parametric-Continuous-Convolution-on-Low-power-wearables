from __future__ import annotations
import argparse, glob, os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Folder containing best_macro_metrics_*.csv (recursive)")
    ap.add_argument("--out_csv", type=str, default="", help="Output CSV path (default: <root>/macro_summary.csv)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "**", "best_macro_metrics_*.csv"), recursive=True))
    if not files:
        raise SystemExit(f"No best_macro_metrics_*.csv found under {args.root}")

    dfs = []
    for p in files:
        df = pd.read_csv(p)
        if df.empty:
            continue
        df["source_path"] = p
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # normalize columns
    if "freq_tag" not in all_df.columns and "eval_tag" in all_df.columns:
        all_df = all_df.rename(columns={"eval_tag": "freq_tag"})
    if "freq_tag" not in all_df.columns:
        raise SystemExit(f"Missing freq_tag/eval_tag in columns: {list(all_df.columns)}")

    for c in ["acc_mean", "prec_mean", "rec_mean", "f1_mean"]:
        if c not in all_df.columns:
            raise SystemExit(f"Missing {c} in columns: {list(all_df.columns)}")

    group_cols = ["freq_tag"]
    for c in ["dataset_name", "conv_type", "gamma_quant", "quant_bits", "conv_kernel_size", "standard_padding", "kernel_support_s"]:
        if c in all_df.columns:
            group_cols.append(c)

    agg = (
        all_df.groupby(group_cols)[["acc_mean", "prec_mean", "rec_mean", "f1_mean"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = ["_".join([x for x in col if x]).rstrip("_") if isinstance(col, tuple) else col for col in agg.columns]

    out_csv = args.out_csv.strip() or os.path.join(args.root, "macro_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    agg.to_csv(out_csv, index=False)

    # Print a compact view
    show = [c for c in agg.columns if c in ("freq_tag", "dataset_name", "conv_type", "f1_mean_mean", "f1_mean_std", "f1_mean_count")]
    if show:
        print(agg[show].sort_values(["conv_type"] if "conv_type" in show else ["freq_tag"]).to_string(index=False))
    print(f"[OK] wrote {out_csv}")

if __name__ == "__main__":
    main()
