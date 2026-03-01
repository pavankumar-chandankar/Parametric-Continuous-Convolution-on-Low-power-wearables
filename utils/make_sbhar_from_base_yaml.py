import argparse
from pathlib import Path
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_cfg", type=str, required=True, help="Your existing WEAR yaml to clone.")
    ap.add_argument("--anno_dir", type=str, required=True, help="data/sbhar/annotations/50hz")
    ap.add_argument("--out_cfg", type=str, required=True, help="Output yaml path.")
    ap.add_argument("--sens_folder", type=str, required=True, help="data/sbhar/raw/inertial")
    args = ap.parse_args()

    base = yaml.safe_load(Path(args.base_cfg).read_text(encoding="utf-8"))

    anno_dir = Path(args.anno_dir)
    annos = sorted(str(p) for p in anno_dir.glob("loso_sbj_*.json"))
    if not annos:
        raise SystemExit(f"No loso_sbj_*.json in: {anno_dir}")

    base["anno_json"] = annos
    base["has_null"] = False

    # dataset block
    base.setdefault("dataset", {})
    base["dataset"]["sens_folder"] = args.sens_folder
    base["dataset"]["window_size"] = 128
    base["dataset"]["window_overlap"] = 50
    base["dataset"]["channels"] = 9

    # network block (your code sometimes uses net_cfg)
    base.setdefault("net_cfg", {})
    base["net_cfg"]["window_size"] = 128
    base["net_cfg"]["channels"] = 9
    base["net_cfg"]["classes"] = 6

    Path(args.out_cfg).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_cfg).write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    print(f"[OK] wrote: {args.out_cfg}")
    print(f"[OK] anno_json entries: {len(annos)}")
    print(f"[OK] sens_folder: {args.sens_folder}")


if __name__ == "__main__":
    main()
