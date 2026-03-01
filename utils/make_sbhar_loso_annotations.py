import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_map", type=str, required=True, help="label_map.json from processed/windows_50hz.")
    ap.add_argument("--sens_folder", type=str, required=True, help="Folder containing sbj_<id>.csv files.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder for loso_sbj_<k>.json.")
    args = ap.parse_args()

    label_map_path = Path(args.label_map)
    sens_folder = Path(args.sens_folder)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # label_map.json is like {"0":"WALKING", ...}; convert to {"WALKING":0,...}
    lm = json.loads(label_map_path.read_text(encoding="utf-8"))
    label_dict = {v: int(k) for k, v in lm.items()}

    subjects = sorted([p.stem for p in sens_folder.glob("sbj_*.csv")])
    if not subjects:
        raise SystemExit(f"No sbj_*.csv found in: {sens_folder}")

    for heldout in subjects:
        db = {}
        for s in subjects:
            db[s] = {"subset": "Validation" if s == heldout else "Training"}

        anno = {
            "database": db,
            "label_dict": label_dict,
        }

        out_path = out_dir / f"loso_{heldout}.json"   # e.g., loso_sbj_1.json
        out_path.write_text(json.dumps(anno, indent=2), encoding="utf-8")

    print(f"[OK] wrote {len(subjects)} LOSO annotation JSONs to: {out_dir}")
    print(f"[OK] example: {out_dir / ('loso_' + subjects[0] + '.json')}")


if __name__ == "__main__":
    main()
