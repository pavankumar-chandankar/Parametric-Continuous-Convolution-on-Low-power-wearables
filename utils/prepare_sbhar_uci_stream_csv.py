import argparse
from pathlib import Path

import numpy as np


def npz_to_stream(x: np.ndarray, y: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    """
    x: (Nwin, T, C), y: (Nwin,)
    Reconstruct a continuous stream by overlap-write:
      stream_x[i*step : i*step+T] = x[i]
      stream_y[i*step : i*step+T] = y[i]
    This guarantees that for windows extracted with the same (T, step), the label at each window end matches y[i].
    """
    assert x.ndim == 3, x.shape
    nwin, T, C = x.shape
    assert y.shape[0] == nwin, (y.shape, nwin)

    L = step * (nwin - 1) + T
    stream_x = np.zeros((L, C), dtype=np.float32)
    stream_y = np.zeros((L,), dtype=np.int64)

    for i in range(nwin):
        s = i * step
        stream_x[s : s + T] = x[i]
        stream_y[s : s + T] = y[i]

    return stream_x, stream_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_dir", type=str, required=True, help="Folder with sbj_<id>.npz (x,y).")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder for sbj_<id>.csv streams.")
    ap.add_argument("--step", type=int, default=64, help="Step size between windows (SBHAR uses 64).")
    args = ap.parse_args()

    win_dir = Path(args.windows_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npzs = sorted(win_dir.glob("sbj_*.npz"))
    if not npzs:
        raise SystemExit(f"No sbj_*.npz found in: {win_dir}")

    for p in npzs:
        obj = np.load(p, allow_pickle=False)
        x = obj["x"].astype(np.float32)  # (Nwin,T,C)
        y = obj["y"].astype(np.int64)    # (Nwin,)

        stream_x, stream_y = npz_to_stream(x, y, step=args.step)

        # CSV columns: t, f0..f{C-1}, label
        t = np.arange(stream_x.shape[0], dtype=np.int64).reshape(-1, 1)
        data = np.concatenate([t, stream_x, stream_y.reshape(-1, 1)], axis=1)

        header = ["t"] + [f"f{i}" for i in range(stream_x.shape[1])] + ["label"]
        out_csv = out_dir / (p.stem + ".csv")  # sbj_<id>.csv
        np.savetxt(out_csv, data, delimiter=",", header=",".join(header), comments="", fmt="%.6f")
        print(f"[OK] {out_csv}  stream_len={data.shape[0]}  C={stream_x.shape[1]}")

    print(f"\n[DONE] Wrote {len(npzs)} per-subject CSV streams to: {out_dir}")


if __name__ == "__main__":
    main()
