"""Qualitative side-by-side comparison of two signs (paper Figure ``compare'').

Renders evenly-spaced frames of two gesturally similar signs, one sign per row,
to illustrate why the encoder maps them close together and the few-shot
classifier confuses them. By default it reproduces the paper's example ---
sign 057 (*dance*) vs. sign 058 (*bathe*) --- overlaying the MediaPipe Holistic
skeleton (exactly the input the model sees) on the RGB frames.

Usage:
    # RGB frames + skeleton overlay (default), reproduces Figure ``compare''
    python -m src.evaluation.compare_signs --out runs/default/compare.png

    # different pair / sample
    python -m src.evaluation.compare_signs --signs 57 58 --names dance bathe \
        --signer 5 --rep 1 --n-frames 6 --out runs/default/compare.png

    # no video decode: draw skeletons only (works from released keypoints alone)
    python -m src.evaluation.compare_signs --skeleton-only \
        --out runs/default/compare_skeleton.png

Sign/keypoint files follow LSA64 naming SSS_PPP_RRR (sign_signer_repetition),
1-based on disk; --signs/--signer/--rep are given in that same 1-based form.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Keypoint layout produced by src/data/extract_keypoints.py:
#   0..32  pose (33)   33..53  left hand (21)   54..74  right hand (21)
N_POSE, N_HAND = 33, 21
POSE_OFF, LHAND_OFF, RHAND_OFF = 0, N_POSE, N_POSE + N_HAND

# Standard MediaPipe topologies (hardcoded so `mediapipe` is not needed to draw).
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
POSE_COLOR, LHAND_COLOR, RHAND_COLOR = "#00e5ff", "#ffd21e", "#7CFC00"


def stem(sign: int, signer: int, rep: int) -> str:
    return f"{sign:03d}_{signer:03d}_{rep:03d}"


def read_video_frames(path: Path) -> np.ndarray:
    """Return all frames of a video as (T, H, W, 3) RGB uint8."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames)


def even_indices(n_total: int, n_pick: int) -> np.ndarray:
    n_pick = min(n_pick, n_total)
    return np.linspace(0, n_total - 1, n_pick).round().astype(int)


def _present(pt: np.ndarray) -> bool:
    # MediaPipe zero-fills missing detections; treat all-zero as absent.
    return bool(np.any(np.abs(pt[:2]) > 1e-6))


def draw_skeleton(ax, kp_frame: np.ndarray, w: float, h: float,
                  lw: float = 1.4, ms: float = 2.2) -> None:
    """Draw the 75-keypoint skeleton. Coords are MediaPipe-normalized [0,1];
    multiply by (w, h) to place them on a frame, or pass w=h=1 for a bare plot.
    """
    groups = [
        (POSE_OFF, POSE_CONNECTIONS, POSE_COLOR),
        (LHAND_OFF, HAND_CONNECTIONS, LHAND_COLOR),
        (RHAND_OFF, HAND_CONNECTIONS, RHAND_COLOR),
    ]
    for off, conns, color in groups:
        for i, j in conns:
            a, b = kp_frame[off + i], kp_frame[off + j]
            if _present(a) and _present(b):
                ax.plot([a[0] * w, b[0] * w], [a[1] * h, b[1] * h],
                        color=color, lw=lw, solid_capstyle="round", zorder=3)
        for idx in range(off, off + len(conns) // 2 + 1):
            p = kp_frame[idx] if idx < len(kp_frame) else None
            if p is not None and _present(p):
                ax.plot(p[0] * w, p[1] * h, "o", color=color, ms=ms, zorder=4)


def load_keypoints(kp_dir: Path, s: str) -> np.ndarray | None:
    f = kp_dir / f"{s}.npy"
    return np.load(f) if f.exists() else None


def build_row(sign, signer, rep, name, videos_dir, kp_dir, n_frames,
              skeleton_only, overlay):
    """Return (label, [(frame_or_None, kp_frame_or_None), ...]) for one sign."""
    s = stem(sign, signer, rep)
    kp = load_keypoints(kp_dir, s) if (overlay or skeleton_only) else None

    if skeleton_only:
        if kp is None:
            raise FileNotFoundError(
                f"--skeleton-only needs {kp_dir/(s + '.npy')} (run extract_keypoints)."
            )
        idx = even_indices(len(kp), n_frames)
        cells = [(None, kp[i]) for i in idx]
    else:
        vpath = videos_dir / f"{s}.mp4"
        if not vpath.exists():
            for ext in (".avi", ".mov"):
                if (videos_dir / f"{s}{ext}").exists():
                    vpath = videos_dir / f"{s}{ext}"
                    break
        if not vpath.exists():
            raise FileNotFoundError(
                f"Video {vpath} not found. Unpack LSA64 into {videos_dir}, "
                f"or use --skeleton-only to render from keypoints alone."
            )
        frames = read_video_frames(vpath)
        n = len(frames) if kp is None else min(len(frames), len(kp))
        idx = even_indices(n, n_frames)
        cells = [(frames[i], (kp[i] if kp is not None else None)) for i in idx]

    return f"{sign:03d} — {name}", cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signs", type=int, nargs=2, default=[57, 58],
                    help="two 1-based sign ids (default: 57 58 = dance, bathe)")
    ap.add_argument("--names", nargs=2, default=["dance", "bathe"])
    ap.add_argument("--signer", type=int, default=1)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--videos-dir", type=Path, default=Path("data/lsa64/videos"))
    ap.add_argument("--keypoints-dir", type=Path,
                    default=Path("data/lsa64/keypoints"))
    ap.add_argument("--no-skeleton", action="store_true",
                    help="RGB frames only, without the skeleton overlay")
    ap.add_argument("--skeleton-only", action="store_true",
                    help="draw skeletons on a blank canvas (no video needed)")
    ap.add_argument("--out", type=Path, default=Path("runs/default/compare.png"))
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    overlay = not args.no_skeleton and not args.skeleton_only

    rows = [
        build_row(sign, args.signer, args.rep, name, args.videos_dir,
                  args.keypoints_dir, args.n_frames, args.skeleton_only, overlay)
        for sign, name in zip(args.signs, args.names)
    ]
    ncols = max(len(cells) for _, cells in rows)

    fig, axes = plt.subplots(2, ncols, figsize=(2.3 * ncols, 5.2),
                             squeeze=False)
    for r, (label, cells) in enumerate(rows):
        for c in range(ncols):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c >= len(cells):
                ax.axis("off")
                continue
            frame, kp_frame = cells[c]
            if frame is not None:
                h, w = frame.shape[:2]
                ax.imshow(frame)
                if kp_frame is not None:
                    draw_skeleton(ax, kp_frame, w, h)
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)
            else:  # skeleton-only: normalized coords, image y points down
                draw_skeleton(ax, kp_frame, 1.0, 1.0, lw=1.6, ms=3.0)
                ax.set_xlim(0, 1)
                ax.set_ylim(1, 0)
                ax.set_aspect("equal")
                ax.set_facecolor("white")
            if c == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
