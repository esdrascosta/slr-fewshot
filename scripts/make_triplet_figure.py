"""Build the triplet-learning diagram (anchor/positive/negative) for LSA64.

Produces figures/triplet_lsa64.png: three sign videos -> Encoder -> embedding
vectors -> embedding space with pull (green) / push (red) arrows.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "data" / "lsa64" / "videos"
KEYPOINTS = ROOT / "data" / "lsa64" / "keypoints"
OUT = ROOT / "figures" / "triplet_lsa64.png"

# (video, gloss, frame fraction chosen so hands are clear of the blurred face)
SAMPLES = [
    ("051_002_001.mp4", "GRACIAS", 0.25),   # anchor
    ("051_005_001.mp4", "GRACIAS", 0.35),   # positive (other signer)
    ("022_002_001.mp4", "AGUA", 0.35),      # negative
]

GREEN = "#2e9e44"
VEC_COLORS = {  # face, edge
    "a": ("#c9dcf5", "#7a9cc6"),
    "p": ("#d7ecd0", "#7fb56e"),
    "n": ("#f5cfcf", "#c98080"),
}
PULL = "#5aa63c"
PUSH = "#c0392b"

def grab_frame(path, frac=0.5):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = int(n * frac)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame from {path}")
    frame = blur_face(frame, Path(path).stem, idx)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def blur_face(frame, stem, frame_idx):
    """Anonymize using pose keypoints: nose (0) center, ear distance (7-8) radius."""
    kp = np.load(KEYPOINTS / f"{stem}.npy")
    pose = kp[min(frame_idx, len(kp) - 1), :33]
    nose, l_ear, r_ear = pose[0], pose[7], pose[8]
    if not nose[:2].any():
        return frame
    h, w = frame.shape[:2]
    cx, cy = int(nose[0] * w), int(nose[1] * h)
    rx = max(int(abs(l_ear[0] - r_ear[0]) * w * 1.1), 40)
    ry = int(rx * 1.3)
    x0, y0 = max(cx - rx, 0), max(cy - ry, 0)
    x1, y1 = min(cx + rx, w), min(cy + ry, h)
    roi = frame[y0:y1, x0:x1]
    k = max((x1 - x0) // 2 * 2 + 1, 31)
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.ellipse(
        mask, (cx - x0, cy - y0), (rx, ry), 0, 0, 360, 255, -1,
    )
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask3 = cv2.merge([mask] * 3) / 255.0
    frame[y0:y1, x0:x1] = (blurred * mask3 + roi * (1 - mask3)).astype(np.uint8)
    return frame


def crop_16x9_center(img, zoom=1.35):
    """Crop toward the signer so they fill more of the thumbnail."""
    h, w = img.shape[:2]
    ch, cw = int(h / zoom), int(w / zoom * (16 / 9) * (h / w))
    cw = min(cw, w)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def draw_video_stack(ax, img, x, y, w, h, label):
    """Three offset frames like a small stack, front one shows the image."""
    off = 0.45
    for i in (2, 1):
        ax.add_patch(
            Rectangle(
                (x + i * off, y + i * off), w, h,
                facecolor="#dfe8f2", edgecolor="black", lw=1.2, zorder=2 + (2 - i),
            )
        )
    ax.imshow(img, extent=[x, x + w, y, y + h], zorder=5, aspect="auto")
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor="none", edgecolor="black", lw=1.4, zorder=6)
    )
    ax.text(
        x + w / 2, y - 1.2, label,
        ha="center", va="top", fontsize=15, fontweight="bold", zorder=6,
    )


def draw_encoder(ax, x, y, w, h):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.25",
            facecolor=GREEN, edgecolor="#1e7030", lw=1.5, zorder=5,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, "Encoder",
        ha="center", va="center", fontsize=19, fontweight="bold",
        color="white", zorder=6,
    )


def draw_vector(ax, x, y_center, key, n=7, cw=2.1, ch=1.5, lw=1.3, zorder=5):
    face, edge = VEC_COLORS[key]
    y0 = y_center - n * ch / 2
    for i in range(n):
        ax.add_patch(
            Rectangle(
                (x, y0 + i * ch), cw, ch,
                facecolor=face, edgecolor=edge, lw=lw, zorder=zorder,
            )
        )
    return y0, y0 + n * ch


def harrow(ax, x0, x1, y):
    ax.add_patch(
        FancyArrow(
            x0, y, x1 - x0, 0, width=0.28, head_width=1.0, head_length=1.0,
            length_includes_head=True, facecolor="black", edgecolor="black", zorder=5,
        )
    )


def dotted_link(ax, p0, p1, rad):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=14,
            linestyle=(0, (2, 3)), color="#9a9a9a", lw=1.4, zorder=3,
        )
    )


def push_pull(ax, x, y, dx, dy, color, sep=0.9):
    """Two parallel arrows (double arrow) from (x, y) along (dx, dy)."""
    # perpendicular offset
    norm = (dx**2 + dy**2) ** 0.5
    ox, oy = -dy / norm * sep / 2, dx / norm * sep / 2
    for s in (-1, 1):
        ax.add_patch(
            FancyArrow(
                x + s * ox, y + s * oy, dx, dy,
                width=0.22, head_width=0.85, head_length=0.9,
                length_includes_head=True, facecolor=color, edgecolor=color, zorder=7,
            )
        )


def main():
    OUT.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(20, 11.4), dpi=140)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 57)
    ax.set_aspect((57 / 100) * (20 / 11.4))
    ax.axis("off")

    rows_y = [46.5, 29.5, 12.5]           # row centers: anchor, positive, negative
    keys = ["a", "p", "n"]
    img_w, img_h = 13.5, 8.6
    enc_x, enc_w, enc_h = 20.5, 14.0, 5.2
    vec_x = 41.5

    vec_tips = []
    for (fname, label, frac), yc, key in zip(SAMPLES, rows_y, keys):
        img = crop_16x9_center(grab_frame(VIDEOS / fname, frac))
        draw_video_stack(ax, img, 1.0, yc - img_h / 2, img_w, img_h, label)
        harrow(ax, 15.6, enc_x - 0.4, yc)
        draw_encoder(ax, enc_x, yc - enc_h / 2, enc_w, enc_h)
        harrow(ax, enc_x + enc_w + 0.4, vec_x - 0.5, yc)
        y0, y1 = draw_vector(ax, vec_x, yc, key)
        ax.text(
            vec_x + 1.05, y1 + 0.6, f"$x^{key}$",
            ha="center", va="bottom", fontsize=21, zorder=6,
        )
        vec_tips.append((vec_x + 2.3, yc))

    # ---- embedding space -------------------------------------------------
    ox, oy = 64.0, 14.0                    # 3D axes origin
    axkw = dict(width=0.22, head_width=1.0, head_length=1.1,
                length_includes_head=True, facecolor="#333333",
                edgecolor="#333333", zorder=4)
    ax.add_patch(FancyArrow(ox, oy, 0, 27, **axkw))          # up
    ax.add_patch(FancyArrow(ox, oy, 34, 0.5, **axkw))        # right
    ax.add_patch(FancyArrow(ox, oy, -8.5, -7.5, **axkw))     # toward viewer

    # embedded vectors: anchor (blue), positive (green), negative far (red)
    emb = {"a": (79.5, 34.5), "p": (71.0, 27.0), "n": (89.5, 29.0)}
    for key, (ex, ey) in emb.items():
        draw_vector(ax, ex, ey, key, n=6, cw=1.7, ch=1.35, zorder=6)

    # dotted trajectories from x^a / x^p / x^n into the space
    dotted_link(ax, vec_tips[0], (emb["a"][0] - 0.3, emb["a"][1] + 3.4), -0.25)
    dotted_link(ax, vec_tips[1], (emb["p"][0] - 0.4, emb["p"][1]), -0.12)
    dotted_link(ax, vec_tips[2], (emb["n"][0] - 0.4, emb["n"][1] - 2.5), 0.30)

    # pull together (green, between anchor and positive)
    push_pull(ax, 73.8, 28.8, 2.5, 2.0, PULL)        # toward anchor
    push_pull(ax, 78.2, 33.2, -2.5, -2.0, PULL)      # toward positive

    # push apart (red)
    push_pull(ax, 78.9, 39.7, -2.4, 2.4, PUSH)       # anchor pushed from negative
    push_pull(ax, 92.2, 30.7, 2.7, -2.4, PUSH)       # negative pushed away

    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
