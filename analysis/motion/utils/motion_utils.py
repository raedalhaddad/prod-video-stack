# ruff: noqa
# noqa: F401,F403

# real_time_video_analysis/motion_utils.py
from __future__ import annotations

try:
    import cv2
except Exception:
    cv2 = None  # type: ignore
from typing import Dict, List, Tuple

import numpy as np


# --- Background subtractor factory (no globals; caller passes params) ----------
def make_bg_subtractor(
    bg_history: int, bg_var_th: int, shadow_tau_x100: int, detect_shadows: bool = True
):
    # history/varThreshold are live-tunable; shadow τ via setShadowThreshold
    mog = cv2.createBackgroundSubtractorMOG2(
        history=int(bg_history),
        varThreshold=int(bg_var_th),
        detectShadows=bool(detect_shadows),
    )
    try:
        mog.setShadowThreshold(float(shadow_tau_x100) / 100.0)
    except Exception:
        pass
    return mog


# --- Mask post-processing (median → open → dilate) -----------------------------
def postprocess_mask(
    fgmask: np.ndarray, median_blur_k: int, morph_open_iters: int, dilate_iters: int
) -> np.ndarray:
    mask = (fgmask == 255).astype(np.uint8) * 255
    if median_blur_k and median_blur_k > 0:
        mask = cv2.medianBlur(mask, int(median_blur_k) | 1)
    if morph_open_iters and morph_open_iters > 0:
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=int(morph_open_iters))
    if dilate_iters and dilate_iters > 0:
        k = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, k, iterations=int(dilate_iters))
    return mask


# --- Component boxes (with tiny gap bridge) -----------------------------------
def motion_boxes_components(
    mask: np.ndarray, min_area: int = 900, gap_px: int = 6
) -> List[Tuple[int, int, int, int, int]]:
    """
    Return list of (x, y, w, h, area_mask) in MASK coords.
    Uses a tiny pre-dilate to bridge pixel gaps before connected-components.
    """
    if gap_px and gap_px > 0:
        iters = max(1, int(round(gap_px / 3.0)))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=iters)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: List[Tuple[int, int, int, int, int]] = []
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if a >= int(min_area):
            out.append((int(x), int(y), int(w), int(h), int(a)))
    return out


# --- Micro-merge nearby small components (mask-space) --------------------------
def _rects_intersect_expanded(a, b, gap_px=4) -> bool:
    ax, ay, aw, ah, _ = a
    bx, by, bw, bh, _ = b
    ax1, ay1, ax2, ay2 = ax - gap_px, ay - gap_px, ax + aw + gap_px, ay + ah + gap_px
    bx1, by1, bx2, by2 = bx - gap_px, by - gap_px, bx + bw + gap_px, by + bh + gap_px
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _rect_union(a, b):
    ax, ay, aw, ah, _ = a
    bx, by, bw, bh, _ = b
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return (x1, y1, x2 - x1, y2 - y1)


def merge_small_boxes(
    boxes_small: List[Tuple[int, int, int, int, int]], min_area: int, gap_px: int = 4
) -> List[Tuple[int, int, int, int, int]]:
    """
    Merge only NEARBY SMALL components:
      - 'small' if area_mask <= small_max (≈ 3× min_area)
      - merge when expanded rects (by merge_gap_px) intersect
    """
    if not boxes_small:
        return []
    boxes = list(boxes_small)
    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            xi, yi, wi, hi, ai = boxes[i]
            accum = (xi, yi, wi, hi)
            total_area = ai
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if _rects_intersect_expanded(boxes[i], boxes[j], gap_px=gap_px):
                    xj, yj, wj, hj, aj = boxes[j]
                    accum = _rect_union((xi, yi, wi, hi, ai), (xj, yj, wj, hj, aj))
                    total_area += aj
                    used[j] = True
                    changed = True
            used[i] = True
            x, y, w, h = accum
            out.append((x, y, w, h, int(total_area)))
        boxes = out
    # enforce min_area after merges
    return [b for b in boxes if b[4] >= int(min_area)]


# --- Convert mask-space boxes to full-res xyxy + areas ------------------------
def motion_boxes_xyxy_and_area(
    boxes_small: List[Tuple[int, int, int, int, int]], mask_shape, frame_shape
) -> List[Dict]:
    """
    Convert motion boxes from small (mask) coords to full-res YOLO-style xyxy and areas.
    - boxes_small: (x, y, w, h, area_mask) where area_mask is contour area in mask pixels.
    Returns: [{"xyxy":[x1,y1,x2,y2], "area": rect_area_full_px, "area_mask": area_mask_px}, ...]
    """
    mh, mw = int(mask_shape[0]), int(mask_shape[1])
    fh, fw = int(frame_shape[0]), int(frame_shape[1])
    sx = fw / float(max(1, mw))
    sy = fh / float(max(1, mh))
    out: List[Dict] = []
    for x, y, w, h, area_mask in boxes_small:
        x1 = int(round(x * sx))
        y1 = int(round(y * sy))
        x2 = int(round((x + w) * sx))
        y2 = int(round((y + h) * sy))
        rect_area = int(max(0, x2 - x1) * max(0, y2 - y1))
        out.append({"xyxy": [x1, y1, x2, y2], "area": rect_area, "area_mask": int(area_mask)})
    return out


# --- Contour boxes (fallback to CC; simple & fast) ----------------------------
def motion_boxes_with_area(
    mask: np.ndarray, min_area: int = 900
) -> List[Tuple[int, int, int, int, int]]:
    """
    Return list of (x, y, w, h, area_mask) in MASK coords using contours.
    Simpler than CC; can be noisier on speckly masks.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Tuple[int, int, int, int, int]] = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < int(min_area):
            continue
        x, y, w, h = cv2.boundingRect(c)
        out.append((int(x), int(y), int(w), int(h), int(a)))
    return out


# --- Shadow suppression (HSV against learned BG image) ------------------------
def drop_shadow_like_pixels(
    mask_small: np.ndarray,
    small_bgr: np.ndarray,
    bg_img: np.ndarray,
    enabled: bool,
    vdrop_pct: int,
    s_delta_x100: int,
    h_delta_deg: int,
) -> tuple[np.ndarray, float]:
    """
    Return (clean_mask, drop_ratio), removing pixels likely to be cast shadows:
    darker V with small changes in S/H relative to the background image.
    All thresholds are provided by the caller (no globals).
    """
    if not enabled or bg_img is None or mask_small is None:
        return mask_small, 0.0
    # Normalize BG to BGR if it arrived as single-channel
    if bg_img.ndim == 2 or (bg_img.ndim == 3 and bg_img.shape[2] == 1):
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR)
    curr_hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    bg_hsv = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)
    m = mask_small > 0
    if not np.any(m):
        return mask_small, 0.0
    cH, cS, cV = cv2.split(curr_hsv)
    bH, bS, bV = cv2.split(bg_hsv)
    bV_f = bV.astype(np.float32)
    cV_f = cV.astype(np.float32)
    # Relative V drop (shadow pixels are darker)
    v_drop = (bV_f - cV_f) / (bV_f + 1e-3)
    # S and H stability
    s_delta = np.abs(cS.astype(np.int16) - bS.astype(np.int16)).astype(np.float32) / 255.0
    h_diff = np.abs(cH.astype(np.int16) - bH.astype(np.int16)).astype(np.float32)
    h_diff = np.minimum(h_diff, 180.0 - h_diff)  # hue wrap-around
    v_ok = v_drop >= (float(vdrop_pct) / 100.0)
    s_ok = s_delta <= (float(s_delta_x100) / 100.0)
    h_ok = h_diff <= float(h_delta_deg)
    drop = v_ok & s_ok & h_ok & m
    cleaned = mask_small.copy()
    cleaned[drop] = 0
    drop_ratio = float(np.count_nonzero(drop)) / float(np.count_nonzero(m) + 1e-6)
    return cleaned, drop_ratio


# --- High-pass gray for motion (illumination-robust) --------------------------
def high_pass_gray(gray: np.ndarray, sigma: float, gain: float = 1.0) -> np.ndarray:
    """
    Build a high-pass version of a grayscale image:
      hp = gray - GaussianBlur(gray, σ=sigma), then optional gain & abs convert.
    Mirrors the inline logic used before feeding MOG2.
    """
    blur = cv2.GaussianBlur(gray, (0, 0), float(sigma))
    hp = cv2.subtract(gray, blur)
    if abs(float(gain) - 1.0) > 1e-3:
        hp = cv2.convertScaleAbs(hp, alpha=float(gain), beta=0.0)
    return hp


# --- Zones helpers ------------------------------------------------------------
def _scale_polys(polys: List[List[Tuple[float, float]]], w: int, h: int) -> List[np.ndarray]:
    out = []
    for poly in polys or []:
        pts = np.array([(int(px * w), int(py * h)) for (px, py) in poly], dtype=np.int32)
        out.append(pts)
    return out


def build_weight_and_hard_masks(
    shape: Tuple[int, int],
    include,
    exclude_hard,
    downweight,
    priority,
    downweight_factor: float = 0.25,
    priority_factor: float = 2.0,
):
    """
    Return (weight_map float32, hard_mask uint8 0/255) at the motion (small) resolution.
    - include:   only these regions count (others zeroed)
    - exclude:   truly ignore regions (hard 0 mask)
    - downweight: give fractional credit (e.g., foliage)
    - priority:   amplify credit (e.g., walkway/door)
    """
    h, w = int(shape[0]), int(shape[1])
    weight = np.ones((h, w), np.float32)
    hard = np.ones((h, w), np.uint8) * 255
    if include:
        weight[:] = 0.0
        hard[:] = 0
        for pts in _scale_polys(include, w, h):
            cv2.fillPoly(weight, [pts], 1.0)
            cv2.fillPoly(hard, [pts], 255)
    for pts in _scale_polys(downweight, w, h):
        cv2.fillPoly(weight, [pts], float(downweight_factor))
    for pts in _scale_polys(priority, w, h):
        cv2.fillPoly(weight, [pts], float(priority_factor))
    for pts in _scale_polys(exclude_hard, w, h):
        cv2.fillPoly(weight, [pts], 0.0)
        cv2.fillPoly(hard, [pts], 0)
    if weight.sum() <= 1e-6:
        weight[:] = 1.0
        hard[:] = 255
    return weight, hard


# --- Zones: build a binary mask from normalized polygons ---------------------
def polys_to_mask(shape: Tuple[int, int] | np.ndarray, polys) -> np.ndarray:
    """
    Convert a list of normalized polygons (0..1) into a uint8 mask (0/255)
    at the provided shape (h, w) or mask.shape.
    """
    if isinstance(shape, tuple):
        h, w = int(shape[0]), int(shape[1])
    else:
        h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w), np.uint8)
    if not polys:
        return out
    for pts in _scale_polys(polys, w, h):
        cv2.fillPoly(out, [pts], 255)
    return out


# --- Exposure / global-flash guard check (pure) -------------------------------
def exposure_guard_check(
    prev_small_gray: np.ndarray | None,
    small_gray: np.ndarray,
    mean_t: int,
    std_t: int,
    pix_t: int,
    pct_t: int,
    grad_t: float,
) -> tuple[bool, float, float, float, float]:
    """
    Compare prev vs curr gray to detect global exposure/WB shifts.
    Returns: (trigger, m, s, pct, gmean)
      m = mean(absdiff), s = std(absdiff),
      pct = fraction of pixels over pix_t,
      gmean = mean |Δgrad| scaled (≈0..20), matches in-script logic.
    """
    if prev_small_gray is None:
        return False, 0.0, 0.0, 0.0, 0.0
    fdiff = cv2.absdiff(small_gray, prev_small_gray)
    m = float(np.mean(fdiff))
    s = float(np.std(fdiff))
    pct = float((fdiff >= int(pix_t)).sum()) / float(fdiff.size)
    gx1 = cv2.Sobel(prev_small_gray, cv2.CV_16S, 1, 0, ksize=3)
    gy1 = cv2.Sobel(prev_small_gray, cv2.CV_16S, 0, 1, ksize=3)
    gx2 = cv2.Sobel(small_gray, cv2.CV_16S, 1, 0, ksize=3)
    gy2 = cv2.Sobel(small_gray, cv2.CV_16S, 0, 1, ksize=3)
    mag1 = cv2.magnitude(gx1.astype(np.float32), gy1.astype(np.float32))
    mag2 = cv2.magnitude(gx2.astype(np.float32), gy2.astype(np.float32))
    gmean = float(np.mean(np.abs(mag2 - mag1))) / 8.0  # ~0..20 range
    trigger = (
        (pct >= (float(pct_t) / 100.0))
        or ((m > float(mean_t)) and (s < float(std_t)))
        or (m > float(mean_t) and gmean < float(grad_t))
    )
    return bool(trigger), m, s, pct, gmean
