# ruff: noqa
# noqa: F401,F403

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None  # type: ignore

from .config import *  # noqa: F401,F403
from .utils.motion_utils import drop_shadow_like_pixels  # separate import for clarity
from .utils.motion_utils import (
    build_weight_and_hard_masks,
    exposure_guard_check,
    high_pass_gray,
    make_bg_subtractor,
    merge_small_boxes,
    motion_boxes_components,
    motion_boxes_with_area,
    motion_boxes_xyxy_and_area,
    polys_to_mask,
    postprocess_mask,
)


@dataclass
class MotionResult:
    # core outputs the caller needs each frame
    mask_small: np.ndarray
    boxes_small: List[Tuple[int, int, int, int, int]]  # (x,y,w,h,area)
    motion_list: List[Dict[str, Any]]
    area_frac_weighted: float
    shadow_drop_frac: float

    # exposure guard state & metrics
    expo_guard_active: bool
    expo_metrics: Tuple[float, float, float, float]  # mean, std, pct, grad_mean

    # wind gate + metrics
    wind_like: bool
    real_override: bool
    flow_inst_mag: Optional[float]
    flow_inst_coh: Optional[float]
    tdc: Optional[float]
    tss: Optional[float]
    dfr: Optional[float]
    flow_win_N: int
    # NEW: time-only eligibility telemetry (always returned)
    flow_win_span_s: float
    flow_last_gap_s: float
    win_eligible: bool
    ovr_min_frac_eff: float  # NEW: effective override area floor (for HUD)

    # smoothed stats / foliage
    mag_ema: Optional[float]
    coh_ema: Optional[float]
    mag_std_10s: float
    foliage_motion_frac: float

    # Wind Governor telemetry
    wind_idx: float
    wind_state: str
    wind_severity: float
    zones_active_frac: float

    # effective thresholds (telemetry / controls)
    flow_mag_thresh_eff: float
    arm_min_time_eff: float
    wind_tiny_factor_eff: float

    # timings (ms) to feed your live perf counters
    t_motion_ms: float
    t_flow_ms: float
    # NEW: expose override dwell timer (seconds) for HUD
    real_ovr_timer_s: float

    # zones (optional v1: instant-only metrics; 3×3 by default)
    zones: Optional[Dict[str, Any]] = None


class MotionEngine:
    """
    Owns stateful motion/background/wind machinery:
      - Background subtractor + fast relearn window
      - Exposure guard (global flash / relight protection)
      - Weight/hard masks + foliage mask at 'small' resolution
      - Optical-flow temporal window (TDC/TSS/DFR) and EMAs
      - Real-motion override timer

    Caller contract:
      - Call step(ts, frame_bgr, small_bgr, small_gray, recording_active, dt, g=globals())
      - If UI sliders change BG params, call rebuild_bg(g=globals()).
    """

    def __init__(self):
        self.bg = None
        self.prev_small_gray: Optional[np.ndarray] = None

        # --- debug (print) state to avoid spamming the terminal ---
        self._dbg_last_ovr_min_frac = None
        self._dbg_last_arm_thresh = None

        # state that persists across frames
        self.fast_lr_until: float = 0.0
        self.expo_guard_until: float = 0.0

        # “small” masks (built lazily when first frame size is known)
        self.weight_map_small: Optional[np.ndarray] = None
        self.hard_mask_small: Optional[np.ndarray] = None
        self.down_mask_small: Optional[np.ndarray] = None  # foliage zones only (binary)

        # temporal windows & EMAs for wind metrics
        self.flow_hist: deque = deque()  # (ts, ux, uy, mag_inst)
        self.mag_hist: deque = deque()  # (ts, mag_ema) for 10s std
        self.mag_ema: Optional[float] = None
        self.coh_ema: Optional[float] = None
        self.real_override_timer: float = 0.0

        # Always-on telemetry for HUD (kept fresh each frame)
        self.last_flow_win_span_s: float = 0.0
        self.last_flow_gap_s: float = 1e9
        self.last_win_eligible: bool = False

        # Wind Governor state
        # EMAs seeded to calm scene to avoid biased wind_idx at startup
        self.tdc_ema: Optional[float] = 1.0  # calm = fully consistent
        self.tss_ema: Optional[float] = 1.0  # calm = spatially stable
        self.dfr_ema: Optional[float] = 0.0  # calm = low randomness
        self.zones_act_ema: Optional[float] = 0.0
        self.wind_idx: float = 0.0
        self.wind_state: str = "Calm"
        self._wind_candidate_since: float = 0.0  # hysteresis sustain timer
        self._wind_last_state_change_ts: float = 0.0

        # Effective thresholds (initialized to base values)
        self.flow_mag_thresh_eff: float = float(globals().get("FLOW_MAG_THRESH", 0.0))
        self.arm_min_time_eff: float = float(globals().get("ARM_MIN_TIME_S", 0.0))
        self.wind_tiny_factor_eff: float = float(globals().get("WIND_TINY_FACTOR", 1.0))

        # stride helpers
        self.motion_frame_i: int = 0
        self.flow_frame_i: int = 0

        # cache last results to support motion stride reuse if needed
        self._cache = dict(
            mask=None, boxes_small=None, motion_list=None, area_frac_weighted=0.0, sh_drop=0.0
        )

        # ---- v2: per-zone temporal history (list of deques; sized to grid) ----
        self._zones_grid: Tuple[int, int] = (0, 0)
        self._zone_hist: List[deque] | None = None  # each holds (ts, ux, uy, mag_inst)

    # ----- helpers to read live params from caller's module globals OR config -----
    def _p(self, name: str, g: Optional[Dict[str, Any]], default: Any = None):
        if g and (name in g):
            return g[name]
        return (
            globals().get(name, default)
            if name in globals()
            else getattr(
                __import__(__name__.rsplit(".", 1)[0] + ".config", fromlist=["*"]), name, default
            )
        )

    def rebuild_bg(self, g: Optional[Dict[str, Any]] = None):
        """Rebuild background subtractor with current slider/config values."""
        BG_HISTORY_ = int(self._p("BG_HISTORY", g, BG_HISTORY))
        BG_VAR_TH_ = int(self._p("BG_VAR_TH", g, BG_VAR_TH))
        SHADOW_TAU_ = int(self._p("SHADOW_TAU", g, SHADOW_TAU))
        self.bg = make_bg_subtractor(BG_HISTORY_, BG_VAR_TH_, SHADOW_TAU_, detect_shadows=True)
        # reset exposure guard & fast-learn windows; keep prev_small_gray to avoid a cold start burst
        self.expo_guard_until = 0.0
        self.fast_lr_until = 0.0

    def fast_lr_active(self, ts: float) -> bool:
        return bool(ts < self.fast_lr_until)

    def _ensure_bg(self, g: Optional[Dict[str, Any]]):
        if self.bg is None:
            self.rebuild_bg(g=g)

    def _ensure_zone_masks(self, small_shape: Tuple[int, int], g: Optional[Dict[str, Any]]):
        if (
            (self.weight_map_small is None)
            or (self.hard_mask_small is None)
            or (self.down_mask_small is None)
        ):
            INCLUDE = self._p("INCLUDE_ZONES", g, INCLUDE_ZONES)
            EXCL = self._p("EXCLUDE_ZONES_HARD", g, EXCLUDE_ZONES_HARD)
            DOWN = self._p("DOWNWEIGHT_ZONES", g, DOWNWEIGHT_ZONES)
            PRIOR = self._p("PRIORITY_ZONES", g, PRIORITY_ZONES)
            DW_F = float(self._p("DOWNWEIGHT_FACTOR", g, DOWNWEIGHT_FACTOR))
            PR_F = float(self._p("PRIORITY_FACTOR", g, PRIORITY_FACTOR))
            W, H = small_shape[1], small_shape[0]
            weight, hard = build_weight_and_hard_masks(
                (H, W), INCLUDE, EXCL, DOWN, PRIOR, downweight_factor=DW_F, priority_factor=PR_F
            )
            self.weight_map_small = weight
            self.hard_mask_small = hard
            self.down_mask_small = polys_to_mask((H, W), DOWN)

    def step(
        self,
        ts: float,
        frame_bgr: np.ndarray,
        small_bgr: Optional[np.ndarray] = None,
        small_gray: Optional[np.ndarray] = None,
        recording_active: bool = False,
        dt: float = 0.0,
        g: Optional[Dict[str, Any]] = None,
    ) -> MotionResult:
        """
        Returns a MotionResult with everything the caller needs for gating, logging, and HUD.
        """
        t_motion0 = time.time()
        self._ensure_bg(g)

        # build small views if caller didn't
        if small_bgr is None:
            small_bgr = cv2.resize(frame_bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        if small_gray is None:
            small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)

        # zone masks once (per resolution)
        self._ensure_zone_masks(small_gray.shape, g)

        # Read live tunables (from sliders or config)
        HP_ON = int(self._p("HP_MOTION_ON", g, HP_MOTION_ON))
        HP_SIG = float(self._p("HP_SIGMA", g, HP_SIGMA))
        HP_GAIN_ = float(self._p("HP_GAIN", g, HP_GAIN))

        MED_K = int(self._p("MEDIAN_BLUR_K", g, MEDIAN_BLUR_K))
        OPEN_I = int(self._p("MORPH_OPEN_ITERS", g, MORPH_OPEN_ITERS))
        DIL_I = int(self._p("DILATE_ITERS", g, DILATE_ITERS))

        SH_ON = int(self._p("SHADOW_SUPPRESS_ON", g, SHADOW_SUPPRESS_ON))
        SH_V = int(self._p("SHAD_VDROP_PCT", g, SHAD_VDROP_PCT))
        SH_Sx = int(self._p("SHAD_S_DELTA_X100", g, SHAD_S_DELTA_X100))
        SH_Hdeg = int(self._p("SHAD_H_DELTA_DEG", g, SHAD_H_DELTA_DEG))

        EXPO_ON = int(self._p("EXPO_GUARD_ON", g, EXPO_GUARD_ON))
        EX_MEAN = int(self._p("EXPO_MEAN_T", g, EXPO_MEAN_T))
        EX_STD = int(self._p("EXPO_STD_T", g, EXPO_STD_T))
        EX_PIX = int(self._p("EXPO_PIX_T", g, EXPO_PIX_T))
        EX_PCT = int(self._p("EXPO_PCT_T", g, EXPO_PCT_T))
        EX_GRAD = float(self._p("EXPO_GRAD_T", g, EXPO_GRAD_T))
        EX_HOLD = int(self._p("EXPO_HOLD_MS", g, EXPO_HOLD_MS))

        BG_HIST = int(self._p("BG_HISTORY", g, BG_HISTORY))
        BG_FAST_X = float(self._p("BG_FAST_LR_X", g, BG_FAST_LR_X))
        BG_FAST_MS_ = int(self._p("BG_FAST_MS", g, BG_FAST_MS))

        MIN_AREA = int(self._p("MOTION_MIN_AREA", g, MOTION_MIN_AREA))
        CC_GAP = int(self._p("CC_GAP_PX", g, CC_GAP_PX))
        MR_GAP = int(self._p("MERGE_GAP_PX", g, MERGE_GAP_PX))
        BOX_MODE_ = self._p("BOX_MODE", g, BOX_MODE)

        # motion stride (caller keeps cadence for UI/CPU; engine caches if needed)
        MOTION_N_IDLE = int(self._p("MOTION_EVERY_N_IDLE", g, 1))
        MOTION_N_REC = int(self._p("MOTION_EVERY_N_REC", g, 2))
        stride = MOTION_N_REC if recording_active else MOTION_N_IDLE
        do_motion = (self.motion_frame_i % max(1, int(stride))) == 0

        # ----- MOTION -----
        if do_motion:
            base_lr = 1.0 / max(2, BG_HIST)
            lr = base_lr * (BG_FAST_X if (ts < self.fast_lr_until) else 1.0)
            if HP_ON:
                hp = high_pass_gray(small_gray, sigma=HP_SIG, gain=HP_GAIN_)
                fg = self.bg.apply(hp, learningRate=float(lr))
            else:
                fg = self.bg.apply(small_bgr, learningRate=float(lr))

            mask = postprocess_mask(fg, MED_K, OPEN_I, DIL_I)
            # optional shadow suppression (needs color BG)
            bg_img = None
            try:
                bg_img = self.bg.getBackgroundImage()
            except Exception:
                bg_img = None

            if bg_img is not None and bg_img.ndim == 3 and bg_img.shape[2] == 3:
                mask, sh_drop = drop_shadow_like_pixels(
                    mask_small=mask,
                    small_bgr=small_bgr,
                    bg_img=bg_img,
                    enabled=bool(SH_ON),
                    vdrop_pct=int(SH_V),
                    s_delta_x100=int(SH_Sx),
                    h_delta_deg=int(SH_Hdeg),
                )
            else:
                sh_drop = 0.0

            # exposure guard
            expo_metrics = (0.0, 0.0, 0.0, 0.0)
            if EXPO_ON:
                trig, m, s, pct, gmean = exposure_guard_check(
                    self.prev_small_gray,
                    small_gray,
                    mean_t=EX_MEAN,
                    std_t=EX_STD,
                    pix_t=EX_PIX,
                    pct_t=EX_PCT,
                    grad_t=EX_GRAD,
                )
                expo_metrics = (float(m), float(s), float(pct), float(gmean))
                if trig:
                    self.expo_guard_until = ts + (float(EX_HOLD) / 1000.0)
                    self.fast_lr_until = max(self.fast_lr_until, ts + (float(BG_FAST_MS_) / 1000.0))
            expo_guard_active = ts < self.expo_guard_until

            # hard mask for contours; zero everything if guard is active
            mask_eff = (
                cv2.bitwise_and(mask, self.hard_mask_small)
                if not expo_guard_active
                else np.zeros_like(mask)
            )

            # foliage fraction (portion of motion inside downweight zones)
            try:
                motion_px = int(np.count_nonzero(mask_eff))
                foliage_frac = (
                    (
                        float(np.count_nonzero(cv2.bitwise_and(mask_eff, self.down_mask_small)))
                        / float(motion_px)
                    )
                    if (motion_px > 0)
                    else 0.0
                )
            except Exception:
                foliage_frac = 0.0

            # boxes (component or contour) + motion list at full-res
            if BOX_MODE_ == "contour":
                boxes_small = motion_boxes_with_area(mask_eff, min_area=MIN_AREA)
            else:
                boxes_small = motion_boxes_components(mask_eff, min_area=MIN_AREA, gap_px=CC_GAP)
                boxes_small = merge_small_boxes(boxes_small, MIN_AREA, MR_GAP)

            motion_list = motion_boxes_xyxy_and_area(boxes_small, mask_eff.shape, frame_bgr.shape)

            # weighted motion fraction (arming score)
            area_frac_weighted = float(
                (mask_eff.astype(np.float32) / 255.0 * self.weight_map_small).sum()
                / (self.weight_map_small.sum() + 1e-6)
            )

            # cache for reuse when striding
            self._cache.update(
                mask=mask,
                boxes_small=boxes_small,
                motion_list=motion_list,
                area_frac_weighted=area_frac_weighted,
                sh_drop=sh_drop,
            )
        else:
            # reuse last motion results when striding
            mask = (
                self._cache["mask"]
                if self._cache["mask"] is not None
                else np.zeros_like(small_gray)
            )
            boxes_small = self._cache["boxes_small"] or []
            motion_list = self._cache["motion_list"] or []
            area_frac_weighted = float(self._cache["area_frac_weighted"])
            sh_drop = float(self._cache["sh_drop"])
            # exposure guard still updates on stride
            expo_metrics = (0.0, 0.0, 0.0, 0.0)
            if EXPO_ON:
                try:
                    trig, m, s, pct, gmean = exposure_guard_check(
                        self.prev_small_gray,
                        small_gray,
                        mean_t=EX_MEAN,
                        std_t=EX_STD,
                        pix_t=EX_PIX,
                        pct_t=EX_PCT,
                        grad_t=EX_GRAD,
                    )
                    expo_metrics = (float(m), float(s), float(pct), float(gmean))
                    if trig:
                        self.expo_guard_until = ts + (float(EX_HOLD) / 1000.0)
                        self.fast_lr_until = max(
                            self.fast_lr_until, ts + (float(BG_FAST_MS_) / 1000.0)
                        )
                except Exception:
                    pass
            expo_guard_active = ts < self.expo_guard_until
            # foliage fraction on stride
            try:
                motion_px = int(np.count_nonzero(mask))
                foliage_frac = (
                    (
                        float(np.count_nonzero(cv2.bitwise_and(mask, self.down_mask_small)))
                        / float(motion_px)
                    )
                    if (motion_px > 0)
                    else 0.0
                )
            except Exception:
                foliage_frac = 0.0

        self.motion_frame_i += 1
        t_motion_ms = (time.time() - t_motion0) * 1000.0

        # ----- WIND via optical flow (with stride + optional ROI crop) -----
        FLOW_EVERY_N_ = int(self._p("FLOW_EVERY_N", g, FLOW_EVERY_N))
        FLOW_MASK_ROI_ = int(self._p("FLOW_MASK_ROI", g, FLOW_MASK_ROI))
        FLOW_MIN_ROI_ = int(self._p("FLOW_MIN_ROI", g, FLOW_MIN_ROI))
        FLOW_ROI_MARGIN_ = int(self._p("FLOW_ROI_MARGIN", g, FLOW_ROI_MARGIN))
        FLOW_MAG_TH = float(self._p("FLOW_MAG_THRESH", g, FLOW_MAG_THRESH))
        FLOW_COH_TH = float(self._p("FLOW_COHERENCE_THRESH", g, FLOW_COHERENCE_THRESH))
        FLOW_WIN_S = float(self._p("FLOW_TEMP_WIN_S", g, FLOW_TEMP_WIN_S))
        # Legacy: MIN_N is now ignored for eligibility (kept for backwards compat/UI only)
        FLOW_WIN_MIN_N_ = int(self._p("FLOW_WIN_MIN_N", g, 10))
        FLOW_MAX_GAP_S = float(self._p("FLOW_MAX_SAMPLE_GAP_S", g, FLOW_MAX_SAMPLE_GAP_S))
        FLOW_DFR_ANG = float(self._p("FLOW_DFR_ANGLE_DEG", g, FLOW_DFR_ANGLE_DEG))

        WIND_KEEP_TDC_MAX_ = float(self._p("WIND_KEEP_TDC_MAX", g, 0.55))
        WIND_KEEP_TSS_MAX_ = float(self._p("WIND_KEEP_TSS_MAX", g, 0.50))
        WIND_KEEP_DFR_MIN_ = float(self._p("WIND_KEEP_DFR_MIN", g, 0.30))
        WIND_TINY_FRAC_ = float(self._p("WIND_TINY_FOOTPRINT_FRAC", g, 0.01))

        REAL_OVR_ON = int(self._p("REAL_OVR_ENABLED", g, REAL_OVR_ENABLED))
        OVR_COH_MIN = float(self._p("REAL_OVR_COH_INST_MIN", g, REAL_OVR_COH_INST_MIN))
        OVR_TDC_MIN = float(self._p("REAL_OVR_TDC_MIN", g, REAL_OVR_TDC_MIN))
        OVR_TSS_MIN = float(self._p("REAL_OVR_TSS_MIN", g, REAL_OVR_TSS_MIN))
        OVR_DFR_MAX = float(self._p("REAL_OVR_DFR_MAX", g, REAL_OVR_DFR_MAX))
        # Make the *default* override area follow the current ARM slider,
        # clamped to [0.0005, 0.010]. Still user-overridable via REAL_OVR_MIN_FRAC.
        ARM_THRESH_ = float(self._p("ARM_THRESH", g, ARM_THRESH))
        _ovr_min_default = max(0.0005, min(0.010, ARM_THRESH_))
        OVR_MIN_FRAC = float(self._p("REAL_OVR_MIN_FRAC", g, _ovr_min_default))
        OVR_DWELL_S = float(self._p("REAL_OVR_DWELL_S", g, REAL_OVR_DWELL_S))
        WIND_EMA_TAU = float(self._p("WIND_EMA_TAU_S", g, WIND_EMA_TAU_S))

        # --- DEBUG: simple print when the effective floor or ARM changes ---
        try:
            using_default = abs(OVR_MIN_FRAC - _ovr_min_default) < 1e-12
            if (
                self._dbg_last_ovr_min_frac is None
                or abs(OVR_MIN_FRAC - self._dbg_last_ovr_min_frac) > 1e-12
                or self._dbg_last_arm_thresh is None
                or abs(ARM_THRESH_ - self._dbg_last_arm_thresh) > 1e-12
            ):
                print(
                    f"[OVR DEBUG] eff={OVR_MIN_FRAC*100:.3f}% "
                    f"(source={'ARM-default' if using_default else 'EXPLICIT'}) | "
                    f"ARM={ARM_THRESH_*100:.3f}% | default_from_arm={_ovr_min_default*100:.3f}%"
                )
                self._dbg_last_ovr_min_frac = OVR_MIN_FRAC
                self._dbg_last_arm_thresh = ARM_THRESH_
        except Exception:
            pass

        wind_like = False
        real_override = False
        mean_mag_inst = None
        coherence_inst = None
        tdc_win = None
        tss_win = None
        dfr_win = None
        flow_win_N = 0

        t_flow_ms = 0.0
        # Keep full-size (analysis-space) arrays if we run flow so zones can use them.
        flow_vx_full: Optional[np.ndarray] = None
        flow_vy_full: Optional[np.ndarray] = None
        mag_full: Optional[np.ndarray] = None
        # Use the effective mask (after hard mask + exposure guard) to decide if we should run flow.
        # If the guard is active, there is effectively no motion to measure.
        has_motion = (
            bool(np.any(cv2.bitwise_and(mask, self.hard_mask_small)))
            if not expo_guard_active
            else False
        )
        do_flow = (
            (self.prev_small_gray is not None)
            and has_motion
            and (FLOW_EVERY_N_ <= 1 or (self.flow_frame_i % FLOW_EVERY_N_) == 0)
        )
        if do_flow:
            t0 = time.time()
            try:
                # Use effective mask for flow ROI/pixels so metrics and area align
                motion_mask = (
                    cv2.bitwise_and(mask, self.hard_mask_small)
                    if not expo_guard_active
                    else np.zeros_like(mask)
                )
                if FLOW_MASK_ROI_:
                    ys, xs = np.where(motion_mask > 0)
                    if ys.size == 0 or xs.size == 0:
                        # No motion after masking/guard → skip flow metrics this frame.
                        motion_pixels = np.array([], dtype=bool)
                        flow = None
                        mag = None
                    else:
                        # Build ROI from motion pixels
                        y0, y1 = int(ys.min()), int(ys.max()) + 1
                        x0, x1 = int(xs.min()), int(xs.max()) + 1
                        # Expand by margin
                        y0 = max(0, y0 - FLOW_ROI_MARGIN_)
                        x0 = max(0, x0 - FLOW_ROI_MARGIN_)
                        y1 = min(small_gray.shape[0], y1 + FLOW_ROI_MARGIN_)
                        x1 = min(small_gray.shape[1], x1 + FLOW_ROI_MARGIN_)
                        # Enforce minimum ROI
                        if (y1 - y0) < FLOW_MIN_ROI_:
                            pad = (FLOW_MIN_ROI_ - (y1 - y0)) // 2
                            y0 = max(0, y0 - pad)
                            y1 = min(small_gray.shape[0], y1 + pad)
                        if (x1 - x0) < FLOW_MIN_ROI_:
                            pad = (FLOW_MIN_ROI_ - (x1 - x0)) // 2
                            x0 = max(0, x0 - pad)
                            x1 = min(small_gray.shape[1], x1 + pad)
                        # Compute flow on the (final) ROI
                        roi_prev = self.prev_small_gray[y0:y1, x0:x1]
                        roi_curr = small_gray[y0:y1, x0:x1]
                        flow = cv2.calcOpticalFlowFarneback(
                            roi_prev,
                            roi_curr,
                            None,
                            pyr_scale=0.5,
                            levels=1,
                            winsize=15,
                            iterations=2,
                            poly_n=5,
                            poly_sigma=1.2,
                            flags=0,
                        )
                        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        motion_pixels = motion_mask[y0:y1, x0:x1] > 0
                        # Expand into full-size arrays for zone reductions
                        flow_vx_full = np.zeros_like(small_gray, dtype=np.float32)
                        flow_vy_full = np.zeros_like(small_gray, dtype=np.float32)
                        mag_full = np.zeros_like(small_gray, dtype=np.float32)
                        flow_vx_full[y0:y1, x0:x1] = flow[..., 0].astype(np.float32)
                        flow_vy_full[y0:y1, x0:x1] = flow[..., 1].astype(np.float32)
                        mag_full[y0:y1, x0:x1] = mag.astype(np.float32)
                else:
                    flow = cv2.calcOpticalFlowFarneback(
                        self.prev_small_gray,
                        small_gray,
                        None,
                        pyr_scale=0.5,
                        levels=1,
                        winsize=15,
                        iterations=2,
                        poly_n=5,
                        poly_sigma=1.2,
                        flags=0,
                    )
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    motion_pixels = motion_mask > 0
                    flow_vx_full = flow[..., 0].astype(np.float32)
                    flow_vy_full = flow[..., 1].astype(np.float32)
                    mag_full = mag.astype(np.float32)

                if np.any(motion_pixels):
                    mean_mag = float(np.mean(mag[motion_pixels]))
                    vx = float(np.mean(flow[..., 0][motion_pixels]))
                    vy = float(np.mean(flow[..., 1][motion_pixels]))
                    coherence = math.hypot(vx, vy) / (mean_mag + 1e-6)

                    # --- Tier-A controls (pre-flow decision): compute effective thresholds using previous severity
                    try:
                        base_mag = float(FLOW_MAG_TH)
                        base_arm = float(self._p("ARM_MIN_TIME_S", g, ARM_MIN_TIME_S))
                        base_tiny = float(self._p("WIND_TINY_FACTOR", g, WIND_TINY_FACTOR))
                        s_prev = float(getattr(self, "wind_severity", 0.0))
                        if int(globals().get("WIND_GOV_CONTROLS", 0)) and (
                            self.wind_state == "Windy"
                        ):
                            mag_gain = float(globals().get("WIND_GOV_MAG_GAIN", 0.30))
                            arm_add = float(globals().get("WIND_GOV_ARM_MIN_ADD_S", 0.40))
                            tiny_g = float(globals().get("WIND_GOV_TINY_GAIN", 1.00))
                            self.flow_mag_thresh_eff = base_mag * (1.0 + mag_gain * s_prev)
                            self.arm_min_time_eff = base_arm + arm_add * s_prev
                            self.wind_tiny_factor_eff = base_tiny * (1.0 + tiny_g * s_prev)
                        else:
                            self.flow_mag_thresh_eff = base_mag
                            self.arm_min_time_eff = base_arm
                            self.wind_tiny_factor_eff = base_tiny
                    except Exception:
                        pass

                    wind_like_raw = (
                        mean_mag < float(self.flow_mag_thresh_eff) and coherence < FLOW_COH_TH
                    )
                    mean_mag_inst = mean_mag
                    coherence_inst = coherence

                    # temporal window update
                    den = math.hypot(vx, vy) + 1e-6
                    ux, uy = (vx / den, vy / den)
                    self.flow_hist.append((ts, ux, uy, mean_mag))
                    # --- Pre-arm warm-fill (global) ---
                    try:
                        if int(globals().get("WARMFILL_ENABLE", 1)) and len(self.flow_hist) == 1:
                            wf = float(globals().get("FLOW_WARMFILL_S", 0.0))
                            if wf > 0.0:
                                # Seed a synthetic older sample so span ≈ wf immediately.
                                self.flow_hist.appendleft((ts - wf, ux, uy, mean_mag))
                    except Exception:
                        pass
                    cutoff = ts - float(FLOW_WIN_S)
                    while self.flow_hist and self.flow_hist[0][0] < cutoff:
                        self.flow_hist.popleft()
                    N = len(self.flow_hist)
                    flow_win_N = N
                    if N >= 2:
                        sx = sum(u for (_t, u, _v, _m) in self.flow_hist)
                        sy = sum(v for (_t, _u, v, _m) in self.flow_hist)
                        tdc = math.hypot(sx, sy) / float(N)
                        mags = [m for (_t, _u, _v, m) in self.flow_hist]
                        mbar = float(sum(mags)) / float(N) if N > 0 else 0.0
                        sdev = float(np.std(mags)) if N > 1 else 0.0
                        tss = 1.0 - (sdev / (mbar + 1e-6))
                        # direction flip rate
                        flips = 0
                        pairs = 0
                        for i in range(1, N):
                            ux1, uy1 = self.flow_hist[i - 1][1], self.flow_hist[i - 1][2]
                            ux2, uy2 = self.flow_hist[i][1], self.flow_hist[i][2]
                            dot = max(-1.0, min(1.0, ux1 * ux2 + uy1 * uy2))
                            ang = math.degrees(math.acos(dot))
                            if ang >= FLOW_DFR_ANG:
                                flips += 1
                            pairs += 1
                        dfr = (float(flips) / float(pairs)) if pairs > 0 else 0.0
                        tdc_win = max(0.0, min(1.0, tdc))
                        tss_win = max(0.0, min(1.0, tss))
                        dfr_win = max(0.0, min(1.0, dfr))

                    # --------- NEW: time-only eligibility + max-gap guard ----------
                    # Fill-level by time (span of retained samples) and ensure samples are fresh.
                    win_span_s = (self.flow_hist[-1][0] - self.flow_hist[0][0]) if N >= 2 else 0.0
                    last_gap_s = (ts - self.flow_hist[-1][0]) if N >= 1 else 1e9
                    # require ~90% of the configured window to be covered, and the last sample to be recent
                    win_eligible = bool(
                        (win_span_s >= (0.90 * FLOW_WIN_S)) and (last_gap_s <= FLOW_MAX_GAP_S)
                    )
                    # cache for HUD
                    self.last_flow_win_span_s = float(win_span_s)
                    self.last_flow_gap_s = float(last_gap_s)
                    self.last_win_eligible = bool(win_eligible)
                    keep_temporal = bool(
                        win_eligible
                        and (
                            (tdc_win is not None and tdc_win <= WIND_KEEP_TDC_MAX_)
                            or (tss_win is not None and tss_win <= WIND_KEEP_TSS_MAX_)
                            or (dfr_win is not None and dfr_win >= WIND_KEEP_DFR_MIN_)
                        )
                    )
                    tiny_veto = float(area_frac_weighted) < float(
                        WIND_TINY_FRAC_ * float(getattr(self, "wind_tiny_factor_eff", 1.0))
                    )

                    # real-motion override dwell
                    if REAL_OVR_ON:
                        ovr_ok = bool(
                            win_eligible
                            and (coherence is not None and coherence >= OVR_COH_MIN)
                            and (tdc_win is not None and tdc_win >= OVR_TDC_MIN)
                            and (tss_win is not None and tss_win >= OVR_TSS_MIN)
                            and (dfr_win is not None and dfr_win <= OVR_DFR_MAX)
                            and (area_frac_weighted >= OVR_MIN_FRAC)
                        )
                        if ovr_ok:
                            self.real_override_timer += dt
                        else:
                            self.real_override_timer = 0.0
                        real_override = self.real_override_timer >= OVR_DWELL_S
                    else:
                        real_override = False
                        self.real_override_timer = 0.0

                    wind_like = bool(wind_like_raw or keep_temporal or tiny_veto)
                    # Update temporal metric EMAs for Wind Governor
                    try:
                        alpha_w = 1.0 - math.exp(
                            -max(1e-3, dt or 0.033)
                            / max(1e-3, float(globals().get("WIND_EMA_TAU_S", 3.0)))
                        )
                    except Exception:
                        alpha_w = 0.3
                    # Update EMAs; if metric missing this frame, decay toward calm baselines
                    if dfr_win is not None:
                        self.dfr_ema = (1.0 - alpha_w) * float(
                            self.dfr_ema or 0.0
                        ) + alpha_w * float(dfr_win)
                    else:
                        self.dfr_ema = (1.0 - alpha_w) * float(
                            self.dfr_ema or 0.0
                        )  # → 0.0 baseline
                    if tdc_win is not None:
                        self.tdc_ema = (1.0 - alpha_w) * float(
                            self.tdc_ema or 1.0
                        ) + alpha_w * float(tdc_win)
                    else:
                        self.tdc_ema = (1.0 - alpha_w) * float(
                            self.tdc_ema or 1.0
                        ) + alpha_w * 1.0  # → 1.0 baseline
                    if tss_win is not None:
                        self.tss_ema = (1.0 - alpha_w) * float(
                            self.tss_ema or 1.0
                        ) + alpha_w * float(tss_win)
                    else:
                        self.tss_ema = (1.0 - alpha_w) * float(
                            self.tss_ema or 1.0
                        ) + alpha_w * 1.0  # → 1.0 baseline
                    if real_override:
                        wind_like = False

                    # EMAs + gustiness std
                    if self.mag_ema is None:
                        self.mag_ema = mean_mag
                    if self.coh_ema is None:
                        self.coh_ema = coherence
                    try:
                        alpha = 1.0 - math.exp(-max(1e-3, dt or 0.033) / max(1e-3, WIND_EMA_TAU))
                    except Exception:
                        alpha = 0.3
                    self.mag_ema = (1.0 - alpha) * self.mag_ema + alpha * mean_mag
                    self.coh_ema = (1.0 - alpha) * self.coh_ema + alpha * coherence
                    self.mag_hist.append((ts, self.mag_ema))
                    cutoff_hist = ts - 10.0
                    while self.mag_hist and self.mag_hist[0][0] < cutoff_hist:
                        self.mag_hist.popleft()
            finally:
                t_flow_ms = (time.time() - t0) * 1000.0

        # Keep span/gap eligibility fresh for HUD even on non-flow frames:
        try:
            cutoff = ts - float(FLOW_WIN_S)
            while self.flow_hist and self.flow_hist[0][0] < cutoff:
                self.flow_hist.popleft()
            N2 = len(self.flow_hist)
            span_s = (self.flow_hist[-1][0] - self.flow_hist[0][0]) if N2 >= 2 else 0.0
            gap_s = (ts - self.flow_hist[-1][0]) if N2 >= 1 else 1e9
            elig = bool((span_s >= (0.90 * FLOW_WIN_S)) and (gap_s <= FLOW_MAX_GAP_S))
            self.last_flow_win_span_s = float(span_s)
            self.last_flow_gap_s = float(gap_s)
            self.last_win_eligible = bool(elig)

            # --- NEW: decay Wind Governor EMAs even when no flow sample this frame ---
            # Pull EMAs toward "calm" baselines on idle frames so wind_idx can fall naturally.
            try:
                alpha_w = 1.0 - math.exp(
                    -max(1e-3, (dt or 0.033))
                    / max(1e-3, float(globals().get("WIND_EMA_TAU_S", 3.0)))
                )
            except Exception:
                alpha_w = 0.3
            if not do_flow:
                # DFR → 0, TDC/TSS → 1
                self.dfr_ema = (1.0 - alpha_w) * float(self.dfr_ema or 0.0)
                base_tdc = float(self.tdc_ema if self.tdc_ema is not None else 1.0)
                base_tss = float(self.tss_ema if self.tss_ema is not None else 1.0)
                self.tdc_ema = (1.0 - alpha_w) * base_tdc + alpha_w * 1.0
                self.tss_ema = (1.0 - alpha_w) * base_tss + alpha_w * 1.0
                # If we've gone well past the freshness gap, clamp to calm immediately.
                if gap_s > (2.0 * float(FLOW_MAX_GAP_S)):
                    self.dfr_ema = 0.0
                    self.tdc_ema = 1.0
                    self.tss_ema = 1.0
        except Exception:
            pass

        self.prev_small_gray = small_gray
        self.flow_frame_i += 1

        # 10s std from EMA history
        try:
            vals = [v for (_t, v) in self.mag_hist]
            mstd = float(np.std(vals)) if len(vals) > 1 else 0.0
        except Exception:
            mstd = 0.0

        # ---------------- ZONES (instant + temporal per-zone; measure-only) ----------------
        zones_out = None
        try:
            if int(globals().get("ZONES_ENABLED", 1)):
                gh = max(1, int(globals().get("ZONES_ROWS", 3)))
                gw = max(1, int(globals().get("ZONES_COLS", 3)))
                H, W = small_gray.shape[:2]
                # Crop to multiples of grid (avoid ragged edges)
                Hc = (H // gh) * gh
                Wc = (W // gw) * gw
                if Hc <= 0 or Wc <= 0:
                    raise RuntimeError("invalid grid crop")
                # Effective motion mask after hard mask / guard
                eff_mask = (
                    cv2.bitwise_and(mask, self.hard_mask_small)
                    if not expo_guard_active
                    else np.zeros_like(mask)
                )
                K = (eff_mask[:Hc, :Wc] > 0).astype(np.float32).reshape(gh, Hc // gh, gw, Wc // gw)
                zone_area = float((Hc // gh) * (Wc // gw))
                # AREA FRAC per zone
                area_px = K.sum(axis=(1, 3))
                area_frac = (area_px / max(1.0, zone_area)).astype(np.float32)
                # MAG & COH per zone (if we ran flow this frame)
                if (
                    (mag_full is not None)
                    and (flow_vx_full is not None)
                    and (flow_vy_full is not None)
                ):
                    M = mag_full[:Hc, :Wc].reshape(gh, Hc // gh, gw, Wc // gw)
                    VX = flow_vx_full[:Hc, :Wc].reshape(gh, Hc // gh, gw, Wc // gw)
                    VY = flow_vy_full[:Hc, :Wc].reshape(gh, Hc // gh, gw, Wc // gw)
                    mag_sum = (M * K).sum(axis=(1, 3))  # (gh, gw)
                    vx_sum = (VX * K).sum(axis=(1, 3))
                    vy_sum = (VY * K).sum(axis=(1, 3))
                    cnt = area_px + 1e-6
                    mag_inst_z = (mag_sum / cnt).astype(np.float32)  # mean magnitude in zone
                    coh_inst_z = (np.sqrt(vx_sum**2 + vy_sum**2) / (mag_sum + 1e-6)).astype(
                        np.float32
                    )
                    energy = mag_sum  # use energy for entropy
                    # v2: per-zone unit directions from vector sums
                    denom = np.sqrt(vx_sum**2 + vy_sum**2) + 1e-6
                    ux_z = (vx_sum / denom).astype(np.float32)
                    uy_z = (vy_sum / denom).astype(np.float32)
                else:
                    # No flow sample this frame; still report area, zero-fill others
                    mag_inst_z = np.zeros_like(area_frac, dtype=np.float32)
                    coh_inst_z = np.zeros_like(area_frac, dtype=np.float32)
                    energy = area_px.astype(np.float32)  # proxy so entropy has a shape
                    ux_z = np.zeros_like(area_frac, dtype=np.float32)
                    uy_z = np.zeros_like(area_frac, dtype=np.float32)
                # Summaries
                zones_active = int(
                    np.count_nonzero(
                        area_frac >= float(globals().get("ZONES_MIN_AREA_FRAC", 0.005))
                    )
                )
                zones_active_frac = float(zones_active) / float((gh * gw) or 1)
                # Update zones_active EMA
                try:
                    alpha_w = 1.0 - math.exp(
                        -max(1e-3, dt or 0.033)
                        / max(1e-3, float(globals().get("WIND_EMA_TAU_S", 3.0)))
                    )
                except Exception:
                    alpha_w = 0.3
                if self.zones_act_ema is None:
                    self.zones_act_ema = zones_active_frac
                else:
                    self.zones_act_ema = (1.0 - alpha_w) * float(
                        self.zones_act_ema or 0.0
                    ) + alpha_w * zones_active_frac
                score = mag_inst_z * coh_inst_z
                top_flat = int(np.argmax(score))
                top_r, top_c = (top_flat // gw), (top_flat % gw)
                e = energy.astype(np.float64)
                es = float(e.sum())
                if es > 0:
                    p = (e / es).ravel()
                    # entropy in nats; OK for relative comparisons
                    energy_entropy = float(-np.sum(p * np.log(p + 1e-12)))
                else:
                    energy_entropy = 0.0

                # ---------- v2: maintain per-zone deques and compute TDC/TSS/DFR ----------
                # (Re)size history on grid change
                if (self._zone_hist is None) or (self._zones_grid != (gh, gw)):
                    self._zones_grid = (gh, gw)
                    self._zone_hist = [deque() for _ in range(gh * gw)]
                # live params
                Z_WIN = float(
                    globals().get("ZONES_TEMP_WIN_S", float(globals().get("FLOW_TEMP_WIN_S", 0.6)))
                )
                Z_GAP = float(
                    globals().get(
                        "ZONES_MAX_GAP_S", float(globals().get("FLOW_MAX_SAMPLE_GAP_S", 0.20))
                    )
                )
                # Append new samples only on frames where we had a flow sample for that zone
                if mag_full is not None:
                    for r in range(gh):
                        for c in range(gw):
                            i = r * gw + c
                            # only append when the zone had any motion pixels (same condition as mag_sum>0)
                            if area_px[r, c] > 0:
                                dq = self._zone_hist[i]
                                dq.append(
                                    (
                                        ts,
                                        float(ux_z[r, c]),
                                        float(uy_z[r, c]),
                                        float(mag_inst_z[r, c]),
                                    )
                                )
                                # --- Pre-arm warm-fill (per-zone) ---
                                try:
                                    if int(globals().get("WARMFILL_ENABLE", 1)) and len(dq) == 1:
                                        zwf = float(
                                            globals().get(
                                                "ZONES_WARMFILL_S",
                                                float(globals().get("FLOW_WARMFILL_S", 0.0)),
                                            )
                                        )
                                        if zwf > 0.0:
                                            dq.appendleft(
                                                (
                                                    ts - zwf,
                                                    float(ux_z[r, c]),
                                                    float(uy_z[r, c]),
                                                    float(mag_inst_z[r, c]),
                                                )
                                            )
                                except Exception:
                                    pass
                # Prune old samples in all zones; compute metrics
                tdc_z = np.zeros((gh, gw), dtype=np.float32)
                tss_z = np.zeros((gh, gw), dtype=np.float32)
                dfr_z = np.zeros((gh, gw), dtype=np.float32)
                span_z = np.zeros((gh, gw), dtype=np.float32)
                gap_z = np.full((gh, gw), 1e9, dtype=np.float32)
                elig_z = np.zeros((gh, gw), dtype=np.uint8)
                for r in range(gh):
                    for c in range(gw):
                        i = r * gw + c
                        dq = self._zone_hist[i]
                        # prune by time window
                        cutoff = ts - Z_WIN
                        while dq and dq[0][0] < cutoff:
                            dq.popleft()
                        N = len(dq)
                        if N >= 1:
                            gap_z[r, c] = float(ts - dq[-1][0])
                        if N >= 2:
                            span_z[r, c] = float(dq[-1][0] - dq[0][0])
                        # eligibility by time coverage + freshness
                        elig = (span_z[r, c] >= (0.90 * Z_WIN)) and (gap_z[r, c] <= Z_GAP)
                        elig_z[r, c] = 1 if elig else 0
                        if N >= 2:
                            sx = sum(u for (_t, u, _v, _m) in dq)
                            sy = sum(v for (_t, _u, v, _m) in dq)
                            tdc = math.hypot(sx, sy) / float(N)
                            mags = [m for (_t, _u, _v, m) in dq]
                            mbar = float(sum(mags)) / float(N) if N > 0 else 0.0
                            sdev = float(np.std(mags)) if N > 1 else 0.0
                            tss = 1.0 - (sdev / (mbar + 1e-6))
                            flips = 0
                            pairs = 0
                            for k in range(1, N):
                                ux1, uy1 = dq[k - 1][1], dq[k - 1][2]
                                ux2, uy2 = dq[k][1], dq[k][2]
                                dot = max(-1.0, min(1.0, ux1 * ux2 + uy1 * uy2))
                                ang = math.degrees(math.acos(dot))
                                if ang >= float(globals().get("FLOW_DFR_ANGLE_DEG", 45.0)):
                                    flips += 1
                                pairs += 1
                            dfr = (float(flips) / float(pairs)) if pairs > 0 else 0.0
                            tdc_z[r, c] = max(0.0, min(1.0, float(tdc)))
                            tss_z[r, c] = max(0.0, min(1.0, float(tss)))
                            dfr_z[r, c] = max(0.0, min(1.0, float(dfr)))

                zones_out = {
                    "grid": [int(gh), int(gw)],
                    "zones_active": zones_active,
                    "top_zone": [int(top_r), int(top_c)],
                    "energy_entropy": energy_entropy,
                    "area_frac_flat": area_frac.astype(float).ravel().tolist(),
                    "mag_inst_flat": mag_inst_z.astype(float).ravel().tolist(),
                    "coh_inst_flat": coh_inst_z.astype(float).ravel().tolist(),
                    # v2 temporal outputs (flattened row-major; measure-only)
                    "win_s": float(Z_WIN),
                    "max_gap_s": float(Z_GAP),
                    "tdc_flat": tdc_z.astype(float).ravel().tolist(),
                    "tss_flat": tss_z.astype(float).ravel().tolist(),
                    "dfr_flat": dfr_z.astype(float).ravel().tolist(),
                    "span_s_flat": span_z.astype(float).ravel().tolist(),
                    "gap_s_flat": gap_z.astype(float).ravel().tolist(),
                    "eligible_flat": [int(x) for x in elig_z.ravel().tolist()],
                    "zones_eligible": int(elig_z.sum()),
                }
        except Exception:
            zones_out = None

        # ---- Wind Governor index & state machine (log-only, no control yet) ----
        try:
            if int(globals().get("WIND_GOV_ENABLED", 1)):
                wts = globals().get("WIND_GOV_WEIGHTS", [0.40, 0.25, 0.20, 0.15])
                dfr_e = float(self.dfr_ema or 0.0)
                zaf_e = float(self.zones_act_ema or 0.0)
                # Calm defaults → TDC/TSS=1.0 so (1−TDC/TSS)=0.0 when unknown
                tdc_e = float(self.tdc_ema if self.tdc_ema is not None else 1.0)
                tss_e = float(self.tss_ema if self.tss_ema is not None else 1.0)
                one_minus_tdc = max(0.0, 1.0 - tdc_e)
                one_minus_tss = max(0.0, 1.0 - tss_e)
                idx = (
                    float(wts[0]) * dfr_e
                    + float(wts[1]) * zaf_e
                    + float(wts[2]) * one_minus_tdc
                    + float(wts[3]) * one_minus_tss
                )
                self.wind_idx = max(0.0, min(1.0, idx))
                now = float(ts)
                # --- Calm gate: if scene is truly quiet, force calm/zero index
                if (zaf_e < 0.005) and (dfr_e < 0.02):
                    self.wind_idx = 0.0
                    self.wind_state = "Calm"
                    self.wind_severity = 0.0
                    self._wind_candidate_since = 0.0
                    self._wind_last_state_change_ts = now
                else:
                    enter_th = float(globals().get("WIND_GOV_ENTER", 0.65))
                    exit_th = float(globals().get("WIND_GOV_EXIT", 0.45))
                    need_enter_s = float(globals().get("WIND_GOV_ENTER_S", 3.0))
                    need_exit_s = float(globals().get("WIND_GOV_EXIT_S", 5.0))
                    if self.wind_state == "Calm":
                        if self.wind_idx > enter_th:
                            self._wind_candidate_since = self._wind_candidate_since or now
                            if (now - self._wind_candidate_since) >= need_enter_s:
                                self.wind_state = "Windy"
                                self._wind_candidate_since = 0.0
                                self._wind_last_state_change_ts = now
                        else:
                            self._wind_candidate_since = 0.0
                    else:
                        if self.wind_idx < exit_th:
                            self._wind_candidate_since = self._wind_candidate_since or now
                            if (now - self._wind_candidate_since) >= need_exit_s:
                                self.wind_state = "Calm"
                                self._wind_candidate_since = 0.0
                                self._wind_last_state_change_ts = now
                        else:
                            self._wind_candidate_since = 0.0
                # Severity ∈ [0,1] when Windy (0 in Calm)
                if self.wind_state == "Windy" and enter_th > exit_th:
                    self.wind_severity = max(
                        0.0, min(1.0, (self.wind_idx - exit_th) / (enter_th - exit_th))
                    )
                else:
                    self.wind_severity = 0.0
            else:
                self.wind_idx = 0.0
                self.wind_state = "Calm"
                self.wind_severity = 0.0
        except Exception:
            self.wind_severity = 0.0

        # ---- Recompute effective thresholds using current severity (for telemetry/logs) ----
        try:
            base_mag = float(self._p("FLOW_MAG_THRESH", g, FLOW_MAG_THRESH))
            base_arm = float(self._p("ARM_MIN_TIME_S", g, ARM_MIN_TIME_S))
            base_tiny = float(self._p("WIND_TINY_FACTOR", g, WIND_TINY_FACTOR))
            s_now = float(getattr(self, "wind_severity", 0.0))
            if int(globals().get("WIND_GOV_CONTROLS", 0)) and (self.wind_state == "Windy"):
                mag_gain = float(globals().get("WIND_GOV_MAG_GAIN", 0.30))
                arm_add = float(globals().get("WIND_GOV_ARM_MIN_ADD_S", 0.40))
                tiny_g = float(globals().get("WIND_GOV_TINY_GAIN", 1.00))
                self.flow_mag_thresh_eff = base_mag * (1.0 + mag_gain * s_now)
                self.arm_min_time_eff = base_arm + arm_add * s_now
                self.wind_tiny_factor_eff = base_tiny * (1.0 + tiny_g * s_now)
            else:
                self.flow_mag_thresh_eff = base_mag
                self.arm_min_time_eff = base_arm
                self.wind_tiny_factor_eff = base_tiny
        except Exception:
            pass

        return MotionResult(
            mask_small=mask,
            boxes_small=boxes_small,
            motion_list=motion_list,
            area_frac_weighted=float(area_frac_weighted),
            shadow_drop_frac=float(sh_drop),
            expo_guard_active=bool(expo_guard_active),
            expo_metrics=tuple(float(x) for x in (expo_metrics or (0.0, 0.0, 0.0, 0.0))),
            wind_like=bool(wind_like),
            real_override=bool(real_override),
            flow_inst_mag=None if mean_mag_inst is None else float(mean_mag_inst),
            flow_inst_coh=None if coherence_inst is None else float(coherence_inst),
            tdc=None if tdc_win is None else float(tdc_win),
            tss=None if tss_win is None else float(tss_win),
            dfr=None if dfr_win is None else float(dfr_win),
            flow_win_N=int(flow_win_N or 0),
            flow_win_span_s=float(self.last_flow_win_span_s),
            flow_last_gap_s=float(self.last_flow_gap_s),
            win_eligible=bool(self.last_win_eligible),
            ovr_min_frac_eff=float(OVR_MIN_FRAC),
            mag_ema=None if self.mag_ema is None else float(self.mag_ema),
            coh_ema=None if self.coh_ema is None else float(self.coh_ema),
            mag_std_10s=float(mstd),
            foliage_motion_frac=float(foliage_frac),
            wind_idx=float(self.wind_idx),
            wind_state=str(self.wind_state),
            wind_severity=float(getattr(self, "wind_severity", 0.0)),
            zones_active_frac=float(zones_active_frac) if "zones_active_frac" in locals() else 0.0,
            flow_mag_thresh_eff=float(
                getattr(self, "flow_mag_thresh_eff", globals().get("FLOW_MAG_THRESH", 0.0))
            ),
            arm_min_time_eff=float(
                getattr(self, "arm_min_time_eff", globals().get("ARM_MIN_TIME_S", 0.0))
            ),
            wind_tiny_factor_eff=float(
                getattr(self, "wind_tiny_factor_eff", globals().get("WIND_TINY_FACTOR", 1.0))
            ),
            t_motion_ms=float(t_motion_ms),
            t_flow_ms=float(t_flow_ms),
            real_ovr_timer_s=float(self.real_override_timer),
            zones=zones_out,
        )
