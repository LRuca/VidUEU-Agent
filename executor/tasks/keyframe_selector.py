from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Sequence, Tuple
import math
import cv2
import numpy as np
import os
import shutil

def _min_max_norm(xs: List[float]) -> List[float]:
    """Normalize a list of numbers into [0, 1]. If all equal, return all zeros."""
    if not xs:
        return []
    mn = min(xs)
    mx = max(xs)
    if mx <= mn:
        return [0.0 for _ in xs]
    return [(x - mn) / (mx - mn) for x in xs]


def _cosine_similarity(a: Sequence[float], b: Sequence[float], eps: float = 1e-8) -> float:
    """Simple cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for xa, xb in zip(a, b):
        xa = float(xa)
        xb = float(xb)
        dot += xa * xb
        na += xa * xa
        nb += xb * xb
    if na <= eps or nb <= eps:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def resolve_iteration_score_weights(value: Any) -> Dict[str, float]:
    """Normalize the outer-loop keyframe score policy.

    The policy names are deliberately small and stable.  Missing or invalid
    values retain the neutral multiplier so older configs keep their scoring
    behavior.  The selector applies these multipliers to its existing quality,
    semantic-change, and temporal-coverage components below.
    """
    source = value if isinstance(value, dict) else {}
    result: Dict[str, float] = {}
    for name in ("image_quality", "semantic_change", "temporal_coverage"):
        try:
            number = float(source.get(name, 1.0))
        except (TypeError, ValueError):
            number = 1.0
        result[name] = max(0.0, min(4.0, number))
    return result


@dataclass
class KeyframeSelector:
    """
    多信号关键帧选择器，用于 T2。
    """

    max_frames: int = 10

    # debug 输出
    debug: bool = True
    debug_max_list: int = 30

    # debug 信息（给 report 用）
    last_detail_scores: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    last_weights: Dict[str, float] = field(default_factory=dict)
    last_scores: Dict[int, float] = field(default_factory=dict)
    last_shot_boundaries: List[int] = field(default_factory=list)

    # Autonomous-iteration policy multipliers.  They are injected by the
    # planner from runtime.iteration_policy and remain neutral for legacy runs.
    iteration_score_weights: Dict[str, float] = field(default_factory=dict)

    # 输出给 T2（写入 output.transition_indices）
    last_transition_indices: List[int] = field(default_factory=list)

    # ✅ 用于把 frame_paths 传给 _auto_shot_boundaries（避免改函数签名）
    _last_frame_paths_for_cut: Optional[List[str]] = field(default=None, init=False, repr=False)

    # [ADDED] cache raw metrics for hard floors / report
    _last_quality_raw_metrics: Optional[List[Dict[str, float]]] = field(default=None, init=False, repr=False)

    # det 子打分权重
    det_obj_weight: float = 0.3
    det_center_weight: float = 0.4
    det_conf_weight: float = 0.3

    # ===================== ✅ embedding 多样性配置 =====================
    # embedding 多样性阈值（越大越宽松；max_sim >= th 会被拒绝）
    emb_diversity_sim_threshold: float = 0.87

    # ✅ diversity 不够下界时的“退火 + 时间惩罚”配置
    diversity_relax_enable: bool = True
    diversity_relax_step: float = 0.01
    diversity_relax_max_rounds: int = 8
    diversity_hard_sim_threshold: float = 0.93  #上界

    # 时间惩罚：w_time(d)=1-beta*exp(-d/tau)
    diversity_time_beta_start: float = 0.25
    diversity_time_beta_step: float = 0.15
    diversity_time_beta_max: float = 0.55
    diversity_time_gap_min: int = 2
    diversity_time_gap_max: int = 6

    # diversity 多 seed 尝试次数
    diversity_seed_trials: int = 6

    # ===================== ✅ transition 过滤配置 =====================
    transition_filter: bool = True
    transition_margin: int = 0

    # 子开关：分别控制每条证据链是否参与
    transition_dissolve_enable: bool = True   # cross-dissolve 拟合链
    transition_emb_enable: bool = True        # embedding 链
    transition_cut_enable: bool = True        # cut_scores run/mask 链（给 emb 提供支撑）

    # --- dissolve thresholds (原 _compute_cross_dissolve_mask 内常量) ---
    transition_pn_min: float = 0.08
    transition_res_max: float = 0.06
    transition_res_ratio_max: float = 0.65
    transition_local_min: float = 0.02
    transition_bal_ratio: float = 0.30
    transition_alpha_min: float = 0.03
    transition_alpha_max: float = 0.97

    # --- embedding thresholds (原 select 内常量) ---
    transition_emb_sim_min: float = 0.70
    transition_emb_span_max: float = 0.92
    transition_emb_delta_min: float = 0.04

    # --- cut thresholds (原 select 内常量) ---
    transition_cut_soft_th: float = 0.14   # 用于生成 transition_mask
    transition_cut_hard_th: float = 0.12   # emb_like 触发时的“更松”cut门槛
    transition_cut_run_len: int = 2        # 原 RUN_LEN

    # dissolve resize
    transition_dissolve_resize_h: int = 96
    transition_dissolve_resize_w: int = 96

    # dissolve balance eps (原来写死 +0.01)
    transition_bal_eps: float = 0.01

    # optional: embedding 是否必须依赖 cut 支撑
    transition_emb_require_cut: bool = True


    # ===================== ✅ IQA (MUSIQ) 配置 =====================
    iqa_musiq_enable: bool = True
    iqa_musiq_conda_env: str = "video_iqa"
    iqa_musiq_device: str = "cuda"      # auto|cuda|cpu
    iqa_musiq_timeout_s: int = 600
    iqa_musiq_weight: float = 0.15      # 作为一个新信号的权重基准（会参与归一化）
    iqa_musiq_pre_topk: int = 0         # 0=全算；>0=只算预筛 topK（提速用）
    # ============================================================

    # ===================== ✅ blurmap 配置 =====================
    blurmap_enable: bool = False

    # 输出/读取目录（空则默认 frames_dir/../blurmap）
    blurmap_dir: str = ""
    blurmap_ext: str = ".png"

    # 缺失时是否自动生成
    blurmap_generate_if_missing: bool = True
    blurmap_conda_env: str = "blurmap_tf1_gpu"
    blurmap_device: str = "auto"         # 透传给 compute_blurmap
    blurmap_timeout_s: int = 1800
    blurmap_require_all: bool = True

    blurmap_val_motion: int = 100
    blurmap_val_defocus: int = 200
    blurmap_grid: int = 16
    blurmap_cell_th: float = 0.15
    blurmap_conf_den: float = 0.20

    # 额外：自检开关
    blurmap_debug_unique: bool = True
    blurmap_debug_unique_maxvals: int = 16

    # ===================== ✅ blurmap thresholds (SEPARATE from legacy) =====================
    # 只在 blurmap_applied=True 时使用（gm_new/lm_new/df_new 这套分数）
    blurmap_global_motion_blur_score_th: float = 0.55
    blurmap_local_motion_blur_score_th: float = 0.55
    blurmap_defocus_blur_score_th: float = 0.60
    # ==============================================================================

    # ==========================================================

    # ===================== ✅ local motion blur 过滤配置 =====================
    local_motion_blur_filter: bool = True
    local_motion_blur_area_min: float = 0.02   # 运动区域占比太小不稳定
    local_motion_blur_area_max: float = 0.60   # 运动区域太大多半是相机运动/全局运动
    local_motion_blur_ratio_th: float = 0.7    # sharp_motion / sharp_static < th => motion 区域明显更糊
    local_motion_blur_static_th: float = 0.01  # 背景要“足够清晰”，否则整帧都糊不算“局部糊”
    local_motion_blur_score_penalty: float = 0.50  # 不能过滤时的降权倍率（<1 越小惩罚越重）

    # tile-based sharpness aggregation (reduce single strong-edge cheating)
    local_motion_blur_tile_grid_h: int = 4
    local_motion_blur_tile_grid_w: int = 4
    local_motion_blur_tile_inner_q: float = 80.0  # percentile inside each tile

    # global-blur fallback (when motion covers most of frame)
    # require enough motion (prev_next) to avoid static low-texture false positives
    local_motion_blur_prev_next_th: float = 0.02

    # Laplacian ratio 阈值：lap_motion / lap_static < th 更像局部糊
    local_motion_blur_lap_ratio_th: float = 0.75

    # anisotropy 增益阈值：anis_motion / anis_static > th 更像方向性拖影
    local_motion_blur_anis_gain_th: float = 1.35

    # score-based 总分阈值（>= 即判定 blur）
    local_motion_blur_score_th: float = 0.6
    # local_motion_blur_score_th: float = 0.25

    # score-based 各证据权重（尽量简单可调）
    local_motion_blur_w_sharp: float = 0.7   # sharp_ratio 证据
    local_motion_blur_w_lap: float = 0.6     # lap_ratio 证据
    local_motion_blur_w_anis: float = 0.35   # anisotropy 证据
    local_motion_blur_w_static: float = 0.25 # static 是否清晰的“加分项”（不再是硬门槛）

    # ✅ 新增：时域掉锐证据（最小侵入，可调）
    local_motion_blur_drop_th: float = 0.80     # sharp_motion / max(sharp_prev, sharp_next) < th 更像 blur
    local_motion_blur_w_drop: float = 0.40      # drop 证据权重
 
    # ========================================================================
        
    # ===================== ✅ global motion blur 过滤配置=====================

    global_motion_blur_score_th: float = 0.7
    # global_motion_blur_score_th: float = 0.30
    global_motion_blur_filter: bool = True

    # motion gate (prev_next = mean(|prev-next|) on small grayscale)
    global_motion_blur_prev_next_th: float = 0.03
    global_motion_blur_prev_next_strict: float = 0.05  # above this, edge_ratio requirement can be relaxed

    # avoid low-texture static false positives
    global_motion_blur_edge_ratio_min: float = 0.01

    # tile-aggregated sharpness threshold (Tenengrad tile_p25)
    global_motion_blur_sharp_p25_th: float = 0.18

    # temporal sharpness drop (relative) threshold
    global_motion_blur_drop_th: float = 0.80

    # score mixing weights
    global_motion_blur_w_sharp: float = 0.8
    global_motion_blur_w_drop: float = 0.4

    # edge-ratio stability gate (for suppressing false positives in global motion blur)
    global_motion_blur_edge_ratio_drop_gate: float = 0.65
    global_motion_blur_edge_ratio_drop_hard: float = 0.45

    # global motion blur tile params
    global_motion_blur_tile_grid_h: int = 4
    global_motion_blur_tile_grid_w: int = 4
    global_motion_blur_tile_inner_q: float = 80.0   # per-tile inner percentile
    # ========================================================================

    # ===================== ✅ defocus / global soft blur（非运动模糊）=====================
    defocus_blur_filter: bool = False

    # 连续段落判定：至少连续 N 帧才认为是“虚焦段”
    defocus_blur_run_len: int = 3

    # edge gate：避免低纹理误杀（地面/墙/天空）
    defocus_blur_edge_ratio_min: float = 0.01

    # tile-aggregated sharpness threshold (Tenengrad tile_p25)
    # 注意：这是“越低越糊”，所以阈值是 “低于它更像虚焦”
    defocus_blur_sharp_p25_th: float = 0.20

    # p50 通常 > p25，用一个倍率对齐（参考你 global_blur 里 1.6）
    defocus_blur_p50_scale: float = 1.6

    # score 触发阈值（单帧）
    defocus_blur_score_th: float = 0.95
    # defocus_blur_score_th: float = 0.50

    # defocus blur tile params
    defocus_blur_tile_grid_h: int = 4
    defocus_blur_tile_grid_w: int = 4
    defocus_blur_tile_inner_q: float = 90
    # ========================================================================

    # ===================== Hard floors 配置 =====================
    hard_floor_enable: bool = True
    hard_floor_allow_empty: bool = True

    # 固定尺度（用于 brightness/contrast/lap_var_small 的“绝对阈值”稳定）
    hard_floor_resize_hw: Tuple[int, int] = (160, 160)

    # A 类：可用性/画质底线（保守：默认基本不启用）
    hard_floor_min_area: float = 0.0              # e.g. 76800; 0=关闭
    hard_floor_bright_min: float = 0.0            # 0..255; 0=关闭
    hard_floor_bright_max: float = 255.0          # 255=关闭
    hard_floor_contrast_min: float = 0.0          # std on resized gray; 0=关闭
    hard_floor_lap_var_small_min: float = 0.0     # Laplacian var on resized gray; 0=关闭

    # ========================================================================

    # ===================== ✅ highlight blowout (局部过曝/高光炸) 过滤配置 =====================
    highlight_blowout_filter: bool = False
    highlight_blowout_rel_p: float = 99.5
    highlight_blowout_rel_min_th: float = 0.92

    # only count near-white & near-neutral pixels as "saturated highlight"
    highlight_blowout_abs_sat_th: float = 0.995
    highlight_blowout_neutral_chroma_th: float = 0.10   # |Cr-0.5|+|Cb-0.5| < th (0..1)
    highlight_blowout_sat_ratio_th: float = 0.35        # inside region

    # region size constraints
    highlight_blowout_region_area_min: float = 0.0015   # 0.15% image
    highlight_blowout_region_area_max: float = 0.06     # 6% image

    # locality contrast
    highlight_blowout_ring_dilate: int = 5
    highlight_blowout_delta_min: float = 0.12

    # texture suppression: true blowout tends to be smooth inside
    highlight_blowout_grad_in_max: float = 0.10         # abs grad mean (on Y)
    highlight_blowout_grad_in_ratio_max: float = 0.65   # grad_in / grad_global

    highlight_blowout_resize_hw: Tuple[int, int] = (160, 160)
    highlight_blowout_margin: int = 0




    # ============================================================

    # 输出给 report（仿 transition_indices）
    last_blowout_indices: List[int] = field(default_factory=list)






    # ===================== debug helpers =====================
    def _dbg(self, *args):
        if self.debug:
            print("[KeyframeSelector]", *args)

    def _dbg_list(self, name: str, xs: List[Any]):
        if not self.debug:
            return
        if xs is None:
            self._dbg(f"{name}=None")
            return
        head = xs[: self.debug_max_list]
        more = "" if len(xs) <= self.debug_max_list else f" ... (total={len(xs)})"
        self._dbg(f"{name} head={head}{more}")

    def _stats(self, xs: List[float]) -> str:
        if not xs:
            return "empty"
        arr = np.array([float(x) for x in xs], dtype=np.float32)
        return f"min={float(arr.min()):.4f}, max={float(arr.max()):.4f}, mean={float(arr.mean()):.4f}"

    # ===================== shot-wise norm utils =====================
    def _estimate_shot_boundaries(self, xs: List[float], min_shot_len: int = 4) -> List[int]:
        """
        从 per-frame 变化序列 xs 中估计镜头边界（返回 shot 起始 index 列表）。
        使用 median + 4*MAD，切点间隔 >= min_shot_len
        """
        n = len(xs)
        if n <= 0:
            return [0]
        arr = np.array([float(x) for x in xs], dtype=np.float32)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) + 1e-6
        th = med + 4.0 * mad

        cand = [i for i in range(1, n) if float(arr[i]) >= th]

        bds = [0]
        last = 0
        for i in cand:
            if i - last >= min_shot_len:
                bds.append(i)
                last = i

        bds = sorted(set(int(b) for b in bds if 0 <= int(b) < n))
        if not bds:
            bds = [0]
        return bds

    def _shotwise_minmax_norm(self, xs: List[float], shot_boundaries: List[int]) -> List[float]:
        n = len(xs)
        if n == 0:
            return []
        if not shot_boundaries or shot_boundaries == [0]:
            return _min_max_norm(xs)

        bds = sorted(set(int(b) for b in shot_boundaries if 0 <= int(b) < n))
        if 0 not in bds:
            bds = [0] + bds

        out = [0.0] * n
        for si, start in enumerate(bds):
            end = bds[si + 1] if si + 1 < len(bds) else n
            seg = [float(x) for x in xs[start:end]]
            seg_n = _min_max_norm(seg)
            out[start:end] = seg_n
        return out
    # ===================== ✅ blurmap helpers =====================
    def _resolve_blurmap_dir(self, frame_path: str) -> str:
        """
        固定 blurmap 目录：segment_name + '_blurmap'
        - 如果 frame 在 .../<segment>/frames/xxx.jpg，则 segment_dir=.../<segment>
        - 如果 frame 直接在 .../<segment>/xxx.jpg，则 segment_dir=.../<segment>
        """
        frames_dir = os.path.dirname(frame_path)
        base = (os.path.basename(frames_dir) or "").lower()

        # 常见结构：<segment>/frames/*.jpg
        if base in ("frames", "frame", "imgs", "images", "keyframes", "kf", "selected_frames"):
            segment_dir = os.path.dirname(frames_dir)
        else:
            segment_dir = frames_dir

        seg_name = os.path.basename(segment_dir) or "segment"
        return os.path.join(segment_dir, f"{seg_name}_blurmap")

    def _blurmap_path_for_frame(self, frame_path: str, idx: Optional[int] = None) -> str:
        ext = str(getattr(self, "blurmap_ext", ".png"))

        # ✅ 优先：如果外部显式指定 blurmap_dir，就用它；否则固定为 <segment_name>_blurmap
        bdir = str(getattr(self, "blurmap_dir", "") or "").strip()
        if not bdir:
            bdir = self._resolve_blurmap_dir(frame_path)

        naming = str(getattr(self, "blurmap_naming", "index") or "index")
        if naming == "index" and idx is not None:
            name = f"{idx:06d}{ext}"
        else:
            name = os.path.splitext(os.path.basename(frame_path))[0] + ext

        return os.path.join(bdir, name)



    def _load_blurmap_metrics(self, frame_paths: List[str]) -> Optional[List[Dict[str, float]]]:
        v_m = int(getattr(self, "blurmap_val_motion", 100))
        v_d = int(getattr(self, "blurmap_val_defocus", 200))
        G = int(getattr(self, "blurmap_grid", 16))
        cell_th = float(getattr(self, "blurmap_cell_th", 0.15))
        conf_den = float(getattr(self, "blurmap_conf_den", 0.35))

        out: List[Dict[str, float]] = []
        missing = 0

        for i, p in enumerate(frame_paths):
            bp = self._blurmap_path_for_frame(p, idx=i)
            if not os.path.exists(bp):
                missing += 1
                out.append({"r_m": 0.0, "r_d": 0.0, "occ_m": 0.0, "blob_m": 0.0, "conf_b": 0.0, "ok": 0.0})
                continue

            im = cv2.imread(bp, cv2.IMREAD_GRAYSCALE)
            if im is None:
                missing += 1
                out.append({"r_m": 0.0, "r_d": 0.0, "occ_m": 0.0, "blob_m": 0.0, "conf_b": 0.0, "ok": 0.0})
                continue

            # ===================== [ADDED] blurmap unique-values debug =====================
            # 只打印一次：第一张成功读到的 blurmap
            if bool(getattr(self, "blurmap_debug_unique", True)):
                try:
                    u = np.unique(im)
                    u = u[: int(getattr(self, "blurmap_debug_unique_maxvals", 16))]
                    self._dbg(f"[blurmap] unique vals sample (idx={i})={u.tolist()} v_m={v_m} v_d={v_d} file={bp}")
                    # 打完一次就关掉，避免刷屏
                    self.blurmap_debug_unique = False
                except Exception:
                    pass
            # =============================================================================

            H, W = im.shape[:2]
            HW = float(max(1, H * W))

            m_m = (im == v_m)
            m_d = (im == v_d)

            r_m = float(np.mean(m_m.astype(np.float32)))
            r_d = float(np.mean(m_d.astype(np.float32)))

            # occ_m: motion blur 是否铺开
            occ = 0.0
            if G > 1:
                hit = 0
                tot = 0
                for yy in range(G):
                    y0 = int(round(yy * H / G)); y1 = int(round((yy + 1) * H / G))
                    for xx in range(G):
                        x0 = int(round(xx * W / G)); x1 = int(round((xx + 1) * W / G))
                        tile = m_m[y0:y1, x0:x1]
                        if tile.size <= 0:
                            continue
                        tot += 1
                        if float(np.mean(tile.astype(np.float32))) >= cell_th:
                            hit += 1
                occ = float(hit / max(1, tot))

            # blob_m: motion blur 最大连通域占比
            blob = 0.0
            try:
                mm_u8 = (m_m.astype(np.uint8) * 255)
                num, labels, stats, _ = cv2.connectedComponentsWithStats(mm_u8, connectivity=8)
                if num > 1:
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    if areas.size > 0:
                        blob = float(np.max(areas) / HW)
            except Exception:
                blob = 0.0

            conf_b = float(min(1.0, max(0.0, (r_m + r_d) / max(1e-6, conf_den))))

            out.append({"r_m": r_m, "r_d": r_d, "occ_m": occ, "blob_m": blob, "conf_b": conf_b, "ok": 1.0})
        
        # ✅ 全失败才 fallback（不受 require_all 影响）
        if (len(frame_paths) - missing) <= 0:
            self._dbg(f"[blurmap] all blurmaps missing -> fallback")
            return None


        require_all = bool(getattr(self, "blurmap_require_all", True))
        if require_all and missing > 0:
            self._dbg(f"[blurmap] missing blurmaps: {missing}/{len(frame_paths)} -> fallback")
            return None

        self._dbg(f"[blurmap] loaded blurmaps ok={(len(frame_paths)-missing)}/{len(frame_paths)}")
        return out


    def _compute_prev_next_small(self, frame_paths: List[str], resize_hw=(160, 160)) -> List[float]:
        n = len(frame_paths)
        if n < 3:
            return [0.0] * n
        ims: List[Optional[np.ndarray]] = []
        for p in frame_paths:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if g is None:
                ims.append(None); continue
            g = cv2.resize(g, resize_hw, interpolation=cv2.INTER_AREA)
            ims.append(g.astype(np.float32) / 255.0)

        out = [0.0] * n
        for i in range(1, n - 1):
            A, B = ims[i - 1], ims[i + 1]
            if A is None or B is None:
                continue
            out[i] = float(np.mean(np.abs(A - B)))
        return out

    def _compute_edge_ratio_small(self, frame_paths: List[str], resize_hw=(160, 160)) -> List[float]:
        n = len(frame_paths)
        out = [0.0] * n
        for i, p in enumerate(frame_paths):
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            g = cv2.resize(g, resize_hw, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            ten = gx * gx + gy * gy
            try:
                edge_ratio, _, _ = self._edge_ratio_mask_from_ten(
                    ten,
                    abs_min=float(getattr(self, "edge_ratio_abs_min_small", 0.002)),
                    mad_k=float(getattr(self, "edge_ratio_mad_k_small", 6.0)),
                )
            except Exception:
                edge_ratio = 0.0
            out[i] = float(edge_ratio)
        return out


    def _compute_motion_area_ratio_small(self, frame_paths: List[str], resize_hw=(160, 160)) -> List[float]:
        n = len(frame_paths)
        if n < 3:
            return [0.0] * n
        ims: List[Optional[np.ndarray]] = []
        for p in frame_paths:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if g is None:
                ims.append(None); continue
            g = cv2.resize(g, resize_hw, interpolation=cv2.INTER_AREA)
            ims.append(g.astype(np.float32) / 255.0)

        out = [0.0] * n
        for i in range(1, n - 1):
            A, M, B = ims[i - 1], ims[i], ims[i + 1]
            if A is None or M is None or B is None:
                continue
            motion = 0.5 * (np.abs(M - A) + np.abs(B - M))
            vals = motion.reshape(-1)
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med))) + 1e-6
            base_th = med + 2.5 * mad
            q_th = float(np.percentile(vals, 75))
            th = max(0.02, q_th, base_th)
            out[i] = float(np.mean((motion > th).astype(np.float32)))
        return out

    def _clip01(self, x: float) -> float:
        if not math.isfinite(x):
            return 0.0
        return float(max(0.0, min(1.0, x)))
    
    def _edge_ratio_mask_from_ten(
        self,
        ten: np.ndarray,
        abs_min: float = 0.002,
        mad_k: float = 6.0,
    ) -> Tuple[float, float, np.ndarray]:
        """
        修复版定义：

        - edge_ratio：用于 gate 的“绝对边缘密度”
            edge_ratio = mean(ten > abs_min)
        这样低纹理帧会自然更小，高纹理帧更大，语义稳定。

        - edge_mask：用于 sharpness 的“强边缘掩码”（仍保留鲁棒阈值）
            th_mask = max(abs_min, median(ten) + mad_k * MAD)
            edge_mask = ten > th_mask
        这样 tile-aggregated sharpness 仍然聚焦强边缘，抗噪更稳。

        返回:
        edge_ratio, th_mask, edge_mask
        """
        vals = ten.reshape(-1).astype(np.float32)

        # --- edge_ratio for gating: ABSOLUTE threshold ---
        # 用 abs_min 做 gate 的阈值，不跟内容自适应，避免“纹理越强 ratio 越小”的反转问题
        abs_th = float(max(0.0, abs_min))
        edge_ratio = float(np.mean((ten > abs_th).astype(np.float32)))

        # --- edge_mask for sharpness aggregation: robust threshold ---
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) + 1e-6
        th_mask = float(max(abs_th, med + float(mad_k) * mad))
        edge_mask = (ten > th_mask)

        return edge_ratio, th_mask, edge_mask


    def _compute_blurmap_scores_new(
        self,
        blurm: List[Dict[str, float]],
        flow_norm: List[float],
        prev_next: List[float],
        cut_scores: List[float],
        edge_ratio: List[float],
        motion_area_ratio: List[float],
    ) -> Tuple[List[float], List[float], List[float]]:
        n = len(blurm)
        gm = [0.0] * n
        lm = [0.0] * n
        df = [0.0] * n

        pn_th = float(getattr(self, "global_motion_blur_prev_next_th", 0.03))
        pn_strict = float(getattr(self, "global_motion_blur_prev_next_strict", 0.05))
        edge_min = float(getattr(self, "defocus_blur_edge_ratio_min", 0.01))

        for i in range(n):
            r_m = float(blurm[i].get("r_m", 0.0))
            r_d = float(blurm[i].get("r_d", 0.0))
            occ_m = float(blurm[i].get("occ_m", 0.0))
            blob_m = float(blurm[i].get("blob_m", 0.0))
            conf_b = float(blurm[i].get("conf_b", 0.0))

            g_flow = self._clip01((float(flow_norm[i]) - 0.2) / (0.7 - 0.2 + 1e-6))
            g_pn = self._clip01((float(prev_next[i]) - pn_th) / (pn_strict - pn_th + 1e-6))
            G_motion = max(g_flow, g_pn)

            G_cut = 1.0 - self._clip01(float(cut_scores[i]) / 0.25)

            # global motion blur：铺开 + motion gate
            E_m = self._clip01(r_m / 0.30) * self._clip01(occ_m / 0.55)
            gm[i] = self._clip01(E_m * (0.5 + 0.5 * G_motion) * (0.6 + 0.4 * G_cut) * (0.5 + 0.5 * conf_b))

            # local motion blur：大块但不铺满 + local motion proxy
            E_loc = self._clip01(blob_m / 0.18) * self._clip01(r_m / 0.22) * (1.0 - self._clip01(occ_m / 0.55))
            g_area = self._clip01(
                (float(motion_area_ratio[i]) - float(getattr(self, "local_motion_blur_area_min", 0.02)))
                / (float(getattr(self, "local_motion_blur_area_max", 0.60)) - float(getattr(self, "local_motion_blur_area_min", 0.02)) + 1e-6)
            )
            G_global = self._clip01((float(flow_norm[i]) - 0.6) / (1.0 - 0.6 + 1e-6))
            G_local_motion = max(g_pn, g_area) * (1.0 - 0.7 * G_global)
            lm[i] = self._clip01(E_loc * (0.5 + 0.5 * G_local_motion) * (0.5 + 0.5 * conf_b))

            # defocus：defocus 占比高 + 非 motion + edge gate
            E_def = self._clip01(r_d / 0.35) * (1.0 - self._clip01(r_m / 0.20))
            G_still = 1.0 - max(g_flow, g_pn)
            er = float(edge_ratio[i])
            G_edge = self._clip01((er - edge_min) / (0.05 - edge_min + 1e-6))
            df[i] = self._clip01(E_def * (0.6 + 0.4 * G_still) * (0.5 + 0.5 * G_edge) * (0.5 + 0.5 * conf_b))

        return gm, lm, df

    def _ensure_blurmap_ready(self, frame_paths: List[str]) -> Optional[str]:
        """
        ✅ 强制重算 blurmap：每次调用都清空对应目录，然后重新生成。
        返回最终使用的 blurmap_dir；失败返回 None（让上层 fallback）。
        """
        if not frame_paths:
            return None

        # 固定：<segment_name>_blurmap
        bdir = self._resolve_blurmap_dir(frame_paths[0])
        self.blurmap_dir = bdir  # 写回，方便 debug/后续读取

        # ===== 强制清空（加安全护栏，避免误删）=====
        base = (os.path.basename(bdir) or "").lower()
        if "blurmap" not in base:
            self._dbg(f"[blurmap] REFUSE to clear suspicious dir (safety): {bdir}")
            return None

        try:
            if os.path.exists(bdir):
                shutil.rmtree(bdir)
            os.makedirs(bdir, exist_ok=True)
            self._dbg(f"[blurmap] cleared dir: {bdir}")
        except Exception as e:
            self._dbg(f"[blurmap] clear dir failed: {e} -> fallback")
            return None
        # =================================================

        # ===== 强制生成 =====
        try:
            from executor.models.blurmap_model import compute_blurmap
            self._dbg(f"[blurmap] force generating blurmaps into: {bdir}")
            compute_blurmap(
                frame_paths,
                out_dir=bdir,
                conda_env=str(getattr(self, "blurmap_conda_env", "blurmap_tf1_gpu")),
                device=str(getattr(self, "blurmap_device", "auto")),
                timeout_s=int(getattr(self, "blurmap_timeout_s", 1800)),
            )
        except Exception as e:
            self._dbg(f"[blurmap] generate failed: {e} -> fallback")
            return None

        # ===== 校验齐全（保留 require_all 语义）=====
        miss = 0
        for i, p in enumerate(frame_paths):
            bp = self._blurmap_path_for_frame(p, idx=i)
            if not os.path.exists(bp):
                miss += 1

        if miss > 0 and bool(getattr(self, "blurmap_require_all", True)):
            self._dbg(f"[blurmap] generate incomplete: missing {miss}/{len(frame_paths)} -> fallback")
            return None

        self._dbg(f"[blurmap] generate done, missing {miss}/{len(frame_paths)}")
        return bdir


    # =================================================================

    # ===================== cut score =====================
    def _compute_cut_scores(self, frame_paths: List[str]) -> List[float]:
        """
        cut_scores[i] 表示 frame i-1 -> i 的变化强度（越大越像硬切/大变化）
        方法：HSV 直方图 Bhattacharyya 距离 + 边缘图差
        """
        n = len(frame_paths)
        if n <= 1:
            return [0.0] * n

        def _read_small_bgr(p: str):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                return None
            h, w = img.shape[:2]
            scale = 160.0 / max(h, w) if max(h, w) > 160 else 1.0
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            return img

        def _hsv_hist(img_bgr):
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            hist = cv2.normalize(hist, None).flatten()
            return hist.astype(np.float32)

        def _edge_map(img_bgr):
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edge = cv2.Canny(gray, 60, 120)
            return edge.astype(np.float32) / 255.0

        cut_scores = [0.0] * n

        prev = _read_small_bgr(frame_paths[0])
        if prev is None:
            return cut_scores

        prev_hist = _hsv_hist(prev)
        prev_edge = _edge_map(prev)

        for i in range(1, n):
            cur = _read_small_bgr(frame_paths[i])
            if cur is None:
                cut_scores[i] = 0.0
                continue

            cur_hist = _hsv_hist(cur)
            cur_edge = _edge_map(cur)

            try:
                hist_dist = float(cv2.compareHist(prev_hist, cur_hist, cv2.HISTCMP_BHATTACHARYYA))
            except Exception:
                hist_dist = 0.0

            try:
                edge_diff = float(np.mean(np.abs(cur_edge - prev_edge)))
            except Exception:
                edge_diff = 0.0

            score = 0.75 * hist_dist + 0.25 * edge_diff
            if not math.isfinite(score):
                score = 0.0
            score = max(0.0, min(1.0, score))

            cut_scores[i] = float(score)

            prev_hist = cur_hist
            prev_edge = cur_edge

        return cut_scores

    def _compute_highlight_blowout_mask(
        self, frame_paths: List[str]
    ) -> Tuple[List[bool], Dict[int, Dict[str, float]]]:
        n = len(frame_paths)
        if n <= 0:
            return [], {}

        rh, rw = getattr(self, "highlight_blowout_resize_hw", (160, 160))
        rh, rw = int(rh), int(rw)
        if rh <= 8 or rw <= 8:
            rh, rw = 160, 160

        rel_p = float(getattr(self, "highlight_blowout_rel_p", 99.5))
        rel_min_th = float(getattr(self, "highlight_blowout_rel_min_th", 0.92))

        sat_th = float(getattr(self, "highlight_blowout_abs_sat_th", 0.995))
        chroma_th = float(getattr(self, "highlight_blowout_neutral_chroma_th", 0.10))
        sat_ratio_th = float(getattr(self, "highlight_blowout_sat_ratio_th", 0.35))

        area_min = float(getattr(self, "highlight_blowout_region_area_min", 0.0015))
        area_max = float(getattr(self, "highlight_blowout_region_area_max", 0.06))

        dil = int(getattr(self, "highlight_blowout_ring_dilate", 5))
        delta_min = float(getattr(self, "highlight_blowout_delta_min", 0.12))

        grad_in_max = float(getattr(self, "highlight_blowout_grad_in_max", 0.10))
        grad_ratio_max = float(getattr(self, "highlight_blowout_grad_in_ratio_max", 0.65))

        mask = [False] * n
        dbg: Dict[int, Dict[str, float]] = {}

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_dil = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(1, dil) * 2 + 1, max(1, dil) * 2 + 1)
        )
        eps = 1e-6

        for i, p in enumerate(frame_paths):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)

            ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            Y = ycc[:, :, 0].astype(np.float32) / 255.0
            Cr = ycc[:, :, 1].astype(np.float32) / 255.0
            Cb = ycc[:, :, 2].astype(np.float32) / 255.0

            # gradient on Y (texture proxy)
            gx = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=3)
            gmag = np.sqrt(gx * gx + gy * gy)
            g_global = float(gmag.mean())

            # relative bright threshold
            try:
                th_rel = float(np.percentile(Y.reshape(-1), rel_p))
            except Exception:
                th_rel = 1.0
            th_rel = max(th_rel, rel_min_th)

            bright = (Y >= th_rel).astype(np.uint8)

            # clean
            try:
                bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k, iterations=1)
                bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k, iterations=1)
            except Exception:
                pass

            H, W = Y.shape[:2]
            HW = float(max(1, H * W))
            bright_ratio = float(bright.mean())

            # if "bright region" is too large, likely white background / UI -> not blowout
            if bright_ratio > 0.35:
                dbg[i] = {
                    "th_rel": float(th_rel),
                    "bright_ratio": float(bright_ratio),
                    "reason": "too_large_bright_region",
                }
                continue

            # saturated & near-neutral mask (true clipped highlight tends to be near-white & low chroma)
            neutral = (np.abs(Cr - 0.5) + np.abs(Cb - 0.5)) <= chroma_th
            sat_neutral = (Y >= sat_th) & neutral

            try:
                num, labels, stats, _ = cv2.connectedComponentsWithStats(bright * 255, connectivity=8)
            except Exception:
                num, labels, stats = 0, None, None

            if num <= 1 or labels is None or stats is None:
                dbg[i] = {
                    "th_rel": float(th_rel),
                    "bright_ratio": float(bright_ratio),
                    "reason": "no_components",
                }
                continue

            is_blowout = False
            best = {
                "best_area": 0.0,
                "best_delta": 0.0,
                "best_sat_ratio": 0.0,
                "best_grad_in": 0.0,
                "best_grad_ratio": 0.0,
                "best_mean_in": 0.0,
                "best_mean_ring": 0.0,
            }

            for cid in range(1, num):
                area = float(stats[cid, cv2.CC_STAT_AREA]) / HW
                if area < area_min or area > area_max:
                    continue

                comp = (labels == cid)
                if not np.any(comp):
                    continue

                # ring
                comp_u8 = comp.astype(np.uint8)
                try:
                    dilated = cv2.dilate(comp_u8, k_dil, iterations=1).astype(bool)
                except Exception:
                    dilated = comp
                ring = dilated & (~comp)
                if not np.any(ring):
                    continue

                mean_in = float(Y[comp].mean())
                mean_ring = float(Y[ring].mean())
                delta = float(mean_in - mean_ring)

                # sat ratio inside component (neutral & near-white)
                sat_ratio = float(np.mean(sat_neutral[comp].astype(np.float32)))

                # texture suppression: blowout region should be smooth (low gradients)
                grad_in = float(gmag[comp].mean())
                grad_ratio = float(grad_in / (g_global + eps))

                # decision
                ok = (delta >= delta_min) and (sat_ratio >= sat_ratio_th) and (grad_in <= grad_in_max) and (grad_ratio <= grad_ratio_max)
                if ok:
                    is_blowout = True

                # best debug
                if (delta > best["best_delta"]) or (delta == best["best_delta"] and sat_ratio > best["best_sat_ratio"]):
                    best.update({
                        "best_area": float(area),
                        "best_delta": float(delta),
                        "best_sat_ratio": float(sat_ratio),
                        "best_grad_in": float(grad_in),
                        "best_grad_ratio": float(grad_ratio),
                        "best_mean_in": float(mean_in),
                        "best_mean_ring": float(mean_ring),
                    })

            mask[i] = bool(is_blowout)
            dbg[i] = {
                "th_rel": float(th_rel),
                "bright_ratio": float(bright_ratio),
                "g_global": float(g_global),

                "best_area": float(best["best_area"]),
                "best_delta": float(best["best_delta"]),
                "best_sat_ratio": float(best["best_sat_ratio"]),
                "best_grad_in": float(best["best_grad_in"]),
                "best_grad_ratio": float(best["best_grad_ratio"]),
                "best_mean_in": float(best["best_mean_in"]),
                "best_mean_ring": float(best["best_mean_ring"]),

                "sat_th": float(sat_th),
                "chroma_th": float(chroma_th),
                "delta_min": float(delta_min),
                "sat_ratio_th": float(sat_ratio_th),
                "grad_in_max": float(grad_in_max),
                "grad_ratio_max": float(grad_ratio_max),
            }

        return mask, dbg




    # ===================== cross dissolve detection (improved) =====================
    def _compute_cross_dissolve_mask(self, frame_paths: List[str]) -> Tuple[List[bool], Dict[int, Dict[str, float]]]:
        """
        更稳的 cross-dissolve 检测：
        - 基于“中间帧是否能被 prev/next 的线性混合拟合”
        mid ≈ α*prev + (1-α)*next
        返回:
        mask[i] True 表示 frame i 很像 cross-dissolve 中间帧
        dbg[i] 记录 alpha / residual / prev_next / prev_mid / mid_next（便于 debug）
        """
        n = len(frame_paths)
        if n < 3:
            return [False] * n, {}

        # ✅ 子开关：只控制 dissolve 这一条检测链路
        dissolve_on = bool(getattr(self, "transition_dissolve_enable", True))
        if not dissolve_on:
            return [False] * n, {}

        # ✅ resize 可配（默认保持你现在的 96x96）
        dh = int(getattr(self, "transition_dissolve_resize_h", 96))
        dw = int(getattr(self, "transition_dissolve_resize_w", 96))
        if dh <= 8 or dw <= 8:
            dh, dw = 96, 96

        smalls: List[Optional[np.ndarray]] = []
        for p in frame_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                smalls.append(None)
                continue
            img = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
            arr = img.astype(np.float32) / 255.0
            smalls.append(arr)

        mask = [False] * n
        dbg: Dict[int, Dict[str, float]] = {}

        PN_MIN = float(getattr(self, "transition_pn_min", 0.08))
        RES_MAX = float(getattr(self, "transition_res_max", 0.06))
        RES_RATIO_MAX = float(getattr(self, "transition_res_ratio_max", 0.65))
        LOCAL_MIN = float(getattr(self, "transition_local_min", 0.02))
        BAL_RATIO = float(getattr(self, "transition_bal_ratio", 0.30))
        ALPHA_MIN = float(getattr(self, "transition_alpha_min", 0.03))
        ALPHA_MAX = float(getattr(self, "transition_alpha_max", 0.97))
        BAL_EPS = float(getattr(self, "transition_bal_eps", 0.01))

        for i in range(1, n - 1):
            A = smalls[i - 1]
            M = smalls[i]
            B = smalls[i + 1]
            if A is None or M is None or B is None:
                continue

            prev_next = float(np.mean(np.abs(A - B)))
            if prev_next < PN_MIN:
                continue

            prev_mid = float(np.mean(np.abs(A - M)))
            mid_next = float(np.mean(np.abs(M - B)))

            D = (A - B).reshape(-1)
            Y = (M - B).reshape(-1)
            denom = float(np.dot(D, D)) + 1e-8
            alpha = float(np.dot(D, Y) / denom)
            alpha = max(0.0, min(1.0, alpha))

            pred = alpha * A + (1.0 - alpha) * B
            residual = float(np.mean(np.abs(pred - M)))

            if (ALPHA_MIN <= alpha <= ALPHA_MAX) and \
            (residual <= RES_MAX) and \
            (residual <= RES_RATIO_MAX * prev_next) and \
            (prev_mid >= LOCAL_MIN) and (mid_next >= LOCAL_MIN) and \
            (abs(prev_mid - mid_next) <= BAL_RATIO * prev_next + BAL_EPS):
                mask[i] = True
                dbg[i] = {
                    "alpha": alpha,
                    "residual": residual,
                    "prev_next": prev_next,
                    "prev_mid": prev_mid,
                    "mid_next": mid_next,
                }

        return mask, dbg


    # ===================== auto shot boundaries =====================
    def _auto_shot_boundaries(self, emb_change_raw: List[float], flow_raw: List[float]) -> List[int]:
        """
        优先用 cut_scores 找切点，并加“无切点”判定（大多数 segment 应该是 [0]）
        """
        frame_paths = getattr(self, "_last_frame_paths_for_cut", None)
        if not frame_paths:
            def _has_variation_local(arr: List[float], eps: float = 1e-6) -> bool:
                if not arr:
                    return False
                try:
                    mn = min(arr)
                    mx = max(arr)
                    if not math.isfinite(mn) or not math.isfinite(mx):
                        return False
                    return (mx - mn) > eps
                except Exception:
                    return False

            basis: Optional[List[float]] = None
            if len(emb_change_raw) == len(flow_raw) and _has_variation_local(emb_change_raw):
                basis = emb_change_raw
            elif _has_variation_local(flow_raw):
                basis = flow_raw

            if basis is None:
                return [0]
            return self._estimate_shot_boundaries(basis, min_shot_len=4)

        n = len(frame_paths)
        cut_scores = self._compute_cut_scores(frame_paths)
        if len(cut_scores) != n:
            return [0]

        max_s = max(cut_scores) if cut_scores else 0.0
        ABS_NO_CUT_TH = 0.25
        if max_s < ABS_NO_CUT_TH:
            return [0]

        arr = np.array([float(x) for x in cut_scores], dtype=np.float32)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) + 1e-6
        th = max(ABS_NO_CUT_TH, med + 4.0 * mad)

        cand = [i for i in range(1, n) if float(arr[i]) >= th]

        min_shot_len = 4
        bds = [0]
        last = 0
        for i in cand:
            if i - last >= min_shot_len:
                bds.append(i)
                last = i

        bds = sorted(set(int(b) for b in bds if 0 <= int(b) < n))
        if not bds:
            bds = [0]
        return bds

    # ===================== YOLO score =====================
    def _compute_yolo_scores(
        self,
        frame_paths: List[str],
        det_results: Optional[Dict[str, Any]],
    ) -> List[float]:
        """
        从 YOLO 原始检测结果构造每帧 det_score ∈ [0,1]
        假设 bbox 为像素坐标 xyxy（x1,y1,x2,y2），与当前 YOLO 输出一致。
        """
        n = len(frame_paths)
        if n == 0 or not det_results:
            return [0.0] * n

        def _get_det_list(item: Any) -> List[Dict[str, Any]]:
            if isinstance(item, dict):
                v = item.get("detections") or item.get("objects") or item.get("results") or []
                return v if isinstance(v, list) else []
            if isinstance(item, list):
                return item
            return []

        def _get_img_wh(p: str, item: Any) -> Tuple[float, float]:
            # 1) prefer stored size in det_results
            if isinstance(item, dict):
                iw = item.get("img_w", None)
                ih = item.get("img_h", None)
                if isinstance(iw, (int, float)) and isinstance(ih, (int, float)) and iw > 1 and ih > 1:
                    return float(iw), float(ih)

            # 2) fallback: read image header (minimal but robust)
            try:
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is not None:
                    h, w = img.shape[:2]
                    if w > 1 and h > 1:
                        return float(w), float(h)
            except Exception:
                pass

            # 3) ultimate fallback
            return 1.0, 1.0

        num_objects_raw: List[float] = [0.0] * n
        center_cov_raw: List[float] = [0.0] * n
        mean_conf_raw: List[float] = [0.0] * n

        for idx, p in enumerate(frame_paths):
            item = det_results.get(p, None)
            det_list = _get_det_list(item)
            if not det_list:
                continue

            W, H = _get_img_wh(p, item)
            img_area = max(W * H, 1.0)

            cx1, cy1 = 0.25 * W, 0.25 * H
            cx2, cy2 = 0.75 * W, 0.75 * H

            confs: List[float] = []
            center_area = 0.0
            valid = 0

            for d in det_list:
                if not isinstance(d, dict):
                    continue
                bbox = d.get("bbox") or d.get("box")
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue

                try:
                    x1, y1, x2, y2 = map(float, bbox)
                except Exception:
                    continue

                # xyxy sanity (avoid negative/flip boxes)
                if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue

                # clamp to image bounds (safe)
                x1 = max(0.0, min(x1, W))
                x2 = max(0.0, min(x2, W))
                y1 = max(0.0, min(y1, H))
                y2 = max(0.0, min(y2, H))
                if x2 <= x1 or y2 <= y1:
                    continue

                valid += 1

                ix1 = max(x1, cx1)
                iy1 = max(y1, cy1)
                ix2 = min(x2, cx2)
                iy2 = min(y2, cy2)
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                center_area += iw * ih

                # conf key fallback
                try:
                    c = d.get("score", None)
                    if c is None:
                        c = d.get("conf", None)
                    if c is None:
                        c = d.get("confidence", 0.0)
                    confs.append(float(c))
                except Exception:
                    confs.append(0.0)

            if valid <= 0:
                continue

            num_objects_raw[idx] = float(valid)
            center_cov_raw[idx] = float(min(1.0, center_area / img_area))
            mean_conf_raw[idx] = float(sum(confs) / len(confs)) if confs else 0.0

        num_norm = _min_max_norm(num_objects_raw)
        center_norm = _min_max_norm(center_cov_raw)
        conf_norm = _min_max_norm(mean_conf_raw)

        det_scores: List[float] = []
        for i in range(n):
            det_score = (
                self.det_obj_weight * num_norm[i]
                + self.det_center_weight * center_norm[i]
                + self.det_conf_weight * conf_norm[i]
            )
            det_scores.append(float(det_score))

        return det_scores


    # ===================== embeddings prepare =====================
    def _prepare_embeddings(self, embeddings: Optional[Any], n: int) -> Optional[List[List[float]]]:
        if embeddings is None:
            return None
        try:
            import numpy as _np  # type: ignore
            if isinstance(embeddings, _np.ndarray):
                if embeddings.ndim != 2 or embeddings.shape[0] != n:
                    return None
                return embeddings.astype("float32").tolist()
        except Exception:
            pass

        if not isinstance(embeddings, (list, tuple)):
            return None
        if len(embeddings) != n:
            return None

        out: List[List[float]] = []
        dim = None
        for row in embeddings:
            if not isinstance(row, (list, tuple)):
                return None
            if dim is None:
                dim = len(row)
                if dim == 0:
                    return None
            elif len(row) != dim:
                return None
            out.append([float(x) for x in row])
        return out

    # ===================== quality score =====================
    def _compute_basic_quality_scores(self, frame_paths: List[str]) -> List[float]:
        """
        简单质量：分辨率、亮度平衡、对比度、清晰度(Laplacian var)

        [MODIFIED]
        - 仍然返回原来的 q_scores（用于 final_score，不改变既有逻辑）
        - 额外缓存 raw metrics（用于 hard floors / report）
          * brightness/contrast 用固定 resize 后的灰度统计
          * lap_var_small 用固定 resize 后的 Laplacian var
        """
        n = len(frame_paths)
        if n == 0:
            self._last_quality_raw_metrics = []
            return []

        areas: List[float] = []
        brights: List[float] = []
        contrs: List[float] = []
        sharp_vars: List[float] = []

        read_fail = 0
        raw_list: List[Dict[str, float]] = []

        resize_hw = tuple(getattr(self, "hard_floor_resize_hw", (160, 160)))
        if not (isinstance(resize_hw, (tuple, list)) and len(resize_hw) == 2):
            resize_hw = (160, 160)
        rH, rW = int(resize_hw[0]), int(resize_hw[1])
        if rH <= 8 or rW <= 8:
            rH, rW = 160, 160

        for p in frame_paths:
            area = 0.0
            mean_full = 0.0
            std_full = 0.0
            sharp_var_full = 0.0

            mean_small = 0.0
            std_small = 0.0
            lap_var_small = 0.0

            read_ok = 0.0
            try:
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError("cv2.imread returned None")
                h, w = img.shape[:2]
                area = float(w * h)

                gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mean_full = float(gray_full.mean())
                std_full = float(gray_full.std())

                lap_full = cv2.Laplacian(gray_full, cv2.CV_64F)
                sharp_var_full = float(lap_full.var())

                # [ADDED] fixed-resize gray stats for hard floors (stable absolute thresholds)
                gray_small = cv2.resize(gray_full, (rW, rH), interpolation=cv2.INTER_AREA)
                mean_small = float(gray_small.mean())
                std_small = float(gray_small.std())
                lap_small = cv2.Laplacian(gray_small, cv2.CV_64F)
                lap_var_small = float(lap_small.var())

                read_ok = 1.0
            except Exception:
                read_fail += 1

            areas.append(area)
            brights.append(mean_full)
            contrs.append(std_full)
            sharp_vars.append(sharp_var_full)

            raw_list.append(
                {
                    "read_ok": float(read_ok),
                    "area": float(area),
                    "bright_mean_small": float(mean_small),
                    "contrast_std_small": float(std_small),
                    "lap_var_small": float(lap_var_small),
                }
            )

        if read_fail > 0:
            self._dbg(f"[quality] cv2.imread failed: {read_fail}/{n}")

        area_n = _min_max_norm(areas)
        contr_n = _min_max_norm(contrs)
        sharp_n = _min_max_norm(sharp_vars)

        bright_balanced: List[float] = []
        for b in brights:
            x = (b - 128.0) / 128.0
            val = max(0.0, 1.0 - abs(x))
            bright_balanced.append(val)
        bright_balanced_n = _min_max_norm(bright_balanced)

        q_scores: List[float] = []
        for i in range(n):
            q = (
                0.1 * area_n[i]
                + 0.3 * bright_balanced_n[i]
                + 0.3 * contr_n[i]
                + 0.3 * sharp_n[i]
            )
            q_scores.append(float(q))

        # [MODIFIED] cache for hard floors / report
        self._last_quality_raw_metrics = raw_list

        return q_scores

    # ===================== ✅ global motion blur score (prev,cur,next) =====================
    def _compute_global_motion_blur_quality_scores(
        self,
        frame_paths: List[str],
        resize_hw: Tuple[int, int] = (160, 160),
    ) -> Tuple[List[float], Dict[int, Dict[str, float]]]:
        """
        Global motion blur detector (for quality penalty):
        - Motion gate by prev_next = mean(|prev-next|) on small grayscale.
        - Sharpness by tile-aggregated Tenengrad: sharp_p25 + (NEW) sharp_p50 for robustness.
        - Optional temporal sharpness drop: cur / max(prev,next), but (NEW) down-weighted when edge_ratio is unstable.

        Returns:
        scores[i] in [0,1] (higher => more likely global motion blur)
        dbg[i] with prev_next / edge_ratio / sharp_p25 / sharp_p50 / drop / edge_ratio_drop, etc.
        """
        n = len(frame_paths)
        if n < 3:
            return [0.0] * n, {}

        # read grayscale small, normalize to 0..1
        ims: List[Optional[np.ndarray]] = []
        for p in frame_paths:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if g is None:
                ims.append(None)
                continue
            g = cv2.resize(g, resize_hw, interpolation=cv2.INTER_AREA)
            ims.append(g.astype(np.float32) / 255.0)

        scores = [0.0] * n
        dbg: Dict[int, Dict[str, float]] = {}

        eps = 1e-6
        gh = int(getattr(self, "global_motion_blur_tile_grid_h", getattr(self, "local_motion_blur_tile_grid_h", 4)))
        gw = int(getattr(self, "global_motion_blur_tile_grid_w", getattr(self, "local_motion_blur_tile_grid_w", 4)))
        inner_q = float(getattr(self, "global_motion_blur_tile_inner_q", getattr(self, "local_motion_blur_tile_inner_q", 80.0)))

        motion_th = float(self.global_motion_blur_prev_next_th)
        motion_strict = float(self.global_motion_blur_prev_next_strict)
        edge_min = float(self.global_motion_blur_edge_ratio_min)
        sharp_th = float(self.global_motion_blur_sharp_p25_th)
        drop_th = float(self.global_motion_blur_drop_th)

        w_sharp = float(self.global_motion_blur_w_sharp)
        w_drop = float(self.global_motion_blur_w_drop)
        w_sum = max(eps, (w_sharp + w_drop))

        # [NEW] controls for edge-stability gating (conservative defaults)
        # If current edge_ratio is much smaller than neighbors, it's often "content/texture change",
        # not necessarily blur -> suppress false positives.
        edge_drop_gate = float(getattr(self, "global_motion_blur_edge_ratio_drop_gate", 0.65))  # <0.65 => suppress
        edge_drop_hard = float(getattr(self, "global_motion_blur_edge_ratio_drop_hard", 0.45))  # <0.45 => strong suppress

        def _clip01(x: float) -> float:
            if not math.isfinite(x):
                return 0.0
            return float(max(0.0, min(1.0, x)))

        def _score_when_lower(val: float, th: float, margin: float = 0.35) -> float:
            if not math.isfinite(val) or not math.isfinite(th) or th <= 0:
                return 0.0
            hi = th * (1.0 + float(margin))
            if val >= hi:
                return 0.0
            if val <= th:
                return 1.0
            return _clip01((hi - val) / (hi - th + eps))

        def _tile_percentile(score_map: np.ndarray, mask: np.ndarray, tiles_q: float) -> float:
            """Each tile -> percentile(inner_q) on masked pixels; then percentile(tiles_q) over tiles."""
            H, W = score_map.shape[:2]
            vals: List[float] = []
            for yy in range(gh):
                y0 = int(round(yy * H / gh))
                y1 = int(round((yy + 1) * H / gh))
                for xx in range(gw):
                    x0 = int(round(xx * W / gw))
                    x1 = int(round((xx + 1) * W / gw))
                    tile = score_map[y0:y1, x0:x1]
                    m = mask[y0:y1, x0:x1]
                    if tile.size <= 0:
                        continue
                    tv = tile[m] if np.any(m) else tile.reshape(-1)
                    if tv.size <= 0:
                        continue
                    try:
                        vals.append(float(np.percentile(tv, inner_q)))
                    except Exception:
                        vals.append(float(np.mean(tv)))
            if not vals:
                return 0.0
            try:
                return float(np.percentile(np.array(vals, dtype=np.float32), tiles_q))
            except Exception:
                return float(np.mean(vals))

        def _tenengrad(img01: np.ndarray) -> np.ndarray:
            gx = cv2.Sobel(img01, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img01, cv2.CV_32F, 0, 1, ksize=3)
            return gx * gx + gy * gy

        def _edge_ratio_from_ten(ten: np.ndarray) -> Tuple[float, float, np.ndarray]:
            abs_min = float(getattr(self, "global_motion_blur_edge_abs_min", 0.002))
            mad_k = float(getattr(self, "global_motion_blur_edge_mad_k", 6.0))
            edge_ratio, edge_th, edge_mask = self._edge_ratio_mask_from_ten(
                ten, abs_min=abs_min, mad_k=mad_k
            )
            return edge_ratio, float(edge_th), edge_mask


        for i in range(1, n - 1):
            A = ims[i - 1]
            M = ims[i]
            B = ims[i + 1]
            if A is None or M is None or B is None:
                continue

            prev_next = float(np.mean(np.abs(A - B)))
            if prev_next < motion_th:
                dbg[i] = {"prev_next": prev_next, "score": 0.0}
                continue

            ten_m = _tenengrad(M)
            edge_ratio, edge_th_m, edge_mask_m = _edge_ratio_from_ten(ten_m)

            # avoid low-texture false positives unless motion is very strong
            if (edge_ratio < edge_min) and (prev_next < motion_strict):
                dbg[i] = {
                    "prev_next": prev_next,
                    "edge_ratio": edge_ratio,
                    "sharp_p25": 0.0,
                    "sharp_p50": 0.0,
                    "drop": 1.0,
                    "s_sharp": 0.0,
                    "s_drop": 0.0,
                    "score": 0.0,
                }
                continue

            # [MODIFIED] compute both p25 & p50 to reduce "only extreme low p25 triggers"
            sharp_p25 = _tile_percentile(ten_m, edge_mask_m, tiles_q=25.0)
            sharp_p50 = _tile_percentile(ten_m, edge_mask_m, tiles_q=50.0)

            # temporal drop (relative): compare to prev/next sharpness
            ten_a = _tenengrad(A)
            ten_b = _tenengrad(B)

            edge_ratio_prev, edge_th_a, edge_mask_a = _edge_ratio_from_ten(ten_a)
            edge_ratio_next, edge_th_b, edge_mask_b = _edge_ratio_from_ten(ten_b)

            sharp_p25_prev = _tile_percentile(ten_a, edge_mask_a, tiles_q=25.0)
            sharp_p25_next = _tile_percentile(ten_b, edge_mask_b, tiles_q=25.0)

            # [NEW] edge_ratio stability feature: current edge density vs neighbors
            ref_edge = max(edge_ratio_prev, edge_ratio_next, eps)
            edge_ratio_drop = float(edge_ratio / ref_edge)

            ref = max(sharp_p25_prev, sharp_p25_next, eps)
            drop = float(sharp_p25 / ref)

            # [MODIFIED] s_sharp uses BOTH p25 and p50:
            # - p25 catches "bottom tiles are weak"
            # - p50 catches "overall soft" (often your 'blurry but not tagged' cases)
            s_sharp25 = _score_when_lower(sharp_p25, sharp_th, margin=0.35)
            # NOTE: p50 is typically higher than p25, so compare to a slightly higher threshold.
            # This keeps it from being overly aggressive while still boosting recall.
            s_sharp50 = _score_when_lower(sharp_p50, sharp_th * 1.6, margin=0.35)
            s_sharp = float(max(s_sharp25, s_sharp50))

            # drop score (only meaningful when current/neighbor edge density is reasonably stable)
            s_drop_raw = _score_when_lower(drop, drop_th, margin=0.25)

            # [NEW] down-weight drop when edge_ratio is unstable (likely content/texture change)
            drop_weight = 1.0
            if edge_ratio_drop < edge_drop_gate:
                # linear decay to 0 at hard threshold
                if edge_ratio_drop <= edge_drop_hard:
                    drop_weight = 0.0
                else:
                    drop_weight = float((edge_ratio_drop - edge_drop_hard) / (edge_drop_gate - edge_drop_hard + eps))
                drop_weight = _clip01(drop_weight)

            s_drop = float(s_drop_raw * drop_weight)

            # base score
            score = (w_sharp * s_sharp + w_drop * s_drop) / w_sum
            score = _clip01(score)

            # [NEW] final suppression for strong edge-ratio drop (extra anti-false-positive)
            # If current edge density is much smaller than neighbors, it's often "low texture view",
            # not blur, so suppress the final score.
            if edge_ratio_drop < edge_drop_gate:
                # keep some score (so true blur can still pass if s_sharp is very high),
                # but reduce moderately.
                score = float(score * (0.4 + 0.6 * drop_weight))  # drop_weight already in [0,1]
                score = _clip01(score)

            scores[i] = float(score)
            dbg[i] = {
                "prev_next": float(prev_next),
                "edge_ratio": float(edge_ratio),
                "edge_ratio_prev": float(edge_ratio_prev),
                "edge_ratio_next": float(edge_ratio_next),
                "edge_ratio_drop": float(edge_ratio_drop),
                "edge_th": float(edge_th_m),
                "sharp_p25": float(sharp_p25),
                "sharp_p50": float(sharp_p50),
                "sharp_p25_prev": float(sharp_p25_prev),
                "sharp_p25_next": float(sharp_p25_next),
                "drop": float(drop),
                "s_sharp25": float(s_sharp25),
                "s_sharp50": float(s_sharp50),
                "s_sharp": float(s_sharp),
                "s_drop_raw": float(s_drop_raw),
                "drop_weight": float(drop_weight),
                "s_drop": float(s_drop),
                "score": float(score),
            }

        return scores, dbg
    # ===========================================================================

    # ===================== ✅ defocus / global soft blur score (single frame) =====================
    def _compute_defocus_blur_quality_scores(
        self,
        frame_paths: List[str],
        resize_hw: Tuple[int, int] = (160, 160),
    ) -> Tuple[List[float], Dict[int, Dict[str, float]]]:
        """
        Defocus / global soft blur detector (for tagging / hard floor):
        - No motion gate (works for long defocus segments).
        - Sharpness by tile-aggregated Tenengrad: sharp_p25 + sharp_p50 (robust).
        - Edge gate: if edge_ratio too low, suppress to avoid low-texture false positives.

        Returns:
        scores[i] in [0,1] (higher => more likely defocus / global soft blur)
        dbg[i] with edge_ratio / sharp_p25 / sharp_p50 / s_sharp25/50 / score, etc.
        """
        n = len(frame_paths)
        if n <= 0:
            return [], {}

        # read grayscale small, normalize to 0..1
        ims: List[Optional[np.ndarray]] = []
        for p in frame_paths:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if g is None:
                ims.append(None)
                continue
            g = cv2.resize(g, resize_hw, interpolation=cv2.INTER_AREA)
            ims.append(g.astype(np.float32) / 255.0)

        scores = [0.0] * n
        dbg: Dict[int, Dict[str, float]] = {}

        eps = 1e-6

        gh = int(getattr(self, "defocus_blur_tile_grid_h", getattr(self, "local_motion_blur_tile_grid_h", 4)))
        gw = int(getattr(self, "defocus_blur_tile_grid_w", getattr(self, "local_motion_blur_tile_grid_w", 4)))
        inner_q = float(getattr(self, "defocus_blur_tile_inner_q", getattr(self, "local_motion_blur_tile_inner_q", 80.0)))


        edge_min = float(getattr(self, "defocus_blur_edge_ratio_min", 0.01))
        sharp_th = float(getattr(self, "defocus_blur_sharp_p25_th", 0.20))
        p50_scale = float(getattr(self, "defocus_blur_p50_scale", 1.6))

        def _clip01(x: float) -> float:
            if not math.isfinite(x):
                return 0.0
            return float(max(0.0, min(1.0, x)))

        def _score_when_lower(val: float, th: float, margin: float = 0.35) -> float:
            # val 越低越糊： val<=th -> 1, val>=th*(1+margin) -> 0
            if not math.isfinite(val) or not math.isfinite(th) or th <= 0:
                return 0.0
            hi = th * (1.0 + float(margin))
            if val >= hi:
                return 0.0
            if val <= th:
                return 1.0
            return _clip01((hi - val) / (hi - th + eps))

        def _tenengrad(img01: np.ndarray) -> np.ndarray:
            gx = cv2.Sobel(img01, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img01, cv2.CV_32F, 0, 1, ksize=3)
            return gx * gx + gy * gy

        def _tile_percentile(score_map: np.ndarray, mask: np.ndarray, tiles_q: float) -> float:
            H, W = score_map.shape[:2]
            vals: List[float] = []
            for yy in range(gh):
                y0 = int(round(yy * H / gh))
                y1 = int(round((yy + 1) * H / gh))
                for xx in range(gw):
                    x0 = int(round(xx * W / gw))
                    x1 = int(round((xx + 1) * W / gw))
                    tile = score_map[y0:y1, x0:x1]
                    m = mask[y0:y1, x0:x1]
                    if tile.size <= 0:
                        continue
                    tv = tile[m] if np.any(m) else tile.reshape(-1)
                    if tv.size <= 0:
                        continue
                    try:
                        vals.append(float(np.percentile(tv, inner_q)))
                    except Exception:
                        vals.append(float(np.mean(tv)))
            if not vals:
                return 0.0
            try:
                return float(np.percentile(np.array(vals, dtype=np.float32), tiles_q))
            except Exception:
                return float(np.mean(vals))

        for i in range(n):
            M = ims[i]
            if M is None:
                continue

            ten_m = _tenengrad(M)

            # edge mask & ratio
            abs_min = float(getattr(self, "defocus_blur_edge_abs_min", 0.002))
            mad_k = float(getattr(self, "defocus_blur_edge_mad_k", 6.0))
            edge_ratio, edge_th, edge_mask = self._edge_ratio_mask_from_ten(
                ten_m, abs_min=abs_min, mad_k=mad_k
            )


            # low-texture gate: avoid false positives
            if edge_ratio < edge_min:
                dbg[i] = {
                    "edge_ratio": float(edge_ratio),
                    "edge_th": float(edge_th),
                    "sharp_p25": 0.0,
                    "sharp_p50": 0.0,
                    "s_sharp25": 0.0,
                    "s_sharp50": 0.0,
                    "score": 0.0,
                }
                continue

            sharp_p25 = _tile_percentile(ten_m, edge_mask, tiles_q=25.0)
            sharp_p50 = _tile_percentile(ten_m, edge_mask, tiles_q=50.0)

            s_sharp25 = _score_when_lower(sharp_p25, sharp_th, margin=0.35)
            s_sharp50 = _score_when_lower(sharp_p50, sharp_th * float(p50_scale), margin=0.35)

            score = float(max(s_sharp25, s_sharp50))
            score = _clip01(score)

            scores[i] = float(score)
            dbg[i] = {
                "edge_ratio": float(edge_ratio),
                "edge_th": float(edge_th),
                "sharp_p25": float(sharp_p25),
                "sharp_p50": float(sharp_p50),
                "s_sharp25": float(s_sharp25),
                "s_sharp50": float(s_sharp50),
                "score": float(score),
            }

        return scores, dbg
    # ================================================================================

    # ===================== ✅ local motion blur (prev,cur,next) =====================
    def _compute_local_motion_blur_flags(
        self,
        frame_paths: List[str],
        resize_hw: Tuple[int, int] = (160, 160),
    ) -> Tuple[List[bool], Dict[int, Dict[str, float]]]:
        """
        用 (prev,cur,next) 的帧差构造 motion mask，并比较 motion 区域与 static 区域的清晰度差异。
        返回：
          flags[i]：是否判定 frame i 存在“局部运动模糊”
          dbg[i]：motion_area_ratio / sharp_ratio / lap_ratio / anis_gain / blur_score 等
        """
        n = len(frame_paths)
        if n < 3:
            return [False] * n, {}

        # 读灰度并缩放到统一尺度，归一化到 0..1
        ims: List[Optional[np.ndarray]] = []
        for p in frame_paths:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if g is None:
                ims.append(None)
                continue
            g = cv2.resize(g, resize_hw, interpolation=cv2.INTER_AREA)
            ims.append(g.astype(np.float32) / 255.0)

        flags = [False] * n
        dbg: Dict[int, Dict[str, float]] = {}

        eps = 1e-6
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # --- helpers ---
        def _clip01(x: float) -> float:
            if not math.isfinite(x):
                return 0.0
            return float(max(0.0, min(1.0, x)))

        def _score_when_lower(val: float, th: float, margin: float = 0.25) -> float:
            """
            val 越低越像 blur：
              val <= th            -> score 接近 1
              val >= th*(1+margin) -> score 接近 0
            """
            if not math.isfinite(val) or not math.isfinite(th) or th <= 0:
                return 0.0
            hi = th * (1.0 + float(margin))
            if val >= hi:
                return 0.0
            if val <= th:
                return 1.0
            return _clip01((hi - val) / (hi - th + eps))

        def _score_when_higher(val: float, th: float, margin: float = 0.25) -> float:
            """
            val 越高越像 blur：
              val >= th            -> score 接近 1
              val <= th*(1-margin) -> score 接近 0
            """
            if not math.isfinite(val) or not math.isfinite(th) or th <= 0:
                return 0.0
            lo = th * (1.0 - float(margin))
            if val <= lo:
                return 0.0
            if val >= th:
                return 1.0
            return _clip01((val - lo) / (th - lo + eps))

        def _tile_percentile(
            score_map: np.ndarray,
            mask: np.ndarray,
            grid_hw: Tuple[int, int],
            inner_q: float,
            tiles_q: float,
        ) -> float:
            # Tile-wise robust aggregation: per-tile percentile(inner_q) -> percentile(tiles_q) over tiles.
            gh, gw = int(grid_hw[0]), int(grid_hw[1])
            H, W = score_map.shape[:2]
            if H <= 0 or W <= 0 or gh <= 0 or gw <= 0:
                return 0.0
            vals: List[float] = []
            for yy in range(gh):
                y0 = int(round(yy * H / gh))
                y1 = int(round((yy + 1) * H / gh))
                for xx in range(gw):
                    x0 = int(round(xx * W / gw))
                    x1 = int(round((xx + 1) * W / gw))
                    tile = score_map[y0:y1, x0:x1]
                    msk = mask[y0:y1, x0:x1]
                    if tile.size <= 0:
                        continue
                    tv = tile[msk] if np.any(msk) else tile.reshape(-1)
                    if tv.size <= 0:
                        continue
                    try:
                        vals.append(float(np.percentile(tv, inner_q)))
                    except Exception:
                        vals.append(float(np.mean(tv)))
            if not vals:
                return 0.0
            try:
                return float(np.percentile(np.array(vals, dtype=np.float32), tiles_q))
            except Exception:
                return float(np.mean(vals))

        for i in range(1, n - 1):
            A = ims[i - 1]
            M = ims[i]
            B = ims[i + 1]
            if A is None or M is None or B is None:
                continue

            # 如果 prev-next 本身几乎没变化，没必要做 motion blur 判定（会不稳定）
            prev_next = float(np.mean(np.abs(A - B)))
            pn_th = float(getattr(self, "local_motion_blur_prev_next_th", 0.02))
            if prev_next < pn_th:
                dbg[i] = {
                    "motion_area_ratio": 0.0,
                    "sharp_motion": 0.0,
                    "sharp_static": 0.0,
                    "sharp_ratio": 1.0,
                    "lap_motion": 0.0,
                    "lap_static": 0.0,
                    "lap_ratio": 1.0,
                    "anis_motion": 1.0,
                    "anis_static": 1.0,
                    "anis_gain": 1.0,
                    "sharp_drop": 1.0,
                    "drop_score": 0.0,
                    "blur_score": 0.0,
                    "prev_next": prev_next,
                }
                continue

            # motion map: (|M-A| + |B-M|)/2
            diff1 = np.abs(M - A)
            diff2 = np.abs(B - M)
            motion = 0.5 * (diff1 + diff2)

            # ✅ 阈值：2.5*MAD（放松），并引入分位数作为下限（更随视频变化）
            vals = motion.reshape(-1)
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med))) + 1e-6

            base_th = med + 2.5 * mad  # 3.0 -> 2.5
            q_th = float(np.percentile(vals, 75))  # 分位数下限（可按需改 70/80）
            th = max(0.02, q_th, base_th)

            mask = (motion > th).astype(np.uint8)  # 0/1

            # 简单形态学去噪：open + close
            try:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_open, iterations=1)
                # ✅ 轻微膨胀：把拖影/边缘附近吃进 motion 区域（减少“只抓到锐边缘”的漏检）
                mask = cv2.dilate(mask, k_open, iterations=1)
            except Exception:
                pass

            motion_area_ratio = float(mask.mean())
            # 运动区域太小：不稳定，跳过
            if motion_area_ratio < float(self.local_motion_blur_area_min):
                dbg[i] = {
                    "motion_area_ratio": motion_area_ratio,
                    "sharp_motion": 0.0,
                    "sharp_static": 0.0,
                    "sharp_ratio": 1.0,
                    "lap_motion": 0.0,
                    "lap_static": 0.0,
                    "lap_ratio": 1.0,
                    "anis_motion": 1.0,
                    "anis_static": 1.0,
                    "anis_gain": 1.0,
                    "sharp_drop": 1.0,
                    "drop_score": 0.0,
                    "blur_score": 0.0,
                    "prev_next": prev_next,
                    "th": float(th),
                    "q_th": float(q_th),
                    "base_th": float(base_th),
                }
                continue

            # 运动区域太大：更像全局运动/相机运动 -> local blur 直接略过
            if motion_area_ratio > float(self.local_motion_blur_area_max):
                dbg[i] = {
                    "mode": "skip_large_motion",
                    "motion_area_ratio": float(motion_area_ratio),
                    "prev_next": float(prev_next),
                    "th": float(th),
                    "q_th": float(q_th),
                    "base_th": float(base_th),
                    "blur_score": 0.0,
                }
                continue


            mask_bool = mask.astype(bool)
            inv_mask = ~mask_bool

            # ===================== Tenengrad =====================
            gx = cv2.Sobel(M, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(M, cv2.CV_32F, 0, 1, ksize=3)
            ten = gx * gx + gy * gy

            try:
                edge_th = float(np.percentile(ten.reshape(-1), 70))
            except Exception:
                edge_th = 0.0
            edge_mask = ten > edge_th

            gh = int(getattr(self, "local_motion_blur_tile_grid_h", 4))
            gw = int(getattr(self, "local_motion_blur_tile_grid_w", 4))
            inner_q = float(getattr(self, "local_motion_blur_tile_inner_q", 80.0))


            motion_edge = mask_bool & edge_mask
            static_edge = inv_mask & edge_mask

            motion_use = motion_edge if np.any(motion_edge) else mask_bool
            static_use = static_edge if np.any(static_edge) else inv_mask

            sharp_motion = _tile_percentile(ten, motion_use, (gh, gw), inner_q=inner_q, tiles_q=50.0)
            sharp_static = _tile_percentile(ten, static_use, (gh, gw), inner_q=inner_q, tiles_q=50.0)
            sharp_motion_p25 = _tile_percentile(ten, motion_use, (gh, gw), inner_q=inner_q, tiles_q=25.0)
            sharp_static_p25 = _tile_percentile(ten, static_use, (gh, gw), inner_q=inner_q, tiles_q=25.0)

            sharp_ratio = float(sharp_motion / (sharp_static + eps))

            # ===================== Laplacian ratio =====================
            try:
                lap = cv2.Laplacian(M, cv2.CV_32F, ksize=3)
                lap_abs = np.abs(lap)
            except Exception:
                lap_abs = None

            if lap_abs is not None:
                lap_motion = _tile_percentile(lap_abs, motion_use, (gh, gw), inner_q=inner_q, tiles_q=50.0)
                lap_static = _tile_percentile(lap_abs, static_use, (gh, gw), inner_q=inner_q, tiles_q=50.0)
                lap_motion_p25 = _tile_percentile(lap_abs, motion_use, (gh, gw), inner_q=inner_q, tiles_q=25.0)
                lap_static_p25 = _tile_percentile(lap_abs, static_use, (gh, gw), inner_q=inner_q, tiles_q=25.0)
                lap_ratio = float(lap_motion / (lap_static + eps))
            else:
                lap_motion, lap_static, lap_ratio = 0.0, 0.0, 1.0
                lap_motion_p25, lap_static_p25 = 0.0, 0.0

            # ===================== Sobel anisotropy =====================
            abs_gx = np.abs(gx)
            abs_gy = np.abs(gy)

            if np.any(mask_bool):
                ex_m = float(abs_gx[mask_bool].mean())
                ey_m = float(abs_gy[mask_bool].mean())
                anis_motion = float(max(ex_m, ey_m) / (min(ex_m, ey_m) + eps))
            else:
                anis_motion = 1.0

            if np.any(static_edge):
                ex_s = float(abs_gx[static_edge].mean())
                ey_s = float(abs_gy[static_edge].mean())
                anis_static = float(max(ex_s, ey_s) / (min(ex_s, ey_s) + eps))
            elif np.any(inv_mask):
                ex_s = float(abs_gx[inv_mask].mean())
                ey_s = float(abs_gy[inv_mask].mean())
                anis_static = float(max(ex_s, ey_s) / (min(ex_s, ey_s) + eps))
            else:
                anis_static = 1.0

            anis_gain = float(anis_motion / (anis_static + eps))

            # ===================== ✅ temporal sharp drop（最小侵入） =====================
            sharp_drop = 1.0
            drop_score = 0.0
            if np.any(mask_bool):
                try:
                    gx_a = cv2.Sobel(A, cv2.CV_32F, 1, 0, ksize=3)
                    gy_a = cv2.Sobel(A, cv2.CV_32F, 0, 1, ksize=3)
                    ten_a = gx_a * gx_a + gy_a * gy_a

                    gx_b = cv2.Sobel(B, cv2.CV_32F, 1, 0, ksize=3)
                    gy_b = cv2.Sobel(B, cv2.CV_32F, 0, 1, ksize=3)
                    ten_b = gx_b * gx_b + gy_b * gy_b
                    sharp_prev = _tile_percentile(ten_a, motion_use, (gh, gw), inner_q=inner_q, tiles_q=50.0)
                    sharp_next = _tile_percentile(ten_b, motion_use, (gh, gw), inner_q=inner_q, tiles_q=50.0)
                    ref = max(sharp_prev, sharp_next)
                    sharp_drop = float(sharp_motion / (ref + eps)) if ref > 0 else 1.0
                    drop_score = _score_when_lower(sharp_drop, float(self.local_motion_blur_drop_th), margin=0.25)
                except Exception:
                    sharp_drop = 1.0
                    drop_score = 0.0

            # ===================== score-based 判定 =====================
            s_sharp = _score_when_lower(sharp_ratio, float(self.local_motion_blur_ratio_th), margin=0.30)
            s_lap = _score_when_lower(lap_ratio, float(self.local_motion_blur_lap_ratio_th), margin=0.30)
            s_anis = _score_when_higher(anis_gain, float(self.local_motion_blur_anis_gain_th), margin=0.25)
            s_static = _score_when_higher(sharp_static, float(self.local_motion_blur_static_th), margin=0.50)

            w_sum = (
                float(self.local_motion_blur_w_sharp)
                + float(self.local_motion_blur_w_lap)
                + float(self.local_motion_blur_w_anis)
                + float(self.local_motion_blur_w_static)
                + float(self.local_motion_blur_w_drop)
            )
            w_sum = max(eps, w_sum)

            blur_score = (
                float(self.local_motion_blur_w_sharp) * s_sharp
                + float(self.local_motion_blur_w_lap) * s_lap
                + float(self.local_motion_blur_w_anis) * s_anis
                + float(self.local_motion_blur_w_static) * s_static
                + float(self.local_motion_blur_w_drop) * float(drop_score)
            ) / w_sum

            blur_score = _clip01(float(blur_score))
            is_blur = bool(blur_score >= float(self.local_motion_blur_score_th))


            flags[i] = bool(is_blur)
            dbg[i] = {
                "motion_area_ratio": motion_area_ratio,
                "sharp_motion": sharp_motion,
                "sharp_static": sharp_static,
                "sharp_ratio": sharp_ratio,
                "sharp_motion_p25": float(sharp_motion_p25),
                "sharp_static_p25": float(sharp_static_p25),
                "lap_motion": lap_motion,
                "lap_static": lap_static,
                "lap_ratio": lap_ratio,
                "lap_motion_p25": float(lap_motion_p25),
                "lap_static_p25": float(lap_static_p25),
                "anis_motion": anis_motion,
                "anis_static": anis_static,
                "anis_gain": anis_gain,
                "sharp_drop": float(sharp_drop),
                "drop_score": float(drop_score),
                "blur_score": float(blur_score),
                "prev_next": prev_next,
                "th": float(th),
                "q_th": float(q_th),
                "base_th": float(base_th),
                "edge_th": float(edge_th),
            }

        return flags, dbg
    # ===========================================================================

    # ===================== adjust k =====================
    def _adjust_k_by_score_clusters(self, final_scores: List[float], desired_k: int) -> int:
        n = len(final_scores)
        if n <= 1 or desired_k <= 1:
            k0 = max(1, min(desired_k, n, self.max_frames))
            self._last_k_lower = k0
            self._last_k_upper = k0
            return k0

        scores_sorted = sorted(final_scores, reverse=True)
        max_s = scores_sorted[0]
        min_s = scores_sorted[-1]
        score_range = max_s - min_s

        if score_range < 1e-3:
            k0 = max(1, min(desired_k, n, self.max_frames))
            self._last_k_lower = k0
            self._last_k_upper = k0
            return k0

        diffs = [scores_sorted[i] - scores_sorted[i + 1] for i in range(n - 1)]
        gap_th = 0.15 * score_range

        cluster_sizes: List[int] = []
        cur_size = 1
        for d in diffs:
            if d >= gap_th:
                cluster_sizes.append(cur_size)
                cur_size = 1
            else:
                cur_size += 1
        cluster_sizes.append(cur_size)

        k_center = max(1, min(desired_k, n, self.max_frames))
        margin = max(1, int(round(0.4 * k_center)))

        k_lower = max(1, k_center - margin)
        k_upper = min(n, self.max_frames, k_center + margin)

        self._last_k_lower = int(k_lower)
        self._last_k_upper = int(k_upper)

        total = 0
        for idx, size in enumerate(cluster_sizes):
            if idx == 0:
                if size >= k_upper:
                    total = k_upper
                    break
                total = size
                continue

            if total >= k_upper:
                break

            remain = k_upper - total
            if size <= remain:
                total += size
            else:
                candidate_total = min(k_upper, total + size)
                if abs(k_center - candidate_total) < abs(k_center - total):
                    total = candidate_total
                break

        total = max(1, min(total, k_upper))
        if total < k_lower:
            total = k_lower

        if total != k_center:
            self._dbg(f"adjust num_frames from {k_center} to {total} (window [{k_lower}, {k_upper}])")
        return total

    # ===================== main select =====================
    def select(
        self,
        frame_paths: Optional[List[str]] = None,
        det_results: Optional[Dict[str, Any]] = None,
        flow_scores: Optional[List[float]] = None,
        num_frames: int = 8,
        video_type: str = "unknown",
        motion_hint: Optional[str] = None,
        asr_text: str = "",
        ocr_text: str = "",
        enabled_tools: Optional[List[str]] = None,
        semantic_scores: Optional[List[float]] = None,
        emb_change_scores: Optional[List[float]] = None,
        embeddings: Optional[Any] = None,
        # 兼容旧接口
        frames: Optional[List[str]] = None,
        detections: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """
        返回关键帧的下标列表（相对于 frame_paths 的索引）。
        """

        if frame_paths is None and frames is not None:
            frame_paths = frames
        if det_results is None and detections is not None:
            det_results = detections

        frame_paths = frame_paths or []
        n = len(frame_paths)

        self._last_frame_paths_for_cut = frame_paths
        self.last_transition_indices = []
        self.last_blowout_indices = []


        if n == 0:
            self.last_detail_scores = {}
            self.last_weights = {}
            self.last_scores = {}
            self.last_shot_boundaries = []
            return []

        enabled_tools = enabled_tools or []
        use_blurmap = bool(getattr(self, "blurmap_enable", False)) or ("blurmap" in enabled_tools)


        self._dbg(f"select() n={n}, num_frames={num_frames}, transition_filter={self.transition_filter}, margin={self.transition_margin}")
        self._dbg(f"video_type={video_type}, motion_hint={motion_hint}, tools={enabled_tools}")

        candidates = list(range(n))


        def _has_variation(xs: List[float], eps: float = 1e-6) -> bool:
            if not xs:
                return False
            try:
                mn = min(xs)
                mx = max(xs)
                if not math.isfinite(mn) or not math.isfinite(mx):
                    return False
                return (mx - mn) > eps
            except Exception:
                return False

        # 1) quality raw
        try:
            q_raw = self._compute_basic_quality_scores(frame_paths)
            if len(q_raw) != n:
                q_raw = [1.0] * n
        except Exception as e:
            self._dbg(f"_compute_basic_quality_scores failed: {e}")
            q_raw = [1.0] * n


        # 1.5) global motion blur (compute if needed for hard floor / report)
        global_blur_scores: List[float] = []
        global_blur_dbg: Dict[int, Dict[str, float]] = {}

        need_global_blur = bool(self.global_motion_blur_filter) and (n >= 3)


        if need_global_blur:
            try:
                global_blur_scores, global_blur_dbg = self._compute_global_motion_blur_quality_scores(frame_paths)
                if len(global_blur_scores) != n:
                    global_blur_scores = [] 
                    global_blur_dbg = {}
            except Exception as e:
                self._dbg(f"_compute_global_motion_blur_quality_scores failed: {e}")
                global_blur_scores = []
                global_blur_dbg = {}

        self._dbg(f"[global_blur] score stats: {self._stats(global_blur_scores)}")


        # robust norm -> q_global_norm
        def _robust_norm(xs: List[float], p_low: float = 10.0, p_high: float = 90.0) -> List[float]:
            if not xs:
                return []
            arr = np.array([float(x) for x in xs], dtype=np.float32)
            lo = float(np.percentile(arr, p_low))
            hi = float(np.percentile(arr, p_high))
            if (not math.isfinite(lo)) or (not math.isfinite(hi)) or hi <= lo + 1e-6:
                return _min_max_norm([float(x) for x in xs])
            out = []
            for x in xs:
                v = (float(x) - lo) / (hi - lo)
                v = max(0.0, min(1.0, v))
                out.append(float(v))
            return out

        q_global_norm = _robust_norm([float(x) for x in q_raw]) if q_raw else [0.0] * n
        self._dbg(f"[quality] q_raw stats: {self._stats(q_raw)}")
        self._dbg(f"[quality] q_global_norm stats: {self._stats(q_global_norm)}")

        # 2) det/flow/sem/emb_change raw
        det_raw = self._compute_yolo_scores(frame_paths, det_results or {})
        flow_raw = [0.0] * n
        if flow_scores is not None and len(flow_scores) == n:
            for i, x in enumerate(flow_scores):
                try:
                    v = float(x)
                    if not math.isfinite(v):
                        v = 0.0
                except Exception:
                    v = 0.0
                flow_raw[i] = float(v)

        sem_raw = [0.0] * n
        if semantic_scores is not None and len(semantic_scores) == n:
            for i, s in enumerate(semantic_scores):
                try:
                    v = float(s)
                    if not math.isfinite(v):
                        v = 0.0
                    v = max(0.0, v)
                except Exception:
                    v = 0.0
                sem_raw[i] = float(v)

        emb_change_raw = [0.0] * n
        if emb_change_scores is not None and len(emb_change_scores) == n:
            for i, s in enumerate(emb_change_scores):
                try:
                    v = float(s)
                    if not math.isfinite(v):
                        v = 0.0
                    v = max(0.0, v)
                except Exception:
                    v = 0.0
                emb_change_raw[i] = float(v)

        # 3) shot boundaries + shotwise norm
        shot_bds = self._auto_shot_boundaries(emb_change_raw=emb_change_raw, flow_raw=flow_raw)
        self.last_shot_boundaries = shot_bds

        q_scores = self._shotwise_minmax_norm([float(x) for x in q_raw], shot_bds)
        det_scores = self._shotwise_minmax_norm([float(x) for x in det_raw], shot_bds) if det_raw else [0.0] * n
        flow_norm = self._shotwise_minmax_norm([float(x) for x in flow_raw], shot_bds)
        sem_scores = self._shotwise_minmax_norm([float(x) for x in sem_raw], shot_bds)
        emb_change_norm = self._shotwise_minmax_norm([float(x) for x in emb_change_raw], shot_bds)


        has_det = _has_variation(det_scores)
        has_flow = _has_variation(flow_norm)
        has_sem = _has_variation(sem_scores)
        has_emb_change = _has_variation(emb_change_norm)

        # ===================== ✅ IQA (MUSIQ) =====================
        musiq_raw = [0.0] * n
        musiq_norm = [0.0] * n

        use_iqa = bool(getattr(self, "iqa_musiq_enable", False)) or ("iqa" in enabled_tools) or ("musiq" in enabled_tools)
        self._dbg(f"[iqa] use_iqa={use_iqa}, iqa_musiq_enable={getattr(self,'iqa_musiq_enable',None)}, tools={enabled_tools}")
        self._dbg(f"[iqa] env={getattr(self,'iqa_musiq_conda_env','')}, device={getattr(self,'iqa_musiq_device','')}, timeout={getattr(self,'iqa_musiq_timeout_s','')}, pre_topk={getattr(self,'iqa_musiq_pre_topk','')}")

        if use_iqa:
            try:
                from executor.models.iqa_musiq import compute_musiq_scores

                pre_topk = int(getattr(self, "iqa_musiq_pre_topk", 0) or 0)
                if pre_topk > 0:
                    # 用 q_global_norm 做预筛（快、你已经算了）
                    order = sorted(range(n), key=lambda i: float(q_global_norm[i]) if i < len(q_global_norm) else 0.0, reverse=True)
                    pick = set(order[: min(pre_topk, n)])
                    to_calc = [frame_paths[i] for i in range(n) if i in pick]
                    calc_scores = compute_musiq_scores(
                        to_calc,
                        conda_env=str(getattr(self, "iqa_musiq_conda_env", "video_iqa")),
                        device=str(getattr(self, "iqa_musiq_device", "auto")),
                        timeout_s=int(getattr(self, "iqa_musiq_timeout_s", 600)),
                    )
                    self._dbg(f"[iqa] musiq_raw len={len(musiq_raw)} expected={n} head={musiq_raw[:5] if isinstance(musiq_raw,list) else musiq_raw}")
                    # 回填到 musiq_raw（未算的保持 0）
                    j = 0
                    for i in range(n):
                        if i in pick:
                            musiq_raw[i] = float(calc_scores[j]) if j < len(calc_scores) else 0.0
                            j += 1
                else:
                    musiq_raw = compute_musiq_scores(
                        frame_paths,
                        conda_env=str(getattr(self, "iqa_musiq_conda_env", "video_iqa")),
                        device=str(getattr(self, "iqa_musiq_device", "auto")),
                        timeout_s=int(getattr(self, "iqa_musiq_timeout_s", 600)),
                    )
                    if len(musiq_raw) != n:
                        musiq_raw = [0.0] * n

                # MUSIQ higher is better -> 直接做 shotwise minmax
                musiq_norm = self._shotwise_minmax_norm([float(x) for x in musiq_raw], shot_bds)
            except Exception as e:
                self._dbg(f"[iqa] musiq compute failed: {e}")
                musiq_raw = [0.0] * n
                musiq_norm = [0.0] * n

        self._dbg(f"[iqa] musiq_raw stats: {self._stats(musiq_raw)}")
        self._dbg(f"[iqa] musiq_norm stats: {self._stats(musiq_norm)}")
        # ==========================================================


        # 4) weights
        policy_weights = resolve_iteration_score_weights(self.iteration_score_weights)
        w_quality = 0.1 * policy_weights["image_quality"]
        w_det = 0.2 if has_det else 0.0
        w_flow = 0.2 if has_flow else 0.0
        w_flow *= policy_weights["temporal_coverage"]
        w_sem = (0.15 if has_sem else 0.0) * policy_weights["semantic_change"]
        w_emb_change = (0.35 if has_emb_change else 0.0) * policy_weights["semantic_change"]
        w_diversity = 0.1 if embeddings is not None else 0.0

        # ✅ IQA (MUSIQ) weight (only if musiq_norm is valid & has variation)
        musiq_norm_safe: List[float] = []
        try:
            if isinstance(musiq_norm, list) and len(musiq_norm) == n:
                musiq_norm_safe = [float(x) for x in musiq_norm]
        except Exception:
            musiq_norm_safe = []

        has_iqa = _has_variation(musiq_norm_safe) if musiq_norm_safe else False
        w_iqa = float(getattr(self, "iqa_musiq_weight", 0.15)) if has_iqa else 0.0

        motion_hint_l = (motion_hint or "").lower()
        vtype = (video_type or "").lower()

        if any(k in vtype for k in ("tutorial", "how-to", "operation", "demo")):
            w_det *= 1.2
            w_flow *= 1.2
        if any(k in vtype for k in ("lecture", "slide", "presentation")):
            w_flow *= 0.5
            w_sem *= 1.2
        if "fast" in motion_hint_l or "high" in motion_hint_l:
            w_flow *= 1.5

        if not has_det:
            w_quality += w_det
            w_det = 0.0

        # ✅ include w_iqa in normalization
        sum_w = w_quality + w_det + w_flow + w_sem + w_emb_change + w_iqa
        if sum_w <= 0:
            w_quality, w_det, w_flow, w_sem, w_emb_change, w_iqa = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
            sum_w = 1.0

        w_quality /= sum_w
        w_det /= sum_w
        w_flow /= sum_w
        w_sem /= sum_w
        w_emb_change /= sum_w
        w_iqa /= sum_w

        self.last_weights = {
            "w_quality": float(w_quality),
            "w_det": float(w_det),
            "w_flow": float(w_flow),
            "w_sem": float(w_sem),
            "w_emb_change": float(w_emb_change),
            "w_iqa_musiq": float(w_iqa),
            "w_diversity": float(w_diversity),
            "policy_image_quality": policy_weights["image_quality"],
            "policy_semantic_change": policy_weights["semantic_change"],
            "policy_temporal_coverage": policy_weights["temporal_coverage"],
        }
        self._dbg("weights:", self.last_weights)


        # 5) cut_scores（保证 report 一定拿到）
        try:
            cut_scores = self._compute_cut_scores(frame_paths)
            if len(cut_scores) != n:
                cut_scores = [0.0] * n
        except Exception:
            cut_scores = [0.0] * n
        self._dbg(f"[cut] cut_scores stats: {self._stats(cut_scores)}")

        # 6) final score + init detail
        final_scores: List[float] = [0.0] * n
        detail_scores: Dict[int, Dict[str, Any]] = {}
        score_map: Dict[int, float] = {}

        for i in range(n):
            q = float(q_scores[i])
            det = float(det_scores[i])
            flow = float(flow_norm[i])
            sem = float(sem_scores[i])
            emb_ch = float(emb_change_norm[i])

            iqa = float(musiq_norm[i]) if i < len(musiq_norm) else 0.0

            final_score = (
                w_quality * q
                + w_det * det
                + w_flow * flow
                + w_sem * sem
                + w_emb_change * emb_ch
                + w_iqa * iqa
            )

            final_scores[i] = float(final_score)
            score_map[i] = float(final_score)

            qg = float(q_global_norm[i]) if i < len(q_global_norm) else 0.0

            detail_scores[i] = {
                "final": float(final_score),
                "q": q,
                "det": det,
                "flow": flow,
                "sem": sem,
                "emb_change": emb_ch,
                "q_global": qg,

                "cut_score": float(cut_scores[i]) if i < len(cut_scores) else 0.0,
                "is_transition": 0.0,
                "is_filtered": 0.0,
                "filter_tag": "",

                # ✅ local motion blur 输出字段
                "is_local_motion_blur": 0.0,
                "motion_area_ratio": 0.0,
                "sharp_motion": 0.0,
                "sharp_static": 0.0,
                "sharp_ratio": 1.0,

                "lap_motion": 0.0,
                "lap_static": 0.0,
                "lap_ratio": 1.0,
                "anis_motion": 1.0,
                "anis_static": 1.0,
                "anis_gain": 1.0,

                "sharp_drop": 1.0,
                "drop_score": 0.0,

                # ✅ global motion blur (quality penalty) debug fields
                "global_motion_blur": 0.0,
                "is_global_motion_blur": 0.0,   # ✅ [ADDED] 给 report 用（G）
                "global_blur_prev_next": 0.0,
                "global_blur_edge_ratio": 0.0,
                "global_blur_sharp_p25": 0.0,
                "global_blur_drop": 1.0,
                "blur_score": 0.0,

                # ✅ defocus / global soft blur debug fields
                "defocus_blur": 0.0,
                "is_defocus_blur": 0.0,
                "defocus_edge_ratio": 0.0,
                "defocus_sharp_p25": 0.0,
                "defocus_sharp_p50": 0.0,

                "iqa_musiq_raw": float(musiq_raw[i]) if i < len(musiq_raw) else 0.0,
                "iqa_musiq": float(iqa),

                # ✅ highlight fields
                "highlight_clip_ratio": 0.0,
                "highlight_tile_max_ratio": 0.0,
                "highlight_p99_5": 0.0,
                "is_highlight_blowout": 0.0,
                "hf_highlight_blowout": 0.0,


                # hard-floor 用
                "hf_defocus_blur": 0.0,

                "blurmap_r_m": 0.0,
                "blurmap_r_d": 0.0,
                "blurmap_occ_m": 0.0,
                "blurmap_blob_m": 0.0,
                "blurmap_conf_b": 0.0,

                # hard-floor report fields
                "hf_read_ok": 1.0,
                "hf_area": 0.0,
                "hf_bright_mean": 0.0,
                "hf_contrast_std": 0.0,
                "hf_lap_var_small": 0.0,
                "hf_global_motion_blur": 0.0,
                "hf_local_motion_blur": 0.0,
                "hf_hard_fail": 0.0,
                "hf_hard_tags": "",
            }

        self.last_scores = score_map

        # ===================== ✅ blurmap score_new（覆盖 global/local/defocus） =====================
        blurmap_applied = False
        blurmap_metrics = None

        if use_blurmap:
            try:
                _bdir = self._ensure_blurmap_ready(frame_paths)
                if _bdir is not None:
                    # 如果你想把 resolve 后的目录写回 self，方便 debug
                    self.blurmap_dir = _bdir
                    blurmap_metrics = self._load_blurmap_metrics(frame_paths)
                else:
                    blurmap_metrics = None
            except Exception as e:
                self._dbg(f"[blurmap] ensure/load failed: {e}")
                blurmap_metrics = None


        if blurmap_metrics is not None and len(blurmap_metrics) == n:
            prev_next_small = self._compute_prev_next_small(frame_paths, resize_hw=(160, 160))
            edge_ratio_small = self._compute_edge_ratio_small(frame_paths, resize_hw=(160, 160))
            motion_area_small = self._compute_motion_area_ratio_small(frame_paths, resize_hw=(160, 160))

            gm_new, lm_new, df_new = self._compute_blurmap_scores_new(
                blurm=blurmap_metrics,
                flow_norm=flow_norm,
                prev_next=prev_next_small,
                cut_scores=cut_scores,
                edge_ratio=edge_ratio_small,
                motion_area_ratio=motion_area_small,
            )

            th_gb = float(getattr(self, "blurmap_global_motion_blur_score_th", 0.55))
            th_lb = float(getattr(self, "blurmap_local_motion_blur_score_th", 0.55))
            th_df = float(getattr(self, "blurmap_defocus_blur_score_th", 0.95))

            # ✅ 各自开关：仅控制 “is_*” hard-flag（不影响写入指标/调试字段）
            enable_gb = bool(getattr(self, "global_motion_blur_filter", True))
            enable_lb = bool(getattr(self, "local_motion_blur_filter", True))
            enable_df = bool(getattr(self, "defocus_blur_filter", True))

            for ii in range(n):
                gb = float(gm_new[ii])
                lb = float(lm_new[ii])
                df = float(df_new[ii])

                # global motion blur
                detail_scores[ii]["global_motion_blur"] = gb
                detail_scores[ii]["hf_global_motion_blur"] = gb
                detail_scores[ii]["is_global_motion_blur"] = 1.0 if (
                    enable_gb and math.isfinite(gb) and gb >= th_gb
                ) else 0.0

                # local motion blur（保持你 report/hf 用的字段名不变）
                detail_scores[ii]["blur_score"] = lb
                detail_scores[ii]["hf_local_motion_blur"] = lb
                detail_scores[ii]["is_local_motion_blur"] = 1.0 if (
                    enable_lb and math.isfinite(lb) and lb >= th_lb
                ) else 0.0
                detail_scores[ii]["motion_area_ratio"] = float(motion_area_small[ii])

                # defocus
                detail_scores[ii]["defocus_blur"] = df
                detail_scores[ii]["hf_defocus_blur"] = df
                detail_scores[ii]["is_defocus_blur"] = 1.0 if (
                    enable_df and math.isfinite(df) and df >= th_df
                ) else 0.0

                # 可选：写 blurmap debug（不影响任何旧逻辑）
                detail_scores[ii]["blurmap_r_m"] = float(blurmap_metrics[ii].get("r_m", 0.0))
                detail_scores[ii]["blurmap_r_d"] = float(blurmap_metrics[ii].get("r_d", 0.0))
                detail_scores[ii]["blurmap_occ_m"] = float(blurmap_metrics[ii].get("occ_m", 0.0))
                detail_scores[ii]["blurmap_blob_m"] = float(blurmap_metrics[ii].get("blob_m", 0.0))
                detail_scores[ii]["blurmap_conf_b"] = float(blurmap_metrics[ii].get("conf_b", 0.0))

            # ✅ defocus 的 run_len 连续段逻辑：也必须受 defocus 开关控制
            if enable_df:
                run_len = int(getattr(self, "defocus_blur_run_len", 3))
                run_len = max(1, run_len)
                flags = [False] * n
                run = 0
                for i in range(n):
                    if float(df_new[i]) >= th_df:
                        run += 1
                    else:
                        if run >= run_len:
                            for j in range(i - run, i):
                                if 0 <= j < n:
                                    flags[j] = True
                        run = 0
                if run >= run_len:
                    for j in range(n - run, n):
                        if 0 <= j < n:
                            flags[j] = True

                num_df = 0
                for ii in range(n):
                    if flags[ii]:
                        detail_scores[ii]["is_defocus_blur"] = 1.0
                        num_df += 1
                    else:
                        detail_scores[ii]["is_defocus_blur"] = 0.0
            else:
                # 明确清零，避免上面被写成 1
                for ii in range(n):
                    detail_scores[ii]["is_defocus_blur"] = 0.0
                run_len = int(getattr(self, "defocus_blur_run_len", 3))
                num_df = 0

            self._dbg(
                f"[blurmap] applied score_new (global/local/defocus), "
                f"enable(gb={enable_gb},lb={enable_lb},df={enable_df}), "
                f"defocus(run_len={run_len})={num_df}/{n}"
            )
            blurmap_applied = True
        else:
            if use_blurmap:
                self._dbg("[blurmap] not available -> fallback to legacy blur logic")
        # =====================================================================


        # 6.5) record global motion blur signals into detail_scores (for report + hard floors)
        if (not blurmap_applied) and need_global_blur:
            if need_global_blur:
                th_gb = float(getattr(self, "global_motion_blur_score_th", 0.85))
                for ii in range(n):
                    try:
                        gb = float(global_blur_scores[ii])
                    except Exception:
                        gb = 0.0

                    detail_scores[ii]["global_motion_blur"] = float(gb)
                    detail_scores[ii]["hf_global_motion_blur"] = float(gb)

                    # ✅ 报告字段：只要算了就写 is_global_motion_blur（不依赖 hard_floor）
                    if math.isfinite(gb) and math.isfinite(th_gb) and gb >= th_gb:
                        detail_scores[ii]["is_global_motion_blur"] = 1.0
                    else:
                        detail_scores[ii]["is_global_motion_blur"] = 0.0

                    info = global_blur_dbg.get(ii) if global_blur_dbg else None
                    if info:
                        detail_scores[ii]["global_blur_prev_next"] = float(info.get("prev_next", 0.0))
                        detail_scores[ii]["global_blur_edge_ratio"] = float(info.get("edge_ratio", 0.0))
                        detail_scores[ii]["global_blur_sharp_p25"] = float(info.get("sharp_p25", 0.0))
                        detail_scores[ii]["global_blur_drop"] = float(info.get("drop", 1.0))

        # ===================== ✅ 6.55) defocus / global soft blur（非运动模糊） =====================
        if not blurmap_applied:
            # 原来的 defocus 计算 + 写入 + run_len
            defocus_scores: List[float] = []
            defocus_dbg: Dict[int, Dict[str, float]] = {}
            need_defocus = bool(getattr(self, "defocus_blur_filter", True)) and (n >= 1)

            if need_defocus:
                try:
                    defocus_scores, defocus_dbg = self._compute_defocus_blur_quality_scores(frame_paths)
                    if len(defocus_scores) != n:
                        defocus_scores, defocus_dbg = [], {}
                except Exception as e:
                    self._dbg(f"_compute_defocus_blur_quality_scores failed: {e}")
                    defocus_scores, defocus_dbg = [], {}

            self._dbg(f"[defocus] score stats: {self._stats(defocus_scores)}")

            # 写入 detail（单帧分数 & dbg）
            if defocus_scores and len(defocus_scores) == n:
                th_df = float(getattr(self, "defocus_blur_score_th", 0.75))
                for ii in range(n):
                    sc = float(defocus_scores[ii])
                    detail_scores[ii]["defocus_blur"] = sc
                    detail_scores[ii]["hf_defocus_blur"] = sc

                    info = defocus_dbg.get(ii) if defocus_dbg else None
                    if info:
                        detail_scores[ii]["defocus_edge_ratio"] = float(info.get("edge_ratio", 0.0))
                        detail_scores[ii]["defocus_sharp_p25"] = float(info.get("sharp_p25", 0.0))
                        detail_scores[ii]["defocus_sharp_p50"] = float(info.get("sharp_p50", 0.0))

                    # 先按单帧阈值写一个（后面会被 run_len 逻辑覆盖成“段落级”更稳标签）
                    detail_scores[ii]["is_defocus_blur"] = 1.0 if (math.isfinite(sc) and sc >= th_df) else 0.0

            # ✅ 连续段落判定：连续 N 帧 >= th 才最终认为是 defocus 段
            if defocus_scores and len(defocus_scores) == n:
                th_df = float(getattr(self, "defocus_blur_score_th", 0.75))
                run_len = int(getattr(self, "defocus_blur_run_len", 3))
                run_len = max(1, run_len)

                flags = [False] * n
                run = 0
                for i in range(n):
                    if float(defocus_scores[i]) >= th_df:
                        run += 1
                    else:
                        if run >= run_len:
                            for j in range(i - run, i):
                                if 0 <= j < n:
                                    flags[j] = True
                        run = 0
                if run >= run_len:
                    for j in range(n - run, n):
                        if 0 <= j < n:
                            flags[j] = True

                # 用段落结果覆盖 is_defocus_blur（更稳）
                num_df = 0
                for ii in range(n):
                    if flags[ii]:
                        detail_scores[ii]["is_defocus_blur"] = 1.0
                        num_df += 1
                    else:
                        detail_scores[ii]["is_defocus_blur"] = 0.0

                self._dbg(f"[defocus] detected defocus(run_len={run_len}): {num_df}/{n}")
        # ===========================================================================

        # 6.6) record fixed-resize raw metrics for hard floors
        raw_metrics = getattr(self, "_last_quality_raw_metrics", None) or []
        if raw_metrics and len(raw_metrics) == n:
            for ii in range(n):
                rm = raw_metrics[ii] or {}
                detail_scores[ii]["hf_read_ok"] = float(rm.get("read_ok", 1.0))
                detail_scores[ii]["hf_area"] = float(rm.get("area", 0.0))
                detail_scores[ii]["hf_bright_mean"] = float(rm.get("bright_mean_small", 0.0))
                detail_scores[ii]["hf_contrast_std"] = float(rm.get("contrast_std_small", 0.0))
                detail_scores[ii]["hf_lap_var_small"] = float(rm.get("lap_var_small", 0.0))


        # 8) cross-dissolve / transition detection
        trans_set: set[int] = set()
        dissolve_dbg: Dict[int, Dict[str, float]] = {}

        if self.transition_filter and n >= 3:
            # ---- per-chain switches (default keep old behavior) ----
            dissolve_on = bool(getattr(self, "transition_dissolve_enable", True))
            emb_on = bool(getattr(self, "transition_emb_enable", True))
            cut_on = bool(getattr(self, "transition_cut_enable", True))

            # ---- thresholds (default keep old constants) ----
            SIM_MIN = float(getattr(self, "transition_emb_sim_min", 0.60))
            SPAN_MAX = float(getattr(self, "transition_emb_span_max", 0.95))
            DELTA_MIN = float(getattr(self, "transition_emb_delta_min", 0.03))

            CUT_SOFT_TH = float(getattr(self, "transition_cut_soft_th", 0.10))
            CUT_HARD_TH = float(getattr(self, "transition_cut_hard_th", 0.08))
            RUN_LEN = int(getattr(self, "transition_cut_run_len", 1))

            # ---- dissolve chain ----
            dissolve_like = [False] * n
            dissolve_dbg = {}
            if dissolve_on:
                dissolve_like, dissolve_dbg = self._compute_cross_dissolve_mask(frame_paths)

            # ---- embedding chain ----
            embedding_vecs = self._prepare_embeddings(embeddings, n)
            emb_like = [False] * n
            if emb_on and (embedding_vecs is not None) and n >= 3:
                for i in range(1, n - 1):
                    sim_prev = _cosine_similarity(embedding_vecs[i], embedding_vecs[i - 1])
                    sim_next = _cosine_similarity(embedding_vecs[i], embedding_vecs[i + 1])
                    sim_span = _cosine_similarity(embedding_vecs[i - 1], embedding_vecs[i + 1])
                    m = min(sim_prev, sim_next)
                    if (m >= SIM_MIN) and (sim_span <= SPAN_MAX) and ((m - sim_span) >= DELTA_MIN):
                        emb_like[i] = True

            # ---- cut chain (build transition_mask) ----
            transition_mask = [False] * n
            if cut_on:
                run = 0
                for i in range(1, n):
                    if float(cut_scores[i]) >= CUT_SOFT_TH:
                        run += 1
                    else:
                        if run >= RUN_LEN:
                            for j in range(i - run, i):
                                if 0 <= j < n:
                                    transition_mask[j] = True
                        run = 0
                if run >= RUN_LEN:
                    for j in range(n - run, n):
                        if 0 <= j < n:
                            transition_mask[j] = True

            # ---- fuse ----
            for i in range(n):
                if dissolve_like[i]:
                    trans_set.add(i)
                elif emb_like[i]:
                    require_cut = bool(getattr(self, "transition_emb_require_cut", True))
                    if require_cut:
                        if cut_on and transition_mask[i]:
                            trans_set.add(i)
                    else:
                        if (cut_on and transition_mask[i]) or (float(cut_scores[i]) >= CUT_HARD_TH):
                            trans_set.add(i)


            # ---- margin expand ----
            if trans_set and self.transition_margin and self.transition_margin > 0:
                m = int(self.transition_margin)
                expanded = set()
                for i in trans_set:
                    for j in range(i - m, i + m + 1):
                        if 0 <= j < n:
                            expanded.add(j)
                trans_set = expanded

        self.last_transition_indices = sorted(list(trans_set))

        self._dbg(f"[transition] detected transitions: {len(self.last_transition_indices)}/{n}")
        self._dbg_list("[transition] transition_indices", self.last_transition_indices)

        if self.debug and dissolve_dbg:
            keys = sorted(dissolve_dbg.keys())[:10]
            for k in keys:
                info = dissolve_dbg[k]
                self._dbg(
                    f"[transition dbg] idx={k} alpha={info.get('alpha', 0):.3f} "
                    f"res={info.get('residual', 0):.3f} pn={info.get('prev_next', 0):.3f} "
                    f"pm={info.get('prev_mid', 0):.3f} mn={info.get('mid_next', 0):.3f}"
                )

        # ===================== ✅ highlight blowout detection =====================
        blowout_set: set[int] = set()
        blowout_dbg: Dict[int, Dict[str, float]] = {}

        if bool(getattr(self, "highlight_blowout_filter", True)) and n >= 1:
            try:
                blowout_mask, blowout_dbg = self._compute_highlight_blowout_mask(frame_paths)
                for ii, f in enumerate(blowout_mask):
                    if f:
                        blowout_set.add(ii)
            except Exception as e:
                self._dbg(f"[blowout] compute failed: {e}")
                blowout_set = set()
                blowout_dbg = {}

        # margin expand（仿 transition）
        if blowout_set and int(getattr(self, "highlight_blowout_margin", 0)) > 0:
            m = int(getattr(self, "highlight_blowout_margin", 0))
            expanded = set()
            for i in blowout_set:
                for j in range(i - m, i + m + 1):
                    if 0 <= j < n:
                        expanded.add(j)
            blowout_set = expanded

        self.last_blowout_indices = sorted(list(blowout_set))
        self._dbg(f"[blowout] detected highlight blowout: {len(self.last_blowout_indices)}/{n}")
        self._dbg_list("[blowout] blowout_indices", self.last_blowout_indices)

        # 写入 detail_scores（用于 report / hard floor）
        for ii in range(n):
            info = blowout_dbg.get(ii)
            if info:
                # --- highlight debug fields (align with _compute_highlight_blowout_mask dbg keys) ---
                detail_scores[ii]["highlight_bright_ratio"]   = float(info.get("bright_ratio", 0.0))
                detail_scores[ii]["highlight_g_global"]       = float(info.get("g_global", 0.0))
                detail_scores[ii]["highlight_best_area"]      = float(info.get("best_area", 0.0))
                detail_scores[ii]["highlight_best_delta"]     = float(info.get("best_delta", 0.0))
                detail_scores[ii]["highlight_best_sat_ratio"] = float(info.get("best_sat_ratio", 0.0))

                # 可选：把你原来这3个字段也保留，但用“兼容取值”，避免历史字段断掉
                detail_scores[ii]["highlight_clip_ratio"]      = float(info.get("clip_ratio", info.get("bright_ratio", 0.0)))
                detail_scores[ii]["highlight_tile_max_ratio"]  = float(info.get("tile_max_ratio", info.get("best_sat_ratio", 0.0)))
                detail_scores[ii]["highlight_p99_5"]           = float(info.get("p99_5", info.get("g_global", 0.0)))

            detail_scores[ii]["is_highlight_blowout"] = 1.0 if (ii in blowout_set) else 0.0
            detail_scores[ii]["hf_highlight_blowout"] = 1.0 if (ii in blowout_set) else 0.0
        # ========================================================================


        # ===================== ✅ 9.5) local motion blur 检测（prev,cur,next） =====================
        if not blurmap_applied:
            # 原来的 local blur 检测 + 写入
            local_blur_set: set[int] = set()
            local_blur_dbg: Dict[int, Dict[str, float]] = {}
            need_local_blur = bool(self.local_motion_blur_filter) and (n >= 3)
            if need_local_blur:
                flags, local_blur_dbg = self._compute_local_motion_blur_flags(frame_paths)
                for ii, f in enumerate(flags):
                    if f:
                        local_blur_set.add(ii)

                # 写入 detail（即使后面硬筛/或不筛，报告也能看到）
                for ii in range(n):
                    info = local_blur_dbg.get(ii)
                    if info:
                        detail_scores[ii]["motion_area_ratio"] = float(info.get("motion_area_ratio", 0.0))
                        detail_scores[ii]["sharp_motion"] = float(info.get("sharp_motion", 0.0))
                        detail_scores[ii]["sharp_static"] = float(info.get("sharp_static", 0.0))
                        detail_scores[ii]["sharp_ratio"] = float(info.get("sharp_ratio", 1.0))

                        detail_scores[ii]["lap_motion"] = float(info.get("lap_motion", 0.0))
                        detail_scores[ii]["lap_static"] = float(info.get("lap_static", 0.0))
                        detail_scores[ii]["lap_ratio"] = float(info.get("lap_ratio", 1.0))
                        detail_scores[ii]["anis_motion"] = float(info.get("anis_motion", 1.0))
                        detail_scores[ii]["anis_static"] = float(info.get("anis_static", 1.0))
                        detail_scores[ii]["anis_gain"] = float(info.get("anis_gain", 1.0))

                        detail_scores[ii]["sharp_drop"] = float(info.get("sharp_drop", 1.0))
                        detail_scores[ii]["drop_score"] = float(info.get("drop_score", 0.0))

                        bs = float(info.get("blur_score", 0.0))
                        detail_scores[ii]["blur_score"] = bs
                        detail_scores[ii]["hf_local_motion_blur"] = bs
                        detail_scores[ii]["is_local_motion_blur"] = 1.0 if (bs >= float(self.local_motion_blur_score_th)) else 0.0


                self._dbg(f"[local_blur] detected local_motion_blur(flags): {len(local_blur_set)}/{n}")
                if self.debug and local_blur_dbg:
                    keys = sorted([k for k in local_blur_dbg.keys() if k in local_blur_set])[:10]
                    for k in keys:
                        info = local_blur_dbg[k]
                        self._dbg(
                            f"[local_blur dbg] idx={k} area={info.get('motion_area_ratio',0):.3f} "
                            f"ratio={info.get('sharp_ratio',1):.3f} lapr={info.get('lap_ratio',1):.3f} "
                            f"anisG={info.get('anis_gain',1):.3f} drop={info.get('sharp_drop',1):.3f} "
                            f"score={info.get('blur_score',0):.3f} "
                            f"sm={info.get('sharp_motion',0):.4f} ss={info.get('sharp_static',0):.4f}"
                        )


        # ===========================================================================
        # 9.7) hard floors tagging
        hard_fail_set: set[int] = set()
        if self.hard_floor_enable:
            for ii in range(n):
                tags_hf: List[str] = []

                # A 类 hard floors（绝对值，来自 fixed-resize raw metrics）
                read_ok = float(detail_scores[ii].get("hf_read_ok", 1.0))
                area = float(detail_scores[ii].get("hf_area", 0.0))
                bright = float(detail_scores[ii].get("hf_bright_mean", 0.0))
                contr = float(detail_scores[ii].get("hf_contrast_std", 0.0))
                lapv = float(detail_scores[ii].get("hf_lap_var_small", 0.0))

                if read_ok < 0.5:
                    tags_hf.append("hf_read_fail")

                th_area = float(self.hard_floor_min_area)
                if math.isfinite(th_area) and th_area > 0.0 and math.isfinite(area) and area > 0.0 and area < th_area:
                    tags_hf.append("hf_low_res")

                bmin = float(self.hard_floor_bright_min)
                bmax = float(self.hard_floor_bright_max)
                if math.isfinite(bright):
                    if math.isfinite(bmin) and bmin > 0.0 and bright < bmin:
                        tags_hf.append("hf_underexposed")
                    if math.isfinite(bmax) and bmax < 255.0 and bright > bmax:
                        tags_hf.append("hf_overexposed")

                cmin = float(self.hard_floor_contrast_min)
                if math.isfinite(cmin) and cmin > 0.0 and math.isfinite(contr) and contr < cmin:
                    tags_hf.append("hf_low_contrast")

                lmin = float(self.hard_floor_lap_var_small_min)
                if math.isfinite(lmin) and lmin > 0.0 and math.isfinite(lapv) and lapv < lmin:
                    tags_hf.append("hf_low_sharpness")

                # blur hard floors: global motion blur（保持你原来的阈值 hard_th）
                if self.global_motion_blur_filter:
                    gb = float(detail_scores[ii].get("hf_global_motion_blur", 0.0))
                    th_gb = float(
                        getattr(self, "blurmap_global_motion_blur_score_th", 0.55)
                        if blurmap_applied
                        else getattr(self, "global_motion_blur_score_th", 0.85)
                    )
                    if math.isfinite(gb) and math.isfinite(th_gb) and gb >= th_gb:
                        tags_hf.append("hf_global_motion_blur")
                        detail_scores[ii]["is_global_motion_blur"] = 1.0

                if self.local_motion_blur_filter:
                    bs = float(detail_scores[ii].get("hf_local_motion_blur", 0.0))
                    th_lb = float(
                        getattr(self, "blurmap_local_motion_blur_score_th", 0.55)
                        if blurmap_applied
                        else float(self.local_motion_blur_score_th)
                    )
                    if math.isfinite(bs) and math.isfinite(th_lb) and bs >= th_lb:
                        tags_hf.append("hf_local_motion_blur")

                
                # defocus hard floor（段落级标签）
                if bool(getattr(self, "defocus_blur_filter", True)):
                    is_df = float(detail_scores[ii].get("is_defocus_blur", 0.0))
                    if is_df >= 0.5:
                        tags_hf.append("hf_defocus_blur")


                # transition hard floors（只在 transition_filter 打开时参与 hardfloor）
                if self.transition_filter and (ii in trans_set):
                    tags_hf.append("hf_transition")

                # highlight blowout hard floor
                if bool(getattr(self, "highlight_blowout_filter", True)) and (ii in blowout_set):
                    tags_hf.append("hf_highlight_blowout")


                if tags_hf:
                    hard_fail_set.add(ii)
                    detail_scores[ii]["hf_hard_fail"] = 1.0
                    detail_scores[ii]["hf_hard_tags"] = "+".join(tags_hf)
                else:
                    detail_scores[ii]["hf_hard_fail"] = 0.0
                    detail_scores[ii]["hf_hard_tags"] = ""
        # ===========================================================================


        # 10) 写入 filter_tag / is_transition / is_filtered
        for i in range(n):
            tags: List[str] = []

            hf_tags = str(detail_scores[i].get("hf_hard_tags", "") or "")
            if hf_tags:
                tags.extend([t for t in hf_tags.split("+") if t])

            if i in trans_set:
                detail_scores[i]["is_transition"] = 1.0

            if tags:
                # 去重但保持相对顺序
                seen = set()
                uniq = []
                for t in tags:
                    if t and t not in seen:
                        uniq.append(t)
                        seen.add(t)
                detail_scores[i]["filter_tag"] = "+".join(uniq)
                detail_scores[i]["is_filtered"] = 1.0

        self.last_detail_scores = detail_scores

        # ===================== [ADDED] 10.5) apply hard-floor filtering to candidates (allow empty) =====================
        if self.hard_floor_enable and candidates and hard_fail_set:
            before = len(candidates)
            candidates_after = [i for i in candidates if i not in hard_fail_set]
            removed = before - len(candidates_after)
            self._dbg(f"[hard_floor] removed {removed}/{before} candidates (allow_empty={self.hard_floor_allow_empty})")

            candidates = candidates_after

            if (not candidates) and self.hard_floor_allow_empty:
                self._dbg("[hard_floor] candidates became empty -> return [] (by design)")
                return []
        # ===========================================================================

        # 11) 调整 k（并确保不会超过 candidates 数量）
        k_desired = max(1, min(int(num_frames), n, self.max_frames))
        k = self._adjust_k_by_score_clusters(final_scores, k_desired)
        if candidates:
            k = min(k, len(candidates))

        k_lower = int(getattr(self, "_last_k_lower", k))
        k_upper = int(getattr(self, "_last_k_upper", k))
        if candidates:
            k_upper = min(k_upper, len(candidates))
            k_lower = min(k_lower, k_upper)

        self._dbg(
            f"[select] candidates={len(candidates)}/{n}, k={k}, "
            f"target_range=[{k_lower},{k_upper}] "
        )


        # ===========================================================================

        # 12) 选帧（支持 diversity）
        embedding_vecs = self._prepare_embeddings(embeddings, n)
        use_diversity = embedding_vecs is not None and float(self.last_weights.get("w_diversity", 0.0)) > 0.0

        def _rank_score(i: int) -> float:
            s = float(final_scores[i])
            return s

        # 分数完全相同：均匀抽样
        if all(abs(final_scores[i] - final_scores[0]) < 1e-6 for i in range(1, n)):
            step = max(1, len(candidates) // max(k, 1))
            indices = [candidates[j] for j in range(0, len(candidates), step)][:k]
            indices = sorted(indices)
            self._dbg_list("[select] selected (uniform)", indices)
            bad = [i for i in indices if i in trans_set]
            if bad:
                self._dbg("[WARN] selected includes transition frames:", bad)
            return indices

        scored = [(_rank_score(i), i) for i in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)

        if not use_diversity:
            top = sorted(scored[:k], key=lambda x: x[1])
            indices = [idx for _, idx in top]
            self._dbg_list("[select] selected", indices)
            bad = [i for i in indices if i in trans_set]
            if bad:
                self._dbg("[WARN] selected includes transition frames:", bad)
            return indices

        # ============================================================
        # ✅ 分数段逐步引入 + 多 seed 贪心 diversity（保留原逻辑）
        #    不够下界时：阈值退火 + 时间惩罚（帧索引差）
        # ============================================================

        def _build_score_bands(sorted_scored: List[Tuple[float, int]]) -> List[Tuple[int, int]]:
            if not sorted_scored:
                return []
            scores_only = [float(s) for s, _ in sorted_scored]
            mx = scores_only[0]
            mn = scores_only[-1]
            r = mx - mn
            if r < 1e-8:
                return [(0, len(sorted_scored))]

            gap_th = 0.15 * r
            bands: List[Tuple[int, int]] = []
            start = 0
            for ii in range(len(scores_only) - 1):
                if (scores_only[ii] - scores_only[ii + 1]) >= gap_th:
                    bands.append((start, ii + 1))
                    start = ii + 1
            bands.append((start, len(sorted_scored)))
            return bands

        def _greedy_diverse(pool: List[Tuple[float, int]], seed_idx: Optional[int], cap: int) -> List[int]:
            if not pool:
                return []

            pool_indices = [idx for _, idx in pool]
            pool_set = set(pool_indices)

            selected: List[int] = []
            selected_set: set[int] = set()

            if seed_idx is not None and seed_idx in pool_set:
                selected.append(seed_idx)
                selected_set.add(seed_idx)
                if len(selected) >= cap:
                    return selected

            for _, idx in pool:
                if idx in selected_set:
                    continue
                cand_vec = embedding_vecs[idx]
                if selected:
                    max_sim = max(_cosine_similarity(cand_vec, embedding_vecs[j]) for j in selected)
                    if max_sim >= self.emb_diversity_sim_threshold:
                        continue
                selected.append(idx)
                selected_set.add(idx)
                if len(selected) >= cap:
                    break

            return selected

        def _score_sum(idxs: List[int]) -> float:
            return float(sum(_rank_score(i) for i in idxs))

        def _time_weight(d: int, beta: float, tau: float) -> float:
            if tau <= 1e-6:
                return 1.0
            try:
                w = 1.0 - float(beta) * math.exp(-float(d) / float(tau))
            except Exception:
                w = 1.0
            if not math.isfinite(w):
                return 1.0
            return float(max(0.0, min(1.0, w)))

        def _min_dist_to_selected(idx: int, selected: List[int]) -> int:
            if not selected:
                return 10**9
            return min(abs(int(idx) - int(j)) for j in selected)

        def _greedy_diverse_time(
            pool: List[Tuple[float, int]],
            seed_idx: Optional[int],
            cap: int,
            sim_th: float,
            hard_th: Optional[float],
            beta: float,
            tau: float,
        ) -> List[int]:
            if not pool:
                return []

            pool_indices = [idx for _, idx in pool]
            pool_set = set(pool_indices)

            selected: List[int] = []
            selected_set: set[int] = set()

            if seed_idx is not None and seed_idx in pool_set:
                selected.append(seed_idx)
                selected_set.add(seed_idx)
                if len(selected) >= cap:
                    return selected

            while len(selected) < cap:
                best_idx: Optional[int] = None
                best_val = -1e18

                for _, idx in pool:
                    if idx in selected_set:
                        continue

                    if selected:
                        cand_vec = embedding_vecs[idx]
                        max_sim = max(_cosine_similarity(cand_vec, embedding_vecs[j]) for j in selected)
                        if hard_th is not None and max_sim >= float(hard_th):
                            continue
                        if max_sim >= float(sim_th):
                            continue

                        d = _min_dist_to_selected(idx, selected)
                        w = _time_weight(d, beta=beta, tau=tau)
                    else:
                        w = 1.0

                    base = float(_rank_score(idx))
                    val = base * float(w)

                    if val > best_val:
                        best_val = val
                        best_idx = idx

                if best_idx is None:
                    break

                selected.append(best_idx)
                selected_set.add(best_idx)

            return selected

        bands = _build_score_bands(scored)
        if not bands:
            self._dbg("[diversity] no bands -> empty")
            return []

        if self.debug:
            self._dbg(f"[diversity] bands={bands}")

        best_overall: List[int] = []
        best_overall_len = -1
        best_overall_sum = -1e18

        pool_end = 0
        for bi, (bs, be) in enumerate(bands):
            pool_end = max(pool_end, be)
            pool = scored[:pool_end]

            M = max(1, int(self.diversity_seed_trials))
            seed_candidates = [idx for _, idx in pool[:M]]

            best_this: List[int] = []
            best_len = -1
            best_sum = -1e18

            for seed in seed_candidates:
                sel = _greedy_diverse(pool, seed_idx=seed, cap=max(1, k_upper))
                l = len(sel)
                s = _score_sum(sel)
                if (l > best_len) or (l == best_len and s > best_sum):
                    best_this = sel
                    best_len = l
                    best_sum = s

            if (best_len > best_overall_len) or (best_len == best_overall_len and best_sum > best_overall_sum):
                best_overall = best_this
                best_overall_len = best_len
                best_overall_sum = best_sum

            self._dbg(f"[diversity] band#{bi} pool={len(pool)} best_len={best_len} target=[{k_lower},{k_upper}]")

            if best_len >= k_lower:
                if best_len > k_upper:
                    order = {idx: rank for rank, (_, idx) in enumerate(pool)}
                    best_this_sorted = sorted(best_this, key=lambda i: order.get(i, 10**9))
                    best_this = best_this_sorted[:k_upper]
                    best_len = len(best_this)

                out = sorted(best_this)
                self._dbg_list("[select] selected (diversity bands)", out)
                bad = [i for i in out if i in trans_set]
                if bad:
                    self._dbg("[WARN] selected includes transition frames:", bad)
                return out

            continue

        # ===================== ✅ 不够下界：阈值退火 + 时间惩罚补齐 =====================
        if self.diversity_relax_enable and k_lower > 0 and best_overall_len < k_lower:
            full_pool = scored
            M = max(1, int(self.diversity_seed_trials))
            seed_candidates = [idx for _, idx in full_pool[:M]]

            try:
                approx_gap = int(len(candidates) // max(1, int(k_upper)))
            except Exception:
                approx_gap = 3
            gap_soft = int(max(int(self.diversity_time_gap_min), min(int(self.diversity_time_gap_max), max(2, approx_gap))))
            tau = float(max(1, gap_soft))

            base_T = float(self.emb_diversity_sim_threshold)
            hard_T = float(self.diversity_hard_sim_threshold)
            eps_T = 1e-6

            def _dbg_sel_stats(tag: str, sel: List[int], limit_show: int = 16) -> None:
                if not self.debug:
                    return
                if not sel:
                    self._dbg(f"{tag} sel=empty")
                    return
                ss = sorted(int(x) for x in sel)
                if len(ss) >= 2:
                    min_gap = min(abs(ss[i] - ss[i - 1]) for i in range(1, len(ss)))
                else:
                    min_gap = 10**9

                max_sim = 0.0
                if embedding_vecs is not None and len(ss) >= 2:
                    try:
                        for i1 in range(len(ss)):
                            v1 = embedding_vecs[ss[i1]]
                            for i2 in range(i1 + 1, len(ss)):
                                sim = _cosine_similarity(v1, embedding_vecs[ss[i2]])
                                if sim > max_sim:
                                    max_sim = sim
                    except Exception:
                        max_sim = 0.0

                head = ss[: min(limit_show, len(ss))]
                more = "" if len(ss) <= limit_show else f" ... (total={len(ss)})"
                self._dbg(f"{tag} len={len(ss)} min_gap={min_gap} max_sim={max_sim:.4f} head={head}{more}")

            self._dbg(
                f"[diversity-rescue] start len={best_overall_len} < k_lower={k_lower}; "
                f"base_T={base_T:.3f}, step={self.diversity_relax_step:.3f}, hard_T={hard_T:.3f}, "
                f"gap_soft={gap_soft}, tau={tau:.1f}"
            )
            _dbg_sel_stats("[diversity-rescue] best_overall", best_overall)

            best_rescue: List[int] = best_overall[:] if best_overall else []
            best_rescue_len = len(best_rescue)
            best_rescue_sum = _score_sum(best_rescue)

            max_rounds = max(1, int(self.diversity_relax_max_rounds))
            step = float(self.diversity_relax_step)

            for rr in range(max_rounds):
                sim_th = min(hard_T - eps_T, base_T + step * float(rr + 1))
                beta = min(
                    float(self.diversity_time_beta_max),
                    float(self.diversity_time_beta_start) + float(self.diversity_time_beta_step) * float(rr),
                )

                best_this: List[int] = []
                best_len = -1
                best_sum = -1e18

                for seed in seed_candidates:
                    sel = _greedy_diverse_time(
                        full_pool,
                        seed_idx=seed,
                        cap=max(1, k_upper),
                        sim_th=float(sim_th),
                        hard_th=float(hard_T),
                        beta=float(beta),
                        tau=float(tau),
                    )
                    l = len(sel)
                    s = _score_sum(sel)
                    if (l > best_len) or (l == best_len and s > best_sum):
                        best_this = sel
                        best_len = l
                        best_sum = s

                self._dbg(f"[diversity-rescue] round#{rr} sim_th={sim_th:.3f} beta={beta:.3f} best_len={best_len}")
                _dbg_sel_stats(f"[diversity-rescue] round#{rr} best_this", best_this)

                if (best_len > best_rescue_len) or (best_len == best_rescue_len and best_sum > best_rescue_sum):
                    best_rescue = best_this
                    best_rescue_len = best_len
                    best_rescue_sum = best_sum
                    _dbg_sel_stats(f"[diversity-rescue] round#{rr} update best_rescue", best_rescue)

                if best_len >= k_lower:
                    out = sorted(best_this[:k_upper])
                    self._dbg_list("[select] selected (diversity rescue)", out)
                    _dbg_sel_stats("[diversity-rescue] OUT", out)
                    bad = [i for i in out if i in trans_set]
                    if bad:
                        self._dbg("[WARN] selected includes transition frames:", bad)
                    return out

            self._dbg(f"[diversity-rescue] still < k_lower after rounds; fallback to hard-only (hard_T={hard_T:.3f})")
            beta = float(self.diversity_time_beta_max)
            best_this = []
            best_len = -1
            best_sum = -1e18
            for seed in seed_candidates:
                sel = _greedy_diverse_time(
                    full_pool,
                    seed_idx=seed,
                    cap=max(1, k_upper),
                    sim_th=float(hard_T - eps_T),
                    hard_th=float(hard_T),
                    beta=float(beta),
                    tau=float(tau),
                )
                l = len(sel)
                s = _score_sum(sel)
                if (l > best_len) or (l == best_len and s > best_sum):
                    best_this = sel
                    best_len = l
                    best_sum = s

            self._dbg(f"[diversity-rescue] hard-only best_len={best_len} beta={beta:.3f}")
            _dbg_sel_stats("[diversity-rescue] hard-only best_this", best_this)

            if best_len >= k_lower:
                out = sorted(best_this[:k_upper])
                self._dbg_list("[select] selected (diversity hard-only)", out)
                _dbg_sel_stats("[diversity-rescue] OUT(hard-only)", out)
                bad = [i for i in out if i in trans_set]
                if bad:
                    self._dbg("[WARN] selected includes transition frames:", bad)
                return out

            self._dbg(f"[diversity-rescue] hard-only still < k_lower; final fallback ignore sim, time-spread fill")
            selected: List[int] = []
            selected_set: set[int] = set()

            for idx in (best_rescue or []):
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                if len(selected) >= k_upper:
                    break

            if self.debug:
                self._dbg(f"[diversity-rescue] final-fallback init_from_best_rescue len={len(selected)}")
                _dbg_sel_stats("[diversity-rescue] final-fallback init_sel", selected)

            target_need = min(k_upper, max(1, k_lower))
            while len(selected) < target_need and len(selected) < len(candidates):
                best_idx = None
                best_val = -1e18
                for _, idx in full_pool:
                    if idx in selected_set:
                        continue
                    d = _min_dist_to_selected(idx, selected)
                    w = _time_weight(d, beta=beta, tau=tau) if selected else 1.0
                    val = float(_rank_score(idx)) * float(w)
                    if val > best_val:
                        best_val = val
                        best_idx = idx
                if best_idx is None:
                    break
                selected.append(best_idx)
                selected_set.add(best_idx)

            out = sorted(selected[:k_upper])
            self._dbg(f"[diversity-rescue] final fallback DONE len={len(out)} target=[{k_lower},{k_upper}] (need={target_need})")
            self._dbg_list("[select] selected (diversity final fallback)", out)
            _dbg_sel_stats("[diversity-rescue] OUT(final-fallback)", out)
            bad = [i for i in out if i in trans_set]
            if bad:
                self._dbg("[WARN] selected includes transition frames:", bad)
            return out

        # === ultimate fallback (shouldn't reach often) ===
        top = sorted(scored[:k], key=lambda x: x[1])
        indices = [idx for _, idx in top]
        self._dbg_list("[select] selected (diversity fallback-topk)", indices)
        return indices
