# -*- coding: utf-8 -*-
"""
电路板扫描 + TSDF 融合重建（两阶段：采集 → 离线融合）
======================================================

阶段 1 · 采集：
    只抓帧 (color_bgr + depth_mm + timestamp) 存入内存，不做任何融合/预处理，
    保证严格 1s/帧 采满 21~22 帧覆盖 360°+20°。

阶段 2 · 离线融合：
    1. 深度预处理：范围裁剪、反光剔除、双边滤波、小空洞 inpaint；
    2. 背景剔除：按内参反投影每像素到相机坐标，计算该点到转轴的径向
       距离 r，r > 140mm 的深度清零（电路板必然在转台台面中心附近）；
    3. 自适应 volume bounds：收集所有帧的有效 3D 点，按 5%~95% 分位
       再加点 padding，防止 volume 过大爆显存；
    4. 逐帧构造 world→cam 外参 → integrate 进 ScalableTSDFVolume；
    5. MarchingCubes 抽 mesh → 子采样降面 → 去噪 → 最大连通域 → 平滑。

坐标系（单位 mm）：同标定一致，世界 W = 第 0 帧相机坐标系快照。
"""

import json
import os
import sys
import time

import numpy as np
import cv2
import open3d as o3d

# 脚本在 src/ 下，把仓库根目录加进 sys.path 以便 import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (OUTPUT_DIR, AXIS_PARAM_FILE, RESULT_DIR, ACQUISITION,
                    DEPTH_MIN_MM, DEPTH_MAX_MM, ensure_dirs)
from camera_utils import OrbbecCamera

# 允许直接把 2D 前景分割（中值背景 + 可选 SAM2）嵌入到流程
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
_FG_HAS_SEG = False
try:
    import segment_foreground as _fg_seg_mod
    _FG_HAS_SEG = True
except Exception as _fg_seg_err:
    _fg_seg_mod = None
    _FG_HAS_SEG = False


# ---------------------------------------------------------------------------
# 变换工具
# ---------------------------------------------------------------------------
def make_extrinsic_world_to_cam(axis_origin, axis_dir, theta):
    """world→cam 的 4x4 外参：绕转轴 (O, dir) 旋转 theta 弧度。"""
    axis_dir = np.asarray(axis_dir, dtype=np.float64)
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    axis_origin = np.asarray(axis_origin, dtype=np.float64)
    K = np.array([
        [0.0, -axis_dir[2], axis_dir[1]],
        [axis_dir[2], 0.0, -axis_dir[0]],
        [-axis_dir[1], axis_dir[0], 0.0],
    ], dtype=np.float64)
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    t = axis_origin - R @ axis_origin
    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T


def rotate_world_point(axis_origin, axis_dir, theta, P_world):
    """把 P_world 绕转轴旋转 theta → P_cam（3xN 或 Nx3 都支持）。"""
    P = np.asarray(P_world, dtype=np.float64)
    single = False
    if P.ndim == 1:
        P = P.reshape(1, 3)
        single = True
    elif P.shape[0] == 3 and P.shape[1] != 3:  # 3xN
        P = P.T
    axis_dir = np.asarray(axis_dir, dtype=np.float64)
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    axis_origin = np.asarray(axis_origin, dtype=np.float64)
    K = np.array([
        [0.0, -axis_dir[2], axis_dir[1]],
        [axis_dir[2], 0.0, -axis_dir[0]],
        [-axis_dir[1], axis_dir[0], 0.0],
    ], dtype=np.float64)
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    P_cam = (P - axis_origin) @ R.T + axis_origin
    if single:
        return P_cam[0]
    return P_cam  # Nx3


def make_pinhole_intr(fx, fy, cx, cy, w, h):
    return o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)


# ---------------------------------------------------------------------------
# 深度预处理 + 背景剔除
# ---------------------------------------------------------------------------
def preprocess_depth_for_tsdf(depth_raw_mm, color_bgr_aligned):
    """Astra 深度图预处理：返回 float32 mm，无效=0。

    方案 1（应对 PCB 绿油/圆盘大平面深度稀疏）：
      - 反光清零后不直接丢，改做「5×5 非零邻域均值填补」：
        仅当 3×3 有效邻域数 >= 6 或 5×5 >= 16 时填补（防止 inpaint
        式伪影，只做"信任邻域的保守插值"）。
      - 双边滤波后再做一轮 3×3 邻域填补（保边后补齐大平面）。
    """
    h, w = depth_raw_mm.shape
    depth = depth_raw_mm.astype(np.float32).copy()

    # 1) 范围裁剪（甜点区 300~470 mm）
    mask_valid = (depth > 300.0) & (depth < 470.0)
    depth[~mask_valid] = 0.0

    # 2) 反光高亮区域标记（金属焊盘/镜面）—— 先清零，稍后做保守补洞
    color_hsv = cv2.cvtColor(color_bgr_aligned, cv2.COLOR_BGR2HSV)
    v_chan = color_hsv[..., 2]
    s_chan = color_hsv[..., 1]
    mask_specular = (v_chan > 230) & (s_chan < 60)
    depth[mask_specular] = 0.0

    # 2.1) 保守邻域补洞（向量化）：对内部 0 像素做 3×3/5×5 非零邻域均值填补
    h, w = depth.shape
    valid_mask = depth > 0
    need_fill = (depth == 0)
    # 只填内部（2px 边距）+ 不是大面积整黑（超过合理空洞就不填）
    border = np.ones((h, w), dtype=bool)
    border[2:h-2, 2:w-2] = False
    need_fill = need_fill & (~border)

    if need_fill.any():
        # -- 3×3 有效邻域计数 & 累加和 --
        vm = valid_mask.astype(np.float32)
        d_valid = np.where(vm > 0, depth, 0.0).astype(np.float32)
        k3 = np.ones((3, 3), dtype=np.float32)
        count3 = cv2.filter2D(vm, -1, k3, borderType=cv2.BORDER_CONSTANT)  # 0~9
        sum3 = cv2.filter2D(d_valid, -1, k3, borderType=cv2.BORDER_CONSTANT)
        mean3 = np.divide(sum3, np.maximum(count3, 1.0), where=count3 > 0)

        # -- 5×5 有效邻域（仅在 3×3 不够 6 但 >=3 时查） --
        k5 = np.ones((5, 5), dtype=np.float32)
        count5 = cv2.filter2D(vm, -1, k5, borderType=cv2.BORDER_CONSTANT)  # 0~25
        sum5 = cv2.filter2D(d_valid, -1, k5, borderType=cv2.BORDER_CONSTANT)
        mean5 = np.divide(sum5, np.maximum(count5, 1.0), where=count5 > 0)

        # 判定：3×3 >= 3 填 3×3 均值；否则 3×3 >= 2 且 5×5 >= 8 填 5×5 均值
        # —— 补洞阈值从 (6, 3+16) 降到 (3, 2+8)：电路板深度稀疏是大面积连续
        #    的（不是离散噪点），放宽条件能把绿油大平面的"间歇性 0 像素"补齐，
        #    代价是边缘可能有 ±1px 扩散，但双边滤波会保边，TSDF 融合时 5mm
        #    trunc 会把这种小误差直接吸收。
        fill3 = need_fill & (count3 >= 3)
        fill5 = need_fill & (count3 < 3) & (count3 >= 2) & (count5 >= 8)
        depth[fill3] = mean3[fill3]
        depth[fill5] = mean5[fill5]

    # 3) 小中值滤波（去椒盐），然后双边保边
    mask_bin = (depth > 0).astype(np.uint8)
    depth_med = cv2.medianBlur(depth.astype(np.float32), 3)
    # 只在原始有效处填中值滤波结果，无效点保持 0（避免边界被膨胀）
    depth_med[mask_bin == 0] = 0.0

    # 4) 双边滤波（保边缘）—— 把深度值缩到 0~255 再做，输出再还原
    depth_valid = depth_med.copy()
    valid_msk = depth_valid > 0
    if valid_msk.any():
        dmin, dmax = float(depth_valid[valid_msk].min()), float(depth_valid[valid_msk].max())
        drange = max(dmax - dmin, 1.0)
        depth_8u = np.zeros_like(depth_valid, dtype=np.uint8)
        depth_8u[valid_msk] = np.clip(
            (depth_valid[valid_msk] - dmin) / drange * 255.0, 0, 255).astype(np.uint8)
        depth_8u_bi = cv2.bilateralFilter(depth_8u, 5, 50, 5)
        depth_bi = np.zeros_like(depth_valid, dtype=np.float32)
        depth_bi[valid_msk] = (depth_8u_bi[valid_msk].astype(np.float32)
                               ) / 255.0 * drange + dmin
    else:
        depth_bi = depth_valid
    # 反光清零 / 范围外区域：若双边没能恢复（仍 0）→ 向量化 3×3 填补
    # —— 阈值从 cnt>=3 → cnt>=2：进一步放宽，保证 2D 深度图大平面上没有
    #    明显的孔洞，让 TSDF 体素化后 iso-surface 能整块连续。
    residual_zero = (depth_bi == 0) & (~border)
    if residual_zero.any():
        vm2 = (depth_bi > 0).astype(np.float32)
        dv2 = np.where(vm2 > 0, depth_bi, 0.0).astype(np.float32)
        cnt = cv2.filter2D(vm2, -1, k3, borderType=cv2.BORDER_CONSTANT)
        sm = cv2.filter2D(dv2, -1, k3, borderType=cv2.BORDER_CONSTANT)
        mn = np.divide(sm, np.maximum(cnt, 1.0), where=cnt > 0)
        pick = residual_zero & (cnt >= 2)
        depth_bi[pick] = mn[pick]

    # 最终范围校验：只保留 300~470mm 的有效深度（用"最终深度值"重新判定，
    # 不基于原始 mask_valid——保守补洞新增的值是合法的，不能因为原始深度=0 就又清零）
    final_valid = (depth_bi > 300.0) & (depth_bi < 470.0)
    depth_bi[~final_valid] = 0.0
    return depth_bi.astype(np.float32)


def mask_background_by_axis(depth_mm, fx, fy, cx, cy, axis_origin, axis_dir,
                            r_max_mm=140.0, z_min_mm=None, z_max_mm=None):
    """按转轴径向距离剔除背景（直接修改 depth_mm，返回同引用）。

    每个有效深度像素 → 相机坐标 P_cam → 计算相对于转轴的：
        r = |(P_cam - O) × dir|    （径向距离，转台台面半径方向）
        h = (P_cam - O) · dir      （沿转轴的高度）
    电路板肯定在 r < r_max_mm 范围（转台中心 ±140mm 足够）。
    """
    if depth_mm.ndim != 2:
        return depth_mm
    h, w = depth_mm.shape
    valid = depth_mm > 0
    if not valid.any():
        return depth_mm

    ys, xs = np.where(valid)
    zs = depth_mm[ys, xs].astype(np.float64)  # mm
    # 反投影：X = (x - cx) * Z / fx, Y = (y - cy) * Z / fy, Z = z
    X = (xs.astype(np.float64) - cx) * zs / fx
    Y = (ys.astype(np.float64) - cy) * zs / fy
    Z = zs
    P = np.stack([X, Y, Z], axis=1)  # Nx3

    O = np.asarray(axis_origin, dtype=np.float64)
    d = np.asarray(axis_dir, dtype=np.float64)
    d = d / np.linalg.norm(d)

    diff = P - O  # Nx3
    # 径向：|diff × d|
    cross = np.cross(diff, d[None, :])  # Nx3
    r = np.linalg.norm(cross, axis=1)   # N, mm
    # 沿轴高度 h
    h_axis = diff @ d  # N, mm

    # 判定丢弃
    drop = r > r_max_mm
    if z_min_mm is not None:
        drop = drop | (h_axis < z_min_mm)
    if z_max_mm is not None:
        drop = drop | (h_axis > z_max_mm)

    if drop.any():
        depth_mm[ys[drop], xs[drop]] = 0.0
        kept = (~drop).sum()
        tot = len(drop)
        # print(f"  [背景剔除] 丢 {drop.sum()} 像素，保留 {kept}/{tot}")
    return depth_mm


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    dirs = ensure_dirs()
    out_dir = dirs["result"]

    # ====== 离线重建模式：TSDF_OFFLINE=1 时从磁盘加载帧数据，跳过相机采集 ======
    _OFFLINE = (os.environ.get("TSDF_OFFLINE", "0") == "1")
    frames_dir_offline = os.path.join(OUTPUT_DIR, "frames")

    # =========================
    # 1. 加载转轴标定
    # =========================
    if not os.path.exists(AXIS_PARAM_FILE):
        raise FileNotFoundError(f"找不到标定文件 {AXIS_PARAM_FILE}")
    with open(AXIS_PARAM_FILE, "r", encoding="utf-8") as f:
        axis = json.load(f)
    axis_origin = np.array(axis["axis_origin"], dtype=np.float64)
    axis_dir = np.array(axis["axis_dir"], dtype=np.float64)
    omega = float(axis["angular_velocity_rad_s"])
    cycle_sec = 2.0 * np.pi / omega
    print(f"[转轴标定] axis_origin = {np.round(axis_origin, 3)} mm")
    print(f"[转轴标定] axis_dir    = {np.round(axis_dir, 5)}")
    print(f"[转轴标定] omega       = {omega:.6f} rad/s，一圈 {cycle_sec:.3f} s")

    # =========================
    # 2. 采集参数
    # =========================
    wait_stable = float(ACQUISITION["wait_stable_sec"])
    # 2026-09-01 加密：0.25s → 0.20s/帧（omega≈0.333 rad/s ≈ 19.1°/s → 约 3.82°/帧）
    # 覆盖 360°+20° ≈ 380° ≈ 100 帧；max_frames 设 200（应付 omega 漂移）。
    # —— 与 80 帧方案相比：视角 ×1.25 加密，中心/横向空洞可从更多斜角观测，
    #   配合 voxel=2.0 trunc=15.0 的强桥连，能把 Top1/2 两半真正合并为 1 整块。
    interval = 0.20
    overlap_deg = float(ACQUISITION["overlap_deg"])
    max_frames = 200
    # omega 修正：把标定后 omega 和实测一圈秒数对比，取标定值并允许微调
    # （如果扫完扇形条带错位严重，可手动 ±0.001 rad/s 再试）

    # ====== 离线重建：从磁盘加载帧数据，跳过相机采集 ======
    if _OFFLINE:
        npz_files = sorted(
            [f for f in os.listdir(frames_dir_offline) if f.startswith("frame_") and f.endswith(".npz")]
        ) if os.path.isdir(frames_dir_offline) else []
        if not npz_files:
            raise FileNotFoundError(f"离线模式：找不到帧数据 {frames_dir_offline}/frame_XXX.npz")
        captured = []
        for fn in npz_files:
            d = np.load(os.path.join(frames_dir_offline, fn), allow_pickle=True)
            captured.append({
                "color_bgr": d["color_bgr"],
                "depth_mm": d["depth_mm"],
                "theta_rad": float(d["theta_rad"]),
                "timestamp": float(d["timestamp"]),
            })
        # 从快照恢复标定参数
        snap = os.path.join(frames_dir_offline, "axis_snapshot.json")
        if os.path.exists(snap):
            with open(snap) as f:
                snap_data = json.load(f)
            axis_origin = np.array(snap_data["axis_origin"])
            axis_dir = np.array(snap_data["axis_dir"])
            omega = float(snap_data["omega"])
        angle_deg = captured[-1]["theta_rad"] * 180.0 / np.pi - captured[0]["theta_rad"] * 180.0 / np.pi
        # 归一化到 [0, 360)
        angle_deg = angle_deg % 360.0
        if angle_deg < 180:
            angle_deg += 360.0
        print(f"\n[离线模式] 加载 {len(captured)} 帧从 {frames_dir_offline}/")
        print(f"  覆盖 {angle_deg:.1f}°, axis_origin={np.round(axis_origin, 2)}")
        # 跳到阶段 2
        # （color_aligned_list / fg_masks 在阶段 2 开头重新计算）

    if not _OFFLINE:
        # =========================
        # 3. 打开相机
        # =========================
        camera = OrbbecCamera()
        try:
            camera.open()
        except Exception as exc:
            raise RuntimeError(f"打开相机失败: {exc}")
        try:
            fx, fy, cx, cy = camera.rgb_intrinsic
            intr = None
            depth_w = depth_h = 0

            # =========================
            # 4. 等待转台匀速 + 丢弃加速段
            # =========================
            print(f"\n请给转台上电，等待 {wait_stable}s 丢弃加速段帧...")
            deadline = time.time() + wait_stable
            while time.time() < deadline:
                try:
                    camera.grab_rgbd()
                except Exception:
                    pass
            for _ in range(5):
                try:
                    camera.grab_rgbd()
                except Exception:
                    pass

            # =========================
            # 5. 阶段 1：纯采集（不做融合）
            # =========================
            print(f"\n[阶段 1 · 采集] 每帧 {interval}s，覆盖 360°+{overlap_deg}°...")
            captured = []  # list of dict: {"color_bgr", "depth_mm", "timestamp", "theta_rad"}
            t0 = time.time()
            frame_idx = 0
            angle_deg = 0.0
            try:
                while frame_idx < max_frames:
                    color_bgr, depth_mm = None, None
                    for _attempt in range(8):
                        try:
                            color_bgr, depth_mm = camera.grab_rgbd()
                        except Exception as exc:
                            color_bgr, depth_mm = None, None
                        n_v = int((depth_mm > 0).sum()) if depth_mm is not None else 0
                        if color_bgr is not None and depth_mm is not None and n_v > 500:
                            break
                        color_bgr, depth_mm = None, None
                        time.sleep(0.01)
                    if color_bgr is None or depth_mm is None:
                        print(f"  [帧 {frame_idx:03d}] ×  8 次子重试全空 (已丢弃)")
                        time.sleep(0.03)
                        continue

                    timestamp = time.time() - t0
                    theta_rad = omega * timestamp
                    angle_deg = np.degrees(theta_rad)

                    h, w = depth_mm.shape
                    if intr is None:
                        intr = make_pinhole_intr(fx, fy, cx, cy, w, h)
                        depth_w, depth_h = w, h
                        print(f"  [初始化] 分辨率 {w}x{h}  "
                              f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
                    captured.append({
                        "color_bgr": color_bgr.copy(),
                        "depth_mm": depth_mm.copy(),
                        "timestamp": float(timestamp),
                        "theta_rad": float(theta_rad),
                    })
                    n_valid = int((depth_mm > 0).sum())
                    print(f"  [帧 {frame_idx:03d}] ✓  t={timestamp:5.2f}s  "
                          f"角度={angle_deg:6.1f}°  有效深度={n_valid:>7d}")

                    if angle_deg >= 360.0 + overlap_deg:
                        print(f"  [采集停止] 覆盖 {angle_deg:.1f}° ≥ 360+{overlap_deg}°")
                        break

                    frame_idx += 1
                    next_t = frame_idx * interval
                    delay = next_t - (time.time() - t0)
                    if delay > 0:
                        time.sleep(delay)
            finally:
                cv2.destroyAllWindows()

            print(f"\n[阶段 1 完成] 捕获 {len(captured)} 帧，"
                  f"覆盖 {angle_deg:.1f}°，用时 {captured[-1]['timestamp']:.1f}s" if captured else "捕获 0 帧！")
            if len(captured) < 12:
                raise RuntimeError(
                    f"仅捕获 {len(captured)} 帧，数据不足。"
                    "请换 USB3.0 口（蓝色口），重新插拔后再试。"
                    f"当前捕获覆盖 {angle_deg:.1f}°。")
        finally:
            camera.close()
            print("[相机] 已关闭。")

        # ====== 1-5. 保存帧数据到磁盘（以后可离线重跑 TSDF 融合）======
        frames_dir = os.path.join(OUTPUT_DIR, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        for idx, fr in enumerate(captured):
            np.savez_compressed(
                os.path.join(frames_dir, f"frame_{idx:03d}.npz"),
                color_bgr=fr["color_bgr"],
                depth_mm=fr["depth_mm"],
                theta_rad=fr["theta_rad"],
                timestamp=fr["timestamp"],
            )
        import json as _json
        with open(os.path.join(frames_dir, "axis_snapshot.json"), "w", encoding="utf-8") as _f:
            _json.dump({"axis_origin": list(axis_origin),
                         "axis_dir": list(axis_dir),
                         "omega": omega,
                         "n_frames": len(captured),
                         "interval": interval,
                         "overlap_deg": overlap_deg}, _f, indent=2)
        print(f"[帧数据] 已保存 {len(captured)} 帧到 {frames_dir}/frame_XXX.npz")

    # =========================
    # 6. 阶段 2：离线深度预处理 + 统计自适应 bounds + 自适应厚度/径向 mask
    # =========================
    print(f"\n[阶段 2 · 离线融合] 对 {len(captured)} 帧做预处理+背景剔除...")

    # ====== 2-0. 2D 前景分割（先从彩色图维度把电路板和背景切开）======
    # 目的：轴 mask 之前就把「转台台面/其他背景深度」清零，避免它们污染
    #       后续 r/h 直方图统计（h 主体跨度 85mm 的根因）、污染 TSDF
    #       融合导致扇形碎片。
    # 步骤：(1)先把每帧 color_aligned 对齐深度分辨率 → (2)把 BGR 转 RGB
    #        → (3)调用 segment_foreground.segment_frames（中值背景+可选 SAM2）
    #        → (4)得到 fg_mask[H,W] 255=前景/0=背景 → 后续 depth_pre 乘上。
    fg_masks = [None] * len(captured)
    color_aligned_list = [None] * len(captured)
    try:
        # 先把每帧 color 对齐深度尺寸
        for idx, fr in enumerate(captured):
            color_bgr = fr["color_bgr"]
            depth_raw = fr["depth_mm"]
            h, w = depth_raw.shape
            if (color_bgr.shape[1], color_bgr.shape[0]) != (w, h):
                color_aligned = cv2.resize(color_bgr, (w, h))
            else:
                color_aligned = color_bgr
            color_aligned_list[idx] = color_aligned

        if _FG_HAS_SEG:
            rgbs = [cv2.cvtColor(ca, cv2.COLOR_BGR2RGB) for ca in color_aligned_list]
            # ================================================================
            # 方案3开关: 环境变量 FG_MOTION_ONLY=1 强制只用运动先验（不调用SAM2），
            #            让"圆盘薄壳 + 电路板"整个保留为前景 → 圆盘在 TSDF 里
            #            充当大平面几何骨架，把本来不连续的电路板大平面接起来形成
            #            连续整块 mesh → 精修阶段再切圆盘薄壳。
            # ================================================================
            _force_motion = (os.environ.get("FG_MOTION_ONLY", "0") == "1")
            predictor = None
            if not _force_motion:
                # 尝试加载 SAM2 小权重（默认 checkpoints/sam2.1_hiera_small.pt）
                # 找不到 / sam2 未安装 / 安装失败 都 fallback 到「仅中值背景运动掩码」
                import torch as _torch
                _has_cuda = False
                try:
                    _has_cuda = _torch.cuda.is_available()
                except Exception:
                    _has_cuda = False
                _device = "cuda" if _has_cuda else "cpu"
                _ckpt = os.environ.get(
                    "SAM2_CKPT",
                    os.path.join(_SCRIPT_DIR, "checkpoints", "sam2.1_hiera_small.pt"),
                )
                _cfg = os.environ.get("SAM2_CONFIG", "")
                if not _cfg or not os.path.exists(_cfg):
                    # 回退到项目目录 -> site-pkg sam2 安装目录内置
                    for _c in [
                        os.path.join(_SCRIPT_DIR, "sam2_configs", "sam2.1", "sam2.1_hiera_s.yaml"),
                        os.path.join(_SCRIPT_DIR, "configs", "sam2.1", "sam2.1_hiera_s.yaml"),
                    ]:
                        if os.path.exists(_c):
                            _cfg = _c
                            break
                    if not _cfg or not os.path.exists(_cfg):
                        try:
                            import sam2 as _s2
                            _base = os.path.dirname(os.path.abspath(_s2.__file__))
                            _c_pkg = os.path.join(_base, "configs", "sam2.1", "sam2.1_hiera_s.yaml")
                            if os.path.exists(_c_pkg):
                                _cfg = _c_pkg
                        except Exception:
                            pass
                predictor = _fg_seg_mod.build_sam2_predictor(_ckpt, _cfg, device=_device)
            print(f"[前景分割] predictor={'SAM2' if predictor is not None else '仅运动先验 (方案3)'}"
                  f"{' (FG_MOTION_ONLY 强制)' if _force_motion else ''}")
            _masks, _boxes = _fg_seg_mod.segment_frames(
                rgbs,
                predictor=predictor,
                per_frame_bbox=True,
                global_box_override=None,
                min_bbox_area_frac=0.002,
            )
            for idx in range(len(captured)):
                m = _masks[idx]
                # 保证 shape=(H,W) uint8，255=前景
                if m is not None and m.size > 0:
                    if m.dtype != np.uint8:
                        m = m.astype(np.uint8)
                    if m.max() <= 1:
                        m = m * 255
                    if m.shape != color_aligned_list[idx].shape[:2]:
                        m = cv2.resize(m, (color_aligned_list[idx].shape[1],
                                           color_aligned_list[idx].shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
                    fg_masks[idx] = m
                else:
                    # 退化为全 255（等于这帧不做 2D 剔除）
                    fg_masks[idx] = np.full(
                        color_aligned_list[idx].shape[:2], 255, dtype=np.uint8
                    )
            # 保存诊断预览（首帧 + 中间帧 mask 叠加）
            _out_vis_dir = os.path.join(RESULT_DIR, "fg_masks")
            os.makedirs(_out_vis_dir, exist_ok=True)
            for _vi in (0, len(captured) // 2, len(captured) - 1):
                ca = color_aligned_list[_vi]
                mm = fg_masks[_vi]
                v = ca.copy()
                v[mm > 0] = (v[mm > 0].astype(np.int16) // 2 +
                             np.array([0, 140, 0], np.int16) // 2).astype(np.uint8)
                if _boxes is not None and _boxes[_vi] is not None:
                    x1, y1, x2, y2 = _boxes[_vi]
                    cv2.rectangle(v, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.imwrite(os.path.join(_out_vis_dir, f"fg_{_vi:03d}.png"), v)
                cv2.imwrite(os.path.join(_out_vis_dir, f"mask_{_vi:03d}.png"), mm)
            print(f"[前景分割] 诊断预览: {_out_vis_dir}")
        else:
            print("[前景分割] segment_foreground 不可用，跳过 2D 剔除")
    except Exception as _fg_err:
        print(f"[前景分割] 失败({_fg_err})，这一轮跳过 2D 剔除（仍保留轴 mask）")
        for idx in range(len(captured)):
            if color_aligned_list[idx] is None:
                # 补 color_aligned 兜底
                fr = captured[idx]
                h, w = fr["depth_mm"].shape
                if (fr["color_bgr"].shape[1], fr["color_bgr"].shape[0]) != (w, h):
                    color_aligned_list[idx] = cv2.resize(fr["color_bgr"], (w, h))
                else:
                    color_aligned_list[idx] = fr["color_bgr"]
            if fg_masks[idx] is None:
                fg_masks[idx] = np.full(
                    color_aligned_list[idx].shape[:2], 255, dtype=np.uint8
                )

    # ====== 2a. 先跑全部帧一阶段预处理（已乘 2D 前景 mask），收集 (r, h_axis) ======
    all_rh = []  # 累计 [r, h_axis]
    processed_frames_pre = []  # (color_aligned, depth_pretreated_only, theta, timestamp)
    for idx, fr in enumerate(captured):
        theta = fr["theta_rad"]
        timestamp = fr["timestamp"]
        color_aligned = color_aligned_list[idx]
        depth_raw = fr["depth_mm"]
        h, w = depth_raw.shape
        # 反光 + 范围裁剪 + 双边（不做轴 mask），看真实 r/h 分布
        depth_pre = preprocess_depth_for_tsdf(depth_raw, color_aligned)
        # ========= 应用 2D 前景 mask（核心！把非电路板像素深度清零）=========
        fg = fg_masks[idx]
        if fg is not None:
            depth_pre[fg == 0] = 0.0
        n_valid = int((depth_pre > 0).sum())
        if n_valid > 0:
            ys, xs = np.where(depth_pre > 0)
            zs = depth_pre[ys, xs].astype(np.float64)
            Xc = (xs - cx) * zs / fx
            Yc = (ys - cy) * zs / fy
            Zc = zs
            P = np.stack([Xc, Yc, Zc], axis=1)
            diff = P - axis_origin[None, :]
            cross = np.cross(diff, axis_dir[None, :])
            r = np.linalg.norm(cross, axis=1)
            h_axis = diff @ axis_dir
            # 等间隔采样最多 3000 点/帧防爆内存
            if len(r) > 3000:
                sel = np.linspace(0, len(r)-1, 3000).astype(np.int64)
                all_rh.append(np.stack([r[sel], h_axis[sel]], axis=1))
            else:
                all_rh.append(np.stack([r, h_axis], axis=1))
        processed_frames_pre.append((color_aligned, depth_pre, theta, timestamp))
        print(f"  [帧 {idx:03d}] 预处理+2Dfg 有效={n_valid:>7d}  "
              f"t={timestamp:5.2f}s θ={np.degrees(theta):6.1f}°")

    if len(all_rh) == 0:
        raise RuntimeError("预处理后无有效深度！检查相机/深度范围。")
    rh_all = np.concatenate(all_rh, axis=0)
    r_all = rh_all[:, 0]
    h_all = rh_all[:, 1]
    r_q = np.percentile(r_all, [1, 10, 25, 50, 75, 90, 99])
    h_q = np.percentile(h_all, [1, 10, 25, 50, 75, 90, 99])
    print(f"\n[自适应厚度/径向] 共 {rh_all.shape[0]} 采样点:")
    print(f"  r 分位 [1/10/25/50/75/90/99] = {np.round(r_q,1)} mm")
    print(f"  h 分位 [1/10/25/50/75/90/99] = {np.round(h_q,1)} mm")

    # ====== 2b. 从 h 分布确定厚度窗口：累计 80% 密度区间 ±2mm ======
    bins = int(np.ceil((h_q[-1] - h_q[0]) / 1.0))  # 1mm/bin
    bins = max(bins, 30)
    hh, edges = np.histogram(h_all, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    # 平滑一下（窗口=3bins≈3mm）
    kernel = np.ones(3, dtype=float) / 3.0
    hh_s = np.convolve(hh.astype(float), kernel, mode="same")

    # 累计 80% 密度区间 ±2mm
    order = np.argsort(-hh_s)
    cumsum_frac = np.cumsum(hh_s[order]) / hh_s.sum()
    keep_n = int(np.searchsorted(cumsum_frac, 0.80)) + 1
    keep_bins = set(int(x) for x in order[:keep_n])
    keep_bin_arr = sorted(keep_bins)
    h_pcb_lo_raw = float(centers[keep_bin_arr[0]])
    h_pcb_hi_raw = float(centers[keep_bin_arr[-1]])
    z_min = h_pcb_lo_raw - 2.0
    z_max = h_pcb_hi_raw + 2.0
    # ====== 2c. 径向 r 阈值：取 r 的 P95 向上取整到整 5，再加 10mm 余量 ======
    r_p95 = float(np.percentile(r_all, 95))
    r_max = float(np.ceil((r_p95 + 10.0) / 5.0) * 5.0)

    print(f"  h 主体密度区间 (80%)= [{h_pcb_lo_raw:.1f}, {h_pcb_hi_raw:.1f}] mm")
    print(f"  → 最终厚度窗口 z ∈ [{z_min:.1f}, {z_max:.1f}] mm (±2mm 余量)")
    print(f"  r P95={r_p95:.1f} → 最终径向 r_max={r_max:.1f} mm (+10mm 余量)")

    # ====== 2d. 对所有帧应用 轴 mask (径向 + 厚度)，并收集 bounds 点 ======
    all_points_world = []
    processed_frames = []
    for idx, (color_aligned, depth_pre, theta, timestamp) in enumerate(processed_frames_pre):
        # 拷贝再 mask，避免污染 debug
        depth_proc = depth_pre.copy()
        depth_proc = mask_background_by_axis(
            depth_proc, fx, fy, cx, cy,
            axis_origin, axis_dir,
            r_max_mm=r_max,
            z_min_mm=z_min,
            z_max_mm=z_max,
        )
        n_valid = int((depth_proc > 0).sum())
        print(f"  [帧 {idx:03d}] 轴 mask 后有效={n_valid:>7d}  "
              f"(t={timestamp:5.2f}s θ={np.degrees(theta):6.1f}°)")

        if n_valid > 0:
            ys, xs = np.where(depth_proc > 0)
            zs = depth_proc[ys, xs].astype(np.float64)
            Xc = (xs - cx) * zs / fx
            Yc = (ys - cy) * zs / fy
            Zc = zs
            P_cam = np.stack([Xc, Yc, Zc], axis=1)
            # 逆旋转回世界坐标
            R = make_extrinsic_world_to_cam(axis_origin, axis_dir, theta)[:3, :3]
            Rt = R.T
            P_world = (P_cam - axis_origin[None, :]) @ Rt + axis_origin[None, :]
            if P_world.shape[0] > 5000:
                sel = np.linspace(0, P_world.shape[0]-1, 5000).astype(np.int64)
                all_points_world.append(P_world[sel])
            else:
                all_points_world.append(P_world)
        processed_frames.append({
            "color_aligned": color_aligned,
            "depth_proc": depth_proc,
            "theta_rad": theta,
        })

    # 自适应 bounds（基于**已做轴 mask 后的**电路板采样点 → 小得多！）
    if len(all_points_world) == 0:
        raise RuntimeError("所有帧预处理后无有效深度！请检查深度范围/电路板是否在视野。")
    pts_all = np.concatenate(all_points_world, axis=0)
    # 对电路板点集，用 1/99 分位 + 很小 padding（5mm）即可。
    lo = np.percentile(pts_all, 1, axis=0)
    hi = np.percentile(pts_all, 99, axis=0)
    padding = np.array([5.0, 5.0, 5.0])  # mm
    volume_min = tuple((lo - padding).tolist())
    volume_max = tuple((hi + padding).tolist())
    print(f"\n[自适应 bounds（只包电路板）] 按 {pts_all.shape[0]} 采样点估计：")
    print(f"  1%分位   : {np.round(lo, 1)} mm")
    print(f"  99%分位  : {np.round(hi, 1)} mm")
    print(f"  bounds(5mm padding): {volume_min} ~ {volume_max} mm")

    # =========================
    # 7. TSDF 融合
    # =========================
    # 方案3（FG_MOTION_ONLY=1 + 大平面骨架桥连）：
    #   禁用 SAM2，让"转台圆盘薄壳 + 电路板"整体保留为前景 → 转台圆盘在 TSDF
    #   里充当一块完美的 120mm 直径大平面几何骨架，Astra 对电路板绿油大平面
    #   给出的深度=0 稀疏像素，会依附在圆盘骨架上被 Marching Cubes 插值桥接成
    #   连续整块；精修阶段再用径向双峰检测剔除外环。
    #   2026-09-01 参数升级：voxel 1.5 → 2.0mm，sdf_trunc 10.0 → 15.0mm
    #   （≈7.5×voxel）。2mm 体素让 Marching Cubes iso-surface 在电路板中心/
    #   横向 5~10mm 空洞上形成单连续面片；15mm 截断提供足够大的插值支撑半径。
    #   代价：板厚会从实测 1.6mm 膨胀到 2.5~3.5mm（TSDF 插值板厚），元件
    #   细节会轻微模糊（但仍然能分辨电容/端子的凸起轮廓）。
    voxel_length = 2.0
    sdf_trunc = 15.0
    tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    print(f"\n[TSDF] voxel={voxel_length}mm  trunc={sdf_trunc}mm")
    print("[TSDF] 开始逐帧融合...")
    integrated = 0
    for idx, fr in enumerate(processed_frames):
        depth_proc = fr["depth_proc"]
        color_aligned = fr["color_aligned"]
        theta = fr["theta_rad"]
        if (depth_proc > 0).sum() < 500:
            print(f"  [帧 {idx:03d}] 有效深度过少，跳过")
            continue
        color_rgb = cv2.cvtColor(color_aligned, cv2.COLOR_BGR2RGB)
        h, w = depth_proc.shape
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d.geometry.Image(color_rgb.astype(np.uint8)),
            depth=o3d.geometry.Image(depth_proc),
            depth_scale=1.0,
            depth_trunc=470.0,
            convert_rgb_to_intensity=False,
        )
        extrinsic = make_extrinsic_world_to_cam(axis_origin, axis_dir, theta)
        tsdf_volume.integrate(rgbd, intr, extrinsic)
        integrated += 1

    print(f"[TSDF 完成] 成功融合 {integrated}/{len(processed_frames)} 帧")
    if integrated < 8:
        raise RuntimeError(f"仅成功融合 {integrated} 帧，无法可靠重建")

    # =========================
    # 8. 抽 Mesh + 后处理
    # =========================
    print("\n[后处理] Marching Cubes 抽三角网格...")
    t_start = time.time()
    mesh = tsdf_volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print(f"  原始 mesh: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面"
          f"  ({time.time()-t_start:.1f}s)")

    # 8a. 自适应 bounds 裁剪
    crop_box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array(volume_min, dtype=np.float64),
        max_bound=np.array(volume_max, dtype=np.float64),
    )
    mesh = mesh.crop(crop_box)
    print(f"  bounds 裁剪后: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

    # 8b. 子采样降面（目标 ≤ 50 万面，后续去噪平滑快很多）
    max_tris = 500000
    if len(mesh.triangles) > max_tris:
        tgt = min(max_tris, int(len(mesh.triangles) * 0.15))
        tgt = max(tgt, 100000)
        print(f"  子采样降面: {len(mesh.triangles)} → 目标 {tgt}...")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=tgt)
        mesh.compute_vertex_normals()
        print(f"  降面后: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

    # 8c. 统计离群顶点（TriangleMesh 没有该方法，转 PointCloud 做 inlier mask 再回填）
    if len(mesh.vertices) > 50:
        verts_np = np.asarray(mesh.vertices)
        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(verts_np)
        pcd_tmp, ind_inlier = pcd_tmp.remove_statistical_outlier(
            nb_neighbors=30, std_ratio=1.5)
        keep_v = np.zeros(len(verts_np), dtype=bool)
        keep_v[np.asarray(ind_inlier, dtype=np.int64)] = True
        tris_np = np.asarray(mesh.triangles)
        keep_tri = keep_v[tris_np[:, 0]] & keep_v[tris_np[:, 1]] & keep_v[tris_np[:, 2]]
        mesh.remove_triangles_by_mask(~keep_tri)
        mesh.remove_unreferenced_vertices()
        print(f"  统计去噪后: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

    # 8d. 连通域选择：保留全部（累计 99.5%）——不丢电路板碎片
    #   2026-09-01 修正：之前用"场景A-aware 圆盘 h 重叠"选择，误把电路板
    #   元件碎片（h 与圆盘不同高）剔除，导致 Poisson 桥接后只剩圆盘骨架。
    #   现在：保留累计 99.5% 连通域，只剔除极小噪声碎片（<0.5% 面积）。
    if len(mesh.triangles) > 50:
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            tri_cls, cls_area, _ = mesh.cluster_connected_triangles()
        if len(cls_area) > 0:
            tri_cls_np = np.asarray(tri_cls, dtype=np.int64)
            cls_area_np = np.asarray(cls_area, dtype=np.float64)
            order = np.argsort(-cls_area_np)   # 按面积降序
            cum_frac = np.cumsum(cls_area_np[order]) / cls_area_np.sum()
            keep_n = int(np.searchsorted(cum_frac, 0.995)) + 1
            keep_cls_set = set(int(c) for c in order[:keep_n])
            keep_tri = np.array([int(c) in keep_cls_set for c in tri_cls_np], dtype=bool)
            mesh.remove_triangles_by_mask(~keep_tri)
            mesh.remove_unreferenced_vertices()
            # 打印 Top6 帮助诊断
            top_fracs = cls_area_np[order] / cls_area_np.sum()
            print(f"  连通域(累计99.5%, 前{keep_n}块): "
                  + ", ".join([f"{f*100:.1f}%" for f in top_fracs[:6]]))
            print(f"  → 保留后: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

    # 8d-bis. 保存原始 TSDF mesh 副本（供 Poisson 桥接或离线重跑用）
    raw_tsdf_path = os.path.join(out_dir, "pcb_tsdf_raw.ply")
    o3d.io.write_triangle_mesh(raw_tsdf_path, mesh,
                               write_vertex_normals=True, write_vertex_colors=False)
    print(f"  [保存原始] {raw_tsdf_path}")

    # 8e. 轻微平滑（只1次，保护引脚/焊盘边缘）
    mesh = mesh.filter_smooth_simple(number_of_iterations=1)
    mesh.compute_vertex_normals()

    # =========================
    # 9. 保存 + 质量评估
    # =========================
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_ply = os.path.join(RESULT_DIR, "pcb_tsdf.ply")
    o3d.io.write_triangle_mesh(out_ply, mesh, write_vertex_colors=True,
                               write_vertex_normals=True)
    print(f"\n[保存] mesh → {out_ply}")

    # 板厚估计：沿转轴方向投影分位差
    if len(mesh.vertices) > 0:
        verts = np.asarray(mesh.vertices)  # Nx3 世界坐标
        axis_dir_n = axis_dir / np.linalg.norm(axis_dir)
        h_axis = (verts - axis_origin[None, :]) @ axis_dir_n  # 沿转轴的高度
        h2, h98 = np.percentile(h_axis, [2, 98])
        est_thickness = float(abs(h98 - h2))
        print(f"[质量] 沿转轴方向 2%~98% 分位跨度 ≈ {est_thickness:.2f} mm")
        print(f"       （板厚标称 1.6mm，这里包含焊盘/元件高度）")

        # 几何包围盒（x/y/z 三方向）
        bbox = mesh.get_axis_aligned_bounding_box()
        sx, sy, sz = bbox.get_extent()
        print(f"[质量] 网格外包围盒: {sx:.1f} × {sy:.1f} × {sz:.1f} mm  (x×y×z)")

        # 点数/面数汇总
        print(f"[质量] {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

    # =========================
    # 10. 可视化
    # =========================
    print("\n[可视化] 打开 Open3D 窗口（ESC 或关闭窗口退出）。")
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30.0)
    o3d.visualization.draw_geometries(
        [mesh, coord],
        window_name=f"TSDF 重建 ({len(mesh.vertices)}v, {len(mesh.triangles)}t)",
        width=1200, height=900,
    )


if __name__ == "__main__":
    main()
