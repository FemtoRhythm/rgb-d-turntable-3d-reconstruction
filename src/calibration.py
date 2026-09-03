# -*- coding: utf-8 -*-
"""
转台旋转轴标定（棋盘格旋转拟合方案）
==============================================================

原理：
- 棋盘格随转台旋转，每个角点（内角点）的运动轨迹在空间中是一个圆；
- 所有角点轨迹圆的圆心都位于转台上，且圆的法向量都与转轴平行；
- 因此：对每一帧标定样本，先用 OpenCV 在彩色图检测棋盘格 2D 角点，再用
  solvePnP 估计棋盘格位姿 + 时间连续性消歧 90° 对称分支，恢复跨帧一致的
  canonical 顺序 3D 角点；对同一角点跨多帧的 3D 轨迹拟合空间圆；由多个圆的
  圆心与法向量共同解出转轴：
      axis_origin: 转轴上一个 3D 点 (x, y, z)
      axis_dir   : 转轴单位方向向量

旋转角度由「一圈周期参数」推算（转台通电常转、无法启停）：
- 采集时记录每帧相对采集起始时刻的时间戳；
- 角度 = 360° × 时间戳 / turntable_full_cycle_sec（使用前实测一圈时间）；
- 转轴拟合本身不依赖角度，仅标定自检（粗旋转拼接）需要角度。

约束：全程仅使用棋盘格角点轨迹，不使用标准球。
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import cv2
from scipy.optimize import least_squares

# 脚本在 src/ 下，把仓库根目录加进 sys.path 以便 import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (CALIB_DATA_DIR, AXIS_PARAM_FILE, CHESSBOARD, CALIBRATION,
                    ACQUISITION, angle_deg_from_timestamp, ensure_dirs)
from camera_utils import assert_unit_mm


# ---------------------------------------------------------------------------
# 空间圆拟合
# ---------------------------------------------------------------------------
def _fit_plane(points):
    """用 SVD 拟合 3D 点所在的平面，返回 (法向量, 平面上一重心点)。"""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]           # 最小奇异值对应的方向即平面法向量
    return normal, centroid


def _circle_basis(normal):
    """由法向量构造平面局部正交基 (e1, e2)。"""
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(normal, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    return e1, e2


def _fit_spatial_circle(points):
    """拟合空间圆（几何最小二乘，精度高于 Kasa 代数法）。

    步骤：SVD 拟合平面 -> 投影 2D -> Kasa 代数法求初值 -> LM 迭代最小化
    点到圆的几何距离（几何残差），得到更精确的圆心与半径。

    参数:
        points: (N, 3) 同一角点跨多帧的 3D 坐标
    返回:
        (center, normal, radius)，center 为圆心 (3,)，normal 为单位法向量 (3,)
    """
    points = np.asarray(points, dtype=np.float64)
    normal, centroid = _fit_plane(points)
    e1, e2 = _circle_basis(normal)

    # 投影到平面 2D 坐标
    centered = points - centroid
    u = centered @ e1
    v = centered @ e2

    # Kasa 代数法求初值（x²+y² = 2a·x + 2b·y - c，圆心(a,b)，r²=a²+b²-c）
    A = np.column_stack([2 * u, 2 * v, -np.ones_like(u)])
    b = u * u + v * v
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a0, b0, c0 = sol
    r0 = float(np.sqrt(max(a0 * a0 + b0 * b0 - c0, 1e-6)))

    # LM 几何拟合：最小化 sum( (sqrt((u-a)^2+(v-b)^2) - R)^2 )，得到无偏圆心/半径
    def _resid(p):
        a, b_, R = p
        return np.sqrt((u - a) ** 2 + (v - b_) ** 2) - R

    res = least_squares(_resid, x0=np.array([a0, b0, r0]), method="lm")
    a, b_, R = res.x

    center2d = np.array([a, b_])
    radius = float(abs(R))

    # 圆心还原到 3D
    center = centroid + center2d[0] * e1 + center2d[1] * e2

    # 法向量符号修正：空间圆拟合的法向量存在 ± 歧义（SVD 最小奇异向量方向不确定）。
    # 角点按帧序绕转轴正转（角度递增，右手定则），故 cross(径向向量, 相邻帧径向向量)
    # 应指向转轴正方向。累加相邻帧叉积增强信号，据此统一 normal 指向，避免 axis_dir
    # 符号随机翻转导致自检（粗旋转拼接）大幅错位。
    cross_sum = np.zeros(3)
    for i in range(len(points) - 1):
        cross_sum += np.cross(points[i] - center, points[i + 1] - center)
    if np.dot(normal, cross_sum) < 0:
        normal = -normal

    return center, normal, radius


def _circle_residuals(points, center, normal, radius):
    """计算各点相对拟合空间圆的残差（mm）。

    残差 = 点到圆平面的偏差 与 径向偏差 的组合：
        r_i = sqrt(d_plane_i^2 + d_radial_i^2)
    其中 d_plane = |(p - center)·normal|，d_radial = |||p - center|| - radius|。
    """
    pts = np.asarray(points, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    d = pts - center
    d_plane = np.abs(d @ n)
    d_radial = np.abs(np.linalg.norm(d, axis=1) - radius)
    return np.sqrt(d_plane ** 2 + d_radial ** 2)


def _rotate_around_axis(points, axis_origin, axis_dir, theta):
    """绕空间任意轴旋转点集（罗德里格斯公式），单位 mm，用于标定自检。"""
    axis_dir = np.asarray(axis_dir, dtype=np.float64)
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    axis_origin = np.asarray(axis_origin, dtype=np.float64)
    K = np.array([[0, -axis_dir[2], axis_dir[1]],
                  [axis_dir[2], 0, -axis_dir[0]],
                  [-axis_dir[1], axis_dir[0], 0]], dtype=np.float64)
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    pts = np.asarray(points, dtype=np.float64)
    return axis_origin + (pts - axis_origin) @ R.T


# ---------------------------------------------------------------------------
# 棋盘格角点提取
# ---------------------------------------------------------------------------
def _detect_corners(color_bgr):
    """在彩色图检测棋盘格 2D 角点（亚像素）。

    返回:
        (N, 2) 角点像素坐标，检测失败返回 None
    """
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    pattern = CHESSBOARD["pattern_size"]
    ret, corners = cv2.findChessboardCorners(gray, pattern, None)
    if not ret or corners is None:
        return None

    win = CALIBRATION["corner_refine_win"]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, win, (-1, -1), criteria)
    return corners.reshape(-1, 2)


def _chessboard_object_points():
    """生成棋盘格内角点的物体坐标（棋盘格平面内，z=0，单位 mm）。

    顺序与 OpenCV findChessboardCorners 标准行列顺序一致（行优先）。
    """
    cols, rows = CHESSBOARD["pattern_size"]
    s = CHESSBOARD["square_size_mm"]
    objp = np.zeros((rows * cols, 3), dtype=np.float64)
    k = 0
    for r in range(rows):
        for c in range(cols):
            objp[k] = [c * s, r * s, 0.0]
            k += 1
    return objp


def _board_center():
    """棋盘格内角点网格中心（物体坐标，mm），用于 90° 对称消歧时平移校正。"""
    cols, rows = CHESSBOARD["pattern_size"]
    s = CHESSBOARD["square_size_mm"]
    return np.array([(cols - 1) * s / 2.0, (rows - 1) * s / 2.0, 0.0], dtype=np.float64)


def _rot_z(theta):
    """绕物体坐标系 z 轴（棋盘格法向）的旋转矩阵，theta 单位弧度。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _rot_angle(Ra, Rb):
    """两个旋转矩阵之间的夹角（度），用于 90° 对称分支消歧。"""
    M = Ra.T @ Rb
    v = np.clip((np.trace(M) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(v))


def _solve_board_pose(corners2d, rgb_intrinsic, rgb_distortion,
                      prev_R_raw, prev_t_raw, prev_R_canon):
    """由棋盘格 2D 角点求解位姿，并恢复「跨帧一致」的 canonical 顺序 3D 角点。

    背景（核心）：8x8 正方形棋盘格具有 90° 旋转对称性，OpenCV findChessboardCorners
    的角点顺序在棋盘格旋转跨越 90° 对称位置时会跳变，导致不同帧同号角点不是同一
    物理点，直接按索引串轨迹会使空间圆拟合残差爆炸（螺旋/环形拖尾的根因）。

    做法：
    1. 用 solvePnP 估计棋盘格位姿；首帧用普通求解，后续帧用 useExtrinsicGuess +
       「上一帧原始位姿」作初值。注意初值必须与检测顺序自洽（即原始位姿），否则
       跨 90° 跳变帧会因初值偏差 90° 而发散。
    2. 枚举 4 个 90° 分支（绕棋盘格法向旋转 0/90/180/270°），选「与上一帧
       canonical 位姿旋转夹角最小」的分支，用时间连续性打破 90° 对称，得到
       跨帧一致的角点顺序。
    3. 由位姿 + 已知棋盘格几何直接计算 canonical 顺序 3D 角点（mm），比深度
       反投影更稳定（不受深度噪声影响）。

    返回:
        (pts, R_raw, t_raw, R_canon)：pts 为 canonical 顺序 3D 角点 (N,3)（mm）；
        R_raw/t_raw 为原始（检测顺序）位姿，供下一帧 useExtrinsicGuess 初值；
        R_canon 为消歧后的 canonical 位姿。solvePnP 失败返回 None。
    """
    objp = _chessboard_object_points()
    c_center = _board_center()
    fx, fy, cx, cy = rgb_intrinsic
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.asarray(rgb_distortion, dtype=np.float64)

    corners = np.asarray(corners2d, dtype=np.float64).reshape(-1, 1, 2)
    if len(corners) != len(objp):
        return None

    if prev_R_raw is None:
        ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        rvec0 = cv2.Rodrigues(prev_R_raw)[0]
        tvec0 = np.asarray(prev_t_raw, dtype=np.float64).reshape(3, 1)
        ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, rvec0, tvec0, True,
                                      cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    R_raw, _ = cv2.Rodrigues(rvec)
    t_raw = np.asarray(tvec, dtype=np.float64).reshape(3)

    # 90° 消歧：枚举 4 分支，选与上一帧 canonical 位姿旋转夹角最小的
    if prev_R_canon is None:
        best_q = 0
    else:
        best_q, best_angle = 0, np.inf
        for q in range(4):
            Rq = R_raw @ _rot_z(np.deg2rad(q * 90)).T
            ang = _rot_angle(prev_R_canon, Rq)
            if ang < best_angle:
                best_angle, best_q = ang, q

    theta = best_q * 90.0
    R_canon = R_raw @ _rot_z(np.deg2rad(theta)).T
    t_canon = t_raw - R_raw @ (_rot_z(np.deg2rad(theta)).T @ c_center - c_center)
    objp_c = (_rot_z(np.deg2rad(theta)).T @ (objp - c_center).T).T + c_center
    pts = (R_raw @ objp_c.T).T + t_raw   # 等价于 R_canon @ objp + t_canon
    return pts, R_raw, t_raw, R_canon


# ---------------------------------------------------------------------------
# 转轴求解
# ---------------------------------------------------------------------------
def solve_axis(corner_tracks):
    """由角点轨迹集合求解转轴参数，并返回空间圆拟合残差。

    参数:
        corner_tracks: list[(N_i, 3)]，每个角点一条轨迹（跨帧 3D 坐标）
    返回:
        (axis_origin, axis_dir, residuals)
        axis_dir 为单位向量；residuals 为所有有效圆上各点拟合残差（mm）的一维数组。
    """
    centers = []
    normals = []
    residuals = []

    for pts in corner_tracks:
        pts = pts[~np.isnan(pts).any(axis=1)]   # 剔除缺失/无效帧（NaN）
        if len(pts) < CALIBRATION["min_frames_for_circle"]:
            continue
        try:
            center, normal, radius = _fit_spatial_circle(pts)
        except np.linalg.LinAlgError:
            continue
        centers.append(center)
        normals.append(normal)
        residuals.append(_circle_residuals(pts, center, normal, radius))

    if len(centers) < 3:
        raise RuntimeError("有效角点轨迹圆数量不足，无法可靠求解转轴。请检查采集样本。")

    centers = np.array(centers)
    normals = np.array(normals)
    residuals = np.concatenate(residuals)

    # 法向量方向统一（与第一个同向）
    n0 = normals[0]
    for i in range(len(normals)):
        if np.dot(normals[i], n0) < 0:
            normals[i] = -normals[i]

    # 轴方向：每个角点轨迹圆的法向量都与转轴平行，取平均（稳健）。
    # 注意：不能用「圆心点 PCA」求轴方向——当棋盘格接近水平放置时，所有圆心
    # 在转轴方向上几乎重合（spread≈0），PCA 主方向会被圆心定位噪声主导，错误
    # 指向垂直转轴的方向（本场景实测夹角 88°）。故轴方向以法向量平均为准。
    axis_dir = normals.mean(axis=0)
    axis_dir /= np.linalg.norm(axis_dir)

    # 轴上一点：圆心均值（圆心都位于转轴上）
    axis_origin = centers.mean(axis=0)

    # 交叉验证：圆心到转轴直线的垂直距离应接近 0（验证轴方向正确性）
    d_axis = np.linalg.norm(np.cross(centers - axis_origin, axis_dir), axis=1)
    print(f"[轴方向校验] 圆心到转轴直线垂直距离：均值={d_axis.mean():.3f} mm，"
          f"最大={d_axis.max():.3f} mm（应接近 0）")

    return axis_origin, axis_dir, residuals


def _estimate_omega(corner_tracks, timestamps, axis_origin, axis_dir):
    """从角点旋转轨迹拟合转台角速度 omega（rad/s）。

    比秒表实测一圈时间更精确：每个角点随转台绕轴旋转，其在垂直轴的平面内的
    极角随时间线性变化，斜率为角速度。对全部角点轨迹线性拟合，取中位数抗离群。
    廉价展示转台转速不稳、秒表实测误差大，用此反推更可靠。

    参数:
        corner_tracks: list[(N,3)] 每个角点一条跨帧轨迹（同一物理角点 N 帧 3D 坐标，mm）
        timestamps:    list[float] 每帧相对首帧的时间戳（秒，与轨迹帧序一致）
        axis_origin:   (3,) 转轴原点（mm）
        axis_dir:      (3,) 转轴单位方向向量
    返回:
        float: 角速度 rad/s（axis_dir 已按旋转方向定号，故 omega 恒正）
    """
    axis_dir = np.asarray(axis_dir, dtype=np.float64)
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    axis_origin = np.asarray(axis_origin, dtype=np.float64)

    # 垂直轴的平面正交基 (e1, e2)，满足 e1 × e2 = axis_dir（右手系）
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis_dir, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis_dir, e1)

    t = np.asarray(timestamps, dtype=np.float64)
    omegas = []
    for pts in corner_tracks:
        pts = np.asarray(pts, dtype=np.float64)
        if len(pts) < 3:
            continue
        rel = pts - axis_origin
        u = rel @ e1
        v = rel @ e2
        phi = np.unwrap(np.arctan2(v, u))
        A = np.column_stack([t, np.ones_like(t)])
        try:
            sol, *_ = np.linalg.lstsq(A, phi, rcond=None)
        except np.linalg.LinAlgError:
            continue
        omegas.append(float(sol[0]))

    if not omegas:
        raise RuntimeError("角速度拟合失败：无有效角点轨迹。")

    omega = float(np.median(np.array(omegas)))
    if omega <= 0:
        omega = abs(omega)
    return omega


# ---------------------------------------------------------------------------
# 转轴倾斜校验
# ---------------------------------------------------------------------------
def _check_axis_tilt(axis_dir):
    """校验拟合转轴 axis_dir 与「理想竖直方向」的夹角（可配置，默认仅告警）。

    背景：只有当相机「俯视转台」（转轴在相机坐标系中接近竖直）时，把 ideal_axis_dir
    设为 [0,0,1] 才有物理意义。若相机斜视转台，转轴在相机坐标系中天然倾斜
    （本工程实测与 [0,0,1] 夹角约 54°），该夹角属于正常现象，不应判为标定失败。

    返回:
        (tilt_deg, fail)
        tilt_deg: axis_dir 与理想方向夹角（度）
        fail:     是否应判定标定失败（= 夹角超阈值 且 axis_tilt_fail_enabled=True）
    """
    ideal = np.array(CALIBRATION["ideal_axis_dir"], dtype=np.float64)
    ideal = ideal / np.linalg.norm(ideal)
    d = np.asarray(axis_dir, dtype=np.float64)
    d = d / np.linalg.norm(d)

    cos_ang = float(np.clip(np.dot(d, ideal), -1.0, 1.0))
    tilt_deg = float(np.degrees(np.arccos(cos_ang)))
    thresh = float(CALIBRATION["max_axis_tilt_deg"])
    fail = (tilt_deg > thresh) and bool(CALIBRATION["axis_tilt_fail_enabled"])
    return tilt_deg, fail


# ---------------------------------------------------------------------------
# 标定自检验证
# ---------------------------------------------------------------------------
def selfcheck_axis(all_corners_3d, angles_deg, axis_origin, axis_dir):
    """标定自检：用求解的转轴对相邻帧角点做粗旋转拼接，检查重合误差。

    思路：转台匀速转动时，相邻帧相对旋转角 delta = angles[k] - angles[k-1]，
    把第 k 帧角点绕转轴反向旋转 -delta 应回到第 k-1 帧姿态。逐对相邻帧比较
    对应角点距离，取中位数（稳健，抗加速段个别帧）作为拼接质量指标。若误差
    过大（出现环形拖尾），说明转轴/角度不可靠。

    返回:
        (ok, err) — ok 是否通过；err 为中位重合误差（mm）
    """
    cfg = CALIBRATION
    if len(all_corners_3d) < 2:
        return True, 0.0

    errs = []
    for k in range(1, len(all_corners_3d)):
        pts_prev = np.asarray(all_corners_3d[k - 1])
        pts = np.asarray(all_corners_3d[k])
        delta = np.deg2rad(angles_deg[k] - angles_deg[k - 1])
        # 反向旋转回上一帧姿态（-delta）
        rotated = _rotate_around_axis(pts, axis_origin, axis_dir, -delta)
        m = min(len(pts_prev), len(rotated))
        valid = np.isfinite(pts_prev[:m]).all(axis=1) & np.isfinite(rotated[:m]).all(axis=1)
        if valid.sum() == 0:
            continue
        diff = np.linalg.norm(pts_prev[:m][valid] - rotated[:m][valid], axis=1)
        errs.extend(diff.tolist())

    if not errs:
        return True, 0.0

    errs = np.array(errs)
    max_err = float(np.max(errs))
    mean_err = float(np.mean(errs))
    median_err = float(np.median(errs))
    print(f"[自检] 相邻帧粗旋转拼接重合误差：均值={mean_err:.3f} mm，"
          f"中位={median_err:.3f} mm，最大={max_err:.3f} mm")
    if median_err > cfg["selfcheck_max_error_mm"]:
        print(f"[自检失败] 中位重合误差 {median_err:.3f} mm 超过阈值 "
              f"{cfg['selfcheck_max_error_mm']} mm，转轴标定不可靠（可能出现环形拖尾），"
              f"请勿继续工件重建。")
        return False, median_err
    print(f"[自检通过] 中位重合误差 {median_err:.3f} mm <= 阈值 "
          f"{cfg['selfcheck_max_error_mm']} mm。")
    return True, median_err


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(data_dir=None):
    """执行转轴标定，输出 axis_params.json。"""
    data_dir = data_dir or CALIB_DATA_DIR
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"标定样本目录不存在: {data_dir}")

    # 读取帧清单与相机内参
    angle_list_path = os.path.join(data_dir, "angles.json")
    cam_param_path = os.path.join(data_dir, "camera_params.json")
    if not os.path.exists(angle_list_path):
        raise FileNotFoundError("缺少 angles.json，请先运行采集模块的标定采集。")
    if not os.path.exists(cam_param_path):
        raise FileNotFoundError("缺少 camera_params.json，请先运行采集模块的标定采集。")

    with open(angle_list_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(cam_param_path, "r", encoding="utf-8") as f:
        cam_params = json.load(f)

    rgb_intrinsic = cam_params["rgb_intrinsic"]
    rgb_distortion = cam_params["rgb_distortion"]

    # 转台一圈周期（角度推算用，使用前实测填入）
    cycle_sec = float(ACQUISITION["turntable_full_cycle_sec"])
    print(f"[转台配置] 一圈周期 = {cycle_sec} 秒（角度 = 360° × 时间戳 / 该周期）")

    # 逐帧检测角点并求位姿 -> 恢复跨帧一致的 canonical 顺序 3D 角点
    all_corners_3d = []      # 每帧的角点 3D (M, 3)，canonical 顺序（跨帧同号角点对应同一物理点）
    angles_deg = []          # 有效帧的旋转角度（度，采集时按一圈周期推算）
    timestamps = []          # 有效帧相对首帧的时间戳（秒），用于角速度精校正
    t0 = None
    prev_R_raw = prev_t_raw = prev_R_canon = None
    for rec in records:
        color_path = os.path.join(data_dir, rec["color"])
        color_bgr = cv2.imread(color_path)
        if color_bgr is None:
            print(f"[采集帧 {rec['index']}] 无法读取彩色图，丢弃该帧。")
            continue

        corners2d = _detect_corners(color_bgr)
        if corners2d is None:
            print(f"[采集帧 {rec['index']}] 棋盘格 2D 角点检测失败，丢弃该帧，不参与拟合。")
            continue

        # 【跨帧角点对齐】solvePnP 求位姿 + 时间连续性消歧 90° 对称分支，
        # 得到 canonical 顺序 3D 角点，保证每帧同号角点是同一物理角点。
        result = _solve_board_pose(corners2d, rgb_intrinsic, rgb_distortion,
                                   prev_R_raw, prev_t_raw, prev_R_canon)
        if result is None:
            print(f"[采集帧 {rec['index']}] 棋盘格位姿求解失败（solvePnP 异常），丢弃该帧。")
            continue

        corners3d, prev_R_raw, prev_t_raw, prev_R_canon = result

        # 单位校验：断言角点坐标在 mm 量级（防止毫米/米混用）
        assert_unit_mm(corners3d, label=f"帧 {rec['index']} 角点")

        print(f"[采集帧 {rec['index']}] 角点 {len(corners3d)} 个（canonical 顺序），"
              f"位姿求解成功。")

        all_corners_3d.append(corners3d)
        ts = float(rec.get("timestamp", 0.0))
        if t0 is None:
            t0 = ts
        timestamps.append(ts - t0)
        # 角度：优先用采集时记录的 angle_deg；缺失则用时间戳 + 一圈周期推算
        if "angle_deg" in rec:
            angles_deg.append(float(rec["angle_deg"]))
        else:
            angles_deg.append(angle_deg_from_timestamp(ts, 0.0, cycle_sec))

    # 有效帧数检查（不足则直接退出）
    if len(all_corners_3d) < CALIBRATION["min_valid_frames"]:
        raise RuntimeError(
            f"标定有效帧数不足：{len(all_corners_3d)} < {CALIBRATION['min_valid_frames']}，"
            f"标定样本过少，退出。请重新采集更多角度的样本。")

    # 重组为「每个角点一条轨迹」（含 NaN 占位，保持角点索引对齐）
    n_frames = len(all_corners_3d)
    m = min(len(c) for c in all_corners_3d)
    corner_tracks = []
    for j in range(m):
        track = np.array([all_corners_3d[i][j] for i in range(n_frames)], dtype=np.float64)
        corner_tracks.append(track)

    # 求解转轴（含空间圆拟合残差）
    axis_origin, axis_dir, residuals = solve_axis(corner_tracks)
    assert abs(np.linalg.norm(axis_dir) - 1.0) < 1e-6, "axis_dir 未归一化为单位向量"

    # 【残差校验】输出全部/均值/最大残差
    print("\n===== 空间圆拟合残差统计 =====")
    print(f"残差样本数: {len(residuals)}")
    print(f"残差均值: {float(np.mean(residuals)):.4f} mm")
    print(f"残差最大: {float(np.max(residuals)):.4f} mm")
    print(f"残差列表(mm): {np.round(residuals, 4).tolist()}")
    max_residual = float(np.max(residuals))
    if max_residual > CALIBRATION["max_residual_mm"]:
        print(f"\n[标定失败] 最大残差 {max_residual:.4f} mm 超过阈值 "
              f"{CALIBRATION['max_residual_mm']} mm，转轴标定结果不可靠，不保存 json，终止。")
        raise RuntimeError(
            "转轴标定残差过大，标定失败。请检查棋盘格是否太靠近转轴中心、"
            "深度是否稳定、是否有坏角点参与拟合。")

    # 【转轴倾斜校验】axis_dir 与理想竖直方向夹角（可配置，默认仅告警）
    tilt_deg, tilt_fail = _check_axis_tilt(axis_dir)
    ideal = CALIBRATION["ideal_axis_dir"]
    thresh = CALIBRATION["max_axis_tilt_deg"]
    print(f"\n[转轴倾斜校验] axis_dir 与理想方向 {ideal} 夹角 = {tilt_deg:.3f}° "
          f"(阈值 {thresh}°)")
    if tilt_fail:
        print(f"[标定失败] 转轴倾斜 {tilt_deg:.3f}° 超过阈值 {thresh}°，"
              f"判定转轴倾斜严重，终止。")
        raise RuntimeError("转轴倾斜超阈值，标定失败。")
    if tilt_deg > thresh:
        print(f"[告警] 转轴倾斜 {tilt_deg:.3f}° 超过阈值 {thresh}°，但 "
              f"axis_tilt_fail_enabled=False，仅告警不失败（相机斜视场景属正常现象）。")
    else:
        print(f"[转轴倾斜校验通过] 夹角 {tilt_deg:.3f}° <= 阈值 {thresh}°。")

    # 角速度：先由一圈周期粗算，再用角点旋转轨迹精校正（秒表实测误差大，
    # 用角点轨迹反推更精确），并用精确 omega 重算每帧角度供自检与输出。
    omega_rough = 2.0 * np.pi / cycle_sec
    omega = _estimate_omega(corner_tracks, timestamps, axis_origin, axis_dir)
    print(f"\n[角速度] 一圈周期粗算 omega = {omega_rough:.6f} rad/s "
          f"(等效一圈 {cycle_sec} 秒)")
    print(f"[角速度] 角点轨迹精校正 omega = {omega:.6f} rad/s = {np.degrees(omega):.3f}°/s "
          f"(等效一圈 {2 * np.pi / omega:.3f} 秒)")
    angles_deg = [float(np.degrees(omega * ts)) for ts in timestamps]

    # 标定自检验证（粗旋转拼接重合误差）
    ok, selfcheck_err = selfcheck_axis(all_corners_3d, angles_deg, axis_origin, axis_dir)
    if not ok:
        print("\n[标定失败] 自检未通过：粗旋转拼接出现明显错位（环形拖尾），"
              "转轴/角度不可靠。请勿继续工件重建。")
        raise RuntimeError("转轴标定自检未通过。")

    # 保存结果（单位注释写明 mm）
    ensure_dirs()
    result = {
        "axis_origin": axis_origin.tolist(),
        "axis_dir": axis_dir.tolist(),
        "angular_velocity_rad_s": omega,
        "unit": "mm",
        "note": ("所有空间坐标单位均为毫米(mm)；axis_dir 为单位向量(|dir|≈1)。"
                 "相机一旦挪动必须重新标定；仅在相机-转台相对位置不变时有效。"),
        "max_residual_mm": max_residual,
        "selfcheck_max_error_mm": selfcheck_err,
        "axis_tilt_deg": round(tilt_deg, 4),
        "num_corners": m,
        "num_frames": n_frames,
        "calib_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(AXIS_PARAM_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n===== 转轴标定结果 =====")
    print(f"axis_origin          : {axis_origin}  (mm)")
    print(f"axis_dir             : {axis_dir}  (单位向量, |dir|={np.linalg.norm(axis_dir):.6f})")
    print(f"angular_velocity     : {omega:.6f} rad/s  (= {np.degrees(omega):.3f} deg/s)")
    print(f"最大空间圆残差       : {max_residual:.4f} mm")
    print(f"自检最大重合误差     : {selfcheck_err:.4f} mm")
    print(f"已保存到 {AXIS_PARAM_FILE}")
    return result


if __name__ == "__main__":
    run()
