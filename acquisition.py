# -*- coding: utf-8 -*-
"""
采集模块
========

转台「匀速转动」、速度未知场景下的自动采集：
- 程序按固定时间间隔自动采集若干帧，并记录每帧的时间戳（相对首帧，秒）。
- 每帧对应的旋转角度无需人工输入，由 calibration.py 根据角点轨迹自动求解角速度后推算。

两套流程：
1. 转轴标定采集：棋盘格固定在转台上，保存彩色点云 ply + 彩色图 png +
   对齐深度图 npy（供 calibration.py 检测角点、拟合转轴、求解角速度）。
2. 工件扫描采集：工件放在转台上，保存彩色点云 ply（供 scan_and_tsdf.py 重建）。
"""

import json
import os
import time

import numpy as np
import cv2
import open3d as o3d

from config import ACQUISITION, CHESSBOARD, ensure_dirs, angle_deg_from_timestamp
from camera_utils import OrbbecCamera, save_camera_params


def _save_pointcloud(pcd, path):
    o3d.io.write_point_cloud(path, pcd)
    print(f"  保存点云: {path} ({len(pcd.points)} 点)")


def _show_preview(color_bgr, depth_mm):
    """显示彩色图 + 深度伪彩预览（非阻塞）。"""
    if color_bgr is None:
        return
    depth_vis = np.clip(depth_mm, 150, 2000)
    depth_vis = ((depth_vis - 150) / (2000 - 150) * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imshow("Color", color_bgr)
    cv2.imshow("Depth", depth_colored)
    cv2.waitKey(1)


def _quick_check_chessboard(color_bgr):
    """标定采集时快速检测棋盘格 2D 角点是否可见（不做亚像素精确化）。

    返回:
        bool: 是否检测到棋盘格（供采集阶段提示，失败帧不保存）。
    """
    if color_bgr is None:
        return False
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    ret, _ = cv2.findChessboardCorners(gray, CHESSBOARD["pattern_size"], None)
    return bool(ret)


def _is_black_frame(color_bgr):
    """检测彩色帧是否为全黑/过暗帧（无法用于重建，应跳过不保存）。

    返回:
        bool: True 表示全黑/过暗帧（平均亮度低于阈值）。
    """
    if color_bgr is None:
        return True
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < 5.0   # 平均亮度 < 5 判定为黑帧


def _wait_turntable_stable(camera, interval, min_stable=4, max_wait=90.0):
    """动态等待转台匀速（标定采集专用，棋盘格可见）。

    匀速旋转时，棋盘格角点质心相邻帧（间隔 interval 秒）位移距离恒定；加速段位移
    单调递增。连续 min_stable 帧位移量稳定即认为匀速并返回。相比固定等待，此方法
    不受「程序打印提示后才手动上电」造成的上电延迟影响，能确保第 0 帧落在匀速段，
    避免首帧落在加速段导致角增量偏小、自检出现离群帧。
    """
    print("\n请给转台上电。程序将自动检测转速，转台匀速后自动开始采集...")
    prev_centroid = None
    prev_dist = None
    stable = 0
    start = time.time()
    last_check = 0.0
    while time.time() - start < max_wait:
        now = time.time()
        if now - last_check < interval:
            time.sleep(0.05)
            continue
        last_check = now
        try:
            color, _ = camera.grab_rgbd()
        except Exception:
            color = None
        if color is None:
            continue
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD["pattern_size"], None)
        if not ret or corners is None:
            prev_centroid = prev_dist = None
            stable = 0
            print("  等待棋盘格可见（请确认棋盘格固定在转台上且转台已上电）...")
            continue
        centroid = corners.reshape(-1, 2).mean(axis=0)
        if prev_centroid is not None:
            dist = float(np.linalg.norm(centroid - prev_centroid))
            # 位移过小说明转台未动（未上电/静止），不参与稳定性判断
            if prev_dist is not None and prev_dist > 1.0 and dist > 1.0:
                ratio = dist / prev_dist
                if 0.6 <= ratio <= 1.5:
                    stable += 1
                    print(f"  转速检测：质心位移 {dist:.1f}px（稳定 {stable}/{min_stable}）")
                    if stable >= min_stable:
                        print("  转台已匀速，开始正式采集。")
                        return
                else:
                    stable = 0
                    print(f"  转速检测：质心位移 {dist:.1f}px（仍在加速/波动，重置）")
            prev_dist = dist
        prev_centroid = centroid
    print("  警告：等待匀速超时，按当前状态开始采集。")


def _acquire(camera, mode):
    """通用自动采集（转台「通电常转、无法启停/定点停止/步进」）。

    mode: "calib" 转轴标定采集；"scan" 工件扫描采集。
    两套模式共用同一套定时抓帧逻辑：
      1. 上电后先丢弃 wait_stable_sec 秒内的加速段帧（转速不稳）；
      2. 转速稳定后，程序内部定时器按固定间隔抓取 RGB-Depth 同步对齐帧，
         保存点云 ply，记录每帧相对采集起始时刻的时间戳；
      3. 由「时间戳 ÷ 一圈周期」推算每帧旋转角度 angle_deg；
      4. 采集满 360°（加 overlap_deg 首尾重叠）后自动结束。
    """
    if mode == "calib":
        out_dir = ensure_dirs()["calib"]
        prefix = "calib"
        save_rgbd = True      # 标定需额外保存彩色图与深度图
        print("\n===== 转轴标定采集（通电常转） =====")
        print("请将棋盘格牢固固定在转台上，相机位置固定不再改动。")
    elif mode == "scan":
        out_dir = ensure_dirs()["scan"]
        prefix = "scan"
        save_rgbd = False
        print("\n===== 工件扫描采集（通电常转） =====")
        print("请将工件固定在转台上，相机位置保持与标定时一致。")
    else:
        raise ValueError(f"未知采集模式: {mode}")

    cycle_sec = float(ACQUISITION["turntable_full_cycle_sec"])
    wait_stable = float(ACQUISITION["wait_stable_sec"])
    interval = float(ACQUISITION["interval_sec"])
    overlap_deg = float(ACQUISITION["overlap_deg"])
    max_frames = int(ACQUISITION["max_frames"])

    # 打印转台配置与推算参数
    deg_per_sec = 360.0 / cycle_sec
    print(f"[转台配置] 一圈周期 = {cycle_sec} 秒")
    print(f"[转台配置] 稳定等待 = {wait_stable} 秒（丢弃加速段帧）")
    print(f"[转台配置] 角速度 = {deg_per_sec:.3f}°/s")
    print(f"[转台配置] 采集间隔 = {interval} 秒，每帧约 {deg_per_sec * interval:.3f}°")
    print(f"[转台配置] 采集满 360° + {overlap_deg}° 首尾重叠后自动结束")

    # 标定采集保存相机内参
    if save_rgbd:
        cam_param_path = os.path.join(out_dir, "camera_params.json")
        save_camera_params(cam_param_path, camera)
        print(f"相机内参已保存: {cam_param_path}")

    # ---- 第 1 步：等待转速稳定，丢弃加速段帧 ----
    if mode == "calib":
        # 标定采集（棋盘格可见）：自动检测转台匀速，避免「上电延迟 + 加速段」残留
        _wait_turntable_stable(camera, interval)
    else:
        # 工件扫描采集（无棋盘格）：固定等待 + 清空 SDK 帧队列
        print(f"\n请给转台上电。程序将等待 {wait_stable} 秒丢弃加速段帧（不保存）...")
        stable_deadline = time.time() + wait_stable
        while time.time() < stable_deadline:
            try:
                camera.grab_rgbd()   # 抓取并丢弃，同时清空 SDK 帧队列
            except Exception:
                pass
        for _ in range(5):
            try:
                camera.grab_rgbd()
            except Exception:
                pass

    # ---- 第 2 步：转速稳定后定时抓帧，满 360° 自动结束 ----
    print(f"\n转速稳定，开始正式采集（间隔 {interval} 秒）...")
    t0 = time.time()
    records = []
    i = 0
    try:
        while i < max_frames:
            # 采集当前帧
            try:
                color, depth = camera.grab_rgbd()
            except Exception as exc:
                print(f"[错误] 第 {i} 帧采集失败: {exc}")
                color, depth = None, None

            timestamp = time.time() - t0
            angle_deg = angle_deg_from_timestamp(timestamp, 0.0, cycle_sec)

            saved = False
            if color is not None and depth is not None:
                # 【全黑帧过滤】彩色图过暗/全黑时直接跳过，不保存该帧
                if _is_black_frame(color):
                    print(f"第 {i} 帧为全黑/过暗帧，跳过不保存。")
                else:
                    # 标定采集：实时检测棋盘格 2D 角点状态（失败帧将在标定阶段丢弃）
                    if save_rgbd:
                        if _quick_check_chessboard(color):
                            print(f"第 {i} 帧棋盘格检测: OK")
                        else:
                            print(f"第 {i} 帧棋盘格检测: 失败（该帧将在标定时被丢弃，不参与拟合）")

                    pcd = camera.build_pointcloud(
                        color, depth, voxel_size_mm=ACQUISITION.get("voxel_size_mm", 1.0))
                    if len(pcd.points) == 0:
                        print(f"[错误] 第 {i} 帧点云为空，跳过。")
                    else:
                        name = f"{prefix}_{i:03d}"
                        ply_path = os.path.join(out_dir, name + ".ply")
                        _save_pointcloud(pcd, ply_path)
                        rec = {"index": i, "timestamp": round(timestamp, 4),
                               "angle_deg": round(angle_deg, 4),
                               "ply": os.path.basename(ply_path)}
                        if save_rgbd:
                            color_path = os.path.join(out_dir, name + "_color.png")
                            depth_path = os.path.join(out_dir, name + "_depth.npy")
                            cv2.imwrite(color_path, color)
                            np.save(depth_path, depth)
                            rec["color"] = os.path.basename(color_path)
                            rec["depth"] = os.path.basename(depth_path)
                        records.append(rec)
                        saved = True

            # 打印每帧时间戳与推算角度
            print(f"第 {i} 帧: t={timestamp:.3f}s, 角度={angle_deg:.2f}°"
                  + ("" if saved else "（未保存）"))

            if ACQUISITION.get("show_preview", True) and color is not None:
                _show_preview(color, depth)

            # 满一圈（加首尾重叠）自动结束
            if angle_deg >= 360.0 + overlap_deg:
                print(f"[采集结束] 已覆盖 {angle_deg:.2f}° >= 360°+{overlap_deg}°。")
                break

            # 精确等到下一个采集时刻
            i += 1
            next_t = i * interval
            delay = next_t - (time.time() - t0)
            if delay > 0:
                time.sleep(delay)
    finally:
        cv2.destroyAllWindows()

    _save_angle_list(out_dir, records)
    last_angle = records[-1]["angle_deg"] if records else 0.0
    print(f"\n采集完成，共 {len(records)} 帧，覆盖角度 {last_angle:.2f}°，保存在 {out_dir}")


def _save_angle_list(out_dir, records):
    """保存帧清单 json（记录时间戳）。"""
    path = os.path.join(out_dir, "angles.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"帧清单已保存: {path}")


def run(mode):
    """采集模块入口。mode: "calib" / "scan"。"""
    camera = OrbbecCamera()
    try:
        camera.open()
        _acquire(camera, mode)
    finally:
        camera.close()


if __name__ == "__main__":
    import sys
    m = sys.argv[1] if len(sys.argv) > 1 else "calib"
    run(m)
