# -*- coding: utf-8 -*-
"""
相机封装：Orbbec Astra Stereo S U3 采集与彩色点云生成
======================================================

功能：
1. 打开相机、使能激光（深度检测依赖 IR 投影）
2. 采集 RGB-D（彩色图 + 对齐到彩色的深度图）
3. 生成 XYZRGB 彩色点云（彩色相机坐标系，单位 mm）
4. 彩色像素坐标 -> 3D 点（供转轴标定角点映射）

关键说明：
- pyorbbecsdk-community 的 frame.get_data() 存在 stride=0 缺陷（返回的
  numpy 数组所有元素读到同一字节），因此统一使用 get_data_pointer() +
  ctypes 直接读取帧内存。
- 深度原始值为 Y16 格式（uint16），depth_scale=1.0，单位即毫米。
- 点云坐标统一采用「彩色相机坐标系」，单位毫米；所有 3D 计算（标定、
  重建）均沿用该坐标系与单位，保证一致。
"""

import ctypes
import json

import numpy as np
import cv2
import open3d as o3d
import pyorbbecsdk as ob

from config import CAMERA, DEPTH_MIN_MM, DEPTH_MAX_MM

# 使 ctypes 能读取 capsule 指针
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p


# ---------------------------------------------------------------------------
# 帧数据读取（绕过 stride=0 缺陷）
# ---------------------------------------------------------------------------
def get_frame_data(frame):
    """正确读取帧的原始字节数据。"""
    cap = frame.get_data_pointer()
    ptr = ctypes.pythonapi.PyCapsule_GetPointer(cap, b"frame_data_pointer")
    return np.frombuffer(ctypes.string_at(ptr, frame.get_data_size()), dtype=np.uint8)


def _decode_color(frame):
    """将彩色帧解码为 BGR uint8 图像。"""
    data = get_frame_data(frame)
    fmt = frame.get_format()
    if fmt == ob.OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == ob.OBFormat.RGB:
        img = data.reshape((frame.get_height(), frame.get_width(), 3))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if fmt == ob.OBFormat.BGRA:
        img = data.reshape((frame.get_height(), frame.get_width(), 4))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    raise RuntimeError(f"不支持的彩色格式: {fmt}")


def _decode_depth(frame):
    """将深度帧解码为 uint16（单位由 depth_scale 换算为 mm）。"""
    data = get_frame_data(frame).view(np.uint16)
    depth = data.reshape((frame.get_height(), frame.get_width())).astype(np.float32)
    return depth * frame.get_depth_scale()


class OrbbecCamera:
    """Orbbec Astra Stereo S U3 相机封装。"""

    def __init__(self):
        self._pipeline = None
        self._device = None
        self._align_filter = None
        self.rgb_intrinsic = None      # [fx, fy, cx, cy]
        self.rgb_distortion = None     # [k1,k2,p1,p2,k3,k4,k5,k6]
        self._undist_map = None        # 预计算的去畸变归一化坐标 (H, W, 2)

    # ------------------------------------------------------------------
    # 打开 / 关闭
    # ------------------------------------------------------------------
    def open(self):
        ctx = ob.Context()
        dev_list = ctx.query_devices()
        if dev_list.get_count() == 0:
            raise RuntimeError("未检测到 Orbbec 设备，请检查 USB3.0 连接与驱动。")
        self._device = dev_list.get_device_by_index(0)

        # 使能激光（关闭 LDP 距离保护、开启激光投影）
        self._set_bool(ob.OBPropertyID.OB_PROP_LDP_BOOL, False)
        self._set_bool(ob.OBPropertyID.OB_PROP_LASER_BOOL, True)

        self._pipeline = ob.Pipeline()
        config = ob.Config()
        # ================================================================
        # 彩色分辨率：默认 640×480 MJPG@30fps（与深度 640×400 组成 SDK 唯一
        # 支持的「RGB-D 外参表匹配配对」）。
        # - 2026-08-31: 尝试切到 1280×720 MJPG，但 SDK AlignFilter 报错
        #   "Can not find matched camera param!"（1280×720 + 640×400 不在出
        #   厂外参配对表中），被迫回退。
        # - 下一轮彩色升级计划：用默认分辨率先 get_camera_param() 拿到
        #   T_color_from_depth 外参，再用 OpenCV 手写投影/反投影做深度-彩色
        #   对齐（绕过 SDK AlignFilter）。
        # ================================================================
        config.enable_stream(
            self._pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            .get_default_video_stream_profile())
        # 深度：默认 640×400 Y11@30fps（硬件上限）
        config.enable_stream(
            self._pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            .get_default_video_stream_profile())
        self._pipeline.start(config)

        # 深度对齐到彩色流
        self._align_filter = ob.AlignFilter(ob.OBStreamType.COLOR_STREAM)

        # 读取并缓存相机内参，预计算去畸变映射
        self._load_camera_params()
        info = self._device.get_device_info()
        conn_type = str(info.get_connection_type())
        print(f"[相机] 已打开 {info.get_name()}  SN={info.get_serial_number()}  连接={conn_type}")
        # USB 连接速度由 USB 控制器 PHY 层在枚举时协商，SDK 只能读取、无法强制改变。
        # 深度传感器数据量大，必须走 USB3.0；若降级为 USB2.0，深度流会反复丢帧。
        if "USB2" in conn_type:
            print("[警告] 深度传感器当前为 USB2.0 连接（应为 USB3.0），带宽不足会导致深度流丢帧。")
            print("       请重新插拔 USB 线，或换到主板原生 USB3.0 口（蓝色口），确认显示 USB3.0 后再采集。")

    def close(self):
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def _set_bool(self, prop_id, value):
        try:
            if self._device.is_property_supported(prop_id, ob.OBPermissionType.PERMISSION_READ_WRITE):
                self._device.set_bool_property(prop_id, value)
        except Exception as exc:
            print(f"[警告] 属性 {prop_id} 设置失败: {exc}")

    # ------------------------------------------------------------------
    # 内参
    # ------------------------------------------------------------------
    def _load_camera_params(self):
        """从 SDK 读取内参，并预计算彩色去畸变归一化坐标映射。"""
        cp = self._pipeline.get_camera_param()

        if CAMERA["rgb_intrinsic"] is not None:
            self.rgb_intrinsic = np.array(CAMERA["rgb_intrinsic"], dtype=np.float64)
        else:
            ri = cp.rgb_intrinsic
            self.rgb_intrinsic = np.array([ri.fx, ri.fy, ri.cx, ri.cy], dtype=np.float64)

        if CAMERA["rgb_distortion"] is not None:
            self.rgb_distortion = np.array(CAMERA["rgb_distortion"], dtype=np.float64)
        else:
            rd = cp.rgb_distortion
            self.rgb_distortion = np.array(
                [rd.k1, rd.k2, rd.p1, rd.p2, rd.k3, rd.k4, rd.k5, rd.k6],
                dtype=np.float64)

        # 预计算去畸变映射（相机固定，内参不变，只需计算一次）
        self._compute_undistort_map()

    def _compute_undistort_map(self):
        """为彩色图每个像素预计算去畸变后的归一化坐标 (x_n, y_n)。"""
        # 需要彩色分辨率；从首次采集时确定，这里用默认 640x480
        # 实际分辨率在 grab_rgbd 中按需重建
        self._undist_map = None
        self._undist_shape = None

    def _ensure_undistort_map(self, h, w):
        """按需生成 (h, w) 分辨率的去畸变归一化坐标映射。"""
        if self._undist_map is not None and self._undist_shape == (h, w):
            return
        fx, fy, cx, cy = self.rgb_intrinsic
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = self.rgb_distortion

        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        pts = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
        # undistortPoints -> 归一化针孔坐标
        undist = cv2.undistortPoints(pts, camera_matrix, dist)
        self._undist_map = undist.reshape(h, w, 2).astype(np.float64)
        self._undist_shape = (h, w)

    # ------------------------------------------------------------------
    # 采集
    # ------------------------------------------------------------------
    def grab_rgbd(self):
        """采集一帧 RGB-D。

        返回:
            color_bgr: (H, W, 3) uint8 彩色图
            depth_mm:  (H, W) float32 对齐到彩色的深度图（mm），无有效值处为 0
        """
        frames = self._pipeline.wait_for_frames(2000)
        if frames is None:
            raise RuntimeError("获取帧超时。")

        # 深度对齐到彩色
        aligned = self._align_filter.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = frames.get_color_frame()

        if depth_frame is None or color_frame is None:
            raise RuntimeError("深度或彩色帧为空。")

        color_bgr = _decode_color(color_frame)
        depth_mm = _decode_depth(depth_frame)
        return color_bgr, depth_mm

    # ------------------------------------------------------------------
    # 2D -> 3D
    # ------------------------------------------------------------------
    def color_pixels_to_3d(self, uv, depth_mm):
        """将彩色图像素坐标映射为 3D 点（彩色坐标系，mm）。

        参数:
            uv: (N, 2) 像素坐标 [(u, v), ...]，整数或浮点
            depth_mm: (H, W) 对齐深度图（mm）
        返回:
            (N, 3) 3D 点，无效深度处返回 NaN
        """
        uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
        h, w = depth_mm.shape
        self._ensure_undistort_map(h, w)

        # 双线性采样去畸变归一化坐标
        x_n = self._bilinear(self._undist_map[:, :, 0], uv)
        y_n = self._bilinear(self._undist_map[:, :, 1], uv)
        z = self._bilinear(depth_mm, uv)

        pts = np.stack([x_n * z, y_n * z, z], axis=1)
        # 无效深度标记 NaN
        invalid = (z <= DEPTH_MIN_MM) | (z >= DEPTH_MAX_MM) | (z <= 0)
        pts[invalid] = np.nan
        return pts

    @staticmethod
    def _bilinear(img, uv):
        """对单通道图像做双线性插值采样。uv: (N, 2)。"""
        h, w = img.shape
        u = np.clip(uv[:, 0], 0, w - 1.001)
        v = np.clip(uv[:, 1], 0, h - 1.001)
        u0 = np.floor(u).astype(int)
        v0 = np.floor(v).astype(int)
        u1 = np.minimum(u0 + 1, w - 1)
        v1 = np.minimum(v0 + 1, h - 1)
        du = u - u0
        dv = v - v0
        val = (img[v0, u0] * (1 - du) * (1 - dv) +
               img[v0, u1] * du * (1 - dv) +
               img[v1, u0] * (1 - du) * dv +
               img[v1, u1] * du * dv)
        return val

    # ------------------------------------------------------------------
    # 点云生成
    # ------------------------------------------------------------------
    def build_pointcloud(self, color_bgr, depth_mm, voxel_size_mm=None):
        """由 RGB-D 生成 XYZRGB 点云（彩色坐标系，mm）。

        返回:
            o3d.geometry.PointCloud（有效点，颜色 0~1）
        """
        h, w = depth_mm.shape
        self._ensure_undistort_map(h, w)

        x_n = self._undist_map[:, :, 0]
        y_n = self._undist_map[:, :, 1]
        z = depth_mm

        # 反投影到 3D（mm）
        x = x_n * z
        y = y_n * z

        valid = (z > DEPTH_MIN_MM) & (z < DEPTH_MAX_MM)
        pts = np.stack([x[valid], y[valid], z[valid]], axis=1)
        colors = color_bgr[valid][:, ::-1].astype(np.float64) / 255.0  # BGR -> RGB, 0~1

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if voxel_size_mm is not None and voxel_size_mm > 0:
            pcd = pcd.voxel_down_sample(voxel_size_mm)
        return pcd


def pixels_to_3d(uv, depth_mm, rgb_intrinsic, rgb_distortion):
    """独立函数：彩色像素坐标 -> 3D 点（彩色坐标系，mm）。

    供离线标定模块使用（不依赖相机对象），与 OrbbecCamera.color_pixels_to_3d
    使用同一套去畸变 + 反投影逻辑。

    参数:
        uv: (N, 2) 彩色像素坐标
        depth_mm: (H, W) 对齐深度图（mm）
        rgb_intrinsic: [fx, fy, cx, cy]
        rgb_distortion: [k1,k2,p1,p2,k3,k4,k5,k6]
    返回:
        (N, 3) 3D 点，无效深度处为 NaN
    """
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    fx, fy, cx, cy = rgb_intrinsic
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.asarray(rgb_distortion, dtype=np.float64)

    # 去畸变 -> 归一化针孔坐标
    undist = cv2.undistortPoints(uv.reshape(-1, 1, 2), camera_matrix, dist)
    x_n = undist[:, 0, 0]
    y_n = undist[:, 0, 1]

    # 双线性采样深度
    h, w = depth_mm.shape
    u = np.clip(uv[:, 0], 0, w - 1.001)
    v = np.clip(uv[:, 1], 0, h - 1.001)
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = np.minimum(u0 + 1, w - 1)
    v1 = np.minimum(v0 + 1, h - 1)
    du = (u - u0).astype(np.float64)
    dv = (v - v0).astype(np.float64)
    z = (depth_mm[v0, u0] * (1 - du) * (1 - dv) +
         depth_mm[v0, u1] * du * (1 - dv) +
         depth_mm[v1, u0] * (1 - du) * dv +
         depth_mm[v1, u1] * du * dv)

    pts = np.stack([x_n * z, y_n * z, z], axis=1)
    invalid = (z <= DEPTH_MIN_MM) | (z >= DEPTH_MAX_MM) | (z <= 0)
    pts[invalid] = np.nan
    return pts


# ---------------------------------------------------------------------------
# 相机参数序列化（供标定模块保存/加载，保证 2D->3D 可复现）
# ---------------------------------------------------------------------------
def save_camera_params(path, camera):
    """保存相机内参到 json。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "rgb_intrinsic": camera.rgb_intrinsic.tolist(),
            "rgb_distortion": camera.rgb_distortion.tolist(),
            "unit": "mm",
        }, f, indent=2, ensure_ascii=False)


def load_camera_params(path):
    """加载相机内参 json。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 单位校验
# ---------------------------------------------------------------------------
def assert_unit_mm(points, label="points"):
    """断言 3D 坐标单位为毫米（mm），防止毫米/米混用。

    判断依据：相机工作距离通常在 0.1~2 米量级，即坐标绝对值应在
    150~3000 mm 量级；若出现 >10000（10 米）的量级，说明单位被缩放
    （如误用米）。本函数打印关键坐标量级，并在异常时抛出 AssertionError。

    参数:
        points: (N, 3) 3D 坐标
        label: 用于日志的标签
    返回:
        (mean_abs, max_abs) 绝对值的均值与最大值
    """
    pts = np.asarray(points, dtype=np.float64)
    finite = pts[np.isfinite(pts)]
    if finite.size == 0:
        raise AssertionError(f"[单位校验] {label} 无有效坐标，无法校验单位。")

    mean_abs = float(np.mean(np.abs(finite)))
    max_abs = float(np.max(np.abs(finite)))
    print(f"[单位校验] {label}: 坐标绝对值 均值={mean_abs:.2f} 最大={max_abs:.2f}，"
          f"按毫米量级判定（应远小于 10000）")

    if max_abs > 10000.0:
        raise AssertionError(
            f"[单位校验失败] {label} 坐标最大值 {max_abs:.2f} 超过 10000 mm，"
            f"疑似单位混用（米/毫米）。请检查深度 scale 与反投影单位。")
    return mean_abs, max_abs
