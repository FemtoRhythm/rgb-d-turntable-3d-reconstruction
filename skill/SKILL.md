# RGB-D 转台三维重建技能（TSDF 融合）

## 适用场景

固定 RGB-D 相机 + 匀速不可控转台，对小型物体（PCB、元件、工艺品）做 360° 三维重建。
替代传统 ICP+泊松方法，用 TSDF 体素融合获得更鲁棒、无累积漂移的网格。

## 硬件要求

| 组件 | 规格 |
|------|------|
| 相机 | Orbbec Astra Stereo S U3（SV1301S_U3），USB 3.0 |
| 转台 | 匀速旋转，一圈约 19s，不可启停/步进 |
| 标定板 | 棋盘格 8×8 内角点，方格边长 18mm |

## 环境依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pyorbbecsdk open3d numpy scipy opencv-python
# SAM2（可选，前景分割用）
pip install sam-2
```

## 完整流程（4 步）

### 步骤 1：转轴标定

```bash
# 1a. 采集：棋盘格放转台中心，转台上电
python src/acquisition.py calib
# → 22 帧覆盖 380°，保存到 output/calibration/

# 1b. 解算：拟合空间圆 → 求转轴原点/方向/角速度
python src/calibration.py
# → 输出 output/axis_params.json
```

**质量验收**：
- 空间圆残差均值 ≤ 0.5mm，最大 ≤ 1.0mm
- 相邻帧重合误差中位 ≤ 2mm

### 步骤 2：扫描采集 + 帧保存

```bash
# 电路板放转台中心，转台断电
$env:FG_MOTION_ONLY = "1"   # 方案3：保留转台圆盘骨架作为几何支撑
python src/scan_and_tsdf.py
# → 采集 ~100 帧（0.20s/帧，覆盖 380°）
# → 帧数据保存到 output/frames/frame_XXX.npz
# → TSDF 融合（voxel=2.0mm, trunc=15.0mm）
# → 输出 output/result/pcb_tsdf_raw.ply（原始副本）+ pcb_tsdf.ply
```

**离线重跑**（不扫描相机）：
```bash
$env:TSDF_OFFLINE = "1"
python src/scan_and_tsdf.py
```

### 步骤 3：Poisson 桥接（跨气隙合并）

TSDF 融合可能生成多个不连通薄壳（圆盘骨架两半 + 近距伪影），Poisson 重建可跨
<15mm 气隙将它们合并为单整块。

```python
# 内联脚本（详见项目历史中的 Poisson 桥接逻辑）
# 1. 加载 pcb_tsdf_raw.ply
# 2. Poisson(depth=9, scale=1.1, linear_fit=True)
# 3. KNN 裁剪 ≤15mm
# 4. 保留最大连通域
# 5. 保存为 pcb_tsdf.ply
```

### 步骤 4：精修

```bash
python src/refine_pcb_mesh.py
# → 输出 output/result/pcb_tsdf_refined.ply
# → 自动启动可视化窗口（红=原始，蓝=精修后）
```

**精修逻辑**：
1. 转轴坐标变换：h = (P - axis_origin)·axis_dir，r = |P - 投影到轴|
2. 场景识别：圆盘窄峰检测（span<8mm, frac>40%）→ 场景 A
   - 补救：径向 r 分布二次确认（Q90/Q50>1.25 → 强制场景 A）
3. 径向剔除：面密度双峰检测 + 密度谷底 + 10mm 余量 → 剔外环
4. 厚度剔除：h' 下限 = peak_P2 - TSDF插值厚度 - 0.5mm（clip [-6, -1.5]）
5. 连通域保留 Top1

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| voxel_length | 2.0mm | TSDF 体素大小，越大桥连越强 |
| sdf_trunc | 15.0mm | 截断距离（≈7.5×voxel） |
| 采集间隔 | 0.20s | 约 3.8°/帧，100 帧覆盖 380° |
| Poisson depth | 9 | 跨气隙桥接能力 |
| KNN 裁剪 | 15mm | 保留离原始点云 ≤15mm 的面 |
| 连通域保留 | 99.5% | 不丢电路板碎片 |

## 硬件限制

| 限制 | 影响 |
|------|------|
| Astra 仅校准 640×480+640×400 | 1280×720 彩色不可用（内参全 0） |
| 电路板绿油反光 → 深度=0 | TSDF 无法融合电路板大平面 |
| 高出圆盘 ≥5mm 元件 → 深度=0 | 无元件凸起几何 |
| 转台圆盘薄壳作为骨架 | TSDF 生成圆盘骨架，精修剔除外环 |

## 文件结构

```
.
├── config.py                 # 全局参数（相机 + 标定板 + 采集配置）
├── src/
│   ├── camera_utils.py       # Orbbec SDK 封装（open/grab_rgbd/close）
│   ├── acquisition.py        # 标定数据采集（棋盘格 + 转台）
│   ├── calibration.py        # 转轴解算（空间圆拟合 + 自检）
│   ├── scan_and_tsdf.py      # 主流程：采集→TSDF融合→后处理
│   ├── refine_pcb_mesh.py    # 网格精修（径向+厚度剔除+连通域）
│   ├── segment_foreground.py # SAM2 前景分割（可选）
│   └── mesh_render_views.py  # 正交视图渲染工具
├── requirements.txt          # Python 依赖
└── .gitignore
```

## 质量指标

| 指标 | 目标 | 当前 |
|------|------|------|
| 转轴残差 | ≤0.5mm | 0.167mm |
| 自检重合 | ≤2mm | 0.380mm |
| Top1 连通域 | ≥90% | 99.9% |
| 板厚 | 2.5~3.5mm | 6.55mm（Poisson 插值） |
| 径向 MAX | ≤130mm | 124.9mm |
