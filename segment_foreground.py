# -*- coding: utf-8 -*-
"""
segment_foreground.py — 转台物体前景分割，生成 masks/ 供 COLMAP/TSDF 融合 使用

核心思路（针对"相机固定 + 物体旋转"的转台场景）：
  1. 运动先验定位：
     - 中值背景建模：对所有/前 N 帧彩色灰度图取 pixel-wise median → 得到
       纯背景估计（转台旋转时物体像素只占部分帧，中值≈静止背景）。
     - 每帧 |frame - bg| > thresh → 得到每帧独立运动掩码 → 每帧独立 bbox
       （解决转台偏心/旋转时「固定 bbox 截掉部分电路板」的问题）。
     - 支持手动固定 bbox 或「全局 bbox union」回退。
  2. SAM2 精细分割：用每帧 bbox（或全局 bbox）作为 box prompt 喂给
     SAM2ImagePredictor，逐帧分割出物体精细边界（multimask_output 选最高
     置信度 mask）。
  3. 形态学后处理：
     - CLOSE + OPEN 去噪点 + 填小裂缝
     - 保留最大连通域（电路板为单一刚性物体）
     - 孔洞填充（确保 mask 内部没有空洞被误判为背景）

输入支持：
  (a) --images DIR           : 目录读 .png/.jpg
  (b) --input_json FILE.json : 读 angles.json 或 scan_frames.json，取 "color" 字段
                                （相对路径以 JSON 所在目录为基）
  (c) 作为库调用 segment_frames(rgb_list, ...) : 直接在内存中处理
                                （供 scan_and_tsdf.py 在两阶段融合前调用）

输出：
  masks/frame_0000.png ... 单通道二值图（255=前景物体，0=背景）。
  作为库调用时返回 masks(H×W uint8) 列表，长度与输入一致。
"""

import argparse
import json
import os

import cv2
import numpy as np

# ================================================================
# 步骤 1：运动先验定位（中值背景建模 + 每帧独立 bbox）
# ================================================================

def build_median_background(gray_frames, sample_every=1):
    """用 gray_frames 列表的中值做稳健背景估计。

    为省内存，每 sample_every 帧取 1 帧；对 35 帧来说 sample_every=1 即可。
    """
    stack = np.stack(gray_frames[::sample_every], axis=0)  # (N,H,W)
    return np.median(stack, axis=0).astype(np.uint8)


def motion_mask_per_frame(gray, bg, diff_thresh=25, min_area_frac=0.002,
                          kernel_size=9):
    """对单帧算运动掩码并返回 (mask_uint8, bbox_xyxy)；失败返回 (None, None)。

    min_area_frac：掩码像素占比下限，低于此判定为「无运动物体」（避免整
    图噪声被误当成前景）。
    """
    d = cv2.absdiff(gray, bg)
    d = cv2.GaussianBlur(d, (5, 5), 0)
    motion = (d > diff_thresh).astype(np.uint8) * 255

    k = np.ones((kernel_size, kernel_size), np.uint8)
    motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, k)
    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    h, w = gray.shape
    if motion.sum() < (h * w * min_area_frac) * 255:
        return None, None

    ys, xs = np.nonzero(motion)
    if len(xs) < 200:
        return None, None

    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    bw, bh = x2 - x1, y2 - y1
    if bw < 10 or bh < 10:
        return None, None

    # 8% margin 外扩
    mx = int(bw * 0.08)
    my = int(bh * 0.08)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w - 1, x2 + mx)
    y2 = min(h - 1, y2 + my)
    return motion, (x1, y1, x2, y2)


def bbox_union(boxes):
    """取一组 bbox 的并集。"""
    xs = [b[0] for b in boxes] + [b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[3] for b in boxes]
    return min(xs), min(ys), max(xs), max(ys)


# ================================================================
# 步骤 2：SAM2 box-prompt 精细分割（可选）
# ================================================================

def _sam2_available():
    try:
        import sam2  # noqa: F401
        return True
    except Exception:
        return False


def build_sam2_predictor(checkpoint, config, device="cpu"):
    """构建 SAM2 predictor；失败返回 None。"""
    if not _sam2_available():
        print("[sam2] sam2 未安装，跳过精细分割（仅使用运动掩码）")
        return None
    if not os.path.exists(checkpoint):
        print(f"[sam2] checkpoint 不存在: {checkpoint}，跳过精细分割")
        return None
    if not os.path.exists(config):
        print(f"[sam2] config 不存在: {config}，跳过精细分割")
        return None
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print(f"[sam2] 加载模型: {os.path.basename(config)} / {os.path.basename(checkpoint)}")
        model = build_sam2(config, checkpoint, device=device, apply_postprocessing=False)
        return SAM2ImagePredictor(model)
    except Exception as e:
        print(f"[sam2] 加载失败({e})，仅使用运动掩码")
        return None


def sam2_segment_one(predictor, img_rgb, box_xyxy):
    """对单帧用 box prompt 分割，返回 (H,W) bool mask；失败退 None。"""
    if predictor is None:
        return None
    x1, y1, x2, y2 = box_xyxy
    if x2 <= x1 or y2 <= y1:
        return None
    box_np = np.array([[x1, y1, x2, y2]], dtype=np.float32)
    try:
        predictor.set_image(img_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=None, point_labels=None, box=box_np, multimask_output=True
        )
        best = masks[int(np.argmax(scores))].astype(bool)
        return best
    except Exception as e:
        print(f"  [sam2 warn] 推理失败: {e}")
        return None


# ================================================================
# 步骤 3：掩码后处理（填孔 / 去小噪 / 最大连通域保留）
# ================================================================

def postprocess_mask(mask_bool_or_u8):
    """mask: bool 或 0-255 uint8；返回 0/255 uint8 精修后 mask。

    处理顺序：CLOSE → OPEN → 孔洞填充 → 最大连通域。
    """
    if mask_bool_or_u8.dtype == bool:
        u = (mask_bool_or_u8.astype(np.uint8)) * 255
    else:
        u = mask_bool_or_u8.astype(np.uint8)
        if u.max() <= 1:
            u = u * 255

    k5 = np.ones((5, 5), np.uint8)
    u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k5)
    u = cv2.morphologyEx(u, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # 孔洞填充：从 (0,0) 开始 floodFill 像素 0 → 填充后得到「与背景连通的 0 区」
    # 与 255 区不连通的内部 0 就是孔。把这些孔置 255。
    h, w = u.shape
    fill_mask = np.zeros((h + 2, w + 2), np.uint8)   # floodFill 需要 tmp 尺寸 (H+2, W+2)
    u_out = u.copy()
    # 只填充 0 值像素：loDiff=upDiff=0；flags 指定像素联通性
    cv2.floodFill(u_out, fill_mask, (0, 0), 255,
                  loDiff=(0, 0, 0, 0), upDiff=(0, 0, 0, 0),
                  flags=8 | cv2.FLOODFILL_FIXED_RANGE)
    # u_out 中保持 0 的像素 = 与图像边界不相连的 0 = 孔洞 → 把它们填为 255
    holes = (u_out == 0).astype(np.uint8) * 255
    u = np.clip(u.astype(np.int32) + holes.astype(np.int32), 0, 255).astype(np.uint8)

    # 最大连通域
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (u > 0).astype(np.uint8), connectivity=8
    )
    if num <= 1:
        return np.zeros((h, w), np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    largest = (labels == idx).astype(np.uint8) * 255
    return largest


# ================================================================
# 内存 API：直接对 RGB list（BGR也兼容，内部不做cvt只要SAM2进来是RGB）
# ================================================================

def segment_frames(rgb_imgs, predictor=None, per_frame_bbox=True,
                   global_box_override=None, min_bbox_area_frac=0.002):
    """对一组图像做前景分割（三步完整流程）。

    参数
    ----------
    rgb_imgs : list[np.ndarray] shape (H,W,3)，RGB 顺序。
    predictor : SAM2ImagePredictor | None
        None 表示只做运动先验 + 后处理，跳过 SAM2。
    per_frame_bbox : bool
        True  → 每帧独立 bbox（转台偏心场景推荐）
        False → 用所有帧 bbox 的并集（或 global_box_override）
    global_box_override : tuple(x1,y1,x2,y2) | None
        手动固定框（覆盖运动分析）。
    min_bbox_area_frac : float
        每帧运动掩码像素占比下限。

    返回
    ----------
    masks_u8 : list[np.ndarray] 0/255 uint8 (H,W)
    bboxes   : list[tuple|None] 每帧最终用的 bbox（便于诊断画图）
    """
    N = len(rgb_imgs)
    if N == 0:
        return [], []

    H, W = rgb_imgs[0].shape[:2]
    grays = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in rgb_imgs]

    # --- 步骤 1 先验定位 ---
    if global_box_override is not None:
        boxes = [global_box_override] * N
    else:
        bg_gray = build_median_background(grays)
        boxes = []
        for g in grays:
            _, b = motion_mask_per_frame(
                g, bg_gray, min_area_frac=min_bbox_area_frac
            )
            boxes.append(b)

        # 失败帧 → 用最近一次成功 bbox；全部失败 → 画面中心 40%
        any_ok = any(b is not None for b in boxes)
        if any_ok:
            # 先算一个 union 做最后兜底
            good = [b for b in boxes if b is not None]
            fallback = bbox_union(good)
            last_ok = fallback
            for i in range(N):
                if boxes[i] is None:
                    boxes[i] = last_ok
                else:
                    last_ok = boxes[i]
        else:
            fallback = (int(W * 0.3), int(H * 0.3), int(W * 0.7), int(H * 0.7))
            boxes = [fallback] * N

        if not per_frame_bbox:
            union = bbox_union(boxes)
            boxes = [union] * N

    # --- 步骤 2 + 3：逐帧 SAM2 + 后处理 / 或直接用运动后处理 ---
    masks_u8 = []
    bg_gray2 = None
    if predictor is None:
        bg_gray2 = build_median_background(grays)

    for i in range(N):
        box = boxes[i]
        if predictor is not None:
            seg = sam2_segment_one(predictor, rgb_imgs[i], box)
            if seg is None or seg.sum() < 50:
                # SAM2 失败 → 退化成运动掩码
                if bg_gray2 is None:
                    bg_gray2 = build_median_background(grays)
                motion_u8, _ = motion_mask_per_frame(
                    grays[i], bg_gray2, min_area_frac=min_bbox_area_frac
                )
                seg_b = (motion_u8 > 0) if motion_u8 is not None else None
            else:
                seg_b = seg
            if seg_b is None:
                seg_b = np.zeros((H, W), dtype=bool)
            masks_u8.append(postprocess_mask(seg_b))
        else:
            if bg_gray2 is None:
                bg_gray2 = build_median_background(grays)
            motion_u8, _ = motion_mask_per_frame(
                grays[i], bg_gray2, min_area_frac=min_bbox_area_frac
            )
            if motion_u8 is None:
                motion_u8 = np.zeros((H, W), dtype=np.uint8)
            masks_u8.append(postprocess_mask(motion_u8 > 0))

    return masks_u8, boxes


# ================================================================
# CLI
# ================================================================

def _load_image_list(args):
    """根据 --images 或 --input_json 返回 (color_paths, names, base_dir)。"""
    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = os.path.dirname(os.path.abspath(args.input_json))
        if isinstance(data, list):
            frames = data
        else:
            frames = data.get("frames", []) or data.get("calibration", []) or data.get("scan", []) or []
        rels = []
        for fr in frames:
            if isinstance(fr, dict):
                rel = fr.get("color") or fr.get("rgb") or fr.get("image")
            else:
                rel = str(fr)
            if rel:
                rels.append(rel)
        paths = [os.path.join(base, r) for r in rels]
        names = [os.path.basename(p) for p in paths]
        return paths, names, base

    base = os.path.abspath(args.images)
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    names = sorted([f for f in os.listdir(args.images)
                    if f.lower().endswith(exts)])
    paths = [os.path.join(base, n) for n in names]
    return paths, names, base


def _default_sam2_config():
    """默认 sam2 小模型 config：先读项目目录，再读 site-pkg 安装内自带。"""
    cands = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "sam2_configs", "sam2.1", "sam2.1_hiera_s.yaml"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "configs", "sam2.1", "sam2.1_hiera_s.yaml"),
    ]
    if _sam2_available():
        import sam2 as _s
        _base = os.path.dirname(os.path.abspath(_s.__file__))
        cands += [
            os.path.join(_base, "configs", "sam2.1", "sam2.1_hiera_s.yaml"),
            os.path.join(_base, "sam2.1_hiera_s.yaml"),
        ]
    for p in cands:
        if os.path.exists(p):
            return p
    return "configs/sam2.1/sam2.1_hiera_s.yaml"


def _default_sam2_ckpt():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "checkpoints", "sam2.1_hiera_small.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images",
                    help="彩色图像目录（默认 images）；--input_json 优先")
    ap.add_argument("--input_json", default=None,
                    help="从 JSON（angles.json / scan_frames.json）取 color 路径")
    ap.add_argument("--output", default="masks", help="掩码输出目录")
    ap.add_argument("--checkpoint", default=_default_sam2_ckpt())
    ap.add_argument("--config", default=_default_sam2_config())
    ap.add_argument("--device", default="cuda" if _sam2_available() and
                    __import__("torch", fromlist=["cuda"]).cuda.is_available()
                    else "cpu")
    ap.add_argument("--bbox", default=None,
                    help="手动固定框 x1,y1,x2,y2（覆盖自动运动定位）")
    ap.add_argument("--per-frame-bbox", action="store_true", default=True,
                    dest="per_frame",
                    help="每帧独立 bbox（默认开启，转台偏心更准）")
    ap.add_argument("--global-bbox", action="store_false", dest="per_frame",
                    help="使用全局 bbox 并集（关闭 per-frame）")
    ap.add_argument("--no-sam2", action="store_true", help="只用运动先验，不调用 SAM2")
    ap.add_argument("--preview", action="store_true",
                    help="额外保存第一帧 + 中间帧 mask 预览图")
    args = ap.parse_args()

    paths, names, _ = _load_image_list(args)
    if not paths:
        print("[error] 没找到任何输入图像")
        return 1
    os.makedirs(args.output, exist_ok=True)
    print(f"[info] 输入图像 {len(paths)} 张")

    # 读 RGB
    rgbs = []
    ok_idx = []
    for i, p in enumerate(paths):
        im = cv2.imread(p)
        if im is None:
            print(f"  [warn] 读失败 {p}")
            continue
        rgbs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ok_idx.append(i)
    if not rgbs:
        print("[error] 所有图像都读失败")
        return 1

    # 手动 bbox
    override = None
    if args.bbox:
        parts = [int(v) for v in args.bbox.split(",")]
        if len(parts) == 4:
            override = tuple(parts)
            print(f"[info] 手动 bbox（覆盖运动）: {override}")

    # 加载 SAM2
    predictor = None
    if not args.no_sam2:
        predictor = build_sam2_predictor(args.checkpoint, args.config, device=args.device)

    # 三步核心
    masks, boxes = segment_frames(
        rgbs, predictor=predictor,
        per_frame_bbox=args.per_frame,
        global_box_override=override,
    )

    # 保存
    for j, i in enumerate(ok_idx):
        out = os.path.join(args.output, names[i])
        # 统一 .png 后缀（和输入同名；scan_and_tsdf 读的时候要能对应 frame_{i:04d}_color.png）
        cv2.imwrite(out, masks[j])

    if args.preview:
        j0 = 0
        i0 = ok_idx[j0]
        first_bgr = cv2.imread(paths[i0])
        m = masks[j0]
        vis = first_bgr.copy()
        vis[m > 0] = (vis[m > 0].astype(np.int16) // 2 +
                      np.array([0, 140, 0], np.int16) // 2).astype(np.uint8)
        x1, y1, x2, y2 = boxes[j0]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(vis, f"frame {names[i0]}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        prev_path = os.path.join(args.output, "_preview_0.png")
        cv2.imwrite(prev_path, vis)
        # 中间帧
        jm = len(ok_idx) // 2
        im_m = cv2.imread(paths[ok_idx[jm]])
        mm = masks[jm]
        vm = im_m.copy()
        vm[mm > 0] = (vm[mm > 0].astype(np.int16) // 2 +
                      np.array([0, 140, 0], np.int16) // 2).astype(np.uint8)
        x1, y1, x2, y2 = boxes[jm]
        cv2.rectangle(vm, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.imwrite(os.path.join(args.output, "_preview_mid.png"), vm)
        print(f"[info] 预览: {prev_path}")

    print(f"[ok] 完成，{len(ok_idx)} 张 mask 已写入 {args.output}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
