# -*- coding: utf-8 -*-
"""
mesh_render_views.py — 把 PLY mesh 离线渲染成 3 个正交视角（俯/侧/前）+ 彩色叠图
（沙箱拦截 Open3D GUI 时用于离线检查重建质量）。

用法:
    .venv\\Scripts\\python.exe mesh_render_views.py output\\result\\pcb_tsdf.ply
    .venv\\Scripts\\python.exe mesh_render_views.py output\\result\\pcb_tsdf_refined.ply
"""
import argparse, os
import numpy as np
import cv2
import open3d as o3d


def _render_offscreen(mesh, w=1280, h=960):
    """用 Open3D 非可视化 headless 渲染，不行就 fallback 到简单 BEV/侧视投影 2D 画布。"""
    try:
        vis = o3d.visualization.rendering.OffscreenRenderer(w, h)
        return vis
    except Exception:
        return None


def _project_3views(vertices, tris=None, colors=None):
    """Fallback：直接把 3D 点投到 3 个正交 2D 平面画成 PNG（无需任何 GL）。

    views: [XY 俯视（Z 叠颜色深浅）, XZ 正视, YZ 侧视]
    """
    # 轴对齐 AABB
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    span = hi - lo
    max_span = float(span.max())
    pad = max_span * 0.08

    views = [
        ("XY_top", 0, 1, 2),   # 俯视: X→x, Y→y, Z→颜色深浅
        ("XZ_front", 0, 2, 1), # 正视: X→x, Z→y, Y→颜色深浅
        ("YZ_side", 1, 2, 0),  # 侧视: Y→x, Z→y, X→颜色深浅
    ]
    outs = {}
    W, H = 1400, 1050
    # 若有三角面，逐面 scanline 填充（简单包围盒 + 深度 buffer）
    for name, ax, ay, ac in views:
        plane = np.zeros((H, W, 3), dtype=np.uint8)
        depth_buf = np.full((H, W), np.inf, dtype=np.float32)

        xs = vertices[:, ax]
        ys = vertices[:, ay]
        cs = vertices[:, ac]
        c_lo, c_hi = float(np.percentile(cs, 1)), float(np.percentile(cs, 99))
        c_rng = max(c_hi - c_lo, 1e-3)

        def _px(v):
            return np.clip(((v - (lo[ax] - pad)) / (max_span + 2*pad)) * (W - 1), 0, W - 1).astype(np.int32)
        def _py(v):
            # y 轴反向（图像原点上）
            return np.clip((H - 1) - (((v - (lo[ay] - pad)) / (max_span + 2*pad)) * (H - 1)), 0, H - 1).astype(np.int32)

        if colors is not None:
            vcol = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        else:
            # 按深度轴 ac 值做伪彩色 jet
            t = np.clip((cs - c_lo) / c_rng, 0, 1)
            # 简易蓝→绿→黄→红 ramp
            ramp = (np.stack([
                np.clip(1.5 - np.abs(t*4 - 3), 0, 1),  # R
                np.clip(1.5 - np.abs(t*4 - 2), 0, 1),  # G
                np.clip(1.5 - np.abs(t*4 - 1), 0, 1),  # B
            ], axis=1) * 255).astype(np.uint8)
            vcol = ramp

        if tris is not None and len(tris) > 0:
            # 预处理三角形 screen 坐标
            tri_px = _px(xs[tris])  # (T,3)
            tri_py = _py(ys[tris])
            # zbuffer: 用 -ac（因为 ac 小的在前？画真实深度用 ac 做 painter's alg：画前先按深度排序）
            tri_depth = cs[tris].mean(axis=1)
            order = np.argsort(-tri_depth)  # 先画远的，后画近的
            tri_px = tri_px[order]
            tri_py = tri_py[order]
            tri_depth = tri_depth[order]
            tri_col = vcol[tris].mean(axis=1)[order]

            # 逐三角形包围盒 + 水平 scanline
            for i in range(len(tris)):
                xmin, xmax = int(tri_px[i].min()), int(tri_px[i].max())
                ymin, ymax = int(tri_py[i].min()), int(tri_py[i].max())
                if xmin == xmax or ymin == ymax:
                    continue
                xmin = max(xmin, 0); xmax = min(xmax, W - 1)
                ymin = max(ymin, 0); ymax = min(ymax, H - 1)
                cx = [float(tri_px[i, j]) for j in range(3)]
                cy = [float(tri_py[i, j]) for j in range(3)]
                # 重心坐标
                A = cx[0]*(cy[1]-cy[2]) + cx[1]*(cy[2]-cy[0]) + cx[2]*(cy[0]-cy[1])
                if abs(A) < 1e-6:
                    continue
                col = (int(tri_col[i, 0]), int(tri_col[i, 1]), int(tri_col[i, 2]))
                for yy in range(ymin, ymax + 1):
                    for xx in range(xmin, xmax + 1):
                        w0 = (xx*(cy[1]-cy[2]) + yy*(cx[2]-cx[1]) + cx[1]*cy[2] - cx[2]*cy[1]) / A
                        w1 = (xx*(cy[2]-cy[0]) + yy*(cx[0]-cx[2]) + cx[2]*cy[0] - cx[0]*cy[2]) / A
                        w2 = 1.0 - w0 - w1
                        if w0 < 0 or w1 < 0 or w2 < 0:
                            continue
                        d = w0 * cs[tris[order[i], 0]] + w1 * cs[tris[order[i], 1]] + w2 * cs[tris[order[i], 2]]
                        if d < depth_buf[yy, xx]:
                            depth_buf[yy, xx] = d
                            plane[yy, xx] = col
        else:
            # Fallback: scatter 画点
            px = _px(xs)
            py = _py(ys)
            for i in range(len(vertices)):
                r = 1
                plane[py[i]-r:py[i]+r+1, px[i]-r:px[i]+r+1] = vcol[i]

        outs[name] = plane
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", help="输入 .ply 路径")
    ap.add_argument("--out_dir", default=None, help="输出目录，默认与 ply 同目录下的 views/")
    args = ap.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.ply, enable_post_processing=True)
    print(f"[读入] {args.ply}")
    print(f"       {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    colors = None
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)

    # 主视图: 3 个正投影视角
    views = _project_3views(verts, tris if len(tris) else None, colors)

    out_dir = args.out_dir
    if not out_dir:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.ply)),
                               "views_" + os.path.splitext(os.path.basename(args.ply))[0])
    os.makedirs(out_dir, exist_ok=True)
    for name, img in views.items():
        path = os.path.join(out_dir, f"{name}.png")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"  → {path}  {img.shape[1]}x{img.shape[0]}")

    # 组合视图
    rows = []
    imgs = [views["XY_top"], views["XZ_front"], views["YZ_side"]]
    labels = ["XY_top (Z depth color)", "XZ_front (Y depth color)", "YZ_side (X depth color)"]
    H = max(im.shape[0] for im in imgs) + 40
    W_tot = sum(im.shape[1] for im in imgs) + 40 * (len(imgs) - 1)
    canvas = np.full((H, W_tot, 3), 20, dtype=np.uint8)
    x = 0
    for lab, im in zip(labels, imgs):
        canvas[40:40+im.shape[0], x:x+im.shape[1]] = im
        cv2.putText(canvas, lab, (x + 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        x += im.shape[1] + 40
    out_path = os.path.join(out_dir, "_3views_combined.png")
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"[OK] 合成视图: {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
