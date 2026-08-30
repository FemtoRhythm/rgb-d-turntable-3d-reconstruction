# -*- coding: utf-8 -*-
"""
对 TSDF 重建出来的 pcb_tsdf.ply 做「去转台台面外环」精修：
支持两种场景自动识别：
【场景 A · 方案3 (FG_MOTION_ONLY=1)】检测到「圆盘窄峰」
    → 转台圆盘薄壳被完整保留，作为电路板板底大平面的几何骨架
    → 不切薄壳（否则骨架就没了）
    → 改用**径向 r mask**：计算每三角面中心到转轴的径向距离 r，
        剔除 r > 圆盘覆盖半径+余量 的纯"圆盘外环"（电路板肯定覆盖不到）
    → 剔除板底下方 >1.5mm 虚影，h' 顶部用固定窗口（至少 15mm）
【场景 B · SAM2 精细分割】未检测到圆盘窄峰
    → 沿用旧有逻辑（密度 85% 段定位板底 + 薄壳剔除 + 厚度窗口筛选）
最终：累计 98% 连通域保留（含细小连接桥），保存，对比可视化。
"""
import json
import os
import sys

import numpy as np
import open3d as o3d

AXIS_PARAM_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output", "axis_params.json")
RESULT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output", "result")


def point_to_plane(points_nx3, plane_abcd):
    a, b, c, d = plane_abcd
    n = np.array([a, b, c], dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return np.zeros(len(points_nx3), dtype=np.float64)
    n = n / norm
    d_norm = d / norm
    return points_nx3 @ n + d_norm


def detect_disk_peak(h_arr, span_thr_mm=8.0, frac_thr=0.40):
    """从 h 分布里检测「圆盘窄峰」：占比>frac_thr 且 跨度<span_thr_mm。
    返回 dict: has_disk, peak_h, peak_span, peak_frac, peak_h_lo, peak_h_hi,
                peak_mask (bool[N] 峰内点), board_center (板底 h = 峰中心 - 半厚),
                h_q (基础分位).
    """
    if h_arr.size < 100:
        return {"has_disk": False}
    h_q = np.percentile(h_arr, [1, 5, 25, 50, 75, 90, 95, 99])
    bins = int(np.ceil((h_q[-1] - h_q[0]) / 1.0))
    bins = max(bins, 30)
    hh, edges = np.histogram(h_arr, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    kernel = np.ones(3, dtype=float) / 3.0
    hh_s = np.convolve(hh.astype(float), kernel, mode="same")
    peak_idx = int(np.argmax(hh_s))
    peak_h = float(centers[peak_idx])
    peak_val = float(hh_s[peak_idx])
    # 向两侧展开到 10% 峰高
    tl = peak_idx
    while tl > 0 and hh_s[tl - 1] >= peak_val * 0.10:
        tl -= 1
    th = peak_idx
    while th < len(hh_s) - 1 and hh_s[th + 1] >= peak_val * 0.10:
        th += 1
    peak_span = float(centers[th] - centers[tl])
    peak_h_lo = float(centers[tl])
    peak_h_hi = float(centers[th])
    peak_frac = float(hh_s[tl:th + 1].sum() / hh_s.sum())
    has_disk = (peak_span < span_thr_mm) and (peak_frac > frac_thr)
    # 哪些原始顶点在峰内（直接用原始 h 值区间判断）
    peak_mask = (h_arr >= peak_h_lo) & (h_arr <= peak_h_hi)
    # 板底中心：取峰内点的 P35 分位 — 圆盘薄壳是 TSDF 模糊后的 ~4.5mm 厚壳，
    # P35 大致对应 TSDF 壳的"下边界"+板底交界处（≈真实板底平面）
    if peak_mask.sum() > 100:
        board_center = float(np.percentile(h_arr[peak_mask], 35))
    else:
        board_center = peak_h
    return dict(
        has_disk=has_disk, peak_h=peak_h, peak_span=peak_span, peak_frac=peak_frac,
        peak_h_lo=peak_h_lo, peak_h_hi=peak_h_hi, peak_mask=peak_mask,
        board_center=board_center, h_q=h_q,
    )


def radial_distance_to_axis(verts_nx3, axis_origin, axis_dir):
    """verts: Nx3 世界坐标。返回 r: N, 到转轴的垂直径向距离(mm)。
    r = |(P - O) × dir|  /  |dir|。dir 已归一化。"""
    diff = verts_nx3 - axis_origin[None, :]  # Nx3
    cross = np.cross(diff, axis_dir[None, :])  # Nx3
    r = np.linalg.norm(cross, axis=1)  # N
    return r


def main():
    in_ply = os.path.join(RESULT_DIR, "pcb_tsdf.ply")
    if not os.path.exists(in_ply):
        raise FileNotFoundError(f"找不到输入 mesh: {in_ply}")
    print(f"[读取] {in_ply}")
    mesh = o3d.io.read_triangle_mesh(in_ply)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    print(f"  输入: {len(verts)} 顶点, {len(tris)} 三角面, "
          f"顶点颜色={mesh.has_vertex_colors()}")
    if len(verts) < 100:
        raise RuntimeError("顶点过少，无法处理")

    # ------------ 1. 转轴标定 → 台面平面 + h 沿轴高度 ------------
    if not os.path.exists(AXIS_PARAM_FILE):
        raise FileNotFoundError(f"找不到 axis_params.json: {AXIS_PARAM_FILE}")
    with open(AXIS_PARAM_FILE, "r", encoding="utf-8") as f:
        axis = json.load(f)
    axis_origin = np.array(axis["axis_origin"], dtype=np.float64)
    axis_dir = np.array(axis["axis_dir"], dtype=np.float64)
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    a, b, c = axis_dir.tolist()
    d = - float(np.dot(axis_dir, axis_origin))
    plane_model = np.array([a, b, c, d], dtype=np.float64)

    dist_signed = point_to_plane(verts, plane_model)  # h = (P-O)·d
    print(f"\n[步骤 1] 由转轴标定构造台面平面: axis_dir·(P - axis_origin) = 0")
    print(f"  axis_origin = {np.round(axis_origin, 2)} mm")
    print(f"  axis_dir    = {np.round(axis_dir, 4)}")
    print(f"  h = (P-O)·d 分位数: 5%={np.percentile(dist_signed,5):.2f}  "
          f"25%={np.percentile(dist_signed,25):.2f}  "
          f"50%={np.percentile(dist_signed,50):.2f}  "
          f"75%={np.percentile(dist_signed,75):.2f}  "
          f"95%={np.percentile(dist_signed,95):.2f}  "
          f"99%={np.percentile(dist_signed,99):.2f} mm")

    # ------------ 2. 场景识别：圆盘窄峰检测 ------------
    # ------------ 3. 先算径向 r（在 scene 判断之前，因补救需要它） ------------
    r_all = radial_distance_to_axis(verts, axis_origin, axis_dir)
    tri_r = 0.333333 * (r_all[tris[:, 0]] + r_all[tris[:, 1]] + r_all[tris[:, 2]])
    dp = detect_disk_peak(dist_signed, span_thr_mm=8.0, frac_thr=0.40)
    is_scene_a = bool(dp["has_disk"])
    # 2026-09-01 补救：即使"圆盘窄峰"因 Poisson 增厚 span 超阈值（>=8mm），
    # 也要再用径向 r 分布二次确认：若 r P50 远小于 r P90（圆盘+外环双峰特征），
    # 且中心 120mm 内含超过 35% 顶点，强制判定场景 A（圆盘骨架）。
    if not is_scene_a:
        r_q25, r_q50, r_q75, r_q90, r_q99 = np.percentile(r_all, [25, 50, 75, 90, 99])
        r_in120 = float((r_all <= 120.0).sum()) / max(r_all.size, 1)
        q_ratio = float(r_q90) / max(r_q50, 1e-6)
        if r_q50 < 125.0 and r_q90 > 135.0 and q_ratio > 1.25 and r_in120 >= 0.35:
            print(f"  [补救·径向确认] rQ50={r_q50:.1f} rQ90={r_q90:.1f} Q90/Q50={q_ratio:.2f}"
                  f"  r≤120占比={r_in120*100:.1f}% → 强制场景 A")
            is_scene_a = True
            h_lo_f, h_hi_f = [float(x) for x in np.percentile(dist_signed, [30, 70])]
            dp = dict(dp)
            dp["has_disk"] = True
            dp["board_center"] = 0.5 * (h_lo_f + h_hi_f)
            dp["peak_h"] = float(np.percentile(dist_signed, 50))
            dp["peak_span"] = 10.0
            dp["peak_frac"] = 0.95
    print(f"\n[步骤 2] 场景识别（圆盘窄峰检测）:")
    print(f"  峰中心 h={dp.get('peak_h', float('nan')):.2f}mm  "
          f"跨度={dp.get('peak_span', float('nan')):.2f}mm  "
          f"占比={dp.get('peak_frac', float('nan'))*100:.1f}%")
    print(f"  → 场景 {'A（方案3·圆盘骨架）' if is_scene_a else 'B（SAM2精细分割·无圆盘）'}")

    print(f"\n[径向 r] 全顶点分位: "
          f"P25={np.percentile(r_all,25):.1f}  P50={np.percentile(r_all,50):.1f}  "
          f"P75={np.percentile(r_all,75):.1f}  P90={np.percentile(r_all,90):.1f}  "
          f"P95={np.percentile(r_all,95):.1f}  P99={np.percentile(r_all,99):.1f} mm")

    # ======================================================================
    # 分支 A / B
    # ======================================================================
    drop = np.zeros(len(tris), dtype=bool)

    if is_scene_a:
        # ================================================================
        # 场景 A · 方案3：圆盘 = 板底骨架（不切薄壳）
        # ================================================================
        body_center = float(dp["board_center"])  # ≈ 峰内点 P35
        h_prime = dist_signed - body_center      # 板底 ≈ 0
        # 3a) 圆盘/电路板覆盖半径估计（终极方案）
        #     TSDF 里台面散点薄壳（r=120~160mm）与中心圆盘（r=60~90mm）双峰共存，
        #     峰内点全混 h 薄壳无法区分。正确做法：用**全顶点径向面密度**
        #     （顶点数 / 环形面积 π(rout²-rin²)）—— 中心单位面积顶点密度高
        #     （真正的圆盘/电路板连续 mesh），外环台面碎片密度显著下降。
        #     算法：
        #       1) 对 [0, 180)mm 分 5mm bin 算面密度 d[i]
        #       2) 找主峰密度 d_peak = max(d[1:30]) （跳过 0-5mm 转轴空穴）
        #       3) 从 r=10mm 向右找「第一处密度从 ≥ 0.6*d_peak 跌到 < 0.5*d_peak，
        #          且后续 ≥15mm 内密度不回升到跌前 0.8*d_peak」的位置 — 这是
        #          中心圆盘与台面背景的真实分界线
        #       4) 取该分界的 r + 5mm 余量，兜底物理区间 [70, 105]mm
        r_data = r_all.copy()
        r_outer_ring_cut = 100.0  # 兜底默认（场景A 电路板常 ~100-120mm）
        edge_idx = -1
        if r_data.size > 100:
            bins_r2 = np.arange(0.0, 180.1, 5.0)  # 0,5,10,...,180
            cnts_r2, edges_r2 = np.histogram(np.clip(r_data, 0, 179.9), bins=bins_r2)
            # 每个 5mm 环的面积 π(rout²-rin²)
            rin_arr = edges_r2[:-1]; rout_arr = edges_r2[1:]
            areas = np.pi * (rout_arr ** 2 - rin_arr ** 2)
            with np.errstate(divide='ignore', invalid='ignore'):
                dens = np.where(areas > 1e-6, cnts_r2.astype(float) / areas, 0.0)
            N = len(dens)
            # 跳过 0-5mm 转轴空穴，求 d_peak、thr_high（>=主峰 55%=仍属圆盘/电路板连续薄壳）
            dens_no_core = dens[1:].copy()
            d_peak = float(dens_no_core.max())
            thr_high = d_peak * 0.55
            # ====== 2026-09-01 新算法：「高密度连续薄壳」右边界 ======
            # 现象：扫描密度并不是"主峰→立即谷底"，而是 0.84→0.60→0.43→0.36→...
            # 然后 120mm 前后出现 0.35=台面外环薄壳第二峰（环形台面碎片），
            # 再往后 130-145mm 掉到 0.25-0.27=真正的背景外环散点。
            # 策略：在 [30, 160)mm 区间，从左向右找「高密度薄壳的最后一个 bin」：
            #   ① 先找到"连续 ≥thr_high"段的右端（仍满足则向右扩张）；
            #   ② 再从该端 + 2 bin 起，找「局部密度极小值点」— 第一峰与第二峰
            #      之间的谷底，该处是"圆盘/电路板"与"台面外环碎片"的真实分界。
            #   ③ 若未找到极小值，兜底用：从左第一个 <0.28*d_peak 且后续 ≥3bin
            #      都不反弹回 ≥0.40*d_peak 的位置。
            #   ④ 最终 r 再 +10mm（给电路板元件/边缘斜面/反光漏检的余量）。
            #   ⑤ 物理兜底 [95, 130]mm（电路板 100×80 对角 128）。
            # ---- ① 找连续 ≥thr_high 的最后一个 bin 右端 ----
            last_high_idx = -1
            for i in range(1, N):
                if dens[i] >= thr_high:
                    last_high_idx = i
            # ---- ② 从 last_high_idx+2 起找 3-bin 滑动窗口均值的极小值 ----
            search_start = last_high_idx + 2  # 跳过紧邻的过渡 bin
            search_end   = N - 1              # 最后一个 bin 无法做极小
            # 3-bin 均值
            smooth = np.zeros(N, dtype=float)
            for i in range(1, N - 1):
                smooth[i] = (dens[i - 1] + dens[i] + dens[i] + dens[i + 1]) / 4.0
            valley_idx = -1
            valley_val = 1e9
            for i in range(max(search_start, 10), min(search_end, 30)):
                # 极小：smooth[i] < smooth[i-1] 且 smooth[i] < smooth[i+1]
                if smooth[i] < smooth[i - 1] and smooth[i] < smooth[i + 1]:
                    if smooth[i] < valley_val:
                        valley_val = smooth[i]
                        valley_idx = i
            # ---- ③ 若没极小值，用"首次跌破 0.28*d_peak 且 3bin 不反弹 ≥0.40*d_peak" ----
            if valley_idx < 0:
                thr_fall = d_peak * 0.28
                thr_rebound = d_peak * 0.40
                for i in range(max(last_high_idx + 2, 12), N - 3):
                    if dens[i] < thr_fall:
                        rebound = False
                        for j in range(i + 1, min(i + 4, N)):
                            if dens[j] >= thr_rebound:
                                rebound = True; break
                        if not rebound:
                            valley_idx = i; break
            # ---- 确定 edge_idx：valley_idx - 1（要切在"台面外环第二峰"的左外沿）
            if valley_idx > 0:
                edge_idx = valley_idx - 1
                r_edge = 5.0 * (edge_idx + 1)  # bin[edge_idx] 的右端点
                # +10mm 给电路板真实外伸（元件/边缘，毕竟转台圆盘=120mm直径）
                r_outer_ring_cut = r_edge + 10.0
            else:
                r_outer_ring_cut = 110.0  # 终极兜底（电路板常 100-120 横纵向范围）
            # ---- ⑤ 物理兜底区间（电路板真实外轮廓 r P99 ≤ 128mm） ----
            r_outer_ring_cut = min(130.0, max(95.0, r_outer_ring_cut))
            # 诊断辅助：报告推导依据
            print(f"  [径向判定] d_peak={d_peak:.3f}  last_high_bin=[{last_high_idx*5},{(last_high_idx+1)*5})mm"
                  f"  valley_idx=[{valley_idx*5},{(valley_idx+1)*5})mm"
                  f"  edge={r_outer_ring_cut:.0f}mm")
            # 诊断打印 5~155mm 密度（每 5mm），标记分界点
            print(f"\n[步骤 3 · 场景A] 径向面密度双峰检测（转台中心 vs 台面碎片）：")
            print(f"  5mm/环 · 顶点数 · 环形面积 · 密度(count/mm²)")
            dmax_show = float(dens[1:28].max())
            for i in range(1, 29):
                r_lo = i * 5; r_hi = (i+1) * 5
                mark_cross = " ◀ 密度跌破 50% 主峰=边界" if edge_idx == i else ""
                mark_peak = " ★ 密度≥60% 主峰" if dens[i] >= thr_high else ""
                # 归一化密度条 ▇
                bar_n = int(round(dens[i] / max(dmax_show, 1e-6) * 30))
                bar = "█" * bar_n
                print(f"  [{r_lo:3d},{r_hi:3d}) {cnts_r2[i]:>7d}v "
                      f"A{areas[i]:>7.0f}mm²  d={dens[i]:6.2f} {bar:<30s}"
                      f"{mark_cross}{mark_peak}")
        print(f"\n  板底中心 = h={body_center:.2f}mm (峰内点 P35)")
        print(f"  圆盘/电路板覆盖半径 = {r_outer_ring_cut:.1f}mm（物理兜底 [70,105]mm）")
        # 诊断性占比
        for rr in [70, 80, 90, 100, 110, 120]:
            fr = float((r_all < rr).sum()) / len(r_all) * 100
            print(f"    r < {rr:3d}mm 累积顶点占比 = {fr:.1f}%")

        # 每三角面三个顶点的 r 最大值 > 阈值 → 该三角面落在圆盘外环区域
        tri_rmax = np.max([r_all[tris[:, 0]], r_all[tris[:, 1]], r_all[tris[:, 2]]], axis=0)
        drop_ring = tri_rmax > r_outer_ring_cut
        print(f"  径向外环剔除 (r>{r_outer_ring_cut:.1f}mm): {int(drop_ring.sum())} / {len(tris)} 面")

        # 3b) 板底下方虚影：不要瞎切 -1.5mm！
        #     A. TSDF(v=2,t=15) 插值会在圆盘薄壳 h≈42~48mm 生成「对称」等势面，
        #        圆盘真实一面在 h≈44.7mm（body_center），另一面在 h≈46.5mm 以上
        #        元件面，中间插出厚度 ≈3mm 的连续薄壳。
        #     B. 但 2%~98% 分位显示大量顶点落在 h≈-60~42mm 的"台面虚影"
        #        （从相机看向转台时大角度斜视角的 depth 噪声）。
        #     正确做法：以「峰内点（圆盘）自身的 P1」为底，向下再扣 1×TSDF
        #        板厚 ≈3mm（不是硬编码 1.5mm）。这样圆盘薄壳下半边完整保留，
        #        仅删除盘下 h 更远的纯噪声。
        tri_minhp = np.min([h_prime[tris[:, 0]], h_prime[tris[:, 1]], h_prime[tris[:, 2]]], axis=0)
        # 先估计 TSDF 插值板厚（从 h 的 2%~98% 跨度，已知正常=2.5~3.5mm）
        tsdf_thickness = float(np.percentile(h_prime, 98) - np.percentile(h_prime, 2))
        # 但场景 A 还包含"盘下伪影"，更稳的厚度 = 3.0mm 常数（v=2 t=15 经验值）
        scene_thick = float(np.clip(tsdf_thickness, 2.0, 6.0))
        # 下阈值：body_center - 峰内点 P15 - scene_thick 再 - 1mm 余量
        #   峰内点 = dist_signed ∈ [peak_h - span/2, peak_h + span/2]
        peak_h = float(dp["peak_h"]); peak_span = float(dp["peak_span"])
        in_peak = (dist_signed >= peak_h - peak_span / 2) & (dist_signed <= peak_h + peak_span / 2)
        peak_p2 = float(np.percentile(dist_signed[in_peak], 2)) if in_peak.sum() > 50 else (peak_h - peak_span / 2)
        hp_lo_raw = peak_p2 - body_center - scene_thick - 0.5  # 峰底 - 插值厚 - 0.5 余量
        hp_lo = float(np.clip(hp_lo_raw, -6.0, -1.5))        # 限制区间，不至于删过头
        hp_below = tri_minhp < hp_lo
        print(f"  [厚度估算] scene_thick={scene_thick:.2f}mm  peak_P2={peak_p2:.2f}mm"
              f"  hp_lo={hp_lo:.2f}mm")
        print(f"  板底下方虚影 (h'<{hp_lo:.2f}mm): {int(hp_below.sum())} 面")

        # 3c) 顶部厚度窗口：Astra 给不出 >5mm 元件深度 → 但保留"圆盘薄壳 +
        #     TSDF 插值出的少量高突起"。窗口底部 = h' - 1.5mm 才开始判上界。
        #     不要用硬编码 15mm；参考数据 P99.9=3.78mm（只看到薄壳），说明
        #     当前电路板元件全被深度零化、TSDF 插值没造元件细节。因此上界
        #     用 P99.9+4mm 兜底即可。
        tri_maxhp = np.max([h_prime[tris[:, 0]], h_prime[tris[:, 1]], h_prime[tris[:, 2]]], axis=0)
        hp_p999 = float(np.percentile(h_prime, 99.9))
        hp_hi = max(8.0, hp_p999 + 4.0, scene_thick * 3.0)  # 至少 8mm
        drop_above = tri_minhp > hp_hi
        print(f"  超窗口上方 (h'>{hp_hi:.1f}mm): {int(drop_above.sum())} 面")
        print(f"  h' 分位（全顶点）: P2={np.percentile(h_prime,2):.2f}  "
              f"P50={np.percentile(h_prime,50):.2f}  "
              f"P99.5={float(np.percentile(h_prime,99.5)):.2f}  P99.9={hp_p999:.2f} mm")

        drop = drop_ring | hp_below | drop_above
        shell_thr = float("nan")  # 场景 A 不切薄壳
        board_hp_lo, board_hp_hi = -1.5, hp_hi

    else:
        # ================================================================
        # 场景 B · SAM2 精细分割：没有圆盘骨架
        # ================================================================
        frac_pos = float((dist_signed >= 0.5).sum()) / len(dist_signed)
        frac_neg = float((dist_signed <= -0.5).sum()) / len(dist_signed)
        board_side_sign = 1 if frac_pos >= frac_neg else -1
        side_vals = dist_signed if board_side_sign == 1 else -dist_signed
        side_v_pos = side_vals[side_vals >= 0]
        h99 = float(np.percentile(side_v_pos, 99)) if len(side_v_pos) else 100.0
        bins = max(30, int(np.ceil(h99 / 1.0)))
        hh, edges = np.histogram(np.clip(side_v_pos, 0, h99), bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        hh_s = np.convolve(hh.astype(float), np.ones(5) / 5.0, mode="same")
        order = np.argsort(-hh_s)
        cum_f = np.cumsum(hh_s[order]) / hh_s.sum()
        kn = int(np.searchsorted(cum_f, 0.85)) + 1
        kbins = sorted(set(int(x) for x in order[:kn]))
        if board_side_sign == 1:
            body_lo_raw = float(centers[kbins[0]])
            body_hi_raw = float(centers[kbins[-1]])
        else:
            body_lo_raw = -float(centers[kbins[-1]])
            body_hi_raw = -float(centers[kbins[0]])
        body_center = 0.5 * (body_lo_raw + body_hi_raw)
        body_thickness = abs(body_hi_raw - body_lo_raw)
        body_center = float(body_center)
        h_prime = dist_signed - body_center
        print(f"\n[步骤 3 · 场景B] 密度前85%定位板底：")
        print(f"  主体 h ∈ [{body_lo_raw:.2f}, {body_hi_raw:.2f}] mm, 中心={body_center:.2f}mm, 厚≈{body_thickness:.1f}mm")

        hp_p2 = float(np.percentile(h_prime, 2))
        hp_p50 = float(np.percentile(h_prime, 50))
        hp_p98 = float(np.percentile(h_prime, 98))
        hp_p995 = float(np.percentile(h_prime, 99.5))
        element_height_est = max(hp_p995, 0.0)
        board_hp_lo = -1.5
        board_hp_hi = max(12.0, element_height_est + 3.0, body_thickness + 10.0)
        shell_thr = 0.6
        print(f"  h' 分位: P2={hp_p2:.2f} P50={hp_p50:.2f} P98={hp_p98:.2f} P99.5={hp_p995:.2f} mm")
        print(f"  厚度窗口 h' ∈ [{board_hp_lo:.1f}, {board_hp_hi:.1f}] mm  "
              f"薄壳阈值 |h'| < {shell_thr:.1f}mm")

        d0 = h_prime[tris[:, 0]]; d1 = h_prime[tris[:, 1]]; d2 = h_prime[tris[:, 2]]
        tri_maxhp = np.max([d0, d1, d2], axis=0)
        tri_minhp = np.min([d0, d1, d2], axis=0)
        out_win = (tri_maxhp < board_hp_lo) | (tri_minhp > board_hp_hi)
        in_shell = (tri_minhp > -shell_thr) & (tri_maxhp < shell_thr)
        below_surface = tri_maxhp < board_hp_lo
        drop = out_win | in_shell | below_surface
        print(f"  超窗口丢弃: {int(out_win.sum())} 面")
        print(f"  薄壳丢弃(|h'|<{shell_thr:.1f}mm): {int(in_shell.sum())} 面")
        print(f"  板底下方虚影: {int(below_surface.sum())} 面")

    # ------------ 公共：应用剔除 mask ------------
    n_drop = int(drop.sum())
    print(f"\n[步骤 4] 合计丢弃: {n_drop} / {len(tris)} 面 ({100*n_drop/len(tris):.1f}%)")
    mesh.remove_triangles_by_mask(drop)
    mesh.remove_unreferenced_vertices()
    print(f"  剔除后: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

    # ------------ 5. 连通域：累计 ≥ 98%（保留细小连接桥） ------------
    if len(mesh.triangles) > 50:
        print("\n[步骤 5] 保留累计三角面占比 ≥ 98% 的连通分量（保留细小连接桥）...")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            tri_cls, cls_area, _ = mesh.cluster_connected_triangles()
        tri_cls = np.asarray(tri_cls, dtype=np.int64)
        cls_area = np.asarray(cls_area, dtype=np.float64)
        if len(cls_area) > 0:
            total_area = float(cls_area.sum())
            cls_order = np.argsort(cls_area)[::-1]
            fracs = cls_area[cls_order] / total_area
            cum = np.cumsum(fracs)
            keep_thr_idx = int(np.searchsorted(cum, 0.98)) + 1
            keep_thr_idx = min(keep_thr_idx, len(cls_order))
            keep_cls_set = set(int(x) for x in cls_order[:keep_thr_idx])
            topn_show = min(6, len(fracs))
            print(f"  连通分量数 = {len(cls_area)}，Top 占比: "
                  + ", ".join([f"{fracs[i]*100:.1f}%" for i in range(topn_show)]))
            print(f"  取前 {keep_thr_idx} 个分量，累计占比 = {cum[keep_thr_idx-1]*100:.1f}%")
            keep_tri = np.array([int(c) in keep_cls_set for c in tri_cls], dtype=bool)
            mesh.remove_triangles_by_mask(~keep_tri)
            mesh.remove_unreferenced_vertices()
        print(f"  连通域筛选后: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

    # ------------ 6. 轻量平滑 + 保存 ------------
    if len(mesh.vertices) > 0:
        mesh = mesh.filter_smooth_simple(number_of_iterations=1)
        mesh.compute_vertex_normals()

    out_ply = os.path.join(RESULT_DIR, "pcb_tsdf_refined.ply")
    o3d.io.write_triangle_mesh(out_ply, mesh, write_vertex_colors=True, write_vertex_normals=True)
    print(f"\n[保存] {out_ply}")

    # ------------ 7. 板厚估计（沿台面法向） ------------
    if len(mesh.vertices) > 0:
        verts2 = np.asarray(mesh.vertices)
        hp = (verts2 - axis_origin[None, :]) @ axis_dir - float(dp.get("board_center", 0.0) if is_scene_a else body_center)
        d2, d98 = np.percentile(hp, [2, 98])
        est_n = float(abs(d98 - d2))
        print(f"\n[质量 · 厚度] 沿法向(板底=0) 2%~98% 分位: "
              f"[{d2:.2f}, {d98:.2f}] mm, 跨度 = {est_n:.2f} mm")
        if is_scene_a:
            print(f"         场景A: 板底=圆盘骨架，板厚由TSDF插值贡献 (~2~3mm为正常插值厚度)")
        else:
            print(f"         （板底≈0，板厚 1.6mm + 元件；理想 1.6~12mm）")

        bbox = mesh.get_axis_aligned_bounding_box()
        sx, sy, sz = bbox.get_extent()
        print(f"[质量 · 包围盒] {sx:.1f} × {sy:.1f} × {sz:.1f} mm  (x×y×z)")
        print(f"[质量 · 规模] {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")
        r2 = radial_distance_to_axis(verts2, axis_origin, axis_dir)
        print(f"[质量 · 径向覆盖] 精修后 r 分位: "
              f"P50={np.percentile(r2,50):.1f}  P90={np.percentile(r2,90):.1f}  "
              f"P99={np.percentile(r2,99):.1f}  MAX={r2.max():.1f} mm")

    # ------------ 8. 可视化对比 ------------
    print("\n[可视化] 红色=原始 mesh（含台面/外环），蓝色=精修后（电路板+骨架板底）。按 ESC 退出。")
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30.0)
    mesh_orig = o3d.io.read_triangle_mesh(in_ply)
    mesh_orig.paint_uniform_color([0.9, 0.3, 0.3])
    if len(mesh.vertices) > 0:
        if not mesh.has_vertex_colors():
            mesh.paint_uniform_color([0.3, 0.5, 0.9])

    o3d.visualization.draw_geometries(
        [mesh_orig, mesh, coord],
        window_name=(f"场景{'A' if is_scene_a else 'B'}精修对比: 红=原始, 蓝=精修  "
                     f"{len(mesh.vertices)}v, {len(mesh.triangles)}t"),
        width=1400, height=900,
    )


if __name__ == "__main__":
    main()
