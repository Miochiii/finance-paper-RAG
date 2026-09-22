# -*- coding: utf-8 -*-
"""viz_report.py —— 把实验结果画成论文/汇报用图（纯 Pillow，不引入 matplotlib 依赖）。

输出到 results/figs/：
  fig1_e1_matrix.png       检测器 × 漂移目标 对照矩阵（检出率热力图 + EDD 标注）
  fig2_trajectory.png      指标原始曲线 vs 累积检测统计量（变点 / 阈值 / 报警点）
  fig3_false_alarm.png     同一阈值下各融合方式的误报比例（数值实验）
  fig4_proxy_validity.png  无标注代理指标的有效性（批次内 ρ + Holm 校正）
  fig5_cost_and_dilution.png  融合的延迟代价（Δ=0.4）与弱漂移下的稀释（Δ=0.2）

用法：
  python viz_report.py                 # 全部重画
  python viz_report.py --only fig1,fig2
数据来源：results/ 下的 edetector_*.csv 与 proxy_validity_*.csv（自动取最新一份）。
"""
import argparse
import csv
import glob
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_core.config import PROJECT_DIR  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGS_DIR = os.path.join(RESULTS_DIR, "figs")

# 默认数据源**固定**为报告引用的那次运行：results/ 下还有许多「参数探针」跑批，
# 按时间取最新会拿到缺 E1 行的探针结果（曾导致对照矩阵全 0、EDD 为 inf 而崩）。
# 要画别的运行用 --e1-csv / --series-csv / --validity-csv 覆盖。
E1_CSV = "edetector_20260922_2024.csv"        # E1 主口径：用 E5 推荐 m 复跑的那次
E1_Q85_CSV = "edetector_20260921_1955.csv"    # 旧口径（各指标统一 q85），用于对比 m 的影响
SERIES_CSV = "edetector_series_20260921_1955.csv"
VALIDITY_CSV = "proxy_validity_hmm_20260921_1928.csv"
E5_CSV = "edetector_20260922_2023.csv"        # E5 扫描结果（含 e5_curve / e5_mix 行）
E2_CSV = "edetector_20260922_2032.csv"        # E2 归因结果（含 e2_attr 行）

# ---- 配色（色盲友好的深浅对比）----
BG = (255, 255, 255)
INK = (33, 33, 33)
MUTED = (117, 117, 117)
GRID = (226, 232, 240)
PANEL = (246, 248, 251)
GOOD = (46, 125, 50)
GOOD_L = (200, 230, 201)
BAD = (198, 40, 40)
BAD_L = (255, 205, 210)
BLUE = (21, 101, 192)
BLUE_L = (187, 222, 251)
ORANGE = (239, 108, 0)
PURPLE = (106, 27, 154)
GRAY = (158, 158, 158)

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",    # 黑体
    r"C:\Windows\Fonts\simsun.ttc",    # 宋体
]
_FONT_CACHE = {}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    order = FONT_CANDIDATES[1:] + FONT_CANDIDATES[:1] if bold else FONT_CANDIDATES
    for path in order:
        if os.path.isfile(path):
            try:
                f = ImageFont.truetype(path, size)
                _FONT_CACHE[key] = f
                return f
            except OSError:
                continue
    f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


class Canvas:
    """极简绘图画布：坐标变换 + 文本/矩形/折线/圆点 + 虚线。"""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w, h), BG)
        self.d = ImageDraw.Draw(self.img)

    # ---- 基础 ----
    def text(self, x, y, s, size=20, color=INK, bold=False, anchor="la", line_gap=6):
        f = font(size, bold)
        if "\n" not in s:
            self.d.text((x, y), s, font=f, fill=color, anchor=anchor)
            return
        lh = size + line_gap
        for i, line in enumerate(s.split("\n")):
            self.d.text((x, y + i * lh), line, font=f, fill=color, anchor=anchor)

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1):
        self.d.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=width)

    def line(self, pts, color=INK, width=2, dash=None):
        if not dash:
            self.d.line(pts, fill=color, width=width, joint="curve")
            return
        # 手工虚线
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            seg = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            if seg <= 0:
                continue
            n = max(1, int(seg // (dash * 2)))
            for k in range(n + 1):
                a = k * 2 * dash / seg
                b = min(1.0, (k * 2 * dash + dash) / seg)
                if a >= 1.0:
                    break
                self.d.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a),
                             (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)],
                            fill=color, width=width)

    def circle(self, x, y, r, fill=INK, outline=None, width=2):
        self.d.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=outline, width=width)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.img.save(path)
        return path


class Panel:
    """一个坐标区：像素框 + 线性/对数 y 轴 + 网格与刻度。"""

    def __init__(self, c: Canvas, box, xlim, ylim, title="", ylabel="", logy=False):
        self.c, self.box = c, box
        self.x0, self.y0, self.x1, self.y1 = box
        self.xlim, self.ylim = xlim, ylim
        self.logy = logy
        self.title, self.ylabel = title, ylabel

    def px(self, x):
        (a, b) = self.xlim
        return self.x0 + (x - a) / (b - a) * (self.x1 - self.x0)

    def py(self, y):
        import math
        a, b = self.ylim
        if self.logy:
            a, b, y = math.log10(max(a, 1e-6)), math.log10(max(b, 1e-6)), math.log10(max(y, 1e-6))
        return self.y1 - (y - a) / (b - a) * (self.y1 - self.y0)

    def grid(self, yticks, xticks, xlabels=None, ylabels=None, ysize=17, xsize=17):
        c = self.c
        c.rect(self.x0, self.y0, self.x1, self.y1, fill=PANEL)
        for y in yticks:
            yy = self.py(y)
            c.line([(self.x0, yy), (self.x1, yy)], color=GRID, width=1)
        for i, x in enumerate(xticks):
            xx = self.px(x)
            c.line([(xx, self.y0), (xx, self.y1)], color=GRID, width=1)
            lab = (xlabels[i] if xlabels else str(x))
            c.text(xx, self.y1 + 8, lab, size=xsize, color=MUTED, anchor="ma")
        for i, y in enumerate(yticks):
            lab = (ylabels[i] if ylabels else (f"{y:g}"))
            c.text(self.x0 - 10, self.py(y), lab, size=ysize, color=MUTED, anchor="rm")
        c.rect(self.x0, self.y0, self.x1, self.y1, outline=(203, 213, 225), width=1)
        if self.title:
            c.text(self.x0, self.y0 - 34, self.title, size=21, bold=True)
        if self.ylabel:
            c.text(self.x0 - 10, self.y0 - 34, self.ylabel, size=17, color=MUTED, anchor="ra")

    def hline(self, y, color=BAD, width=2, dash=8):
        self.c.line([(self.x0, self.py(y)), (self.x1, self.py(y))], color=color, width=width, dash=dash)

    def vline(self, x, color=BAD, width=2, dash=8):
        self.c.line([(self.px(x), self.y0), (self.px(x), self.y1)], color=color, width=width, dash=dash)

    def plot(self, xs, ys, color=BLUE, width=3):
        self.c.line([(self.px(x), self.py(y)) for x, y in zip(xs, ys)], color=color, width=width)

    def plot_clip(self, xs, ys, ymax, color=BLUE, width=3):
        """画到超过 ymax 为止——直接 np.minimum 截断会在顶部留一条假的水平线。"""
        pts = []
        for x, y in zip(xs, ys):
            if y > ymax:
                break
            pts.append((self.px(x), self.py(y)))
        if len(pts) >= 2:
            self.c.line(pts, color=color, width=width)
        return pts[-1][0] if pts else None


# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------
def latest(pattern: str, required: str = "", required_values=()) -> str:
    """取最新的、且**确实含所需行**的结果文件（跳过参数探针跑批）。

    required：必须出现在表头里的列名；required_values：某列必须出现的取值集合。
    """
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)), reverse=True)
    if not files:
        raise FileNotFoundError(f"找不到 results/{pattern}")
    for path in files:
        if not required:
            return path
        try:
            with open(path, encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
        except (OSError, UnicodeDecodeError):
            continue
        if not rows or required not in rows[0]:
            continue
        if required_values and not any(r.get("experiment") in required_values for r in rows):
            continue
        return path
    raise FileNotFoundError(f"results/{pattern} 里没有含 {required} 的可用文件")


def pick(pinned: str, pattern: str, required: str = "", required_values=()) -> str:
    """优先用固定文件；不存在时回退到「最新的、含所需行」的文件。"""
    path = os.path.join(RESULTS_DIR, pinned)
    if os.path.isfile(path):
        return path
    return latest(pattern, required, required_values)


def load_e1_edd(path=None):
    """读了 (融合方式, 漂移目标, Δ) → (检出率, EDD均值) 的字典，并返回文件路径。"""
    path = path or pick(E1_CSV, "edetector_2*.csv", "experiment", ("e1_edd",))
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("experiment") != "e1_edd":
                continue
            em = (r.get("edd_mean") or "").strip()
            out[(r["fuse"], r.get("drift_target", ""), float(r["delta"]))] = (
                float(r["detect_rate"]), float("inf") if em in ("", "inf") else float(em))
    if not out:
        raise ValueError(f"{path} 里没有 E1（e1_edd）结果行，请用 --e1-csv 指定正确的跑批结果")
    return out, path


def load_series(path=None):
    path = path or pick(SERIES_CSV, "edetector_series_2*.csv", "change_point")
    cols = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                cols.setdefault(k, []).append(float(v))
    for need in ("t", "x", "M_mix", "threshold", "change_point", "alarm_t_mix"):
        if need not in cols:
            raise ValueError(f"{path} 缺少列 {need}，请用 --series-csv 指定正确的轨迹文件")
    return cols, path


def load_validity(path=None):
    path = path or pick(VALIDITY_CSV, "proxy_validity_hmm_2*.csv", "rho_within")
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            p = (r.get("p_within_holm") or "").strip()
            if p in ("", "1.0"):
                continue
            if float(p) < 0.05:
                rows.append({"proxy": r["proxy"], "truth": r["truth"],
                             "rho": float(r["rho_within"]), "p": float(p),
                             "v1": float(r["rho_v1_人工出题"]), "v2": float(r["rho_v2_块锚定"])})
    rows.sort(key=lambda x: -abs(x["rho"]))
    return rows, path


PROXY_CN = {
    "ans_refusal": "拒答标志", "ret_doc_hhi": "检索集中度 HHI",
    "ret_unique_docs": "检索唯一文档数", "cons_top1_agree": "改写后 top-1 一致",
    "cons_doc_jaccard_p2": "改写后 P@2 重叠",
}
TRUTH_CN = {
    "judge_corr": "judge 正确性", "judge_faith": "judge 忠实性",
    "doc_hit": "gold 文献被召回", "quality_composite": "综合质量分",
}


# --------------------------------------------------------------------------
# 图 1：检测器 × 漂移目标 对照矩阵
# --------------------------------------------------------------------------
def fig1_matrix(root=FIGS_DIR):
    e1, src = load_e1_edd()
    delta = 0.4
    rows = [("单指标#0", "只盯「拒答」的检测器"),
            ("单指标#1", "只盯「集中度」的检测器"),
            ("单指标#2", "只盯「分散度」的检测器"),
            ("全指标混合", "三指标凸混合")]
    cols = [("仅指标#0", "漂移打在\n拒答"), ("仅指标#1", "漂移打在\n集中度"),
            ("仅指标#2", "漂移打在\n分散度")]
    c = Canvas(1320, 780)
    c.text(60, 44, "漂移打在哪个指标上，谁能抓到？", size=36, bold=True)
    c.text(60, 100, f"α_edd=1e-3（报警线 1000）· Δ={delta} · 200 次重复 · 20 步滑窗 · "
                    f"检出只认变点后 300 步内", size=19, color=MUTED)
    x0, y0, cw, chh = 430, 220, 250, 100
    # 列头
    for j, (_, lab) in enumerate(cols):
        c.text(x0 + j * cw + cw / 2, y0 - 62, lab, size=20, color=INK, anchor="ma", bold=True)
    for i, (fuse, lab) in enumerate(rows):
        c.text(x0 - 24, y0 + i * chh + chh / 2 - 12, lab, size=20, anchor="ra")
        for j, (tgt, _) in enumerate(cols):
            rate, edd = e1.get((fuse, tgt, delta), (0.0, float("inf")))
            bx0, by0 = x0 + j * cw, y0 + i * chh
            bx1, by1 = bx0 + cw - 12, by0 + chh - 12
            if rate >= 0.5:
                k = min(1.0, max(0.0, rate))
                fill = (int(232 - 150 * k), int(245 - 100 * k), int(233 - 130 * k))
                outline = GOOD
            else:
                fill = BAD_L if rate > 0 else (241, 241, 241)
                outline = BAD if rate > 0 else GRID
            c.rect(bx0, by0, bx1, by1, fill=fill, outline=outline, width=3)
            c.text((bx0 + bx1) / 2, by0 + 20, f"检出 {rate:.2f}", size=26, bold=True,
                   color=GOOD if rate >= 0.5 else (BAD if rate > 0 else MUTED), anchor="ma")
            etxt = "未检出" if edd == float("inf") else f"{edd:.1f} 步报警"
            c.text((bx0 + bx1) / 2, by0 + 56, etxt, size=18,
                   color=MUTED if edd == float("inf") else INK, anchor="ma")
    # 结论区
    # 结论里的百分比按数据算，避免改了 m 之后文字与图不一致（踩过）
    ratios = []
    for i in range(3):
        e_s = e1.get((rows[i][0], cols[i][0], delta), (0, float("inf")))[1]
        e_m = e1.get(("全指标混合", cols[i][0], delta), (0, float("inf")))[1]
        if e_s not in (0, float("inf")) and e_m != float("inf"):
            ratios.append(e_m / e_s)
    cost = (f"慢 {100 * (min(ratios) - 1):.0f}%–{100 * (max(ratios) - 1):.0f}%"
            if ratios else "略慢")
    c.rect(60, 660, 1260, 740, fill=(232, 245, 233), outline=GOOD, width=2)
    c.text(80, 676, f"只盯一个指标：3/9 组合检出（只有对角线）；三指标凸混合：3/3 全部检出，"
                    f"代价是比「事先知道打在哪」的理想单指标{cost}",
           size=20, color=INK)
    out = os.path.join(root, "fig1_e1_matrix.png")
    c.save(out)
    return out, src


# --------------------------------------------------------------------------
# 图 2：原始指标 vs 累积统计量
# --------------------------------------------------------------------------
def fig2_trajectory(root=FIGS_DIR, proxy="ret_doc_hhi", delta=0.15, n_streams=5, T_post=300,
                    m_override: float = 0.441):
    """现场按 E1 的口径模拟一条轨迹：指标原始曲线 vs 累积统计量。

    直接用 edetector 的构件（校准段定尺度 → 退化方向翻正 → 20 步滑窗 → q85 上界 →
    污染注入 → 凸混合递推），保证图与 E1 表格同源同口径；跑 5 条流，突出中位那条。

    Δ 取 0.15 而非 E1 主表的 0.4：这张图的目的是展示「指标只是悄悄变差一点，
    累积量却能稳定报警」，Δ=0.4 会把 40% 以上的观测直接推到最差状态，指标本身就很刺眼，
    反而讲不出「看不出来」这件事。
    """
    import numpy as np
    import edetector as ed

    rows = ed.load_proxy_matrix(need=[proxy])[1]
    X, _ = ed.series_matrix(rows, [proxy])
    rng = np.random.default_rng([0, 7])
    X = X[rng.permutation(X.shape[0])]
    n_calib = max(8, int(X.shape[0] * 0.2))
    u = ed.fit_unit(X[:n_calib, 0], X[:, 0])
    if ed.DEGRADE_DIRECTION.get(proxy, +1) < 0:      # 翻正：越高越差
        u = 1.0 - u
    u = ed.rolling_mean(u, 20)
    # 上界取 E5 曲线给出的推荐点（q85 在这条序列上过于保守，Δ 小时根本检不出）
    m = float(m_override) if m_override else max(ed.estimate_m(u[:n_calib], "q85"), 0.05)
    pool = u[n_calib:]
    lams = ed.lambda_grid(m)
    thr = 1000.0
    cp = 120
    streams, Ms = [], []
    for _ in range(n_streams):
        pre = ed.block_bootstrap(pool, 1, cp, 20, rng).ravel()
        post = ed.block_bootstrap(pool, 1, T_post, 20, rng).ravel()
        s = ed.inject_drift(np.concatenate([pre, post]), at=cp, delta=delta,
                            direction=+1, mode="contaminate", rng=rng)
        streams.append(s)
        Ms.append(ed.build_detectors(s, m, lams, "mix")[0])
    alarms = [int(ed.first_alarm(M, thr)[0]) for M in Ms]
    idx = int(np.argsort(alarms)[len(alarms) // 2])       # 取报警步数中位的那条
    n = len(streams[0])
    ymax = 1e6                                           # 留足量程，避免统计量冲出坐标区
    c = Canvas(1320, 840)
    c.text(60, 40, "为什么必须用「累积」统计量：肉眼判不准的变化，累积量能稳定报警",
           size=32, bold=True)
    c.text(60, 92, f"示例：{PROXY_CN.get(proxy, proxy)}指标 · 20 步滑窗 · 上界 m={m:.3f}（q85）"
                   f"· Δ={delta}（约 {delta:.0%} 的查询变差）· 变点在第 {cp} 步 · 共 {n} 步 · "
                   f"{n_streams} 条独立流", size=19, color=MUTED)

    p1 = Panel(c, (170, 200, 1250, 400), (0, n), (0, 1.05), title="", ylabel="指标值")
    p1.grid([0.0, 0.25, 0.5, 0.75, 1.0], [0, 100, 200, 300, 400],
            ylabels=["0", "0.25", "0.5", "0.75", "1.0"])
    p1.vline(cp, color=BAD, width=3, dash=10)
    for k, s in enumerate(streams):
        if k != idx:
            p1.plot(range(n), np.minimum(s, 1.05), color=(206, 214, 224), width=2)
    p1.plot(range(n), np.minimum(streams[idx], 1.05), color=GRAY, width=3)
    c.text(170, 166, "① 指标本身（已翻正：越高越差）——变点后确实差了一点，但噪声盖住了，肉眼判不准",
           size=21, bold=True)
    c.rect(p1.px(cp) + 12, p1.y0 + 8, p1.px(cp) + 210, p1.y0 + 38, fill=BG)
    c.text(p1.px(cp) + 20, p1.y0 + 13, f"第 {cp} 步开始变差", size=18, color=BAD)

    p2 = Panel(c, (170, 520, 1250, 740), (0, n), (0.5, ymax), logy=True, title="", ylabel="M（对数刻度）")
    p2.grid([1, 10, 100, 1000, 10000, 100000], [0, 100, 200, 300, 400],
            ylabels=["1", "10", "100", "1k", "10k", "100k"])
    p2.vline(cp, color=BAD, width=3, dash=10)
    for k, M in enumerate(Ms):
        if k != idx:
            p2.plot_clip(range(n), M, ymax, color=(206, 214, 224), width=2)
    p2.hline(thr, color=BAD, width=3, dash=10)
    endx = p2.plot_clip(range(n), Ms[idx], ymax, color=BLUE, width=3)
    c.text(170, 486, "② 累积检测统计量：越过报警线就报警", size=21, bold=True)
    c.rect(p2.x0 + 8, p2.py(thr) - 34, p2.x0 + 190, p2.py(thr) - 6, fill=BG)
    c.text(p2.x0 + 16, p2.py(thr) - 30, f"报警线 1/α = {thr:.0f}", size=19, color=BAD)
    a = alarms[idx]
    if a > 0:
        c.circle(p2.px(a), p2.py(thr), 8, fill=BLUE)
        c.rect(p2.px(a) + 14, p2.py(thr) + 12, p2.px(a) + 330, p2.py(thr) + 46, fill=BG)
        c.text(p2.px(a) + 20, p2.py(thr) + 16,
               f"第 {a} 步报警（变点后 {a - cp} 步）", size=20, color=BLUE, bold=True)
    else:
        c.text(p2.px(cp) + 16, p2.py(thr) + 16, f"{n} 步内未报警", size=20, color=BAD, bold=True)
    c.rect(170, 772, 1250, 834, fill=(255, 249, 224), outline=(255, 193, 7), width=2)
    c.text(188, 780, "滑动窗口比对：窗口一旦整体滑进「变差之后」的区间就再也分不出异常；"
                     "累积统计量把长期偏离攒起来，所以逃不掉。", size=19)
    c.text(188, 806, f"{n_streams} 条独立流的报警步数：" +
           "、".join(str(x) if x > 0 else "未报警" for x in alarms) + "（只有一条不响，其余都在变点后百步内）",
           size=17, color=MUTED)
    out = os.path.join(root, "fig2_trajectory.png")
    c.save(out)
    return out, f"{os.path.basename(E1_CSV)} 口径现场模拟（代理={proxy}，Δ={delta}）"


# --------------------------------------------------------------------------
# 图 3：误报比例（数值实验）
# --------------------------------------------------------------------------
def fig3_false_alarm(root=FIGS_DIR):
    """同一报警线下各融合方式的误报比例（横向条形）。

    注意：误报率是**横轴**，所以刻度属于 xticks（画在框下方）——
    早期版本当成 yticks，刻度文字挤到左侧正好压住行标签。
    """
    # 数据来源：log/E2_误报侧对照与m策略_20260921.md 探针 1
    # （α=1e-3、T=3000、500 次重复、变前流；该组 m 偏紧，属边界情形，排序与理论预测一致）
    data = [("三指标凸混合", 0.40, GOOD), ("只用一个指标", 0.61, ORANGE),
            ("取最大（谁先响听谁的）", 0.69, BAD), ("取最小", 0.00, GRAY)]
    c = Canvas(1430, 700)
    c.text(60, 44, "同一报警线下，哪种融合方式更容易「狼来了」", size=34, bold=True)
    c.text(60, 98, "α=1e-3（报警线 1000）· 没有变差的数据 · 3000 步内报警的比例 · 500 次重复",
           size=19, color=MUTED)
    p = Panel(c, (400, 200, 1180, 520), (0, 0.8), (0, 4))
    p.grid([], [0, 0.2, 0.4, 0.6, 0.8], xlabels=["0", "20%", "40%", "60%", "80%"])
    for i, (lab, v, col) in enumerate(data):
        y = len(data) - i - 0.5          # 凸混合画在最上面
        p.c.rect(p.x0, p.py(y + 0.34), p.px(v), p.py(y - 0.34), fill=col)
        p.c.text(p.x0 - 18, p.py(y) - 13, lab, size=21, anchor="ra")
        p.c.text(p.px(v) + 16, p.py(y) - 14, f"{v:.2f}", size=23, bold=True, color=col)
    c.text(1205, 250, "取最大的误报比例", size=22, color=BAD, bold=True)
    c.text(1205, 282, "≈ 凸混合的 1.7 倍", size=22, color=BAD, bold=True)
    c.text(1205, 330, "理论预期：取最大值不构成", size=18, color=MUTED)
    c.text(1205, 356, "合法检测器，没有误报保证", size=18, color=MUTED)
    c.rect(60, 588, 1370, 664, fill=(255, 249, 224), outline=(255, 193, 7), width=2)
    c.text(78, 600, "口径说明：本组用于展示「排序」——该设置下基线上界偏紧，属方法讨论中的边界情形，"
                    "不能当作合法性验证。", size=18)
    c.text(78, 628, "合规且保守的设置下，3000 步内四种方式都 0 次误报（平凡满足保证，但也没有分辨率）。",
           size=18)
    out = os.path.join(root, "fig3_false_alarm.png")
    c.save(out)
    return out, "log/E2_误报侧对照与m策略_20260921.md"

# --------------------------------------------------------------------------
def fig4_validity(root=FIGS_DIR):
    rows, src = load_validity()
    c = Canvas(1460, 1000)
    c.text(60, 44, "哪些「不用标注」的指标真的能反映质量", size=36, bold=True)
    c.text(60, 100, "166 道标注题 · 判定用批次内合并 Spearman ρ + 同族 Holm 校正 · "
                    "最小可检测效应 |ρ| ≥ 0.15", size=20, color=MUTED)
    c.text(60, 130, "图中为全部通过校正的 14 对「代理 × 真值」；空心圆 = 两个标注批次各自的 ρ（同号才算稳健）",
           size=19, color=MUTED)
    x0, y0, bw, bh, gap = 470, 236, 560, 32, 12
    p = Panel(c, (x0, y0 - 12, x0 + bw, y0 + len(rows) * (bh + gap)), (-0.46, 0.46), (0, 1))
    # 注意：ρ 是**横轴**（条形左右延伸），所以网格线是竖线、刻度在下方——
    # 早期版本误把 ρ 刻度画在左侧，正好压住行标签。
    p.grid([], [-0.4, -0.2, 0.0, 0.2, 0.4],
           xlabels=["-0.4", "-0.2", "0", "+0.2", "+0.4"])
    c.line([(p.px(0.0), p.y0), (p.px(0.0), p.y1)], color=(120, 130, 145), width=2)
    for i, r in enumerate(rows):
        y = y0 + i * (bh + gap) + bh / 2
        col = BLUE if r["rho"] < 0 else GOOD
        left, right = p.px(min(0.0, r["rho"])), p.px(max(0.0, r["rho"]))
        c.rect(left, y - bh / 2, right, y + bh / 2, fill=col)
        for v in (r["v1"], r["v2"]):
            c.circle(p.px(v), y, 5, fill=BG, outline=col, width=3)
        c.text(x0 - 22, y - 13,
               f"{PROXY_CN.get(r['proxy'], r['proxy'])} → {TRUTH_CN.get(r['truth'], r['truth'])}",
               size=20, anchor="ra")
        # 数值与 p 值各占一列，避免与批次点重叠
        c.text(x0 + bw + 34, y - 14, f"{r['rho']:+.3f}", size=21, bold=True, color=col)
        c.text(x0 + bw + 150, y - 12,
               "p < 0.001" if r["p"] < 0.001 else f"p = {r['p']:.3f}", size=18, color=MUTED)
    c.text(x0 + bw / 2, y0 - 52, "ρ（批次内合并，横轴）", size=19, color=MUTED, anchor="ma")
    c.text(x0 + bw + 150, y0 - 46, "Holm 校正后", size=19, color=MUTED)
    c.rect(60, 890, 1400, 976, fill=(245, 245, 245), outline=GRID, width=2)
    c.text(80, 902, "没通过的例子（如实报告）：① 长度类指标（答案长度、结论数）——是批次差异造出的伪信号；",
           size=19)
    c.text(80, 932, "② 检索相似度的散布类（标准差 / 四分位距）——|ρ| 全部 ≤ 0.11；"
                    "③ 早期 40 题上的 +0.48 扩样本后未复现。", size=19)
    out = os.path.join(root, "fig4_proxy_validity.png")
    c.save(out)
    return out, src


# --------------------------------------------------------------------------
# 图 5：延迟代价 与 弱漂移下的稀释
# --------------------------------------------------------------------------
def fig5_cost_dilution(root=FIGS_DIR):
    """左：融合带来的延迟代价（Δ=0.4）；右：m 选得太保守会损失多少检出力。

    右panel 原来是「弱漂移下的稀释」，但 E5 查出来那其实是 **m 取 q85 太保守**造成的
    ——换成推荐 m 后 Δ=0.2 三种漂移也都 1.00 检出。所以这一栏改成
    「旧口径 q85 vs 推荐 m」的直接对比，把那个结论更正过来。
    """
    e1, src = load_e1_edd()
    e1_old, _ = load_e1_edd(os.path.join(RESULTS_DIR, E1_Q85_CSV))
    c = Canvas(1400, 900)
    c.text(60, 44, "融合的代价很小；真正拖后腿的是上界 m 选得太保守", size=34, bold=True)
    c.text(60, 100, "左：Δ=0.4 时的报警延迟（越短越好）　右：Δ=0.2 时的检出率，"
                    "旧口径 q85 vs E5 推荐 m", size=19, color=MUTED)
    diag = ["单指标#0", "单指标#1", "单指标#2"]
    labs = ["拒答", "集中度", "分散度"]

    # --- 左：EDD 对比 ---
    p1 = Panel(c, (150, 250, 680, 620), (-0.4, 3.4), (0, 55), title="", ylabel="报警延迟（步）")
    p1.grid([0, 10, 20, 30, 40, 50], [1, 2, 3], xlabels=labs)
    c.text(150, 206, "① 报警延迟（Δ=0.4）：凸混合 vs 盯对指标的单指标", size=22, bold=True)
    for i in range(3):
        _, edd_s = e1.get((diag[i], f"仅指标#{i}", 0.4), (0, float("inf")))
        _, edd_m = e1.get(("全指标混合", f"仅指标#{i}", 0.4), (0, float("inf")))
        if edd_s == float("inf") or edd_m == float("inf"):
            print(f"[fig5] 跳过漂移目标 #{i}：EDD 未检出（{edd_s} / {edd_m}）")
            continue
        w = 0.34
        for off, edd, col in ((-w / 2, edd_s, GRAY), (w / 2, edd_m, BLUE)):
            x = i + 1 + off
            c.rect(p1.px(x - w / 2), p1.py(edd), p1.px(x + w / 2), p1.py(0), fill=col)
            c.text(p1.px(x), p1.py(edd) - 28, f"{edd:.1f}", size=20, bold=True, color=col, anchor="ma")
        c.text(p1.px(i + 1), p1.py(edd_m) - 58, f"+{100 * (edd_m / edd_s - 1):.0f}%", size=18,
               color=ORANGE, anchor="ma")
    c.rect(150, 676, 680, 744, fill=(232, 245, 233), outline=GOOD, width=2)
    c.text(168, 688, "融合只慢 12%–20%（灰=盯对指标的单指标，蓝=凸混合）", size=19, color=INK)
    c.text(168, 716, "换来的是「漂移类型未知」时仍然能报警的兜底能力。", size=19, color=INK)

    # --- 右：m 的影响（旧口径 q85 vs 推荐 m）---
    p2 = Panel(c, (860, 250, 1330, 620), (-0.4, 3.4), (0, 1.22), title="", ylabel="检出率（Δ=0.2）")
    p2.grid([0, 0.25, 0.5, 0.75, 1.0], [1, 2, 3], xlabels=labs,
            ylabels=["0", "0.25", "0.5", "0.75", "1.0"])
    c.text(860, 206, "② 弱漂移下的检出率：m 取 q85 的代价", size=22, bold=True)
    for i in range(3):
        old = e1_old.get(("全指标混合", f"仅指标#{i}", 0.2), (0, 0))[0]
        new = e1.get(("全指标混合", f"仅指标#{i}", 0.2), (0, 0))[0]
        w = 0.3
        for k, (v, col) in enumerate(((old, BAD), (new, GOOD))):
            x = i + 1 + (k - 0.5) * w
            c.rect(p2.px(x - w / 2), p2.py(v), p2.px(x + w / 2), p2.py(0), fill=col)
            c.text(p2.px(x), p2.py(v) - 24, f"{v:.2f}", size=16, bold=True, color=col, anchor="ma")
    lx, ly = 860, 648
    for col, lab in ((BAD, "旧口径：各指标统一 q85"), (GOOD, "E5 推荐 m（逐指标）")):
        c.rect(lx, ly, lx + 28, ly + 18, fill=col)
        c.text(lx + 36, ly - 2, lab, size=17)
        lx += 250
    c.rect(860, 690, 1330, 790, fill=(232, 245, 233), outline=GOOD, width=2)
    c.text(878, 702, "集中度那一列：检出率 0.15 → 1.00", size=18, color=INK)
    c.text(878, 728, "EDD 213 → 37 步；原先报告的「弱漂移下融合被稀释」", size=18, color=MUTED)
    c.text(878, 754, "其实是 m 过保守——E5 把它查了出来。", size=18, color=MUTED)
    c.rect(150, 812, 1330, 878, fill=(245, 248, 251), outline=GRID, width=2)
    c.text(168, 824, "读图：m 按 E5 曲线逐指标选（合法点里最紧的那个）之后，"
                     "Δ=0.2 和 Δ=0.4 都是「单指标 3/9、混合 3/3」。", size=19)
    c.text(168, 852, "结论：融合的代价只是慢一成多；影响结论的主因是上界选择——"
                     "这也是本文的方法学贡献点之一。", size=19, color=MUTED)
    out = os.path.join(root, "fig5_cost_and_dilution.png")
    c.save(out)
    return out, src

def is_inf_s(v) -> bool:
    """与 edetector.is_inf 同义（这里独立实现，避免为一个判断引入模块依赖）。"""
    if isinstance(v, str):
        return v.strip().lower() in ("", "inf", "infinity", "nan", "none")
    try:
        return not math.isfinite(float(v))
    except (TypeError, ValueError):
        return True


def _load_e5(pinned: str, experiment: str):
    """读 E5 扫描结果（experiment=e5_curve / e5_mix）。"""
    path = pick(pinned, "edetector_2*.csv", "experiment", (experiment,))
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("experiment") == experiment:
                rows.append(r)
    if not rows:
        raise ValueError(f"{path} 里没有 {experiment} 结果行，请用 --e5-csv 指定")
    return rows


def fig6_m_tradeoff(root=FIGS_DIR):
    """E5：m 的保守性与检测延迟的权衡曲线（每个代理一个面板）。

    横轴 m、纵轴 EDD；点色表示**在实际报警线（α_edd）下**的变前误报率——
    绿 ≤0.05（可用）、橙 ≤0.5、红 >0.5（功效列不可读）。竖直红线是合法性边界（变前均值），
    紫色虚线是 E1 当时用的 m，绿圈是按规则选出的推荐点。
    """
    rows = _load_e5(E5_CSV, "e5_curve")
    proxies = list(dict.fromkeys(r["proxy"] for r in rows))
    delta = "0.2"
    c = Canvas(1520, 880)
    c.text(60, 40, "m 选得越保守，报警越慢：E1 的 m 该落在哪里", size=34, bold=True)
    c.text(60, 96, f"E5：按校准段分位数扫 m（Δ={delta}、α_edd=1e-3、报警线 1000）；"
                   f"点色＝实际报警线下的变前误报率（绿 ≤0.05、橙 ≤0.5、红 >0.5）",
           size=19, color=MUTED)
    panel_w, gap, x0 = 440, 40, 130
    for k, p_ in enumerate(proxies):
        sub = sorted([r for r in rows if r["proxy"] == p_ and r["delta"] == delta],
                     key=lambda r: float(r["m"]))
        if not sub:
            continue
        bx0 = x0 + k * (panel_w + gap)
        box = (bx0, 250, bx0 + panel_w - 40, 650)
        ms = [float(r["m"]) for r in sub]
        pm = float(sub[0]["pool_mean"])
        # 纵轴用**截尾 EDD**：未检出的流按视界计入。用「只统计检出的流」的 EDD 均值，
        # 尾部会因为这些流只检出 1% 而假性下降，曲线形状会被误读（这一点在 E1 也踩过）。
        def _ey(r):
            v = r.get("edd_censored")
            return float(v) if (v not in ("", None) and not is_inf_s(v)) else 310.0
        ymax = 320.0
        mlo, mhi = min(ms + [pm]), max(ms + [pm])
        pad = max(0.008, (mhi - mlo) * 0.08)
        pan = Panel(c, box, (mlo - pad, mhi + pad), (0, ymax), title="", ylabel="EDD（步）")
        pan.grid([0, ymax / 4, ymax / 2, 3 * ymax / 4, ymax], [],
                 ylabels=[f"{v:.0f}" for v in (0, ymax / 4, ymax / 2, 3 * ymax / 4, ymax)])
        if pm > mlo - pad:
            c.rect(pan.px(mlo - pad), pan.y0, pan.px(pm), pan.y1, fill=(255, 235, 238))
            c.line([(pan.px(pm), pan.y0), (pan.px(pm), pan.y1)], color=BAD, width=2, dash=8)
            c.text(pan.px(pm) + 6, pan.y0 + 8, "合法性边界\nm = 变前均值", size=15, color=BAD)
        pts = [(pan.px(float(r["m"])), pan.py(_ey(r))) for r in sub]
        if len(pts) >= 2:
            c.line(pts, color=BLUE, width=3)
        for r in sub:
            ar = float(r["arl_alarm_rate_op"])
            col = GOOD if ar <= 0.05 else (ORANGE if ar <= 0.5 else BAD)
            det = float(r["detect_rate"])
            # 检出率不足 0.9 的点加个空心圈：它们是「报得早但常常报不出来」，不能算好点
            c.circle(pan.px(float(r["m"])), pan.py(_ey(r)), 7, fill=col,
                     outline=(INK if det < 0.9 else None), width=2)
        e1m = next((float(r["e1_m"]) for r in sub if r.get("e1_m") not in ("", None)), None)
        if e1m is not None:
            c.line([(pan.px(e1m), pan.y0), (pan.px(e1m), pan.y1)], color=PURPLE, width=2, dash=6)
            c.text(pan.px(e1m) - 6, pan.y0 + 8, "E1 用的 m", size=15, color=PURPLE, anchor="ra")
        ok = [(float(r["m"]), _ey(r)) for r in sub
              if float(r["arl_alarm_rate_op"]) <= 0.05 and float(r["detect_rate"]) >= 0.9]
        if ok:
            rm, re_ = min(ok)
            c.circle(pan.px(rm), pan.py(re_), 11, fill=None, outline=GOOD, width=4)
            dy = 22 if re_ > ymax * 0.5 else -30
            c.text(pan.px(rm), pan.py(re_) + dy, f"推荐 m={rm:.3f}", size=16, color=GOOD, anchor="ma")
        c.text(bx0, 206, PROXY_CN.get(p_, p_), size=23, bold=True)
        c.text(bx0, 672, f"变前均值 {pm:.3f}", size=17, color=MUTED)
    c.text(60, 716, "读法：越往右 m 越保守——误报更少，但 EDD 单调变长，到某个点后直接贴住视界"
                     "（300 步内检不出）。纵轴是截尾 EDD，未检出的流按视界计入。", size=20)
    c.text(60, 750, "所以选 m 不是「越大越安全」，而是在合法区间里挑最紧的那一个："
                    "绿点里最靠左、检出率仍 ≥ 0.9 的点（黑圈点＝检出率 < 0.9，不算）。", size=20)
    c.rect(60, 792, 1460, 850, fill=(237, 231, 246), outline=PURPLE, width=2)
    c.text(80, 804, "本次结论：E1 原先按惯例取 q85，对两个连续型指标都过于保守——"
                    "换成推荐 m 后 Δ=0.2 的检出率从 0.26/0.89 提到 1.00/1.00，", size=18)
    c.text(80, 830, "EDD 从 210/141 步降到 33/34 步；拒答指标的 m 本身已经是最优点。",
           size=18)
    out = os.path.join(root, "fig6_m_tradeoff.png")
    c.save(out)
    return out, os.path.basename(E5_CSV)


def fig7_attribution(root=FIGS_DIR):
    """E2：报警之后的漂移类型归因——四条规则的命中率对比（两种归因延迟）。

    横轴 = 漂移幅度 Δ，每个 Δ 下四根柱子是四条归因规则；虚线是随机猜（1/K）。
    """
    rows = _load_e5(E2_CSV, "e2_attr")
    rules = ["evalue", "recent", "zscore", "shift"]
    rule_cn = {"evalue": "e 值分解（累积）", "recent": "最近增量",
               "zscore": "z 检验（方差归一）", "shift": "窗口均值偏移（朴素）"}
    rule_col = {"evalue": GOOD, "recent": BLUE, "zscore": ORANGE, "shift": GRAY}
    delays = sorted({int(r["delay"]) for r in rows})
    deltas = sorted({float(r["delta"]) for r in rows})
    k_cls = max(1, int(float(rows[0].get("K", 3))))
    c = Canvas(1400, 830)
    c.text(60, 40, "报警之后，能不能指出是哪一个指标在退化？", size=34, bold=True)
    c.text(60, 96, f"E2：漂移只打在其中一个指标上，报警后判断是哪一个（{k_cls} 类，"
                   f"随机猜 {1.0 / k_cls:.2f}）；命中率只在真正报了警的流上统计",
           size=19, color=MUTED)
    for j, delay in enumerate(delays):
        bx0 = 140 + j * 660
        pan = Panel(c, (bx0, 250, bx0 + 560, 600), (-0.4, len(deltas) - 0.6), (0, 1.05),
                    title="", ylabel="归因命中率")
        pan.grid([0, 0.25, 0.5, 0.75, 1.0], list(range(len(deltas))),
                 xlabels=[f"Δ={d:g}" for d in deltas],
                 ylabels=["0", "0.25", "0.5", "0.75", "1.0"])
        c.text(bx0, 206, ("① 报警当下就判断" if delay == 0 else f"② 报警后再等 {delay} 步"),
               size=22, bold=True)
        w = 0.2
        for i, d in enumerate(deltas):
            for k, rule in enumerate(rules):
                vals = [float(r["acc"]) for r in rows
                        if r["rule"] == rule and int(r["delay"]) == delay
                        and abs(float(r["delta"]) - d) < 1e-9 and r["acc"] == r["acc"]]
                if not vals:
                    continue
                v = sum(vals) / len(vals)
                x = i + (k - 1.5) * w
                c.rect(pan.px(x - w / 2), pan.py(v), pan.px(x + w / 2), pan.py(0),
                       fill=rule_col[rule])
                c.text(pan.px(x), pan.py(v) - 22, f"{v:.2f}", size=14, bold=True,
                       color=rule_col[rule], anchor="ma")
        pan.hline(1.0 / k_cls, color=BAD, width=2, dash=7)
        c.text(pan.x1 - 4, pan.py(1.0 / k_cls) + 6, "随机猜", size=15, color=BAD, anchor="ra")
    lx, ly = 140, 630
    for rule in rules:
        c.rect(lx, ly, lx + 26, ly + 17, fill=rule_col[rule])
        c.text(lx + 34, ly - 2, rule_cn[rule], size=18)
        lx += 260
    c.rect(140, 682, 1340, 796, fill=(232, 245, 233), outline=GOOD, width=2)
    c.text(158, 694, "结论：用混合统计量自身的分解（各指标自己的 e 值）来归因，"
                     "在所有漂移幅度下都最好（0.98–1.00）；", size=19)
    c.text(158, 722, "朴素的「窗口均值偏移最大者」最差（小漂移时 0.62），"
                     "方差归一化的 z 检验居中（0.81–0.93）。", size=19)
    c.text(158, 750, "另一个发现：归因要趁报警当下做——等 50 步后「最近增量」从 1.00 掉到 0.48，"
                     "而累积 e 值几乎不掉（1.00→0.98）。", size=19, color=INK)
    c.text(158, 776, "口径：漂移是逐指标注入的，所以「归因」= 找出被注入的那个指标；"
                     "真实系统级漂移会让多个指标同时动，属于下一步。", size=17, color=MUTED)
    out = os.path.join(root, "fig7_attribution.png")
    c.save(out)
    return out, os.path.basename(E2_CSV)


FIGURES = {"fig1": fig1_matrix, "fig2": fig2_trajectory, "fig3": fig3_false_alarm,
           "fig4": fig4_validity, "fig5": fig5_cost_dilution, "fig6": fig6_m_tradeoff,
           "fig7": fig7_attribution}


def main() -> int:
    ap = argparse.ArgumentParser(description="实验结果可视化（输出 PNG 到 results/figs/）")
    ap.add_argument("--only", default="", help="只画指定图，如 fig1,fig2")
    ap.add_argument("--outdir", default=FIGS_DIR)
    args = ap.parse_args()
    want = [k.strip() for k in args.only.split(",") if k.strip()] or list(FIGURES)
    for k in want:
        fn = FIGURES.get(k)
        if not fn:
            print(f"未知图名：{k}（可选 {', '.join(FIGURES)}）")
            continue
        path, src = fn(args.outdir)
        print(f"OK {path}（数据源：{os.path.basename(src)}，{os.path.getsize(path)} 字节）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
