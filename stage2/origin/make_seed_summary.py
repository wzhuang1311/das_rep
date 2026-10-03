r"""
make_seed_summary.py —— 汇总多种子实验结果，生成统计与图表

输入：seed_runs\seed_*_result.log
输出：results\cnn_seed_summary.txt
      results\cnn_seed_results.csv
      results\cnn_seed_stats.png
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
SEED_DIR = os.path.join(HERE, "seed_runs")
OUT_DIR = r"D:\汕头大学\科研\深度学习\text.demo\das_rep\results"
os.makedirs(OUT_DIR, exist_ok=True)

PAPER_F1 = 0.940
NAMES_CN = ["背景噪声", "挖掘", "敲击", "浇水", "摇晃", "走动"]


def parse_log(seed):
    path = os.path.join(SEED_DIR, "seed_%d_result.log" % seed)
    text = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"RESULT seed=(\d+)\s+acc_final=([\d.]+)\s+acc_best=([\d.]+)"
                  r"\s+macroF1=([\d.]+)\s+acc_overall=([\d.]+)\s+time=([\d.]+)", text)
    if not m:
        return None
    blk = text.split("MATRIX")[1].strip().split("\n")
    rows = []
    for ln in blk:
        ln = ln.strip()
        if not ln or not all(c.isdigit() or c == "," for c in ln):
            break
        rows.append([int(x) for x in ln.split(",")])
    acc_hist = [float(a) for a in re.findall(r"Epoch\s+\d+/\d+\s+acc = ([\d.]+)", text)]
    return dict(seed=int(m.group(1)), final=float(m.group(2)), best=float(m.group(3)),
                f1=float(m.group(4)), overall=float(m.group(5)), time_s=float(m.group(6)),
                C=np.array(rows), hist=acc_hist)


def main():
    seeds = sorted(int(re.search(r"seed_(\d+)_result", f).group(1))
                   for f in os.listdir(SEED_DIR)
                   if re.match(r"seed_\d+_result\.log", f))
    runs = [parse_log(s) for s in seeds]
    runs = [r for r in runs if r]
    if not runs:
        print("没有解析到任何结果")
        return

    F = np.array([r["f1"] for r in runs])
    FIN = np.array([r["final"] for r in runs])
    BEST = np.array([r["best"] for r in runs])

    lines = []
    A = lines.append
    A("=" * 84)
    A(" CNN 多种子重复实验结果")
    A("=" * 84)
    A(" 目的：论文采用随机 8:2 划分，单次运行无法判断与论文的差距是")
    A("       方法差异还是随机波动。本实验用多个随机种子重复训练。")
    A("")
    A(" %-6s %11s %11s %12s %11s %10s" %
      ("seed", "末轮acc", "最优acc", "宏平均F1", "总体acc", "耗时(分)"))
    A(" " + "-" * 74)
    for r in runs:
        A(" %-6d %11.4f %11.4f %12.4f %11.4f %10.1f" %
          (r["seed"], r["final"], r["best"], r["f1"], r["overall"], r["time_s"] / 60))
    A(" " + "-" * 74)
    for name, arr in [("均值", FIN), ("标准差", FIN)]:
        pass
    A(" %-6s %11.4f %11.4f %12.4f" % ("均值", FIN.mean(), BEST.mean(), F.mean()))
    sd = F.std(ddof=1) if len(F) > 1 else 0.0
    A(" %-6s %11.4f %11.4f %12.4f" %
      ("标准差", FIN.std(ddof=1) if len(FIN) > 1 else 0,
       BEST.std(ddof=1) if len(BEST) > 1 else 0, sd))
    A(" %-6s %11.4f %11.4f %12.4f" % ("最好", FIN.max(), BEST.max(), F.max()))
    A(" %-6s %11.4f %11.4f %12.4f" % ("最差", FIN.min(), BEST.min(), F.min()))
    A("")
    A("=" * 84)
    A(" 与论文对比")
    A("=" * 84)
    A(" 论文 CNN average accuracy (宏平均F1)      = %.4f" % PAPER_F1)
    A(" 本次复现 宏平均F1                         = %.4f ± %.4f (n=%d)" % (F.mean(), sd, len(F)))
    A(" 本次复现 末轮总体准确率                   = %.4f ± %.4f"
      % (FIN.mean(), FIN.std(ddof=1) if len(FIN) > 1 else 0))
    A(" 本次复现 最优总体准确率                   = %.4f ± %.4f"
      % (BEST.mean(), BEST.std(ddof=1) if len(BEST) > 1 else 0))
    diff = PAPER_F1 - F.mean()
    A("")
    A(" 宏平均F1 差距                             = %+.4f" % (-diff))
    if sd > 0:
        A(" 差距相当于                                = %.2f 个标准差" % (abs(diff) / sd))
        lo, hi = F.mean() - sd, F.mean() + sd
        A(" 均值 ± 1 标准差区间                       = [%.4f, %.4f]" % (lo, hi))
        A(" 该区间是否包含论文值 0.940                = %s"
          % ("是" if lo <= PAPER_F1 <= hi else "否"))
    A("")
    A(" 【重要说明】")
    A(" 本次统计的标准差只反映「训练过程随机性」（固定测试集，改变训练种子）。")
    A(" 论文的「随机 8:2 划分」还会引入「数据划分差异」这一额外方差来源。")
    A(" 由于官方数据集未公开其原始划分方式，数据划分带来的方差无法评估。")
    A(" 因此上述「N 个标准差」仅为下界估计，不能作为「差距是否显著」的定论。")

    # 平均混淆矩阵
    Cm = np.mean([r["C"] for r in runs], axis=0)
    TP = np.diag(Cm); P = TP / Cm.sum(0); R = TP / Cm.sum(1); F1c = 2 * P * R / (P + R)
    A("")
    A("=" * 84)
    A(" 三次运行的平均混淆矩阵（小数）")
    A("=" * 84)
    A(" %-12s %8s %8s %8s %8s %8s %8s" % ("", "背景", "挖掘", "敲击", "浇水", "摇晃", "走动"))
    for i, nm in enumerate(NAMES_CN):
        A(" %-12s %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f" % (nm, *Cm[i]))
    A(" " + "-" * 76)
    A(" %-12s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f" % ("逐类 F1", *F1c))
    A("")
    A(" 平均宏平均 F1 = %.4f     平均总体准确率 = %.4f" % (F1c.mean(), TP.sum() / Cm.sum()))

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT_DIR, "cnn_seed_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(OUT_DIR, "cnn_seed_results.csv"), "w", encoding="utf-8") as f:
        f.write("seed,acc_final,acc_best,macroF1,acc_overall,time_s\n")
        for r in runs:
            f.write("%d,%.4f,%.4f,%.4f,%.4f,%.1f\n" %
                    (r["seed"], r["final"], r["best"], r["f1"], r["overall"], r["time_s"]))
    np.savetxt(os.path.join(OUT_DIR, "cnn_seed_mean_matrix.csv"), Cm,
               fmt="%.2f", delimiter=",")

    # ---------- 画图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))

    ax = axes[0]
    for r in runs:
        ax.plot(range(len(r["hist"])), r["hist"], lw=1.2,
                label="seed %d (末轮 %.4f)" % (r["seed"], r["final"]))
    ax.axhline(F.mean(), color="k", ls="--", lw=1.2, label="均值 %.4f" % F.mean())
    ax.axhline(PAPER_F1, color="r", ls=":", lw=1.5, label="论文 0.940")
    ax.set_xlabel("epoch", fontsize=11)
    ax.set_ylabel("测试集准确率", fontsize=11)
    ax.set_title("CNN 三种子训练曲线", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    ax = axes[1]
    x = np.arange(len(runs))
    b = ax.bar(x, F, 0.5, color="#2b7bba", label="本次复现（宏平均F1）")
    for xi, v in zip(x, F):
        ax.text(xi, v + 0.004, "%.4f" % v, ha="center", fontsize=9.5)
    ax.axhline(PAPER_F1, color="r", ls="--", lw=1.5, label="论文 0.940")
    ax.axhspan(F.mean() - sd, F.mean() + sd, color="gray", alpha=0.18,
               label="均值 ± 1 标准差")
    ax.axhline(F.mean(), color="k", ls=":", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(["seed %d" % r["seed"] for r in runs], fontsize=10)
    ax.set_ylim(0.88, 0.96)
    ax.set_ylabel("宏平均 F1-score", fontsize=11)
    ax.set_title("三次运行结果 vs 论文 (%.4f ± %.4f)" % (F.mean(), sd), fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "cnn_seed_stats.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print()
    print("图已保存:", p)
    print("汇总已保存: results\\cnn_seed_summary.txt, cnn_seed_results.csv")


if __name__ == "__main__":
    main()
