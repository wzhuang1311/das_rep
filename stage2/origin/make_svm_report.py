"""
make_svm_report.py —— 从混淆矩阵生成复现报告图（不重新训练，秒出）

跑法：
    python make_svm_report.py
输出：
    das_rep\results\svm_confusion_ours.png
    das_rep\results\svm_vs_paper.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无 GUI 后端，直接存文件，不弹窗
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = r"D:\汕头大学\科研\深度学习\text.demo\das_rep\results"
os.makedirs(OUT_DIR, exist_ok=True)

NAMES = ["background", "digging", "knocking", "watering", "shaking", "walking"]
NAMES_CN = ["背景噪声", "挖掘", "敲击", "浇水", "摇晃", "走动"]

# ---------------- 实测数据 ----------------
# 我们复现得到的测试集混淆矩阵（来自 das_data_svm.py 的 test_matrix 输出）
C = np.array([[588,   0,   0,   1,   0,   0],
              [ 90, 359,   3,   2,  11,  37],
              [ 44,   1, 459,   0,   0,   2],
              [ 17,  15,   0, 364,  25,  30],
              [  0,   1,   0,   4, 540,   1],
              [ 12,  36,   6,   4,   2, 430]])

# 论文 Table 4 报告值（CNN 行不取，这里只对比 SVM）
PAPER_PREC = np.array([0.764, 0.776, 0.836, 0.954, 0.829, 0.862])
PAPER_NAR  = np.array([0.010, 0.406, 0.142, 0.297, 0.049, 0.198])
PAPER_F1   = np.array([0.863, 0.673, 0.847, 0.809, 0.886, 0.831])

# ---------------- 计算我们的指标 ----------------
TP = np.diag(C).astype(float)
OUR_PREC = TP / C.sum(axis=0)
OUR_REC  = TP / C.sum(axis=1)          # = 1 - NAR（论文口径）
OUR_F1   = 2 * OUR_PREC * OUR_REC / (OUR_PREC + OUR_REC)

OUR_ACC_OVERALL = TP.sum() / C.sum()
OUR_ACC_MACRO   = OUR_REC.mean()       # 论文的 "average accuracy" 就是 6 类召回率均值
PAPER_ACC_MACRO = 1 - PAPER_NAR.mean()

# ============================================================
# 图 1：我们的混淆矩阵
# ============================================================
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(C, cmap="Blues", vmin=0, vmax=C.max())

for i in range(6):
    for j in range(6):
        v = C[i, j]
        ax.text(j, i, str(v), ha="center", va="center", fontsize=11,
                color="white" if v > C.max() * 0.55 else "black",
                fontweight="bold" if i == j else "normal")

ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(NAMES, rotation=30, ha="right", fontsize=10)
ax.set_yticklabels(NAMES, fontsize=10)
ax.set_xlabel("Predicted label", fontsize=13)
ax.set_ylabel("True label", fontsize=13)
ax.set_title("SVM baseline - Confusion Matrix (test set, n=%d)\n"
             "overall accuracy = %.2f%%   macro accuracy = %.2f%%"
             % (C.sum(), OUR_ACC_OVERALL * 100, OUR_ACC_MACRO * 100),
             fontsize=13, pad=14)
plt.colorbar(im, ax=ax, label="sample count")
plt.tight_layout()
p1 = os.path.join(OUT_DIR, "svm_confusion_ours.png")
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print("已保存:", p1)

# ============================================================
# 图 2：与论文 Table 4 的对比
# ============================================================
x = np.arange(6); w = 0.35
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

panels = [("Recall  (= 1 - NAR)", OUR_REC, 1 - PAPER_NAR),
          ("Precision", OUR_PREC, PAPER_PREC),
          ("F1-score", OUR_F1, PAPER_F1)]

for ax, (title, ours, paper) in zip(axes, panels):
    b1 = ax.bar(x - w/2, ours,  w, label="ours (reproduced)", color="#2b7bba")
    b2 = ax.bar(x + w/2, paper, w, label="paper (Table 4)",   color="#c44e52")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
                    "%.3f" % b.get_height(), ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES_CN, rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0, 1.13)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

fig.suptitle("SVM baseline: 复现结果 vs 论文 (Results in Optics 2023) ", fontsize=14)
plt.tight_layout()
p2 = os.path.join(OUT_DIR, "svm_vs_paper.png")
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print("已保存:", p2)

# ============================================================
# 控制台汇总表
# ============================================================
print()
print("=" * 88)
print("SVM 基线复现结果汇总")
print("=" * 88)
print("  测试集样本数: %d   训练集样本数: 12335" % C.sum())
print("  总体准确率 (overall) = %d/%d = %.4f" % (TP.sum(), C.sum(), OUR_ACC_OVERALL))
print("  宏平均准确率 (macro) = %.4f     <- 论文 average accuracy 用的是这个口径" % OUR_ACC_MACRO)
print("  论文报告值           = %.4f" % PAPER_ACC_MACRO)
print()
print("  %-12s %8s %8s %8s %8s %8s %8s" %
      ("事件", "P(我们)", "P(论文)", "R(我们)", "R(论文)", "F1(我们)", "F1(论文)"))
print("  " + "-" * 84)
for i, nm in enumerate(NAMES_CN):
    print("  %-12s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f" %
          (nm, OUR_PREC[i], PAPER_PREC[i], OUR_REC[i], 1 - PAPER_NAR[i],
           OUR_F1[i], PAPER_F1[i]))
print("  " + "-" * 84)
print("  %-12s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f" %
      ("宏平均", OUR_PREC.mean(), PAPER_PREC.mean(), OUR_REC.mean(),
       1 - PAPER_NAR.mean(), OUR_F1.mean(), PAPER_F1.mean()))
print()
print("  NAR 双口径:")
print("    论文公式(2) NAR = %d/(%d+%d) = %.4f  (背景类漏检率)"
      % (C[:, 0].sum() - C[0, 0], C[0, 0], C[:, 0].sum() - C[0, 0],
         (C[:, 0].sum() - C[0, 0]) / C[:, 0].sum()))
print("    README 虚警率       = %d/%d = %.4f" % (C[0, 1:].sum(), C[:, 1:].sum(),
                                                C[0, 1:].sum() / C[:, 1:].sum()))
print("    FNR (漏检非背景事件) = %d/%d = %.4f"
      % (C[:, 0].sum() - C[0, 0], C.sum() - C[0].sum(),
         (C[:, 0].sum() - C[0, 0]) / (C.sum() - C[0].sum())))
print("=" * 88)
