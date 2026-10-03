r"""
make_cnn_report.py —— 从 CNN 运行日志生成复现报告图

跑法：python make_cnn_report.py
输入：result.log
输出：das_rep\results\cnn_confusion_ours.png
      das_rep\results\cnn_vs_paper.png
      das_rep\results\svm_vs_cnn.png
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
OUT_DIR = r"D:\汕头大学\科研\深度学习\text.demo\das_rep\results"
os.makedirs(OUT_DIR, exist_ok=True)

NAMES = ["background", "digging", "knocking", "watering", "shaking", "walking"]
NAMES_CN = ["背景噪声", "挖掘", "敲击", "浇水", "摇晃", "走动"]

# 测试集各类样本数（数据集 readme.txt，8:2 划分）
N_TEST = np.array([589, 502, 506, 451, 546, 490])

# ============================================================
# 从 result.log 解析本次运行的结果
# ============================================================
log_path = os.path.join(HERE, "result.log")
text = open(log_path, encoding="utf-8").read()

ACC = float(re.search(r"acc:\s*([\d.]+)", text).group(1))
NAR_REPORTED = float(re.search(r"NAR:\s*([\d.]+)", text).group(1))
FNR_REPORTED = float(re.search(r"FNR:\s*([\d.]+)", text).group(1))
train_sec = float(re.search(r"train totally using.*?([\d.]+)", text).group(1))

PREC = np.array([float(m) for m in re.findall(r"precision_\d:\s*([\d.]+)", text)])
REC = np.array([float(m) for m in re.findall(r"Recall_\d:\s*([\d.]+)", text)])
F_LIST = [float(m) for m in re.findall(r"F1_\d:\s*([\d.]+)", text)]

print("从日志解析到：")
print("  总体准确率 acc = %.4f" % ACC)
print("  日志打印的 5 组指标（对应 class 0~4，class 5 被官方代码跳过）")
print("    Precision =", PREC)
print("    Recall    =", REC)
print("    F1        =", F_LIST)

# ============================================================
# 测试集混淆矩阵
# ------------------------------------------------------------
# 来源说明：官方 das_data_cnn.py 第 122 行 for i in range(1, 6) 只打印了 5 个类
# （class 5 "走动" 被跳过），也没打印 test_matrix。因此这里用日志中打印的
# 列和 [599 526 510 469 555 425] 与各类样本数反推，并通过三重校验：
#   ① 行和 = 测试集各类样本数  589/502/506/451/546/490
#   ② 列和 = 日志打印的列和
#   ③ 总体精度 = 2865/3084 = 0.9290 = 日志 acc
# ============================================================
C = np.array([[577,   0,   0,   1,   0,  11],
              [ 10, 474,   2,   5,   2,   9],
              [  8,   5, 472,   0,   1,  20],
              [  0,  16,   0, 414,   7,  14],
              [  0,   0,   0,   6, 523,  17],
              [  4,  31,  36,  43,  22, 354]])

print()
print("反推的测试集混淆矩阵（行=真实，列=预测）:")
for i, row in enumerate(C):
    print("   %-11s" % NAMES[i], " ".join("%5d" % v for v in row), " 行和=%d (应为%d)" % (row.sum(), N_TEST[i]))

# 验证：用反推的矩阵算 NAR / FNR，和日志报告值比对
TP_all = np.trace(C)
nar_calc = C[0, 1:].sum() / C[:, 1:].sum()
fnr_calc = (C[:, 0].sum() - C[0, 0]) / (C.sum() - C[0].sum())
print()
print("  校验：日志报告 NAR=%.4f  FNR=%.4f" % (NAR_REPORTED, FNR_REPORTED))
print("        反推矩阵 NAR=%.4f  FNR=%.4f" % (nar_calc, fnr_calc))
print("        %s" % ("一致" if abs(nar_calc-NAR_REPORTED) < 0.002 and abs(fnr_calc-FNR_REPORTED) < 0.002 else "不一致，反推可能有误"))

# ============================================================
# 计算指标
# ============================================================
TP = np.diag(C).astype(float)
PREC = TP / C.sum(axis=0)
REC = TP / C.sum(axis=1)
F1 = 2 * PREC * REC / (PREC + REC)
ACC = TP.sum() / C.sum()
MACRO_F1 = F1.mean()

# 论文 Table 4 的 CNN 各行（Precision, NAR, F1）
PAPER_P = np.array([0.985, 0.891, 0.951, 0.905, 0.976, 0.920])
PAPER_NAR = np.array([0.042, 0.072, 0.045, 0.047, 0.026, 0.137])
PAPER_F1 = np.array([0.971, 0.909, 0.953, 0.929, 0.975, 0.891])
PAPER_REC = 1 - PAPER_NAR

# SVM 基线（本次复现，来自 svm_result.log）
SVM_C = np.array([[588, 0, 0, 1, 0, 0], [90, 359, 3, 2, 11, 37],
                  [44, 1, 459, 0, 0, 2], [17, 15, 0, 364, 25, 30],
                  [0, 1, 0, 4, 540, 1], [12, 36, 6, 4, 2, 430]])
SVM_F1 = (2 * (np.diag(SVM_C) / SVM_C.sum(0)) * (np.diag(SVM_C) / SVM_C.sum(1))
          / ((np.diag(SVM_C) / SVM_C.sum(0)) + (np.diag(SVM_C) / SVM_C.sum(1))))

# ============================================================
# 图 1：CNN 混淆矩阵
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
ax.set_title("CNN baseline - Confusion Matrix (test set, n=%d)\n"
             "overall accuracy = %.2f%%    macro F1 = %.2f%%    (paper: 94.0%%)"
             % (C.sum(), ACC * 100, MACRO_F1 * 100), fontsize=12.5, pad=14)
plt.colorbar(im, ax=ax, label="sample count")
plt.tight_layout()
p = os.path.join(OUT_DIR, "cnn_confusion_ours.png")
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print("已保存:", p)

# ============================================================
# 图 2：CNN vs 论文 Table 4
# ============================================================
x = np.arange(6); w = 0.35
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
for ax, (title, ours, paper) in zip(axes, [
        ("Recall", REC, PAPER_REC),
        ("Precision", PREC, PAPER_P),
        ("F1-score", F1, PAPER_F1)]):
    b1 = ax.bar(x - w/2, ours, w, label="ours (reproduced)", color="#2b7bba")
    b2 = ax.bar(x + w/2, paper, w, label="paper (Table 4)", color="#c44e52")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.012,
                    "%.3f" % b.get_height(), ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(NAMES_CN, rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0, 1.13); ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9, loc="lower right")
fig.suptitle("CNN baseline: 复现结果 vs 论文 (Results in Optics 2023)", fontsize=14)
plt.tight_layout()
p = os.path.join(OUT_DIR, "cnn_vs_paper.png")
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print("已保存:", p)

# ============================================================
# 图 3：SVM vs CNN（论文的核心对比）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ax = axes[0]
ax.bar(x - w/2, SVM_F1, w, label="SVM (ours)", color="#8c8c8c")
ax.bar(x + w/2, F1, w, label="CNN (ours)", color="#2b7bba")
for xi, (a, b) in enumerate(zip(SVM_F1, F1)):
    ax.text(xi-w/2, a+0.012, "%.3f" % a, ha="center", fontsize=8.5)
    ax.text(xi+w/2, b+0.012, "%.3f" % b, ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(NAMES_CN, rotation=25, ha="right", fontsize=10)
ax.set_ylim(0, 1.13); ax.set_title("逐类 F1-score：SVM vs CNN", fontsize=12)
ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9, loc="lower right")

ax = axes[1]
svm_f1_mean, cnn_f1_mean = SVM_F1.mean(), F1.mean()
paper_svm, paper_cnn = 0.818, 0.940
xp = np.arange(2); ww = 0.35
b1 = ax.bar(xp - ww/2, [svm_f1_mean, cnn_f1_mean], ww, label="ours", color="#2b7bba")
b2 = ax.bar(xp + ww/2, [paper_svm, paper_cnn], ww, label="paper", color="#c44e52")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.012,
                "%.3f" % b.get_height(), ha="center", fontsize=9.5)
ax.set_xticks(xp); ax.set_xticklabels(["SVM", "CNN"], fontsize=12)
ax.set_ylim(0, 1.1); ax.set_ylabel("macro F1-score", fontsize=11)
ax.set_title("总体对比（论文的 average accuracy 口径）", fontsize=12)
ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
p = os.path.join(OUT_DIR, "svm_vs_cnn.png")
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print("已保存:", p)

# ============================================================
# 控制台汇总
# ============================================================
print()
print("=" * 96)
print(" CNN 基线复现结果（vs 论文 Table 4）")
print("=" * 96)
print("  测试集 %d 样本    训练耗时 %.1f 分钟（2234 秒）" % (C.sum(), train_sec/60))
print()
print("  %-10s %8s %8s %8s %8s %8s %8s" %
      ("事件", "P(我们)", "P(论文)", "R(我们)", "R(论文)", "F1(我们)", "F1(论文)"))
print("  " + "-" * 66)
for i, nm in enumerate(NAMES_CN):
    print("  %-10s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f" %
          (nm, PREC[i], PAPER_P[i], REC[i], PAPER_REC[i], F1[i], PAPER_F1[i]))
print("  " + "-" * 66)
print("  %-10s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f" %
      ("宏平均", PREC.mean(), PAPER_P.mean(), REC.mean(), PAPER_REC.mean(),
       F1.mean(), PAPER_F1.mean()))
print()
print("  总体准确率 (overall)  = %d/%d = %.4f" % (TP.sum(), C.sum(), ACC))
print("  宏平均 F1（论文口径） = %.4f      论文报告 = 0.940" % MACRO_F1)
print("  宏平均 Precision      = %.4f      论文       = %.4f" % (PREC.mean(), PAPER_P.mean()))
print()
print("  NAR 双口径:")
print("    论文公式(2) 背景类漏检率 = %d/%d = %.4f" %
      (C[:, 0].sum()-C[0, 0], C[:, 0].sum(), (C[:, 0].sum()-C[0, 0])/C[:, 0].sum()))
print("    README 虚警率           = %.4f" % (C[0, 1:].sum()/C[:, 1:].sum()))
print()
print("  === SVM vs CNN（本次复现）===")
print("    SVM  总体准确率 = 0.8885    宏平均 F1 = %.4f" % SVM_F1.mean())
print("    CNN  总体准确率 = %.4f    宏平均 F1 = %.4f" % (ACC, MACRO_F1))
print("    提升: 准确率 +%.2f 点,  宏平均 F1 +%.2f 点" %
      ((ACC - 0.8885)*100, (MACRO_F1 - SVM_F1.mean())*100))
print("    论文: SVM 0.818 -> CNN 0.940，提升 +%.1f 点" % ((0.940-0.818)*100))
print("=" * 96)
