r"""
eval_cnn.py —— 加载已训练的 model.pth，在测试集上重新推理，得到权威的混淆矩阵

跑法：python eval_cnn.py
输出：results\cnn_confusion_true.png + 控制台完整指标
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import CNN
from mydataset import MyDataset

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = r"D:\汕头大学\科研\深度学习\text.demo\das_data"
OUT_DIR = r"D:\汕头大学\科研\深度学习\text.demo\das_rep\results"
NAMES = ["background", "digging", "knocking", "watering", "shaking", "walking"]
NAMES_CN = ["背景噪声", "挖掘", "敲击", "浇水", "摇晃", "走动"]
PAPER_P = np.array([0.985, 0.891, 0.951, 0.905, 0.976, 0.920])
PAPER_NAR = np.array([0.042, 0.072, 0.045, 0.047, 0.026, 0.137])
PAPER_F1 = np.array([0.971, 0.909, 0.953, 0.929, 0.975, 0.891])


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(os.path.join(HERE, "model.pth"), weights_only=False)
    model = model.to(dev).eval()
    print("模型已加载，设备:", dev)

    ds = MyDataset(os.path.join(DATA_ROOT, "test"),
                   os.path.join(DATA_ROOT, "test", "label.txt"))
    loader = DataLoader(dataset=ds, batch_size=100, shuffle=False)
    print("测试集样本数:", len(ds))

    preds, labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            x = batch["data"]
            y = batch["label"]
            x = torch.unsqueeze(x, dim=1).float()
            if dev == "cuda":
                x = x.cuda()
            _, out = model(x)
            preds.extend(torch.max(out, dim=1)[1].cpu().tolist())
            labels.extend(y.tolist())
    print("推理完成，用时 %.1f 秒" % (time.time() - t0))

    C = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)

    TP = np.diag(C).astype(float)
    PREC = TP / C.sum(axis=0)
    REC = TP / C.sum(axis=1)
    F1 = 2 * PREC * REC / (PREC + REC)

    print()
    print("=" * 92)
    print(" CNN 测试集真实混淆矩阵（由 model.pth 重新推理得到）")
    print("=" * 92)
    print("  行=真实类别, 列=预测类别")
    print("  %-12s %6s %6s %6s %6s %6s %6s %8s" %
          ("", *NAMES, "行和"))
    for i in range(6):
        print("  %-12s %6d %6d %6d %6d %6d %6d %8d" %
              (NAMES[i], *C[i], C[i].sum()))
    print("  %-12s %6d %6d %6d %6d %6d %6d" %
          ("列和", *C.sum(axis=0)))
    print()
    print("  总体准确率 = %d/%d = %.4f" % (TP.sum(), C.sum(), acc))
    print("  对角元和   = %d" % int(TP.sum()))

    print()
    print("  %-10s %8s %8s %8s %8s %8s %8s" %
          ("事件", "P(我们)", "P(论文)", "R(我们)", "R(论文)", "F1(我们)", "F1(论文)"))
    print("  " + "-" * 66)
    for i, nm in enumerate(NAMES_CN):
        print("  %-10s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f" %
              (nm, PREC[i], PAPER_P[i], REC[i], 1 - PAPER_NAR[i], F1[i], PAPER_F1[i]))
    print("  " + "-" * 66)
    print("  %-10s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f" %
          ("宏平均", PREC.mean(), PAPER_P.mean(), REC.mean(), (1 - PAPER_NAR).mean(),
           F1.mean(), PAPER_F1.mean()))
    print()
    print("  宏平均 F1（论文 average accuracy 口径）= %.4f    论文报告 = 0.940" % F1.mean())

    # ---------- 画图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.8))

    ax = axes[0]
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=C.max())
    for i in range(6):
        for j in range(6):
            v = C[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=10.5,
                    color="white" if v > C.max() * 0.55 else "black",
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(NAMES, rotation=30, ha="right", fontsize=9.5)
    ax.set_yticklabels(NAMES, fontsize=9.5)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("CNN - Confusion Matrix (test n=%d)\noverall acc = %.2f%%   macro F1 = %.2f%%"
                 % (C.sum(), acc * 100, F1.mean() * 100), fontsize=12)
    plt.colorbar(im, ax=ax, label="count")

    ax = axes[1]
    x = np.arange(6); w = 0.35
    b1 = ax.bar(x - w/2, F1, w, label="ours (reproduced)", color="#2b7bba")
    b2 = ax.bar(x + w/2, PAPER_F1, w, label="paper (Table 4)", color="#c44e52")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
                    "%.3f" % b.get_height(), ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(NAMES_CN, rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0, 1.13); ax.set_ylabel("F1-score", fontsize=11)
    ax.set_title("逐类 F1-score：复现 vs 论文", fontsize=12)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "cnn_confusion_true.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print()
    print("图已保存:", p)

    np.savetxt(os.path.join(OUT_DIR, "cnn_confusion_matrix_true.csv"), C,
               fmt="%d", delimiter=",")
    print("矩阵已存: results\\cnn_confusion_matrix_true.csv")


if __name__ == "__main__":
    main()
