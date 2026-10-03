r"""
make_lda_figure.py —— LDA 特征可视化（论文 Fig. 5 复现）

原理：
    把高维特征（SVM 384 维 / CNN 400 维）用 LDA 降到 3 维，
    画成散点图，观察 6 类事件是否分得开。
    LDA 是有监督降维（用到了标签），目标是"让类间距离最大、类内距离最小"。

跑法：python make_lda_figure.py
输出：results\lda_svm.png、results\lda_cnn.png、results\lda_compare.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = r"D:\汕头大学\科研\深度学习\text.demo\das_rep\results"
os.makedirs(OUT_DIR, exist_ok=True)

# 论文 Fig.5 的配色：红、蓝、黄、绿、紫、黑 = 事件 1~6
COLORS = ["red", "blue", "darkorange", "green", "m", "black"]
NAMES_CN = ["背景噪声", "挖掘", "敲击", "浇水", "摇晃", "走动"]

FILES = {
    "SVM": dict(path=os.path.join(HERE, "5km_10km_svm_feature_data.csv"), nfeat=384),
    "CNN": dict(path=os.path.join(HERE, "feature_data.csv"), nfeat=400),
}


def project(path, nfeat):
    """读特征 CSV，LDA 降到 3 维"""
    data = np.loadtxt(path, delimiter=",")
    X = data[:, :nfeat]
    y = data[:, -1].astype(int)

    lda = LinearDiscriminantAnalysis(n_components=3)
    Z = lda.fit_transform(X, y)

    # 可解释性指标：LDA 各判别轴的类别可分性
    ratio = lda.explained_variance_ratio_
    return Z, y, ratio, X.shape


def draw(ax, Z, y, title, elev=30, azim=45, legend=True):
    handles = []
    for c in range(6):
        m = (y == c)
        h = ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2], s=12, c=COLORS[c],
                       marker=".", label=NAMES_CN[c], alpha=0.7)
        handles.append(h)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("LD1", fontsize=9)
    ax.set_ylabel("LD2", fontsize=9)
    ax.set_zlabel("LD3", fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=8)
    if legend:
        ax.legend(handles, NAMES_CN, fontsize=8, markerscale=1.8,
                  loc="upper left", framealpha=0.85)
    return handles


def main():
    results = {}
    for name, cfg in FILES.items():
        Z, y, ratio, shape = project(cfg["path"], cfg["nfeat"])
        results[name] = (Z, y, ratio)
        print("%s: 输入 %s -> LDA 3 维   判别轴方差解释比 = %s"
              % (name, shape, np.round(ratio, 3)))

    # ---------- 图 1：分别出图（论文 Fig.5 的布局：(a) SVM, (b) CNN）----------
    for name in ["SVM", "CNN"]:
        Z, y, ratio = results[name]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        draw(ax, Z, y, "LDA feature visualization - %s\n"
                       "(explained variance ratio: %s)"
                       % (name, ", ".join("%.2f" % r for r in ratio)))
        plt.tight_layout()
        p = os.path.join(OUT_DIR, "lda_%s.png" % name.lower())
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print("已保存:", p)

    # ---------- 图 2：并排对比 ----------
    fig = plt.figure(figsize=(20, 8.5))
    for k, name in enumerate(["SVM", "CNN"], start=1):
        Z, y, ratio = results[name]
        ax = fig.add_subplot(1, 2, k, projection="3d")
        draw(ax, Z, y, "(%s) %s features  (input dim = %d)"
                       % ("ab"[k-1], name, FILES[name]["nfeat"]))
    fig.suptitle("LDA feature visualization: SVM vs CNN  (paper Fig. 5)", fontsize=14)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "lda_compare.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print("已保存:", p)

    # ---------- 定量分析：类间可分性 ----------
    print()
    print("=" * 80)
    print(" 定量对比：6 类在 LDA 空间的分离程度")
    print("=" * 80)
    print("  %-6s %14s %14s %14s" % ("特征", "类内散度(均值)", "类间散度", "可分性比"))
    print("  " + "-" * 62)
    for name in ["SVM", "CNN"]:
        Z, y, ratio = results[name]
        # 类内散度：每个样本到本类中心的平均距离
        within = []
        centers = []
        for c in range(6):
            pts = Z[y == c]
            ctr = pts.mean(axis=0)
            centers.append(ctr)
            within.append(np.linalg.norm(pts - ctr, axis=1).mean())
        centers = np.array(centers)
        # 类间散度：类中心到全局中心的平均距离
        gctr = Z.mean(axis=0)
        between = np.linalg.norm(centers - gctr, axis=1).mean()
        print("  %-6s %14.3f %14.3f %14.3f"
              % (name, np.mean(within), between, between / np.mean(within)))
    print()
    print("  可分性比 = 类间散度 / 类内散度，越大说明 6 类分得越开")
    print("  注意：LDA 是有监督降维（用到标签），所以它天然会看起来分得开，")
    print("        两张图不能直接推断分类器性能，只能定性观察类别结构。")


if __name__ == "__main__":
    main()
