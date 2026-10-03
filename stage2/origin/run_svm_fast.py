"""
run_svm_fast.py —— 用向量化特征提取重跑 SVM，验证精度不变

跑法：python run_svm_fast.py
对比：基线 svm_result.log 中 acc = 0.8885
"""
import os
import sys
import time
import numpy as np
import scipy.io as scio

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extraction import feature_extraction as official
from reference_feature_extraction import feature_extraction_fast

DATA_ROOT = r"D:\汕头大学\科研\深度学习\text.demo\das_data"


def get_diff_data_fixed(data):
    """官方 get_diff_data 的修复版：先用 int32 再差分，避免 uint16 回绕"""
    data = data.astype(np.int32)
    return data[1:, :] - data[:-1, :]


def get_feature_list(data, use_fast, legacy_qiaodu=False):
    """对 12 个通道各提 16 维特征 -> (12, 16)"""
    out = np.zeros([data.shape[1], 16])
    for i in range(data.shape[1]):
        series = data[:, i]
        if use_fast:
            out[i, :] = feature_extraction_fast(series, legacy_qiaodu=legacy_qiaodu)
        else:
            out[i, :] = official(series)
    return out


def build_features(split, use_fast, legacy_qiaodu=False, verbose=True):
    """读整个 split，返回 (X, y)"""
    labelpath = os.path.join(DATA_ROOT, split, "label.txt")
    with open(labelpath, encoding="utf-8-sig") as fh:
        names = [ln for ln in fh if ln.strip()]

    X = np.zeros([len(names), 384])
    y = np.zeros(len(names))

    if verbose:
        print("  正在处理 %s：共 %d 个样本" % (split, len(names)))
    t0 = time.time()
    for idx, ln in enumerate(names):
        parts = ln.split(" ")
        path = os.path.join(DATA_ROOT, split) + parts[0]
        raw = scio.loadmat(path)["data"]
        diff = get_diff_data_fixed(raw)

        feat_raw = get_feature_list(raw, use_fast, legacy_qiaodu)
        feat_diff = get_feature_list(diff, use_fast, legacy_qiaodu)
        X[idx, :] = np.concatenate((feat_raw, feat_diff), axis=1).reshape(-1)
        y[idx] = int(parts[1])

        if verbose and (idx + 1) % 2000 == 0:
            el = time.time() - t0
            print("    %5d/%d  已用 %.1f 分钟  预计还需 %.1f 分钟"
                  % (idx + 1, len(names), el / 60, el / (idx + 1) * (len(names) - idx - 1) / 60))
    if verbose:
        print("    完成，用时 %.2f 分钟" % ((time.time() - t0) / 60))
    return X, y


def run_svm(X_train, y_train, X_test, y_test, tag):
    """训练 SVM 并输出指标"""
    from sklearn import svm, preprocessing
    from sklearn.metrics import confusion_matrix

    scaler = preprocessing.MinMaxScaler()
    tr = scaler.fit_transform(X_train)      # 【修复】只在训练集上 fit，避免数据泄漏
    te = scaler.transform(X_test)

    clf = svm.SVC(C=1.0, kernel="rbf", degree=3, gamma="auto",
                  coef0=0.0, decision_function_shape="ovo",
                  tol=0.001, shrinking=True, max_iter=-1)
    t0 = time.time()
    clf.fit(tr, y_train)
    fit_t = time.time() - t0
    t0 = time.time()
    pred = clf.predict(te)
    pred_t = time.time() - t0

    C = confusion_matrix(y_test, pred)
    acc = np.trace(C) / C.sum()
    print()
    print("=" * 70)
    print(" 结果 [%s]" % tag)
    print("=" * 70)
    print("  混淆矩阵:")
    for row in C:
        print("   ", " ".join("%5d" % v for v in row))
    print()
    print("  acc(overall) = %.4f    宏平均召回 = %.4f"
          % (acc, (np.diag(C) / C.sum(axis=1)).mean()))
    print("  SVM 拟合 %.1f 秒，预测 %.1f 秒" % (fit_t, pred_t))
    return acc, C


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "correct"
    legacy = (mode == "legacy")
    tag = "向量化 + 复刻官方溢出" if legacy else "向量化 + 正确数学"

    print("=" * 70)
    print(" 用向量化特征提取重跑 SVM —— %s" % tag)
    print(" 基线（官方代码 + uint16 修复）: acc = 0.8885")
    print("=" * 70)

    t_all = time.time()
    X_train, y_train = build_features("train", True, legacy)
    X_test, y_test = build_features("test", True, legacy)
    print("  特征提取总耗时: %.2f 分钟" % ((time.time() - t_all) / 60))

    np.save("_X_train_fast.npy", X_train)
    np.save("_y_train_fast.npy", y_train)
    np.save("_X_test_fast.npy", X_test)
    np.save("_y_test_fast.npy", y_test)

    run_svm(X_train, y_train, X_test, y_test, tag)

    # 顺便存成 CSV，方便和官方产物对比
    np.savetxt("fast_feature_data.csv",
               np.concatenate((X_test, y_test[:, None]), axis=1), delimiter=",")
    print("  特征已存: fast_feature_data.csv")


if __name__ == "__main__":
    main()
