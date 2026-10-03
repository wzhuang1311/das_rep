"""
bench_dataloader.py —— 实测官方 vs 向量化 的数据加载速度

目的：量化 mydataset.py 里 normalize() 双层 for 循环的代价。
跑法：python bench_dataloader.py
"""
import os
import sys
import time
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset, DataLoader

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = r"D:\汕头大学\科研\深度学习\text.demo\das_data"
N_BATCH = 10          # 只测前 10 个 batch，不跑完整 epoch

# ============================================================
# 版本 1：完全照抄官方 mydataset.py
# ============================================================
def normalize_official(data):                      # 归一化到 0-255
    rawdata_max = max(map(max, data))
    rawdata_min = min(map(min, data))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = round(((255 - 0) * (data[i][j] - rawdata_min)
                                / (rawdata_max - rawdata_min)) + 0)
    return data


class DatasetOfficial(Dataset):
    def __init__(self, root_dir, names_file):
        self.root_dir = root_dir
        with open(names_file, encoding="utf-8-sig") as fh:
            self.names_list = [ln for ln in fh if ln.strip()]

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        p = self.root_dir + self.names_list[idx].split(" ")[0]
        raw = scio.loadmat(p)["data"].astype(int)
        data = normalize_official(raw)
        return {"data": data, "label": int(self.names_list[idx].split(" ")[1])}


# ============================================================
# 版本 2：向量化 normalize（其余完全相同）
# ============================================================
def normalize_fast(data):
    mn, mx = data.min(), data.max()
    return np.round(255.0 * (data - mn) / (mx - mn))


class DatasetFast(Dataset):
    def __init__(self, root_dir, names_file):
        self.root_dir = root_dir
        with open(names_file, encoding="utf-8-sig") as fh:
            self.names_list = [ln for ln in fh if ln.strip()]

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        p = self.root_dir + self.names_list[idx].split(" ")[0]
        raw = scio.loadmat(p)["data"].astype(int)
        data = normalize_fast(raw)
        return {"data": data, "label": int(self.names_list[idx].split(" ")[1])}


def bench(dataset, name):
    """只跑前 N_BATCH 个 batch，返回 (耗时, 平均每样本秒数)"""
    loader = DataLoader(dataset=dataset, batch_size=100, shuffle=False)
    t0 = time.time()
    n_samples = 0
    for i, batch in enumerate(loader):
        n_samples += batch["data"].shape[0]
        if i + 1 >= N_BATCH:
            break
    el = time.time() - t0
    return el, el / n_samples, n_samples


def main():
    root = os.path.join(DATA_ROOT, "train")
    txt = os.path.join(root, "label.txt")

    print("=" * 74)
    print(" 数据加载速度对照实验（只测前 %d 个 batch，batch_size=100）" % N_BATCH)
    print("=" * 74)
    print(" 训练集 12335 样本，1 epoch = 124 个 batch，计划 50 epochs")
    print()

    results = {}
    # --- 官方版 ---
    ds = DatasetOfficial(root, txt)
    print("  正在测【官方版】双层 for 循环...", flush=True)
    el, per, n = bench(ds, "official")
    results["official"] = (el, per)
    print("    前 %d 个 batch（%d 样本）耗时 %.2f 秒" % (N_BATCH, n, el))
    print("    平均每样本 %.4f 秒" % per)
    print()

    # --- 向量化版 ---
    ds2 = DatasetFast(root, txt)
    print("  正在测【向量化版】...", flush=True)
    el2, per2, n2 = bench(ds2, "fast")
    results["fast"] = (el2, per2)
    print("    前 %d 个 batch（%d 样本）耗时 %.2f 秒" % (N_BATCH, n2, el2))
    print("    平均每样本 %.4f 秒" % per2)
    print()

    # --- 结论 ---
    print("=" * 74)
    print(" 结论")
    print("=" * 74)
    print("  加速比: %.1f 倍" % (per / per2))
    print()
    train_n, test_n = 12335, 3084
    print("  %-14s %16s %16s" % ("", "官方版", "向量化版"))
    print("  " + "-" * 50)
    for label, per_ in (("1 epoch (训练)", per if False else None),):
        pass
    for name, p in (("官方版", per), ("向量化版", per2)):
        e1 = p * train_n / 60
        print("  %-14s %13.1f 分钟 %13.1f 分钟" % (name + " 1 epoch", e1, e1))
    print()
    print("  换算成 50 epochs（纯归一化耗时，不含模型计算）:")
    print("    官方版   : %.1f 小时" % (per * train_n * 50 / 3600))
    print("    向量化版 : %.1f 分钟" % (per2 * train_n * 50 / 60))
    print()
    print("  注：以上只统计数据加载。前向/反向传播在 GPU 上的时间另计。")


if __name__ == "__main__":
    main()
