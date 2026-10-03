r"""
run_cnn_seeds.py —— 用多个随机种子重复训练 CNN，给出准确率的均值±标准差

为什么需要这个：
    论文的数据划分是"随机 8:2"，换一个随机种子结果就会变。
    单次运行无法判断"我们 92.64% vs 论文 94.0%"这个差距
    是方法差异还是随机波动。多次运行才能给出可信结论。

跑法：
    python run_cnn_seeds.py                 # 默认跑 3 个种子
    python run_cnn_seeds.py --seeds 1 2 3 --epochs 50
输出：
    results\cnn_seed_results.csv            # 每次运行的准确率
    results\cnn_seed_summary.txt            # 汇总（均值±标准差）
    seed_runs\seed_<n>_result.log           # 每次运行的逐 epoch 日志
"""
import argparse
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = r"D:\汕头大学\科研\深度学习\text.demo\das_rep\results"

# ------------------------------------------------------------
# 单次训练脚本：固定种子 -> 训练 -> 存结果
# 与官方 das_data_cnn.py 完全相同的网络/超参数，只加了种子控制
# ------------------------------------------------------------
SINGLE_RUN = r'''
import os, sys, random, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score

sys.dont_write_bytecode = True
sys.path.insert(0, r"{here}")
from models import CNN
from mydataset import MyDataset

DATA_ROOT = r"{data_root}"

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--batch_size", type=int, default=100)
p.add_argument("--out", type=str, required=True)
a = p.parse_args()

# ---------- 固定随机种子 ----------
random.seed(a.seed)
np.random.seed(a.seed)
torch.manual_seed(a.seed)
torch.cuda.manual_seed_all(a.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------- 日志（带 flush，可实时查看）----------
class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, msg):
        self.terminal.write(msg); self.log.write(msg); self.log.flush()
    def flush(self):
        self.log.flush()
sys.stdout = Logger(a.out, sys.stdout)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("种子 = %d   设备 = %s   计划 epochs = %d" % (a.seed, dev, a.epochs))

train_ds = MyDataset(os.path.join(DATA_ROOT, "train"),
                     os.path.join(DATA_ROOT, "train", "label.txt"))
test_ds = MyDataset(os.path.join(DATA_ROOT, "test"),
                    os.path.join(DATA_ROOT, "test", "label.txt"))
train_loader = DataLoader(dataset=train_ds, batch_size=a.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_ds, batch_size=a.batch_size, shuffle=False)

model = CNN()
if dev == "cuda":
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()


def train_step(x, y):
    model.train(); model.zero_grad()
    _, probs = model(x)
    loss = criterion(probs, y)
    loss.backward(); optimizer.step()
    return loss.item()


def evaluate(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for b in loader:
            x = torch.unsqueeze(b["data"], dim=1).float()
            if dev == "cuda":
                x = x.cuda()
            _, out = model(x)
            preds.extend(torch.max(out, dim=1)[1].cpu().tolist())
            labels.extend(b["label"].tolist())
    return accuracy_score(labels, preds), confusion_matrix(labels, preds)


t_start = time.time()
best = 0.0
acc_hist = []
for ep in range(a.epochs):
    t0 = time.time()
    for b in train_loader:
        x = torch.unsqueeze(b["data"], dim=1).float()
        y = b["label"]
        if dev == "cuda":
            x, y = x.cuda(), y.cuda()
        train_step(x, y)
    acc, C = evaluate(test_loader)
    acc_hist.append(acc)
    best = max(best, acc)
    print("Epoch %2d/%d  acc = %.4f   (%.1f 秒)" % (ep, a.epochs-1, acc, time.time()-t0))

elapsed = time.time() - t_start
acc_final = acc_hist[-1]
acc_best = max(acc_hist)

# 末轮混淆矩阵的宏平均 F1
TP = np.diag(C).astype(float)
P_ = TP / C.sum(axis=0); R_ = TP / C.sum(axis=1)
macroF1 = float((2*P_*R_/(P_+R_)).mean())

print()
print("RESULT seed=%d  acc_final=%.4f  acc_best=%.4f  macroF1=%.4f  acc_overall=%.4f  time=%.1f"
      % (a.seed, acc_final, acc_best, macroF1, float(np.trace(C))/C.sum(), elapsed))
print("MATRIX")
for row in C:
    print(",".join(str(int(v)) for v in row))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=100)
    a = ap.parse_args()

    data_root = r"D:\汕头大学\科研\深度学习\text.demo\das_data"
    seed_dir = os.path.join(HERE, "seed_runs")
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    script = os.path.join(seed_dir, "_single_run.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(SINGLE_RUN.format(here=HERE, data_root=data_root))

    print("=" * 74)
    print(" 多种子重复实验：seeds = %s, 每个 %d epochs" % (a.seeds, a.epochs))
    print("=" * 74)

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"          # 防止 plt.show() 挂死
    env["PYTHONIOENCODING"] = "utf-8"

    results = []
    for s in a.seeds:
        log = os.path.join(seed_dir, "seed_%d_result.log" % s)
        print("\n---- 开始 seed = %d ----" % s, flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, "-B", script, "--seed", str(s),
                            "--epochs", str(a.epochs), "--batch_size", str(a.batch_size),
                            "--out", log],
                           cwd=seed_dir, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, encoding="utf-8", errors="replace")
        el = time.time() - t0
        out = r.stdout or ""
        print(out[-800:] if len(out) > 800 else out, flush=True)

        # 【修复】子进程把 sys.stdout 重定向到了 Logger，所以 PIPE 抓不到内容，
        #          必须从日志文件里读结果
        text = out
        if os.path.isfile(log):
            file_text = open(log, encoding="utf-8", errors="replace").read()
            if "RESULT" in file_text:
                text = file_text
        line = [ln for ln in text.splitlines() if ln.startswith("RESULT")]
        if line:
            parts = dict(kv.split("=") for kv in line[-1].split()[1:])
            # 顺带解析混淆矩阵（RESULT 行之后的 MATRIX 段落）
            C = None
            if "MATRIX" in text:
                blk = text.split("MATRIX")[1].strip().split("\n")
                rows = []
                for ln in blk:
                    ln = ln.strip()
                    if not ln or not all(c.isdigit() or c == "," for c in ln):
                        break
                    rows.append([int(x) for x in ln.split(",")])
                if len(rows) == 6:
                    C = rows
            results.append(dict(seed=s,
                                acc_final=float(parts["acc_final"]),
                                acc_best=float(parts["acc_best"]),
                                macroF1=float(parts["macroF1"]),
                                acc_overall=float(parts["acc_overall"]),
                                time_s=float(parts["time"]),
                                C=C))
            print("  seed=%d: 末轮acc=%.4f 最优acc=%.4f 宏平均F1=%.4f"
                  % (s, float(parts["acc_final"]), float(parts["acc_best"]),
                     float(parts["macroF1"])), flush=True)
        else:
            print("  [警告] seed=%d 未拿到 RESULT 行，退出码 %s" % (s, r.returncode))
        print("---- seed = %d 完成，用时 %.1f 分钟 ----" % (s, el / 60), flush=True)

    if not results:
        print("没有任何有效结果")
        return

    import numpy as np
    A = np.array([r["acc_final"] for r in results])
    B = np.array([r["acc_best"] for r in results])
    F = np.array([r["macroF1"] for r in results])
    O = np.array([r["acc_overall"] for r in results])

    lines = []
    lines.append("=" * 78)
    lines.append(" CNN 多种子重复实验结果")
    lines.append("=" * 78)
    lines.append(" %-8s %12s %12s %12s %12s" % ("seed", "末轮acc", "最优acc", "宏平均F1", "总体acc"))
    lines.append(" " + "-" * 72)
    for r in results:
        lines.append(" %-8d %12.4f %12.4f %12.4f %12.4f" %
                     (r["seed"], r["acc_final"], r["acc_best"], r["macroF1"], r["acc_overall"]))
    lines.append(" " + "-" * 72)
    lines.append(" %-8s %12.4f %12.4f %12.4f %12.4f" % ("均值", A.mean(), B.mean(), F.mean(), O.mean()))
    lines.append(" %-8s %12.4f %12.4f %12.4f %12.4f" % ("标准差", A.std(ddof=1) if len(A) > 1 else 0,
                                                       B.std(ddof=1) if len(B) > 1 else 0,
                                                       F.std(ddof=1) if len(F) > 1 else 0,
                                                       O.std(ddof=1) if len(O) > 1 else 0))
    lines.append(" %-8s %12.4f %12.4f %12.4f %12.4f" % ("最好", A.max(), B.max(), F.max(), O.max()))
    lines.append(" %-8s %12.4f %12.4f %12.4f %12.4f" % ("最差", A.min(), B.min(), F.min(), O.min()))
    lines.append("")
    lines.append(" 论文报告 CNN average accuracy (宏平均F1) = 0.940")
    lines.append(" 本次复现 宏平均F1 = %.4f ± %.4f" % (F.mean(), F.std(ddof=1) if len(F) > 1 else 0))
    diff = 0.940 - F.mean()
    lines.append(" 与论文差距 = %+.4f" % (-diff))
    if len(F) > 1:
        sd = F.std(ddof=1)
        if sd > 0:
            lines.append(" 差距相当于 %.2f 个标准差" % (abs(diff) / sd))
        lines.append(" 是否落入 1 个标准差内: %s" % ("是（差距在随机波动范围内）"
                                                 if abs(diff) <= sd else "否（差距超出随机波动）"))
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(OUT_DIR, "cnn_seed_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(OUT_DIR, "cnn_seed_results.csv"), "w", encoding="utf-8") as f:
        f.write("seed,acc_final,acc_best,macroF1,acc_overall,time_s\n")
        for r in results:
            f.write("%d,%.4f,%.4f,%.4f,%.4f,%.1f\n" %
                    (r["seed"], r["acc_final"], r["acc_best"], r["macroF1"], r["acc_overall"], r["time_s"]))
    print("\n汇总已保存: results\\cnn_seed_summary.txt 和 cnn_seed_results.csv")


if __name__ == "__main__":
    main()
