
import os, sys, random, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score

sys.dont_write_bytecode = True
sys.path.insert(0, r"D:\汕头大学\科研\深度学习\text.demo\das_rep\stage2\origin")
from models import CNN
from mydataset import MyDataset

DATA_ROOT = r"D:\汕头大学\科研\深度学习\text.demo\das_data"

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
