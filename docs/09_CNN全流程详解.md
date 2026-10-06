# CNN 全流程详解：从原理到跑出结论

> **这篇讲什么**：CNN 为什么能work（原理）、训练脚本每一行在干什么（逐行）、
> 以及一套可以直接搬到你下一个项目上的完整模板。
>
> **配套**：`05_阶段2_源码逐文件讲解.md`（官方源码分析）、
> `06_阶段3_CNN复现.md`（性能瓶颈）、`08_拿到rawdata该怎么做.md`（行动流程）

---

# 第一部分：原理——CNN 到底在算什么

## 1.1 先说清楚我们的数据长什么样

一个样本是 `(10000, 12)` 的矩阵：

```
        ← 12 个空间点（光纤上的相邻位置）→
   ┌────────────────────────────────────┐
   │                                    │
   │      强度值（uint16，约 7800~8650）  │   ↑
   │                                    │   10000 个
   │                                    │   时间采样点
   │                                    │   ↓
   └────────────────────────────────────┘
```

**关键认识：横轴和纵轴都有物理意义。**
- 纵轴相邻的两行 = 相差 0.1 ms 的两个时刻
- 横轴相邻的两列 = 沿光纤相差 10 m 的两个位置

**所以"相邻"是有意义的**——这正是 CNN 能work的前提。

## 1.2 卷积的本质：用小窗口在数据上滑动，找局部模式

卷积核（kernel）就是一个小窗口，比如 `(200, 3)`：

```
   数据 (10000, 12)                     卷积核 (200, 3)
   ┌──────────────────────┐             ┌─────────┐
   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│             │ ▓▓▓▓▓▓▓ │  ← 200 行
   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  ←──滑──→    │ ▓▓▓▓▓▓▓ │     宽 3 列
   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│             └─────────┘
   └──────────────────────┘
        窗口停在一个位置时：
        输出值 = Σ (窗口内的数据 × 核里对应的权重) + 偏置
```

**这个操作在干什么？** 它在问一个问题：

> "在这 200 个连续时刻、3 个相邻空间点上，数据呈现的形状是**不是**长得像我这个核？"

**核里的数字是学出来的**，不是人设计的。训练开始时它们是随机数，
通过反向传播慢慢调整成"能识别某种模式的模板"。

**这和你学过的鱼书 `Affine` 层的区别**：

| | `Affine`（全连接） | `Conv2d`（卷积） |
|---|---|---|
| 连接方式 | 每个输入连每个输出 | **只连局部一小块** |
| 权重 | 每个位置一套独立权重 | **整个输入共享同一套权重** |
| 参数量 | 10000×12 × 400 = 4800 万 | **3000** |
| 擅长 | 全局关系 | **局部模式、且模式出现在哪都认得** |

**"权重共享"是卷积的核心优势**：挖掘事件出现在时间轴的前段还是后段，
用同一个核都能识别——这叫**平移不变性**。

## 1.3 为什么核是 (200, 3)？—— 这个数字有物理含义

`kernel_size=(200, 3)` 的意思是：**时间方向看 200 个点，空间方向看 3 个点。**

- **时间方向 200 点**：采样率 10 MSamp/s → 200 点 = **20 μs**。
  一次敲击、一次挖掘的冲击持续时间大约在这个量级。
  **窗口太短捕捉不到完整事件，太长会把多个事件混在一起。**
- **空间方向 3 点**：因为事件只影响光纤上的一小段（10 m/点，3 点 = 30 m）。
  **不需要看得太宽，看宽了反而混入无关位置的噪声。**

**这两条不是随便定的，是根据信号的时间尺度和空间尺度定的。**
你换一个项目，就要重新根据物理尺度来定——这是第 3.1 节要讲的。

## 1.4 一个真实的数字对比：为什么不该用全连接

```
如果第一层用全连接：
  输入 10000 × 12 = 120,000 个值
  输出哪怕只要 400 个
  → 参数量 = 120000 × 400 = 4800 万

实际用卷积：
  Conv1 参数量 = 1×5×200×3 + 5 = 3005
```

**相差 16000 倍。** 而且这不是"省点空间"，是"能不能训得动"的区别：
4800 万参数需要海量数据才能训练，7.4 万参数的模型在 12335 个样本上刚刚好。

## 1.5 Stride 和 Padding：控制输出尺寸的两个旋钮

```
stride=(50, 1)   窗口每次移动 50 行、1 列
                 ↑ 时间方向跳 50 步，相当于降采样 50 倍
                 ↑ 空间方向不跳，保留全部空间信息
```

**为什么时间方向要大跳，空间方向不跳？**
因为时间上有 10000 个点、太密（相邻点高度相关），
空间上只有 12 个点、本来就稀疏。**这是根据数据形状做的取舍。**

`padding=1` 是在边缘补一圈 0，作用是别让输出尺寸掉太快。

## 1.6 ReLU 和 MaxPool：两个"配角"为什么必要

**ReLU（`max(0, x)`）—— 引入非线性**

没有它，多层卷积叠起来数学上等价于一层卷积（线性函数的复合还是线性）。
ReLU 让网络能表达非线性关系。
**另外它便宜**：求导要么 0 要么 1，比 sigmoid 快得多。

**MaxPool（取 2×2 窗口的最大值）—— 降采样 + 保留最强响应**

两个作用：
1. **降低计算量**：尺寸减半，后面所有层的计算量都跟着降
2. **增强鲁棒性**：取最大值意味着"只要窗口里出现过这个模式，就保留下来"，
   模式稍微偏移一点也能被检出

## 1.7 逐层形状变化（我实测验证过）

这是全篇最该背下来的表——**你必须能自己算出这些数字**：

```
输入                  (100, 1, 10000, 12)
                      ↑    ↑   ↑     ↑
                    batch 通道 时间  空间

Conv1: kernel=(200,3), stride=(50,1), padding=1
  时间: floor((10000 + 2×1 - 200)/50) + 1 = floor(9820/50) + 1 = 196 + 1 = 197
  空间: floor((12   + 2×1 - 3 )/1 ) + 1 = floor(11/1)    + 1 = 11  + 1 = 12
  → (100, 5, 197, 12)

Pool1: kernel=2, stride=2, padding=1        ← nn.MaxPool2d(2, padding=1)
  时间: floor((197 + 2×1 - 2)/2) + 1 = floor(197/2) + 1 = 98 + 1 = 99
  空间: floor((12  + 2×1 - 2)/2) + 1 = floor(12/2)  + 1 = 6  + 1 = 7
  → (100, 5, 99, 7)                           ← 与官方注释一致

Conv2: kernel=(20,2), stride=(4,1), padding=1
  时间: floor((99 + 2 - 20)/4) + 1 = floor(81/4) + 1 = 20 + 1 = 21
  空间: floor((7  + 2 - 2 )/1) + 1 = 7 + 1 = 8
  → (100, 10, 21, 8)

Pool2: kernel=2, padding=0
  时间: floor((21 - 2)/2) + 1 = 9 + 1 = 10
  空间: floor((8  - 2)/2) + 1 = 3 + 1 = 4
  → (100, 10, 10, 4)

Flatten:  100 × 10×10×4 = 400      ← 论文 Table 3 的 "Linear: 400"
Linear(400, 6): (100, 6)            ← 6 类得分
```

**公式（背下来）**：

```
输出尺寸 = floor( (输入尺寸 + 2×padding − kernel) / stride ) + 1
```

**⚠️ 注意一个新手必踩的坑**：`nn.MaxPool2d(kernel_size=2, padding=1)` 里
**stride 没写**，PyTorch 默认 `stride = kernel_size`。
但如果写 `nn.MaxPool2d(2)`，stride 也是 2、padding 是 0——**两者输出尺寸不同**。
出错时一定要按公式手算一遍，别猜。

## 1.8 参数量：7421 个是怎么来的

```
Conv1:  权重 5 × 1 × 200 × 3 = 3000     ← out_ch × in_ch × kh × kw
        偏置 5
        ────────────────────────
        小计           3005

Conv2:  权重 10 × 5 × 20 × 2 = 2000
        偏置 10
        ────────────────────────
        小计           2010

Linear: 权重 400 × 6 = 2400
        偏置 6
        ────────────────────────
        小计           2406

总计 = 3005 + 2010 + 2406 = 7421   ✅ 与实测一致
```

**会算参数量有什么用？**
- 判断模型是不是太大（会不会过拟合）
- 判断显存够不够
- **面试/答辩常问**："你这个模型多大？"——你要能立刻算出来

## 1.9 Flatten 之后那 400 维是什么？

它是 10 个通道 × 10（时间）× 4（空间）的特征图展平的结果。

**注意一个重要事实：`flatten` 之后，空间信息就丢了。**
展平只是把 `(10, 10, 4)` 拉成 400 个数，位置关系不再保留。

**为什么可以接受？** 因为经过两层卷积+池化后，
每个位置的特征已经"混合"了原始的局部信息，
此时再展平做全连接，相当于"综合所有局部特征做最终判断"。这是 CNN 的标准做法。

**官方代码里那 400 维被存进了 `feature_data.csv`**，用途是画 LDA 图（论文 Fig. 5）。

---

# 第二部分：逐行拆解训练脚本

## 2.1 脚本骨架（五个部分，顺序固定）

```python
# ========== ① 导入与配置 ==========
import ...
class Logger: ...          # 日志（让输出同时进终端和文件）
sys.stdout = Logger(...)   # ★ 必须在所有 import 之后、其他代码之前

# ========== ② 评估函数 ==========
def test(model, loader, criterion): ...

# ========== ③ 单步训练函数 ==========
def train(model, x, y, optimizer, criterion): ...

# ========== ④ 画图函数 ==========
def draw(...): ...
def draw_result(C): ...

# ========== ⑤ 主流程 ==========
def main(args):
    # 数据 → 模型 → 优化器 → 循环 → 保存
```

## 2.2 日志：为什么要自己写一个 Logger

官方代码开头那段：

```python
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)   # 写到终端
        self.log.write(message)        # 同时写到文件
        self.log.flush()               # ★ 立刻刷盘（官方缺这一行！）

    def flush(self):
        self.log.flush()

sys.stdout = Logger('result.log', sys.stdout)
```

**它在干什么？** 劫持 `sys.stdout`。之后所有 `print()` 都会：
1. 正常显示在终端
2. **同时**写进 `result.log`

**为什么需要？** 训练一小时，你要能事后回看每个 epoch 的准确率。
终端一关就啥都没了。

**⚠️ 官方版本的致命缺陷**：`write()` 里没有 `flush()`，`flush()` 方法是空的 `pass`。
结果是**内容全积在缓冲区，只有程序退出那一刻才写盘**。

后果：
- 训练中途看日志——**空的**
- 程序崩溃——**日志也丢了**（缓冲区内容随进程消失）

**这就是你之前遇到的"怎么看进度都是空的"的原因。** 修复只加一行：`self.log.flush()`。

**为什么 `sys.stdout = Logger(...)` 要放在所有 import 之后？**
因为有些库在 import 时会往 stdout 写东西，如果提前劫持，
这些输出会进日志文件，混在你的训练记录里。

## 2.3 数据加载：三行代码背后的三层结构

```python
train_dataset = MyDataset(args.root, args.txtpath, transform=None)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
```

| 层 | 类 | 职责 | 关键参数 |
|---|-----|------|---------|
| 3 | `DataLoader` | 怎么批量喂 | `batch_size=100`（一次 100 个）、`shuffle=True`（每个 epoch 打乱） |
| 2 | `MyDataset` | 怎么取一个样本 | 实现 `__init__`/`__len__`/`__getitem__` |
| 1 | 磁盘文件 | 原始 `.mat` | — |

**`shuffle=True` 为什么必要？**
如果不打乱，每个 epoch 的 batch 顺序完全一样，
模型可能记住"前 100 个都是背景噪声"这种顺序信息，
导致学习效率下降甚至学偏。

**⚠️ 官方代码的一个小问题**：测试集的 `DataLoader` 也写了 `shuffle=True`。
评估时其实不需要打乱（因为最后是把所有预测汇总算混淆矩阵，顺序无关），
所以**不影响指标，但属于写法不严谨**。

## 2.4 模型与优化器

```python
model = CNN()
if torch.cuda.is_available():
    model = model.cuda()                    # 把模型参数搬到 GPU
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
```

**`model.cuda()` 在做什么？** 把模型的所有参数张量从内存复制到显存。
之后前向传播时数据也必须搬过去——**两者必须在同一设备上**，
否则就是你在第 47 行遇到的报错。

**Adam 优化器**（`lr=1e-4`、`weight_decay=1e-5`）：

Adam 是"带动量的自适应学习率"优化器。和鱼书里手写的 SGD 对比：

| | 鱼书手写 SGD | Adam |
|---|---|---|
| 更新公式 | `w -= lr × grad` | 用梯度的一阶矩和二阶矩自适应调整每个参数的学习率 |
| 每个参数的学习率 | 相同 | **各自不同** |
| 动量 | 无 | 有（抑制震荡） |
| 超参数 | 只有 lr | lr + β₁ + β₂ + weight_decay |

**`weight_decay=1e-5` 是 L2 正则化**，作用是惩罚过大的权重，防止过拟合。
数值很小，说明作者只是想做轻微约束。

**`nn.CrossEntropyLoss()` 到底算什么？**

它 = `LogSoftmax` + `NLLLoss`，两步合一：

```
① 把 6 个原始得分（logits）转成概率：
   p_i = exp(z_i) / Σ_j exp(z_j)        ← Softmax

② 取正确类别的概率，算负对数：
   loss = −log(p_correct)

③ 对 batch 内所有样本取平均
```

**为什么要取负对数？**
因为正确类别的概率越接近 1，`−log(p)` 越接近 0（loss 小）；
概率越接近 0，loss 越大。**这正好是我们要优化的方向。**

**⚠️ 易错点**：用 `CrossEntropyLoss` 时，模型**输出的必须是原始得分，不能先过 softmax**，
否则等于做了两次 softmax，结果会错。
（官方 `models.py` 的 `self.out` 是纯 `Linear`，正确。）

## 2.5 训练循环：逐行

```python
for epoch in range(args.epochs):                    # 50 轮
    tic = time.time()
    train_predict, train_label = [], []
    runloss = 0

    for (cnt, i) in enumerate(train_loader):        # 每轮 124 个 batch
        batch_x = i['data']                         # (100, 10000, 12)
        batch_y = i['label']                        # (100,)

        batch_x = torch.unsqueeze(batch_x, dim=1)   # ★ 变成 (100, 1, 10000, 12)
        batch_x = batch_x.float()                   # ★ 转 float32

        if torch.cuda.is_available():
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

        tlabels, tpredi, tloss = train(model, batch_x, batch_y, optimizer, criterion)
        runloss += tloss
        train_label.extend(tlabels)
        train_predict.extend(tpredi)

    # ---- 本轮结束，统计与评估 ----
    taccuracy = accuracy_score(train_label, train_predict)
    loss = runloss / batches_per_epoch
    acc_score, loss_score, feature_list, C = test(model, test_loader, criterion)
    print("Epoch %d Val_accuracy %.3f Val_loss %.3f" % (epoch, acc_score, loss_score))
```

### 逐行解释

**`torch.unsqueeze(batch_x, dim=1)` —— 这一行最容易漏**

```
(100, 10000, 12)  →  (100, 1, 10000, 12)
                        ↑ 在第 1 维插入"通道数 = 1"
```

**为什么必须插？** 因为 `nn.Conv2d` 要求输入是 **`(N, C, H, W)`** 四维：
- `N` = batch size
- `C` = 通道数（彩色图是 3，灰度图是 1）
- `H, W` = 高、宽

我们的数据是单通道的"灰度图"，所以要插一个 `C=1`。
**这就是"把时空矩阵当成图片"的具体操作。**

**`.float()` —— 为什么必须转**

数据从 `.mat` 读出来是 `int64`（因为代码里 `.astype(int)`），
而卷积层要求 `float32`。不转的话会报：
```
RuntimeError: expected scalar type Float but found Double
```

**`.cuda()` —— 数据必须和模型在同一设备**

`model` 在第 142-143 行已经搬到 GPU 了，
所以 `batch_x`、`batch_y` 也必须搬，否则报设备不一致错误。

**⚠️ 官方代码在这里有个隐患**：`batch_y = batch_y.cuda()` 之后，
在 `test()` 函数里又和 CPU 上的空张量做 `torch.cat` → **就是你在第 47 行修的 bug**。

### `train()` 函数内部：四步走

```python
def train(model, train_x, train_y, optimizer, criterion):
    model.train()                          # ① 切到训练模式
    model.zero_grad()                      # ② 清空上一轮的梯度
    _, probs = model(train_x)              # ③ 前向传播
    loss = criterion(probs, train_y)       # ④ 算损失
    _, pred = torch.max(probs, dim=1)      #    （顺便取预测类别，仅用于统计）
    loss.backward()                        # ⑤ 反向传播：算梯度
    optimizer.step()                       # ⑥ 更新参数
    return train_y.tolist(), pred.tolist(), loss.item()
```

**这六步就是深度学习的核心循环**，和你鱼书学的完全对应：

| PyTorch | 鱼书（你学过的） | 作用 |
|---------|----------------|------|
| `model.zero_grad()` | `self.grads[...] = None`（重置梯度） | 清空累积的梯度 |
| `model(train_x)` | `predict()` / `forward()` | 前向传播 |
| `criterion(...)` | `SoftmaxWithLoss.forward()` | 算损失 |
| `loss.backward()` | **`backward()`** | **反向传播算梯度** ← 你学的那章 |
| `optimizer.step()` | `self.params[key] -= lr * grad[key]` | 用梯度更新参数 |

**为什么梯度要清空？**
因为 PyTorch 的梯度是**累加**的（`grad += new_grad`）。
不清空的话，上一轮的梯度会叠加进来，参数更新方向就错了。

**`loss.item()`**：把单元素张量转成 Python 浮点数。
用它是为了**避免显存泄漏**（保留张量会一直占着计算图）。

**`loss.backward()` 背后发生了什么？**
PyTorch 在每次前向传播时，会自动记录所有运算构成一张**计算图**。
`backward()` 从损失出发，沿计算图反向应用链式法则，算出每个参数的梯度。

**这正是你鱼书里手写的 `backward()`——只不过 PyTorch 帮你自动做了。**
你学的"误差反向传播"就是这一行的原理。

## 2.6 评估函数 `test()`：和 `train()` 的三个区别

```python
def test(model, dataset, criterion):
    model.eval()                     # ★ 区别1：切到评估模式
    ...
    for (step, i) in enumerate(dataset):
        batch_x = torch.unsqueeze(i['data'], dim=1).float()
        if torch.cuda.is_available():
            batch_x = batch_x.cuda()
            batch_y = i['label'].cuda()
        feature, probs = model(batch_x)
        ...
        prediction.extend(pred.tolist())
        labels.extend(batch_y.tolist())
    accuracy = accuracy_score(labels, prediction)
    C = confusion_matrix(labels, prediction)
    return accuracy, val_loss/total_batch_num, feature_list, C
```

| | `train()` | `test()` |
|---|-----------|----------|
| 模式 | `model.train()` | **`model.eval()`** |
| 梯度 | `loss.backward()` | **不算**（官方漏了 `torch.no_grad()`，是个隐患） |
| 参数 | `optimizer.step()` | **不更新** |

**为什么必须切换模式？**
因为 `Dropout`、`BatchNorm` 这类层在训练和推理时行为**完全不同**：
- `Dropout` 训练时随机丢弃神经元，推理时不丢
- `BatchNorm` 训练时用当前 batch 的统计量，推理时用训练期间累积的统计量

**本项目 CNN 没有这两层**，所以切换**看起来**没影响——**但一定要养成习惯**，
以后用 ResNet 之类漏掉这行，结果会莫名其妙变差。

**⚠️ 官方代码漏了 `torch.no_grad()`**：评估时不需要计算图，
不关掉的话会白白占用显存、拖慢速度。正确写法：

```python
with torch.no_grad():
    feature, probs = model(batch_x)
```

## 2.7 保存与出图

```python
if epoch == args.epochs - 1:            # 只在最后一轮做
    torch.save(model, 'model.pth')                       # ① 存模型
    feature_list = feature_list.detach().numpy()
    np.savetxt('feature_data.csv', feature_list, delimiter=',')   # ② 存特征
    draw_result(C)                                       # ③ 画混淆矩阵
draw(train_acc_list, train_loss_list, test_acc_list, test_loss_list)  # ④ 画曲线
```

**`.detach().numpy()` 为什么要两步？**
- `.detach()`：从计算图里摘出来（否则张量还连着梯度，不能转 numpy）
- `.numpy()`：转成 NumPy 数组（`np.savetxt` 只接受 numpy 数组）

**`torch.save(model, ...)` vs `torch.save(model.state_dict(), ...)`**

| | 存整个模型 | 只存参数 |
|---|---|---|
| 写法 | `torch.save(model, 'a.pth')` | `torch.save(model.state_dict(), 'a.pth')` |
| 加载 | `torch.load('a.pth')` | 需要先建模型再 `load_state_dict` |
| 优点 | 省事 | **更稳、更小、更推荐** |
| 缺点 | 依赖类定义，换环境/换代码结构可能加载失败 | 要多写两行 |

**官方用的是存整个模型**。这在你加载时可能遇到问题
（比如 `torch.load` 在较新版本里 `weights_only` 默认变了）。
我写的 `eval_cnn.py` 里就明确写了 `weights_only=False` 来兼容。

## 2.8 官方代码的几个数值小问题（知道你踩过就行）

| 位置 | 问题 | 影响 |
|------|------|------|
| 第 146 行 | `batches_per_epoch = int(len(train_dataset)/batch_size)` = 123（其实是**样本数**除以 batch） | 用于算平均 loss，**分母偏小 1 倍**，显示的训练 loss 偏大——不影响训练本身 |
| 第 173 行 | `loss = runloss / batches_per_epoch` | 承上 |
| `test()` 里 | `feature_list = torch.tensor([])` 建在 CPU，循环里不断 `torch.cat` | 效率低（每次重新分配内存），且**导致你修的设备不一致 bug** |
| 第 188 行 | `print('train totally using %.3f seconds ', train_time)` | **格式串写错了**（`%` 写成了 `,`），输出的是字面量而不是数字 |

> 最后一条很好玩：输出会变成 `train totally using %.3f seconds  2234.42`
> ——格式串原样打印出来了。**这类 bug 不影响运行，只影响你读日志。**

---

# 第三部分：完整可复用模板

## 3.1 第一步永远是：搞清楚一个样本是什么

**在新项目上，你要先回答这 5 个问题，才能定网络结构：**

| 问题 | 本项目答案 | 怎么影响模型 |
|------|-----------|-------------|
| 一个样本的形状？ | `(10000, 12)` 二维 | **二维 → 用 `Conv2d`** |
| 有空间维度吗？ | 有（12 个位置点） | 空间方向用小核（3） |
| 时间尺度多大？ | 采样率 10 MS/s | 事件持续 ~20 μs → 核取 200 |
| 有几个通道？ | 1（单通道"灰度图"） | `in_channels=1` |
| 类别数？ | 6 | 最后一层 `Linear(..., 6)` |

**形状决定模型类型，这是铁律**：

| 样本形状 | 该用什么 | 卷积核 |
|---------|---------|--------|
| `(T,)` 一维时序 | `Conv1d` | `kernel_size=(200,)` |
| `(T, S)` 时空 | **`Conv2d`** ← 本项目 | `kernel_size=(200, 3)` |
| `(H, W)` 图像 | `Conv2d` | `kernel_size=(3, 3)` |
| `(H, W, C)` 多通道图像 | `Conv2d`，注意 `C` 要移到第 1 维 | `kernel_size=(3, 3)` |

## 3.2 数据清洗：分三步，不要一上来就改

```
第一步：检测（有没有问题？）      ← 占 80% 的功夫
第二步：判断（这个问题影响大吗？）  ← 需要领域知识
第三步：决策（改？扔？留？标记？）  ← 改之前想清楚代价
```

**常见的"数据病"**：

| 病症 | 怎么检测 | 处方 | 什么时候**不要**用这个处方 |
|------|---------|------|---------------------------|
| 缺失值 | `np.isnan(x).sum()` | 删样本 / 填充 / 插值 | 缺失本身有含义时 |
| 异常值 | 看范围、画箱线图 | 3σ 截断 | **信号识别里异常可能就是信号！** |
| 重复样本 | 文件哈希、特征比对 | 去重 | 有意增广的数据 |
| 类别不平衡 | 统计每类数量 | 重采样 / 类别权重 | 不平衡 < 2 倍通常不用管 |
| **数据泄漏** | **检查 train/test 有无重叠** | 删重叠 | **永远不要放过** |
| 坏样本 | `std == 0` | 删掉 | — |
| 量纲不一 | 看各特征数值范围 | 标准化/归一化 | 树模型不需要 |

**本项目的体检结论**（用 `data_profile.py` 跑的）：

```
文件完整性  → 12335 + 3084，与 label.txt 一致   → 不用处理
形状/类型   → 全是 data, (10000,12), uint16      → 不用处理
NaN         → 无                                  → 不用处理
类别平衡    → 最多/最少 = 1.31 倍                 → 不用处理（轻度）
train/test  → 无重叠                              → 不用处理
死通道      → 无                                  → 不用处理
```

**所以这个数据集不需要"清洗"，只需要"归一化"。**

> **这本身是个重要认识**：论文级公开数据集通常是干净的。
> 清洗的功夫主要花在**自己采的数据**上。

## 3.3 归一化：为什么必须做，以及怎么做

**为什么必须？** 因为数据范围是 7800~8650，如果不归一化：
- 数值太大，梯度容易爆炸
- 各通道量纲不一，模型会偏向数值大的通道

**论文的做法**：每个样本**各自**映射到 0~255（模拟灰度图）。

| 归一化方式 | 公式 | 什么时候用 |
|-----------|------|-----------|
| Min-Max（本项目） | `(x − min) / (max − min)` | 范围已知、无极端离群值 |
| 标准化 Z-score | `(x − μ) / σ` | 近似正态分布 |
| 按通道归一化 | 每个通道单独算 min/max | 各通道量纲差异大时 |

**⚠️ 一个必须验证的点**：官方的 `normalize` 用的是 Python 内建 `round()`
和双层循环，我改成 `np.round()` 和向量化后，
**必须逐元素比对确认结果一致**（我做了 144 万个元素的比对，不一致数为 0）。

**换成自己的数据时也要做这个验证。**

## 3.4 完整模板（照抄改三处即可）

```python
# ============================================================
#  模板：任意分类任务的训练脚本
#  需要改的只有 ① 数据配置 ② Dataset ③ 网络结构
# ============================================================
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score


# ---------- ① 日志（抄这段，加 flush）----------
class Logger:
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='utf-8')
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()                    # ★ 别漏
    def flush(self):
        self.log.flush()


# ---------- ② Dataset（改这里：怎么读一个样本）----------
class MyDataset(Dataset):
    def __init__(self, root_dir, names_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(names_file, encoding='utf-8-sig') as f:
            self.names = [ln for ln in f if ln.strip()]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        parts = self.names[idx].split(' ')
        path = os.path.join(self.root_dir, parts[0].lstrip('/\\'))
        # ↓↓↓ 这三行是每个项目要改的核心 ↓↓↓
        data = np.load(path)                       # 换成你的读取方式
        data = data.astype(np.float32)
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        # ↑↑↑
        label = int(parts[1])
        return {'data': torch.from_numpy(data), 'label': label}


# ---------- ③ 模型（改这里：网络结构）----------
class MyCNN(nn.Module):
    def __init__(self, in_ch=1, n_class=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 5, kernel_size=(200, 3), stride=(50, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1),
            nn.Conv2d(5, 10, kernel_size=(20, 2), stride=(4, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Linear(10 * 10 * 4, n_class)   # ★ 这个 400 要自己算！

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)          # flatten
        return self.classifier(x)          # 只返回得分


# ---------- ④ 单步训练 ----------
def train_step(model, x, y, optimizer, criterion):
    model.train()
    optimizer.zero_grad()                  # 清梯度
    logits = model(x)                      # 前向
    loss = criterion(logits, y)            # 算损失
    loss.backward()                        # 反向
    optimizer.step()                       # 更新
    return loss.item()


# ---------- ⑤ 评估 ----------
@torch.no_grad()                           # ★ 官方漏了这个
def evaluate(model, loader, criterion, device):
    model.eval()
    preds, labels, total_loss, n_batch = [], [], 0.0, 0
    for batch in loader:
        x = batch['data'].unsqueeze(1).float().to(device)   # ★ 插通道维
        y = batch['label'].to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        n_batch += 1
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        labels.extend(y.cpu().tolist())
    return (accuracy_score(labels, preds),
            total_loss / max(n_batch, 1),
            confusion_matrix(labels, preds))


# ---------- ⑥ 主流程 ----------
def main(args):
    sys.stdout = Logger(args.log)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 固定随机种子（可复现）
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    train_ds = MyDataset(args.train_root, args.train_txt)
    test_ds = MyDataset(args.test_root, args.test_txt)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = MyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    print("样本数  train=%d  test=%d   设备=%s" % (len(train_ds), len(test_ds), device))
    print("模型参数量 = %d" % sum(p.numel() for p in model.parameters()))

    best_acc = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        run_loss = 0.0
        for batch in train_loader:
            x = batch['data'].unsqueeze(1).float().to(device)
            y = batch['label'].to(device)
            run_loss += train_step(model, x, y, optimizer, criterion)
        acc, test_loss, C = evaluate(model, test_loader, criterion, device)
        print("Epoch %3d/%d  train_loss=%.4f  test_acc=%.4f  test_loss=%.4f  (%.1fs)"
              % (epoch, args.epochs - 1, run_loss / len(train_loader),
                 acc, test_loss, time.time() - t0))
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.ckpt)     # 存最优
    print("\n最优测试准确率 = %.4f" % best_acc)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_root', default=r'D:\...\train')
    p.add_argument('--train_txt',  default=r'D:\...\train\label.txt')
    p.add_argument('--test_root',  default=r'D:\...\test')
    p.add_argument('--test_txt',   default=r'D:\...\test\label.txt')
    p.add_argument('--log',    default='train_result.log')
    p.add_argument('--ckpt',   default='best_model.pth')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    main(p.parse_args())
```

**这个模板相比官方代码的改进**（都是你在本项目里踩过的坑）：

| 改进 | 解决什么问题 |
|------|-------------|
| `Logger` 加 `flush()` | 训练中途看不到日志 |
| `torch.no_grad()` + `@torch.no_grad()` | 评估白占显存 |
| 固定随机种子 | 结果不可复现 |
| `state_dict()` 存模型 | 换环境加载失败 |
| `shuffle=False`（测试集） | 写法不严谨 |
| 保留**最优**模型 | 官方只存最后一轮，可能恰好是低谷 |
| 参数量打印 | 一眼知道模型多大 |
| 输出带进度和耗时 | 好判断有没有卡住 |

## 3.5 出图：三种图怎么画

### 图 1：混淆矩阵（论文 Fig. 4）

```python
import matplotlib
matplotlib.use('Agg')                       # ★ 无 GUI 环境必须加，否则 plt.show() 卡死
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']   # ★ 中文显示
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(C, cmap='Blues')             # C 是 confusion_matrix 的输出
for i in range(6):
    for j in range(6):
        ax.text(j, i, str(C[i, j]), ha='center', va='center',
                color='white' if C[i, j] > C.max()*0.55 else 'black')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(NAMES, rotation=30, ha='right'); ax.set_yticklabels(NAMES)
plt.colorbar(im, ax=ax)
plt.savefig('confusion.png', dpi=150, bbox_inches='tight')
```

**看图的方法（这比画图更重要）**：
- **对角线** = 正确分类，颜色越深越好
- **非对角线** = 错误。**要问"错到哪去了"，而不是只看错多少**
- 如果某一列特别深，说明**很多类都被误判成这一类**
  （本项目 SVM 就是很多事件被误判为背景噪声）

### 图 2：训练曲线

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(train_acc, label='train'); ax1.plot(test_acc, label='test')
ax1.set_ylabel('accuracy'); ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(train_loss, label='train'); ax2.plot(test_loss, label='test')
ax2.set_xlabel('epoch'); ax2.set_ylabel('loss'); ax2.legend(); ax2.grid(alpha=0.3)
```

**怎么从曲线看出问题**（这是核心技能）：

| 曲线形态 | 说明 | 怎么办 |
|---------|------|--------|
| train 和 test 都低、都平 | **欠拟合** | 模型太小 / 训练不够 / 学习率太小 |
| train 高、test 低且差距越来越大 | **过拟合** | 加正则化 / 数据增广 / 早停 |
| 两条都剧烈震荡 | 学习率太大 | 调小 lr |
| train 下降但 test 不动 | 数据有问题 或 训练/测试分布不同 | **回头查数据** |
| test 比 train 还高 | 通常正常（测试集简单，或 train 有 dropout） | 不用慌 |
| loss 变成 NaN | 学习率太大 / 梯度爆炸 | 调小 lr / 加梯度裁剪 |

### 图 3：LDA 特征可视化（论文 Fig. 5）

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=3)     # 降到 3 维
Z = lda.fit_transform(X, y)                          # ★ 用到了标签（有监督）
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for c in range(6):
    m = (y == c)
    ax.scatter(Z[m,0], Z[m,1], Z[m,2], s=12, marker='.', label=NAMES[c])
ax.legend(); ax.view_init(elev=30, azim=45)
```

**LDA vs PCA**：
- **PCA** 无监督，只找方差最大的方向
- **LDA** 有监督（用标签），目标是**类间距离最大、类内距离最小**

**所以 LDA 图"看起来分得开"是预期的**，不能据此判断分类器好坏，
只能定性观察类别结构。

## 3.6 怎么"得到结论"——从数字到结论的完整链条

**光有一个准确率数字，不能叫结论。完整链条是：**

```
① 单次结果        acc = 0.9290
        ↓  但单次结果可能碰巧好/坏
② 多种子重复      0.9265 ± 0.0054  (n=3)
        ↓  有了均值和标准差
③ 与基线对比      SVM 0.8861  vs  CNN 0.9265   → CNN 更优（+4.03 点）
        ↓  但"更优"要有依据
④ 统计检验        差距是否超出波动范围？
        ↓  有了定量判断
⑤ 误差分析        错在哪？为什么错？→ 从混淆矩阵读出模式
        ↓  解释"为什么"
⑥ 与文献对比      论文 0.940，我们 0.9265 → 差 1.35 点 → 分析原因
        ↓
⑦ 结论 + 局限     明确说什么成立、什么不确定
```

**本项目走完这条链后的结论**（这就是"结论"的样子）：

> 1. CNN（92.65% ± 0.54%）显著优于 SVM（88.61%），提升 4.03 个点，
>    优势集中在背景噪声类（F1 0.878 → 0.960），验证了论文的核心论断。
> 2. 与论文的 94.0% 相差 1.35 点，相当于 2.5 个标准差（仅计训练随机性）。
>    **但论文数据划分带来的方差无法评估，故不能判定差距显著。**
> 3. 误差集中在"走动"类（Recall 0.829），错误分散到 5 个类别。
>    与论文解释一致——走动样本包含不同频率成分，部分仅含单次踩踏。
> 4. **局限**：多种子实验固定了测试集，未评估数据划分方差。

**注意第 2 条和第 4 条的措辞**——**没有夸大，也没有隐藏不确定性**。
这是科研结论该有的样子。

---

# 第四部分：用你项目的实际数字验证你的理解

## 4.1 "小数据快速迭代"的正确做法

**永远不要一上来就全量跑。** 先做这个检查：

```python
# 只用 40 个样本，跑 2 个 epoch，看 loss 会不会降
small = torch.utils.data.Subset(train_ds, range(40))
small_loader = DataLoader(small, batch_size=8, shuffle=True)
# ... 训练 ...
print("loss:", losses)   # 应该下降
```

**为什么？** 如果全量跑，一个 bug 要等半小时才暴露。
小数据验证只要 10 秒。

**"小数据快速迭代"是所有工程师的通用技巧。**

## 4.2 数据加载的"单元测试"（6 行，必做）

```python
ds = MyDataset(root, label_file)
print('样本总数:', len(ds))                 # 对不对？
s = ds[0]
print('data 形状:', s['data'].shape)        # 形状对吗？
print('label:', s['label'])                 # 标签对吗？
dl = DataLoader(ds, batch_size=4, shuffle=True)
for batch in dl:
    print('batch 形状:', batch['data'].shape)   # 应该是 (4, ...)
    break
```

**这 6 行能排掉 80% 的数据加载 bug**，特别是"标签和文件错配"这种
不会报错、只会让准确率莫名很低的坑。

## 4.3 性能优化的正确顺序

**先测，再改。** 永远不要"觉得哪里慢就改哪里"。

```python
# 测一个 epoch 的两个部分
t0 = time.time()
for batch in train_loader:      # 只遍历数据，不训练
    pass
print("纯数据加载: %.1f 秒/epoch" % (time.time() - t0))

t0 = time.time()
for batch in train_loader:
    x = batch['data'].unsqueeze(1).float().to(device)
    _ = model(x)                # 只前向
print("数据+前向: %.1f 秒/epoch" % (time.time() - t0))
```

本项目的实测结果（这是发现问题的关键一步）：

```
一个 batch（100 样本）:
  数据加载 + 归一化:  15.24 秒   ← CPU
  前向 + 反向传播:     0.48 秒   ← GPU
                      ────────
  GPU 利用率 = 0.48 / 15.72 ≈ 3%
```

**GPU 有 97% 的时间在等 CPU 喂数据。**

**找到瓶颈后，只优化瓶颈**：

| 瓶颈 | 优化手段 | 本项目效果 |
|------|---------|-----------|
| **Python 循环**（本例） | **向量化** | **188 倍** |
| 单进程读数据 | `num_workers=4` | 最多 4 倍（本例已不需要） |
| 模型本身太慢 | 换更小的模型 / 混合精度 | — |
| GPU 太弱 | 换显卡 | — |

**工程原则：先优化写法，再考虑加资源。反过来做通常是浪费。**

## 4.4 你要能回答的 10 个问题（自测）

1. 输入 `(100, 1, 10000, 12)`，`Conv2d(1,5,(200,3),(50,1),padding=1)` 之后是什么形状？怎么算的？
2. 为什么必须 `unsqueeze(dim=1)`？不插这一维会怎样？
3. `CrossEntropyLoss` 内部做了哪两步？为什么模型输出不能先过 softmax？
4. `model.zero_grad()` 不清空会怎样？为什么 PyTorch 要设计成梯度累加？
5. `train()` 和 `test()` 有哪三个区别？漏掉 `model.eval()` 会有什么后果？
6. `loss.backward()` 和你鱼书里手写的 `backward()` 是什么关系？
7. 这个模型 7421 个参数是怎么算出来的？怎么算一个卷积层的参数量？
8. 训练 loss 下降但测试准确率不涨，你会先查什么？
9. 为什么优化性能之前必须先测"瓶颈在哪"？
10. 只有一个准确率数字，为什么不能算"结论"？要补哪些东西？

---

# 第五部分：你的下一个项目——完整清单

```
【阶段 1】搞清楚数据（半天）
  □ 打开一个文件，打印形状/类型/变量名
  □ 写一句话："一个样本是 ______"
  □ 问导师 6 类问题（任务/来源/标签/指标/领域知识/边界）

【阶段 2】数据体检（1 天）
  □ data_profile.py 改 CONFIG，跑一遍
  □ 每类画 3 个样本的图，用眼睛看
  □ 标签对齐核对（顺序、缺失、重复、冲突）

【阶段 3】数据工程（2~5 天）
  □ 切样本（决定切多长、从哪切、要不要重叠）
  □ 生成统一格式 label.txt
  □ 划分 train/val/test（★ 小心数据泄漏）
  □ 写 Dataset + 单元测试 6 行

【阶段 4】建模（2~3 天）
  □ 40 个样本跑 2 个 epoch，确认 loss 能降
  □ 建立基线（先跑最简单模型）
  □ 测性能瓶颈，只优化瓶颈
  □ 调参（每次只改一个变量，记录结果）

【阶段 5】出结论（1~2 天）
  □ 多种子重复 → 均值 ± 标准差
  □ 与基线对比 + 统计检验
  □ 误差分析（从混淆矩阵读模式）
  □ 与文献/论文对比
  □ 写报告（结论 + 证据 + 局限）

【全程】
  □ bug_记录.md 随手记（现象/原因/修复/影响）
  □ 每次实验的输出带参数命名（否则一周后分不清哪张是哪次）
  □ 原始版本和修改版本并存
```

---

*本文档与 `06_数据处理的为什么.md`（心智模型）、
`07_项目全景_文件地图.md`（文件结构）、
`08_拿到rawdata该怎么做.md`（行动流程）配合使用。*
