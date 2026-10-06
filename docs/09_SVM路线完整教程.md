# SVM 路线完整教程：从 rawdata 到结论

> **这篇的目标**：讲透 SVM 这条路线的每一个环节。学完你应该能：
> 拿一份新的 rawdata，自己走完"清洗 → 导入 → 特征 → 建模 → 训练 → 日志 → 出图 → 结论"全流程。
>
> **SVM 路线的独特之处**：它不需要 GPU、不需要反向传播、几十行代码就能跑完。
> **它是你理解"机器学习项目全流程"这条链子的最佳载体**——因为变量少、跑得快、结果可解释。
> 先把这条链子走通，再上 CNN，你会轻松很多。

---

## 目录

| 环节 | 回答什么问题 | 核心文件 |
|------|------------|---------|
| 0 | SVM 到底是什么？为什么这条路要"提特征"？ | — |
| 1 | 数据清洗：拿到 rawdata 先做什么？ | `data_profile.py` |
| 2 | 数据导入：怎么把文件变成模型能吃的矩阵？ | `get_das_data.py` |
| 3 | 特征工程：384 维特征是怎么算出来的？ | `feature_extraction.py` |
| 4 | 预处理：为什么必须归一化？怎么划 train/test？ | — |
| 5 | 模型构建：SVM 怎么训练？核函数是什么？ | `das_data_svm.py` |
| 6 | 日志：怎么让程序边跑边告诉你进度？ | `Logger` 类 |
| 7 | 评估：混淆矩阵和 P/R/F1 怎么算？ | `das_data_svm.py` |
| 8 | 出图：怎么画出能放进报告的图？ | `make_svm_report.py` |
| 9 | 结论：怎么从数字里读出有意义的分析？ | — |
| 10 | 完整代码：一份可以直接改用的模板 | — |

---

## 〇、先搞懂 SVM 是什么，以及这条路为什么特殊

### 0.1 一句话理解 SVM

**SVM（支持向量机）做的事：找一条"最宽的马路"把两类数据分开。**

```
        ○ ○ ○                     ○ = 类别 A
      ○ ○ ○ ○                     ● = 类别 B
    ─────────────  ← 决策边界（马路的中间线）
        ● ● ●
      ● ● ● ●
```

"最宽"是它的核心思想：不是随便找一条能分开的线，而是找**离两边最近的点都尽量远**的那条线。
那些"离边界最近的点"就叫**支持向量（support vector）**——SVM 的名字来源。

**为什么这个思想好？** 因为留出的"余量"（margin）越大，模型对新数据的容错就越好。
这是 SVM 比"随便画条线"更抗过拟合的根本原因。

### 0.2 多分类怎么办？

SVM 本质是**二分类**器。6 类怎么办？用 **one-vs-one**：

```
6 个类别 → 两两配对 → C(6,2) = 15 个二分类器
   (背景 vs 挖掘)、(背景 vs 敲击)、(背景 vs 浇水)...
   (挖掘 vs 敲击)、(挖掘 vs 浇水)...
   ...
预测时：让 15 个分类器投票，得票最多的类获胜
```

代码里就是这一行参数：

```python
svm.SVC(decision_function_shape='ovo')     # ovo = one-vs-one
```

### 0.3 核函数：为什么要"rbf"

如果两类数据不是"一条直线能分开"的（比如一类把另一类包围在中间），
线性边界就无能为力。**核函数**的作用是：把数据"升维"到高维空间，
在那里它们就变线性可分了。

```
二维平面：        升到三维后：
   ○ ○            ○ ○
  ○ ● ○     →        ○ ● ○     ← 在三维里可以用一个平面切开
   ○ ○                ○ ○
（无法用直线分开）      （可以用平面分开）
```

代码里：

```python
svm.SVC(kernel='rbf')     # 径向基核，最常用的非线性核
```

其他常见选择：`'linear'`（线性，数据本身线性可分时用）、`'poly'`（多项式）、`'sigmoid'`。

### 0.4 ★ 关键：为什么 SVM 这条路必须"手工提特征"

这是理解整个 SVM 路线的核心，**请务必看懂这一段**。

**SVM 吃的是什么？** 一个**固定长度的向量**，比如 `[1.5, 0.3, 8.2, ...]`。
它**不知道**这个向量是从哪来的，也不懂"时间"和"空间"的概念。

**而你的原始数据是什么？** 一个 10000×12 的**矩阵**——有时间维、有空间维。

**如果直接把矩阵拉平（10000×12 → 120000 维向量）喂给 SVM 会怎样？**

| 问题 | 后果 |
|------|------|
| 维度 12 万维 | 计算量爆炸（SVM 训练复杂度约 O(n²~n³)） |
| 相邻时间点高度相关 | 信息大量冗余，等于没提 |
| 时间/空间结构被打乱 | 模型失去了它本来能利用的信息 |
| 样本只有 12335 个 | **维度远大于样本数** → 严重过拟合 |

**所以必须"降维"——把 10000×12 的矩阵压缩成几十到几百个有代表性的数字。
这个动作叫「特征提取」（feature extraction）。**

```
原始：10000 × 12 = 120000 个数
          ↓ 特征提取
特征：  12 通道 × 16 特征 × 2（原始+差分）= 384 个数
```

**这 384 个数要尽量保留"能区分事件"的信息。** 怎么做到？见第 3 节。

> **对比 CNN**：CNN 不需要手工提特征，它直接从 10000×12 的矩阵里自己学。
> 这就是"端到端（end-to-end）"的含义。
> **也正因如此，CNN 效果更好（94% vs 82%）——它保留了 SVM 丢掉的时空结构信息。**

**这条对比链，是你复现报告里最有价值的一段分析。**

---

## 一、数据清洗：拿到 rawdata 先做什么？

> **详细原理见 `06_数据处理的为什么.md` 第二节。这里只讲 SVM 路线的具体做法。**

### 1.1 核心观念：清洗 ≠ 改数据

清洗是**三个动作**：

```
① 检测（有没有问题？）      ← 占 80% 的功夫
② 判断（影响大吗？）        ← 需要领域知识
③ 决策（改？扔？留？）      ← 改之前先想清楚代价
```

**新手最容易跳过 ① 直接做 ③**——看到异常值就删掉，
结果删掉了真正的信号。**在信号识别任务里，异常往往就是你要找的东西！**
（本项目的"敲击"事件就是一个尖峰。）

### 1.2 通用体检工具：`data_profile.py`

**它检查什么：**

| 检查项 | 怎么检 | 为什么重要 |
|--------|--------|-----------|
| 文件完整性 | 数文件数 vs label.txt 行数 | 有文件没标签 / 有标签没文件，都会让训练出错 |
| 形状/类型一致 | 抽样读文件，比对 shape 和 dtype | 不一致会在拼接时崩溃 |
| NaN/Inf | `np.isnan(x).sum()` | 会让模型输出 NaN |
| 类别平衡 | 统计每类数量，算最多/最少比 | 严重不均衡时准确率会骗人 |
| **train/test 重叠** | 文件名或内容比对 | **数据泄漏，永远不能放过** |
| 死通道 | 某通道 `std == 0` | 无效特征，浪费维度 |
| 坏样本 | 全 0 / 常量 | 噪声样本 |
| 标签顺序 | label.txt 顺序 vs 磁盘文件名顺序 | 按行读取时顺序错位会导致标签整体错配 |

**使用方法**：改顶部 CONFIG 区，然后 `python data_profile.py`。

```python
# ============================================================
#  CONFIG —— 换项目时只改这里
# ============================================================
DATA_ROOT = r"...\das_data"
CLASSES = {"01_background": (0, "背景噪声"), ...}   # 文件夹名 -> (标签, 中文名)
FILE_PATTERN = "*.mat"          # 数据文件后缀
MAT_KEY = "data"                # .mat 里存数据的变量名
EXPECTED_SHAPE = (10000, 12)    # 期望形状
SPLITS = ["train", "test"]
```

**为什么要把 CONFIG 单独提出来？** 因为"项目相关的常量"和"通用逻辑"要分开。
换数据集时只改上面几行，下面 200 行逻辑一行不动。**这是写工具脚本的正确姿势。**

### 1.3 本项目体检的结论

| 检查项 | 结果 | 需要处理吗 |
|--------|------|-----------|
| 文件完整性 | train 12335 + test 3084，与 label.txt 一致 | ❌ |
| 形状/类型 | 全部 `data`，`(10000,12)`，`uint16` | ❌ |
| NaN | 没有 | ❌ |
| 类别平衡 | 最多/最少 = 1.31 倍 | ❌（轻度） |
| train/test 重叠 | 无重叠 | ❌ |
| 死通道 / 坏样本 | 无 | ❌ |
| 标签顺序 | 与磁盘顺序一致 | ❌ |

**结论：这个数据集不需要"清洗"，只需要"归一化"。**

> **这本身是个重要认识**：论文级公开数据集通常是干净的。
> **清洗的功夫主要花在自己采的数据上**——而那一天迟早会来。

### 1.4 ★ 一个反直觉的发现（本项目测出来的）

| 类别 | 样本全局标准差 |
|------|--------------|
| 背景噪声 | **392.1** ← 最大 |
| 挖掘 | 106.9 ← 比背景还小 |
| 敲击 | 245.6 |

**"标准差大 = 有事件"这个直觉是错的！**

看通道级的 std 就明白了：

```
挖掘:  [16.6 16.3 20.1 37.0 59.9 64.9 59.1 63.9 36.0 22.9 17.6 15.9]
        ↑──── 安静背景 ────↑ ↑──────── 扰动区 ────────↑ ↑── 安静 ──↑

背景:  [45.1 42.5 78.8 70.7 23.0 19.1 17.1 18.4 28.3 37.6 32.9 22.0]
        ↑── 各种环境干扰散布在多个通道 ──↑
```

**背景噪声是"到处都乱但都不太剧烈"，挖掘是"局部非常剧烈、其余安静"。**

**这个发现的因果链（报告里最有价值的一段）：**

```
全局统计量（std、max、mean...）无法区分"局部剧烈"和"全局杂乱"
                    ↓
SVM 用的是全局统计量特征 → 所以只有 82.6%
CNN 看完整的时空模式     → 所以能做到 94%
                    ↓
        ★ 这就是 CNN 更优的根本原因
```

**教学意义**：**先画图，再看统计量，不要反过来。** 统计量会骗你。

### 1.5 画图：唯一能发现"数据和你预期不一样"的方法

```python
import numpy as np, scipy.io as scio, matplotlib.pyplot as plt

data = scio.loadmat(path)['data']          # (10000, 12)
plt.imshow(data.T, aspect='auto', cmap='inferno')
#        ↑ 转置：让横轴=时间(10000)，纵轴=空间(12)
#                 aspect='auto' 允许非正方形像素（否则 12×10000 被压成一条线）
plt.xlabel('时间采样点'); plt.ylabel('空间位置点')
plt.colorbar(label='强度'); plt.show()
```

**每类至少画 3 个样本，逐条核对论文对形态的描述：**

| 类别 | 论文描述 | 你该看到 |
|------|---------|---------|
| 背景噪声 | "杂乱" | 看不出规律 |
| 挖掘 | "有独立峰，且时间分布更长（沙子回流）" | 峰 + 持续一段 |
| 敲击 | "有独立峰" | 单个尖峰，比挖掘短 |
| 浇水 | "连续事件" | 连续但无周期 |
| 摇晃 | "连续事件，有明显周期性" | 能数出规律条纹 |
| 走动 | "单样本中有多个峰（步频）" | 多个分散尖峰 |

> **⚠️ 论文 Fig.3 画的是差分信号，不是原始信号。** 所以直接画 `data` 可能看不出这些特征，
> 要先做差分（见第 3.3 节）。第一遍可以两个都画，对比着看。

---

## 二、数据导入：把文件变成矩阵

### 2.1 SVM 路线的数据组织方式

**核心思路：一次性把所有样本读进内存，拼成两个大数组。**

```
X  (12335, 384)   ← 每一行是一个样本的 384 维特征
y  (12335,)       ← 每一行是它的标签（0~5）
```

**为什么 SVM 可以"一次全读进内存"？** 因为特征只有 384 维：

```
12335 × 384 × 8 字节（float64）≈ 38 MB   ✅ 装得下
```

**对比 CNN**：如果读原始矩阵，`15419 × 10000 × 12 × 4 字节 ≈ 7.4 GB` ❌ 装不下，
所以 CNN 必须用 `Dataset` 按需读取。

**这个"数据量决定架构"的对比，是理解两种加载方式的关键。**

### 2.2 标签文件：最容易被忽略的坑

`label.txt` 每行长这样：

```
/01_background/220112_cxm_background_01_single_data_1.mat 0
↑ 以 / 开头（方便直接拼到根目录后面）      ↑ 文件名      ↑ 标签
```

**读取代码**：

```python
def read_labels(labelpath):
    name_list = []
    with open(labelpath, encoding='utf-8-sig') as f:   # ★ 注意编码
        for line in f:
            line = line.strip()
            if not line:
                continue
            name_list.append(line)
    return name_list
```

**两个必须注意的点：**

| 点 | 为什么 |
|----|--------|
| `encoding='utf-8-sig'` | 兼容带 BOM 的文本文件。**Windows 上生成的 txt 经常带 BOM**，用 `utf-8` 读会把 BOM 读进第一个字符，导致路径错误（本项目就踩过这个坑） |
| `line.strip()` | 去掉行尾的 `\n` 和 `\r`。**Windows 是 `\r\n`**，不 strip 会让路径带上 `\r` |

### 2.3 ★ 关键校验：标签顺序

**官方代码是"按行遍历读取"的**：

```python
for i in range(len(name_list)):
    path = datapath + name_list[i].split(' ')[0]     # 第 i 行 → 读第 i 个文件
```

**这意味着：如果 label.txt 的行顺序和磁盘上的文件顺序不一致，
读到的文件和它的标签就错配了。**

**官方仓库 issue #12、#13 正是反映这个问题**（用户贴出了顺序错乱的实例）。

**所以必做这个校验**：

```python
def check_label_order(data_root, split, classes):
    """校验 label.txt 的行顺序与磁盘文件名顺序是否一致"""
    bad = []
    for cls in classes:
        d = os.path.join(data_root, split, cls)
        disk_order = sorted(os.listdir(d))                    # 磁盘顺序
        label_order = [从 label.txt 里提取的本类文件名，保持原顺序]
        if disk_order != label_order:
            bad.append(cls)
    return bad

# 本项目的校验结果：
# [OK] train 12335 行、test 3084 行，顺序完全一致
```

**这个校验花 1 分钟，能避免"训练一整天，结果全错"的灾难。**

### 2.4 拼接：从一个个文件到两个大数组

```python
def get_das_data(rootpath, labelpath):
    names = read_labels(labelpath)

    # ★ 预分配数组，而不是不断 append
    X = np.zeros([len(names), 384])          # 2000×384 的零矩阵
    y = np.zeros(len(names))

    for i, line in enumerate(names):
        parts = line.split(' ')
        path = os.path.join(rootpath) + parts[0]      # 拼路径
        raw = scio.loadmat(path)['data']              # (10000,12) uint16

        diff = get_diff_data(raw)                     # 差分 → (9999,12)

        feat_raw  = get_feature_list(raw)             # (12,16)
        feat_diff = get_feature_list(diff)            # (12,16)

        # 横向拼接 → (12, 32)，再拉平 → 384
        X[i, :] = np.concatenate((feat_raw, feat_diff), axis=1).reshape(-1)
        y[i] = int(parts[1])

    return X, y
```

**为什么用 `np.zeros` 预分配，而不是 `list.append`？**

| | `append` 到列表 | 预分配数组 |
|---|---|---|
| 内存 | 每次扩容都要复制全部数据 | 一次分配 |
| 速度 | 慢（尤其数据量大时） | 快 |
| 代码 | `X.append(...)` 简单 | `X[i,:] = ...` 稍繁 |

**当你知道最终大小时，永远预分配。** 这是 NumPy 的基本功。

### 2.5 类型陷阱：`uint16` 差分回绕（本项目的重大发现）

**这是 SVM 路线最关键的一个 bug，直接影响了结果。**

`loadmat` 读出来的是 **`uint16`**（无符号 16 位整数，范围 0~65535，**不能表示负数**）。

**而差分是两个数相减，必然出现负数：**

```python
import numpy as np
a = np.array([7800, 8100], dtype=np.uint16)
print(a[0] - a[1])        # 期望 -300，实际得到 65236 ！
```

**为什么？** 就像汽车里程表只有 5 位数字，从 7800 倒退到 8100 之前，
表盘会往回翻一圈，变成 65236。**`uint16` 里根本没有"负数"这个概念。**

**官方代码的问题**：

```python
diff = data[1:, :] - data[:-1, :]     # 在 uint16 世界里做减法 → 负数全部回绕
```

**正确写法**：

```python
def get_diff_data(data):
    data = data.astype(np.int32)                       # ★ 先转成能存负数的类型
    return data[1:, :] - data[:-1, :]
```

**影响有多大？** 差分特征占了 384 维里的 **192 维（一半）**。
修复后 SVM 准确率从论文的 82.6% 提升到 **88.85%**。

> **官方仓库 issue #2 反映的就是这个问题**，报告者修改后准确率变为 0.878。
>
> **这一类 bug 最危险的地方是"不报错"**——程序照跑，只是数字悄悄错了。
> **所以必须做数值检查**：打印差分前后的 min/max，看有没有异常大的正数。

---

## 三、特征工程：384 维特征是怎么算出来的

### 3.1 整体结构

```
一个样本 (10000, 12)
    │
    ├─ 12 个通道，每通道是一条长度 10000 的时序信号
    │
    ├─ 对【原始信号】的每个通道提 16 个特征 → (12, 16)
    │
    ├─ 对【差分信号】的每个通道提 16 个特征 → (12, 16)
    │
    └─ 横向拼接 → (12, 32) → 拉平 → 384 维
```

**为什么要对差分信号也提一遍？**（论文 4.1 节）

- **原始信号**：保留了绝对幅值信息（信号有多强）
- **差分信号**：保留了变化率信息（变化有多快）

**两者互补**。挖掘和敲击都是"突变"，但它们对原始值和变化率的敏感度不同，
所以两种特征都要。**这是论文实验得出的做法，不是随意堆砌。**

### 3.2 七个基础统计量（最好理解的）

| 特征 | 公式 | 物理含义 |
|------|------|---------|
| 最大值 max | `x.max()` | 信号最强到什么程度 |
| 最小值 min | `x.min()` | 信号最弱到什么程度 |
| 峰峰值 pk | `max − min` | 动态范围 |
| 均值 mean | `x.mean()` | 直流分量（信号的中心在哪） |
| 方差 var | `((x-mean)²).mean()` | 波动程度 |
| 标准差 std | `√var` | 波动程度（和方差同源，量纲和原信号一致） |
| 整流平均值 arv | `|x|.mean()` | 取绝对值后的均值，衡量"能量大小" |

### 3.3 频域特征：FFT

**为什么要看频域？** 因为有些事件在时域上看起来差不多，但频率成分不同。
（比如摇晃有周期性，浇水没有——这在时域上都是"连续波动"，但在频域上区别明显。）

```python
fft_trans = np.abs(np.fft.fft(data))              # 傅里叶变换，取幅值
freq_spectrum = fft_trans[1:int(np.floor(n/2)) + 1]   # 取正频率部分
```

**为什么要取 `[1 : n/2+1]`？**

- `fft` 输出的第 0 项是**直流分量**（频率为 0），对"变化"没意义，去掉
- `fft` 的输出是**对称的**（正频率和负频率互为共轭），只需要前一半
- 所以取 `1` 到 `n/2` 这段——这就是正频率部分

**由频域导出的两个特征：**

| 特征 | 公式 | 含义 |
|------|------|------|
| 能量 | `sum(freq²) / len(freq)` | 频域幅值的平均平方，衡量信号总强度 |
| 信息熵 | `-Σ p·log2(p+1e-5)`，其中 `p = freq/Σfreq` | 频域能量分布的"混乱程度"。分布越均匀熵越大 |

> **`1e-5` 是干什么的？** 防止 `log2(0)` 变成负无穷。这是数值计算的标准技巧。
> 凡是做 log 运算，都要加一个小量。

### 3.4 无量纲指标（最有区分度的几个）

**无量纲 = 除以了某个基准量，所以不受信号绝对大小影响。**
这对本项目很重要——因为不同人的挖掘力度、不同时间的背景噪声强度都不同。

| 特征 | 公式 | 物理含义 |
|------|------|---------|
| 波形因子 boxing | `rms / arv` | 波形"方不方"。正弦波约 1.11，方波 = 1 |
| 脉冲因子 maichong | `max / arv` | 有没有突出的尖峰 |
| 峰值因子 fengzhi | `max / rms` | 同上，但基准不同 |
| 裕度因子 yudu | `max / (√\|x\|均值的平方)` | 对**早期微弱故障**敏感 |

**这些指标来自机械故障诊断领域**（论文参考文献 Jia et al. 2019），
是判断"轴承有没有坏"的经典手段。**用到振动信号识别上是合理的迁移。**

> **为什么"脉冲因子"能区分敲击和背景？**
> 敲击是一个尖峰：`max` 很大，但 `arv`（平均绝对值）不大 → 比值大
> 背景是均匀噪声：`max` 和 `arv` 比例接近 → 比值小

### 3.5 峰度与峭度：两个名字很像但不同的东西

**这是本项目最容易搞混的地方，官方代码里也确实容易混淆。**

| 名称 | 官方变量 | 公式 | 特点 |
|------|---------|------|------|
| 峰度 | `dif_kurt` | 用 `pandas.Series(data).kurt()` | 调整后的 Fisher-Pearson 超额峰度 |
| 峭度 | `dif_qiaodu` | `(Σx⁴/n) / rms⁴` | 皮尔逊峭度（不减去均值） |

**pandas 的 `kurt()` 到底算什么？** 我读过源码（`pandas/core/nanops.py` 第 1402-1404 行）：

```python
adj = 3*(n-1)² / ((n-2)(n-3))
G2  = n(n+1)(n-1)·m4 / ((n-2)(n-3)·m2²) - adj
```

其中 `m2 = Σ(x-mean)²`，`m4 = Σ(x-mean)⁴`（**是求和，不是求均值**）。

**为什么要减 3？** 因为正态分布的峰度是 3，
减去 3 之后**正态分布的峰度 = 0**，这样比较起来更直观。

**⚠️ 这两个特征高度重复**（都在描述"分布有多尖"），
而且官方的 16 维特征里，**信息熵被计算了但没有放进返回列表**
（返回的第 16 项是峭度，不是熵）。这是官方代码的一处瑕疵。

### 3.6 完整的特征提取函数

```python
def feature_extraction(data):
    """输入一条时序信号（10000 或 9999 点），返回 16 维特征"""
    data = np.asarray(data, dtype=np.float64)     # ★ 统一转 float
    n = len(data)

    # ---- 频域（只做一次 FFT，官方做了两次，浪费）----
    fft_trans = np.abs(np.fft.fft(data))
    freq_spectrum = fft_trans[1:int(np.floor(n / 2)) + 1]
    freq_sum = np.sum(freq_spectrum)

    # ---- 时域基础统计量 ----
    f_max  = data.max()
    f_min  = data.min()
    f_pk   = int(f_max) - int(f_min)
    f_mean = data.mean()
    f_var  = data.var()
    f_std  = data.std()

    # ---- 派生量 ----
    f_energy = np.sum(freq_spectrum ** 2) / len(freq_spectrum)
    f_rms    = np.sqrt(f_mean ** 2 + f_std ** 2)
    f_arv    = np.abs(data).mean()

    # ---- 无量纲指标 ----
    f_boxing   = f_rms / f_arv
    f_maichong = f_max / f_arv
    f_fengzhi  = f_max / f_rms
    f_yudu     = f_max / (np.sqrt(np.abs(data)).sum() / n) ** 2

    # ---- 峰度（pandas kurt 的等价实现，注意用求和）----
    adj2 = (data - f_mean) ** 2
    m2 = adj2.sum()
    m4 = (adj2 ** 2).sum()
    f_kurt = (n*(n+1)*(n-1)*m4) / ((n-2)*(n-3)*m2**2) - 3*(n-1)**2/((n-2)*(n-3))

    # ---- 峭度（注意：必须用 float64，用 int64 会溢出）----
    f_qiaodu = (((data ** 2) ** 2).sum() / n) / f_rms ** 4

    # ---- 信息熵 ----
    p = freq_spectrum / freq_sum
    f_entropy = -np.sum(np.log2(p + 1e-5) * p)

    return np.array([round(v, 3) for v in [
        f_max, f_min, f_pk, f_mean, f_energy, f_var, f_std, f_rms,
        f_arv, f_boxing, f_maichong, f_fengzhi, f_yudu, f_kurt,
        f_qiaodu, f_entropy]])
```

**函数封装的三个好习惯：**

| 习惯 | 好处 |
|------|------|
| 输入立刻 `np.asarray(data, dtype=np.float64)` | 不管外部传什么进来，内部处理逻辑统一。**避免类型陷阱** |
| 中间变量用有意义的名字 | 可读性。`f_max` 比 `a1` 强一百倍 |
| 最后统一 `round` | 保持和官方一致（官方返回的就是 3 位小数） |

### 3.7 ⚠️ 两个必须知道的数值陷阱

**陷阱一：`int64` 也会溢出**

```python
# 错误写法
(data.astype(np.int64) ** 4).sum()     # 信号值 8294，四次方 ≈ 4.7e15
                                        # 1 万项相加 ≈ 8e19 > int64 上限 9.2e18
                                        # → 溢出成负数！
```

**正确写法**：

```python
((data ** 2) ** 2).sum()      # 用 float64，不会溢出
```

> **为什么会踩这个坑？** 因为官方用的是 Python 的 `list` 推导
> `np.sum([x**4 for x in data])`，而 **Python 整数是任意精度的**，不会溢出。
> 我一开始"顺手改成 int64 更精确"，反而引入了溢出。
> **这个教训值得记：优化时不要想当然，一定要做数值对比验证。**

**陷阱二：除零**

`f_boxing = f_rms / f_arv`——如果某个通道恰好是常数（全 0 或全相同），
`f_arv = 0`，就会 `ZeroDivisionError` 或得到 `inf`。

**防御写法**：

```python
eps = 1e-10
f_boxing = f_rms / (f_arv + eps)
```

> 本项目的数据没有触发这个问题（背景样本值域 7828~8655，不是常数），
> **但这是"跑得通但很脆"的代码**——换一份数据就可能崩。

---

## 四、预处理：归一化与数据划分

### 4.1 为什么必须归一化

**你的 384 维特征里，数值范围差异极大：**

```
最大值    ~8000          ← 量级 10³
能量      ~3200000       ← 量级 10⁶  ← 比最大值大 400 倍！
峭度      ~39
波形因子  ~2             ← 和能量差 160 万倍
```

**问题**：SVM 的 RBF 核计算的是**欧氏距离**：

```
距离 = √[(a₁-b₁)² + (a₂-b₂)² + ... + (a₃₈₄-b₃₈₄)²]
```

**如果某一维的量级是 10⁶，其他维是 10⁰，那这个距离几乎完全由那一维决定，
其他 383 维等于不存在。**

**归一化就是把所有维拉到同一个量级**，让每一维都有发言权。

### 4.2 MinMaxScaler：把每一维拉到 [0,1]

```
新值 = (x − 该维最小值) / (该维最大值 − 该维最小值)
```

**注意是"该维"**——每一维单独算自己的 min/max，不是全局。

```python
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
scaler.fit(X_train)                        # ★ 只用训练集算 min/max
X_train_scaled = scaler.transform(X_train) # 用同一把尺子
X_test_scaled  = scaler.transform(X_test)  # ★ 绝不能在这里重新 fit！
```

### 4.3 ★ 数据泄漏：官方代码在这里有个 bug

**官方代码是这样写的**：

```python
trainingData = minMaxScaler.fit_transform(X_train)
testData = minMaxScaler.fit_transform(X_test)     # ← 又 fit 了一次！
```

**`fit_transform` = 先 `fit`（算 min/max）再 `transform`（应用）。**

第二次 `fit` 意味着：**测试集用它自己的 min/max 来缩放自己**，
而不是用训练集的尺子。**这就是数据泄漏**——测试集的信息渗进了预处理流程。

**正确做法**：

```python
scaler.fit(X_train)                      # 只在训练集上确定尺子
trainingData = scaler.transform(X_train)
testData     = scaler.transform(X_test)  # 用同一把尺子
```

**为什么这很重要（一个类比）**：

> 考试前老师给了你 10 套模拟题（训练集）和 1 套真题（测试集）。
> **正常做法**：按模拟题的难度准备，然后去考真题。
> **数据泄漏**：先看一眼真题有多难，再"针对性准备"——那成绩就不真实了。

**客观评价**：对 MinMaxScaler 而言，这个泄漏的**实际影响通常不大**
（只是缩放比例不同，不改变样本间的相对关系）。
我在本项目实测过：修正前后准确率变化在 1 个百分点以内。

**但方法上必须修正。** 原因有二：
1. 换成 `StandardScaler` 或做 PCA 时，这个泄漏的影响会大得多
2. **"我知道这里有问题，并且量化了它的影响"** 比"我跑出 88%"有价值得多

### 4.4 数据划分：本项目不用你操心（但要知道原理）

**论文说**："训练和测试数据按 8:2 随机选取，两者无重叠。"

**好消息**：作者已经把原始数据切好并分成了 `train\` 和 `test\` 两个目录，
所以**你直接用现成的两套目录就行，不要自己重新随机分**——
否则会和论文的数字对不上。

**但你以后拿到 rawdata 必须自己划，规则如下：**

| 场景 | 正确划法 | 错误划法（会造成泄漏） |
|------|---------|---------------------|
| 独立样本（本项目） | 随机 8:2 | — |
| **同一段信号切出的片段** | **按整段划分** | 随机 ❌（同一段的片段分到两边） |
| 时序预测 | **按时间前后划分** | 随机 ❌（用未来预测过去） |
| 多人/多设备采集 | 按人/设备划分（更严格） | 随机（结果偏乐观） |

**划分完一定要做一件事：检查有没有重叠。**

```python
overlap = set(train_files) & set(test_files)
print('重叠数量:', len(overlap))     # 必须是 0
```

**更进一步**（本项目的做法）：不只比文件名，还要比**内容哈希**——
防止同一段信号被存成两个不同文件名。

---

## 五、模型构建与训练

### 5.1 完整的训练流程（其实只有 6 行）

**SVM 路线最让人舒服的地方：训练就 6 行代码。**

```python
from sklearn import svm

clf = svm.SVC(C=1.0, kernel='rbf', gamma='auto', decision_function_shape='ovo')
clf.fit(X_train, y_train)         # 训练（这一行就是全部）
y_pred = clf.predict(X_test)      # 预测
acc = (y_pred == y_test).mean()   # 准确率
```

**对比 CNN**：需要定义网络、写训练循环、写反向传播、管设备、跑 50 个 epoch。
**SVM 没有"训练过程"这个概念——`fit` 一次就出结果。**

### 5.2 每个超参数的含义

```python
svm.SVC(
    C=1.0,                        # 惩罚系数
    kernel='rbf',                 # 核函数
    gamma='auto',                 # 核函数的宽度参数
    degree=3,                     # 多项式核的阶数（用rbf时无效）
    coef0=0.0,                    # 核函数的常数项（用rbf时无效）
    decision_function_shape='ovo',# 多分类策略
    tol=0.001,                    # 停止训练的容差
    max_iter=-1,                  # 最大迭代次数，-1 = 不限
)
```

**最关键的两个：`C` 和 `gamma`**

| 参数 | 作用 | 太小会怎样 | 太大会怎样 |
|------|------|-----------|-----------|
| **C** | 对"分错"的惩罚力度 | 容忍错分 → **欠拟合**（边界太宽松） | 不容忍任何错分 → **过拟合**（边界扭曲去迁就每个点） |
| **gamma** | 单个样本影响的范围 | 影响范围太大 → 边界太平滑 → **欠拟合** | 影响范围太小 → 边界碎片化 → **过拟合** |

**`gamma='auto'` 是什么？** 等于 `1 / n_features = 1/384 ≈ 0.0026`。
（另一个选项 `'scale'` = `1/(n_features × X.var())`，是 sklearn 现在的默认值。
**官方代码用的是 `'auto'`**，我们保持一致。）

**新手调参的正确顺序**（别一上来就网格搜索）：

```
① 先用默认参数跑一遍，拿到基准分
② 只调 C（试 0.1, 1, 10, 100），记录结果
③ 固定最好的 C，再调 gamma（试 0.001, 0.01, 0.1, 1）
④ 每次只改一个参数 —— 否则你不知道是谁起的作用
```

> **本项目的经验**：即使不调参（用论文的参数），准确率也有 88.85%，
> 已经超过论文的 82.6%。**说明修复数据 bug 比调参重要得多。**
> **这印证了一条经验：数据和特征决定上限，模型和调参只是逼近上限。**

### 5.3 训练需要多久

| 步骤 | 本项目实测 |
|------|-----------|
| 特征提取（12335 样本） | 93.7 分钟 ← **瓶颈** |
| 特征提取（3084 样本） | 24.0 分钟 |
| **SVM 训练 + 预测** | **0.2 分钟（12 秒）** |

**★ 最重要的认识：SVM 本身只占 0.1% 的时间，瓶颈全在特征提取。**

**这决定了优化的方向：** 优化 SVM 训练毫无意义（它已经很快了），
要优化的是**特征提取**。我们把它向量化后，从 117.9 分钟降到 9.6 分钟（12.3 倍）。

**这个"先找瓶颈再优化"的思路，是性能优化的第一原则。**
详细做法见 `bug_记录.md` 的优化章节，以及第 5.4 节。

### 5.4 性能优化：把 for 循环变成 NumPy 运算

**为什么慢？** 看 `feature_extraction.py` 里这几行：

```python
sum = 0
for i in range(len(data)):        # 9999 次 Python 循环
    sum += np.sqrt(abs(data[i]))
yudu = max(data) / pow(sum / len(data), 2)

qiaodu = (np.sum([x ** 4 for x in data]) / len(data)) / pow(rms, 4)
#                ↑ 又一个 9999 次的列表推导
```

**每个样本要提 24 次特征 → 每个样本约 48 万次 Python 循环。**

**向量化改写**：

```python
yudu = max(data) / (np.sqrt(np.abs(data)).sum() / n) ** 2
#                 ↑ 一次数组运算，底层是 C 语言循环 + SIMD 指令

qiaodu = ((data ** 2) ** 2).sum() / n / rms ** 4
```

**核心思想：NumPy 的数组运算在 C 层面循环，并且能用 SIMD 一次处理多个数。
而 Python 的 `for` 循环每次都要做类型检查、对象分配，慢几十倍。**

**★ 但优化必须验证：**

```python
# 对同一批数据，两种实现逐个比对
a = feature_extraction_official(series)
b = feature_extraction_fast(series)
print('最大偏差:', np.abs(a - b).max())      # 必须是 0 或极小
```

**本项目做了 504 次对比，最大偏差 0.000000000000（逐位一致）。**

> **没有这个验证，你就不敢说"优化没改结果"。**
> 而我确实在这里栽过一次：我以为用 `int64` 更精确，结果它溢出了，
> 造出了一个"看起来合理但完全错误"的特征值。
> **是逐位对比把这个错误抓出来的。**

---

## 六、日志：怎么让程序边跑边告诉你进度

### 6.1 为什么需要日志

**你面对的现实**：特征提取要跑 1.5 小时。
**如果没有日志**：你不知道它是"在正常计算"还是"卡死了"。

**日志要解决三个问题**：
1. **同时**输出到屏幕（盯着看）和文件（事后查）
2. **实时**看到（不是跑完才吐出来）
3. **可回溯**（一周后能查清哪次跑的、参数是什么）

### 6.2 官方实现与它的 bug

```python
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream                # 屏幕
        self.log = open(filename, 'w')        # 文件
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)               # ← 只写，不刷新！
    def flush(self):
        pass                                  # ← 空实现！

sys.stdout = Logger('svm_result.log', sys.stdout)   # 劫持标准输出
```

**它的原理**：把 `sys.stdout`（标准输出）替换成自己的对象。
之后所有 `print()` 都会调用 `Logger.write()`，从而同时写到屏幕和文件。

**这个设计思路是对的，但实现有两个 bug：**

| bug | 后果 |
|-----|------|
| `write` 后不 flush | 内容积在缓冲区，**只有程序退出那一刻才落盘**。跑 1.5 小时，你全程看不到进度 |
| `flush` 是空实现 | 连手动刷新都失效 |

**我实测验证过**：

| 运行方式 | 运行中日志大小 | 结束后大小 |
|---------|--------------|-----------|
| 官方原版 | **0 字节** | 319 字节 |
| `python -u` | **0 字节** | 319 字节 |
| **加 `flush()`** | **26→101→177→253 逐秒增长** ✅ | — |

**为什么 `python -u` 也没用？** 因为 `-u` 只管 `sys.stdout` 那一层的缓冲，
**管不了 `open()` 出来的文件对象自己那一层缓冲**。这是两个独立的缓冲。

### 6.3 正确实现

```python
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='utf-8')   # ★ 加编码
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()          # ★ 关键：写完立即刷盘
    def flush(self):
        self.log.flush()          # ★ 同时把 flush 本身修好
```

**三个改动各自解决什么：**

| 改动 | 解决 |
|------|------|
| `self.log.flush()` in write | 实时可见 |
| `flush()` 里也 flush | 兼容 `print(..., flush=True)` |
| `encoding='utf-8'` | 中文不乱码 |

> **另一个坑**：试过 `open(filename, 'w', line_buffering=True)`，
> 结果报 `TypeError: open() got an unexpected keyword argument 'line_buffering'`。
> **`line_buffering` 是 `TextIOWrapper` 构造函数的参数，不能直接传给 `open()`。**
> 最可靠的办法就是显式 `flush()`。

### 6.4 进度提示：让长任务可观测

**光有日志还不够——打印语句太少，还是看不出进展。**

**在循环里加进度输出**：

```python
import time
t0 = time.time()
for i in range(len(name_list)):
    ...                                        # 干活
    if (i + 1) % 200 == 0 or (i + 1) == len(name_list):
        el = time.time() - t0
        eta = el / (i + 1) * (len(name_list) - i - 1)     # ★ 预计剩余时间
        print("  进度 %d/%d (%.1f%%)  已用 %.1f 分钟  预计还需 %.1f 分钟"
              % (i + 1, len(name_list), (i + 1)*100.0/len(name_list),
                 el/60, eta/60))
```

**为什么 `% 200` 而不是每次都打印？**
打印本身有开销，而且刷屏会让你看不到有用信息。**按固定间隔报，既有进度又不干扰。**

**★ ETA（预计剩余时间）的算法**：

```
平均每个样本耗时 = 已用时间 / 已完成数
预计剩余时间     = 平均耗时 × 剩余数量
```

**就这两行，但让 1.5 小时的等待变得可预期。** 这是长任务脚本的标配。

### 6.5 怎么写日志才"可回溯"

**日志要包含三类信息：**

```python
# ① 开始前：环境和参数
print("=" * 60)
print(" 开始时间:", datetime.datetime.now())
print(" 数据路径:", DATA_ROOT)
print(" 超参数: C=%.2f, gamma=%s, kernel=%s" % (C, gamma, kernel))
print("=" * 60)

# ② 过程中：进度和中间指标
print("Epoch %d  acc=%.4f" % (ep, acc))

# ③ 结束后：最终结果
print("测试集准确率: %.4f" % acc)
print("耗时: %.1f 分钟" % elapsed)
```

**为什么要记环境？** 因为一周后你肯定会问自己
"这个 88.85% 是哪次跑的？用的什么参数？"。

**★ 更重要的习惯：给输出文件带参数命名。**

```python
tag = "C%.2f_gamma%s" % (C, gamma)
plt.savefig("results/svm_confusion_%s.png" % tag)
```

**否则一周后 `SVM_confusion_matrix.jpg` 会有 5 个版本，你分不清哪个是哪个。**

---

## 七、评估：混淆矩阵与指标

### 7.1 混淆矩阵：一切指标的源头

**它就是一张"真实标签 × 预测标签"的计数表：**

```python
from sklearn.metrics import confusion_matrix
C = confusion_matrix(y_test, y_pred)       # 输入：真实标签、预测标签
```

**本项目的 SVM 结果：**

| 真实 \ 预测 | 背景 | 挖掘 | 敲击 | 浇水 | 摇晃 | 走动 |
|-----------|------|------|------|------|------|------|
| **背景噪声** | **588** | 0 | 0 | 1 | 0 | 0 |
| **挖掘** | 90 | **359** | 3 | 2 | 11 | 37 |
| **敲击** | 44 | 1 | **459** | 0 | 0 | 2 |
| **浇水** | 17 | 15 | 0 | **364** | 25 | 30 |
| **摇晃** | 0 | 1 | 0 | 4 | **540** | 1 |
| **走动** | 12 | 36 | 6 | 4 | 2 | **430** |

**怎么读这张表：**

```
① 对角线 = 分对了的数量（越深越好）
② 第 i 行 = 真实是第 i 类的那 589 个样本，实际被分到了哪里
③ 第 j 列 = 所有被预测为第 j 类的样本，真实是什么
④ 行和 = 该类的真实样本数（589, 502, 506, 451, 546, 490）
⑤ 列和 = 该类被预测的次数（751, 412, 468, 375, 578, 500）
```

**★ 立刻能看出问题：**

- 挖掘那一行：**90 个被错判成背景噪声**（占该类的 17.9%）← 最大的问题
- 背景噪声那一列（列和 751）：**比真实样本数 589 多了 162 个**
  ← 说明有 162 个其他类的样本被误判成"背景噪声"

**这两条信息加起来，直接定位了模型最大的缺陷。**

### 7.2 四个指标：各自回答什么问题

先定义四个量（对某一类而言）：

| 符号 | 含义 | 本类例子（以"挖掘"为例） |
|------|------|----------------------|
| **TP** | 真的是挖掘，也预测成挖掘 | 359 |
| **FP** | 不是挖掘，却被预测成挖掘 | 412 − 359 = 53 |
| **FN** | 真的是挖掘，却没预测成挖掘 | 502 − 359 = 143 |
| **TN** | 不是挖掘，也没预测成挖掘 | 其余全部 |

**四个指标：**

| 指标 | 公式 | 回答的问题 | 本项目（挖掘） |
|------|------|-----------|--------------|
| **Precision**（精确率） | `TP / (TP + FP)` | **"你说是挖掘的里面，有多少真的是？"** | 359/412 = 0.871 |
| **Recall**（召回率） | `TP / (TP + FN)` | **"真的挖掘里，你找出来多少？"** | 359/502 = 0.715 |
| **F1** | `2·P·R / (P + R)` | **P 和 R 的综合**（调和平均） | 0.786 |
| **Accuracy**（准确率） | `(TP+TN)/总数` | **整体对不对** | 88.85% |

**★ 为什么用调和平均算 F1，而不是算术平均？**

因为 Precision 和 Recall 是"矛盾的"——你可以只对很有把握的样本做预测，
Precision 会很高但 Recall 很低。算术平均会让这种"偏科"看起来还行。
**调和平均对"偏科"惩罚更重**（一个接近 0，F1 就接近 0）。

**举例**：P=1.0, R=0.01

```
算术平均 = 0.505   ← 看起来还行？其实完全没用
调和平均 = 0.0198  ← 这才反映真实情况
```

### 7.3 ★ 论文的 NAR：一个必须澄清的坑

**论文公式 (2) 定义**：

```
NAR = FN / (TP + FN)
```

**这个式子和 `1 − Recall` 完全一样**（因为 `Recall = TP/(TP+FN)`）。
所以论文的 NAR 其实就是**漏检率**。

**但仓库 README（2024-06 更新）又定义**：

```
NAR = Σd₀ⱼ / ΣΣdᵢⱼ      ← 这是"虚警率"（把背景误报成事件的占比）
```

**论文和 README 自己打架了！** 而且代码第 82 行实现的是 README 那版。

| 口径 | 公式 | 本项目结果 |
|------|------|-----------|
| 论文公式(2) | `FN/(TP+FN)`，背景类 | 163/(588+163) = 0.2170 |
| README | 虚警率 | 1/2333 = 0.0004 |
| FNR | 漏检非背景事件的比例 | 163/2495 = 0.0653 |

**处理方式：两个都算、两个都报，并在报告里说明这个不一致。**

> **这比"选一个看起来好的报上去"专业得多。**
> 而且**发现"论文和代码不一致"本身就是一项有价值的复现发现**。

### 7.4 宏平均 vs 总体平均（口径必须说清）

**同一个模型，两种算法能差好几个百分点：**

| 算法 | 公式 | 本项目 SVM |
|------|------|-----------|
| **总体（micro）** | 所有正确数 / 所有样本数 | 2740/3084 = **88.85%** |
| **宏平均（macro）** | 各类指标的算术平均 | (0.878+0.786+0.943+0.881+0.961+0.869)/6 = **88.61%** |

**什么时候差距大？类别不均衡时。**

本项目背景噪声 589 个、浇水 451 个，差 1.3 倍，所以两个数字接近。
**如果一类有 10000 个、另一类只有 10 个，两个数字可能差 30 个百分点。**

**★ 论文用的是哪个？** 我核对过论文 Table 4：

```
SVM: 6 类 F1 的宏平均 = 0.818   论文声称 0.826
CNN: 6 类 F1 的宏平均 = 0.938   论文声称 0.940
```

**结论：论文的 "average accuracy" 就是"6 类 F1 的宏平均"。**

> **所以和论文对比时，必须用宏平均 F1**，不能用总体准确率——
> 否则口径不一致，数字没有可比性。
> **这是做复现最容易出错、也最容易被审稿人抓住的地方。**

### 7.5 指标计算代码

```python
import numpy as np

def compute_metrics(C, names):
    """从混淆矩阵计算逐类指标"""
    TP = np.diag(C).astype(float)          # 对角线 = 各类 TP
    FP = C.sum(axis=0) - TP                # 列和 - TP = FP
    FN = C.sum(axis=1) - TP                # 行和 - TP = FN
    TN = C.sum() - TP - FP - FN

    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    f1        = 2 * precision * recall / (precision + recall)
    accuracy  = (TP + TN) / C.sum()        # 逐类的"一类 vs 其余"准确率

    print("%-12s %10s %10s %10s" % ("事件", "Precision", "Recall", "F1"))
    for i, nm in enumerate(names):
        print("%-12s %10.3f %10.3f %10.3f" % (nm, precision[i], recall[i], f1[i]))
    print("%-12s %10.3f %10.3f %10.3f" % ("宏平均",
          precision.mean(), recall.mean(), f1.mean()))

    # ★ 两个口径都报
    print("\n总体准确率 (micro) = %.4f" % (TP.sum() / C.sum()))
    print("宏平均 F1  (macro) = %.4f   ← 论文口径" % f1.mean())

    return precision, recall, f1
```

**★ 注意一个细节**：官方代码里指标循环写的是 `for i in range(1, 6)`，
**从 1 开始，跳过了第 0 类（背景噪声）**。所以官方日志里只有 5 个类的指标。

**这是官方的一个 bug**（SVM 和 CNN 里都有），修正成 `range(6)` 才能和论文 Table 4 对齐。

### 7.6 误差分析：从矩阵里读出"为什么错"

**这是把"结果"变成"结论"的关键一步。别停在报数字。**

```python
for i, nm in enumerate(names):
    row = C[i].copy(); row[i] = 0        # 去掉对角（正确的那部分）
    total_err = row.sum()
    if total_err == 0: continue
    print("%-12s 共错 %3d 个 (占本类 %.1f%%)" % (nm, total_err, total_err/C[i].sum()*100))
    for j in np.argsort(row)[::-1][:3]:  # 错误最多的前 3 个去向
        if row[j] == 0: continue
        print("      → 错判为 %-12s %3d 个 (%.1f%%)"
              % (names[j], row[j], row[j]/C[i].sum()*100))
```

**本项目的输出：**

```
挖掘     共错 143 个 (占本类 28.5%)
      → 错判为 背景噪声   90 个 (17.9%)
      → 错判为 走动       37 个 (7.4%)
```

**★ 然后要做的是"解释"，而不是"复述"：**

| 现象 | 解释（结合论文和领域知识） |
|------|------------------------|
| 挖掘→背景噪声 17.9% | 挖掘是间歇性的（挖一下、停一下），**停顿期间的特征接近背景噪声**。论文 4.3 节也说"强度很小的扰动事件淹没在噪声里" |
| 挖掘↔走动 互相混淆 | 论文说"走动包含不同频率成分，**有些样本只包含单次踩踏**"。**单次踩踏和挖一下，在时空图上确实很像** |

**★ 这一步的价值：**

```
只报数字  →  "准确率 88.85%"
加误差分析 →  "准确率 88.85%，主要误差是挖掘被误判为背景（17.9%），
              原因是挖掘动作的间歇性停顿，与论文 4.3 节的解释一致"
```

**后者才是科研，前者只是跑了个程序。**

---

## 八、出图：怎么画出能放进报告的图

### 8.1 图的三个层次

| 层次 | 图 | 作用 |
|------|-----|------|
| ① 自查看 | 原始信号图、特征分布图 | 帮你自己发现问题（第 1.5 节） |
| ② 论文对照 | 混淆矩阵、逐类指标对比 | 证明你复现对了 |
| ③ 汇总说服 | SVM vs CNN 对比 | 支撑你的结论 |

**报告里需要的是 ② 和 ③。① 是过程，不放报告（但必须做）。**

### 8.2 混淆矩阵图

```python
import matplotlib
matplotlib.use("Agg")          # ★ 无 GUI 后端，直接存文件
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]   # ★ 中文字体
plt.rcParams["axes.unicode_minus"] = False                        # ★ 负号正常显示

fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(C, cmap="Blues", vmin=0, vmax=C.max())

# 在每个格子写上数字
for i in range(6):
    for j in range(6):
        v = C[i, j]
        ax.text(j, i, str(v), ha="center", va="center", fontsize=11,
                color="white" if v > C.max() * 0.55 else "black",
                #      ↑ 深色底用白字，浅色底用黑字，保证可读
                fontweight="bold" if i == j else "normal")
                #      ↑ 对角线加粗，一眼看出正确率

ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel("Predicted label", fontsize=13)
ax.set_ylabel("True label", fontsize=13)
ax.set_title("SVM baseline - Confusion Matrix (test n=%d)\n"
             "overall acc = %.2f%%   macro F1 = %.2f%%"
             % (C.sum(), ACC*100, F1.mean()*100), fontsize=12.5)
plt.colorbar(im, ax=ax, label="sample count")
plt.tight_layout()
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()                    # ★ 关掉，释放内存
```

**★ 三个必须加的设置：**

| 设置 | 不加会怎样 |
|------|-----------|
| `matplotlib.use("Agg")` | `plt.show()` 在无交互环境**永久卡死**（我实测踩过） |
| `font.sans-serif = ["Microsoft YaHei"]` | **中文全变方块** □□□ |
| `axes.unicode_minus = False` | 负号显示成方块 |

**★ 两个让图更专业的细节：**

- **数字颜色随背景明暗切换**（深色格子白字、浅色格子黑字）——否则深色格子上的黑字看不见
- **对角线加粗**——让读者一眼抓住重点

### 8.3 与论文对比图（报告里最重要的图）

**思路：把"我们的指标"和"论文的指标"并排画柱状图。**

```python
x = np.arange(6); w = 0.35
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

panels = [("Recall",    OUR_REC,  1 - PAPER_NAR),
          ("Precision", OUR_PREC, PAPER_PREC),
          ("F1-score",  OUR_F1,   PAPER_F1)]

for ax, (title, ours, paper) in zip(axes, panels):
    b1 = ax.bar(x - w/2, ours,  w, label="ours (reproduced)", color="#2b7bba")
    b2 = ax.bar(x + w/2, paper, w, label="paper (Table 4)",   color="#c44e52")
    for bars in (b1, b2):
        for b in bars:      # 在柱子顶上标数值
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.012,
                    "%.3f" % b.get_height(), ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(names_cn, rotation=25, ha="right")
    ax.set_ylim(0, 1.13); ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9)
```

**为什么用并排柱状图？** 因为读者能**一眼看出每一类谁高谁低**。
如果用表格，读者要逐个数字比较，很累。

**★ 一个细节：`ax.set_ylim(0, 1.13)`**
留出上方 13% 的空间给数值标签，否则标签会被切掉。

### 8.4 汇总图：SVM vs CNN（论文核心结论）

```python
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
# 左：逐类 F1 对比
# 右：总体对比（本次复现 vs 论文）
```

**这张图是整个复现的"结论图"**——它直接支撑
"CNN 优于 SVM"这个论文核心论断。

### 8.5 图的命名与归档

**★ 这是最容易被忽视、但影响最大的习惯。**

```python
# ✗ 错误：一周后你分不清这是哪次跑的
plt.savefig("confusion_matrix.png")

# ✓ 正确：参数和结果都体现在文件名里
tag = "C1.0_gamma_auto_acc0.8885"
plt.savefig("results/svm_confusion_%s.png" % tag)
```

**并且所有图统一放在 `results\`**，形成"证据仓库"。

---

## 九、结论：怎么从数字里读出意义

### 9.1 结论的三个层次

| 层次 | 内容 | 例子 |
|------|------|------|
| **① 复述** | 我跑出了什么数 | "SVM 准确率 88.85%" |
| **② 对照** | 和论文/基线比 | "论文 82.6%，我们高 6.8 个点" |
| **③ 解释** | **为什么** | "因为有 192 维差分特征受了 uint16 回绕影响；修复后挖掘类提升最大（+12.1 点），因为挖掘是最依赖差分特征的瞬态事件" |

**只有到 ③ 才算"得到结论"。** ①②只是报数。

### 9.2 本项目的结论链（示范）

```
【现象】SVM 88.85%，比论文高 6.8 个点
   ↓
【追问】为什么高？
   ↓
【排查】差异来源有哪些？
   · 数据集版本（15419 vs 15612）—— 已排除，逐类核对一致
   · 测试集划分不同 —— 无法排除
   · 代码修复 —— 我们修了 uint16 回绕
   ↓
【验证】怎么确认是"修复"起的作用？
   · 差分特征占 384 维中的 192 维（一半）
   · 受影响最大的应该是最依赖差分特征的类
   ↓
【对照】逐类提升幅度：
   挖掘 +12.1 点（最大）← 最瞬态的事件
   浇水 +10.4 点
   走动 +7.6 点
   摇晃 +3.8 点
   敲击 +4.9 点
   背景 +0.8 点（最小）← 背景本来就没有"突变"
   ↓
【结论】提升幅度和"该事件对差分特征的依赖程度"正相关，
        支持"uint16 修复是主要原因"这一解释。
```

**★ 注意最后一步**：不能只说"我觉得是修复起的作用"，
而要找出**一个可检验的预测**（"最依赖差分的类提升最大"），然后**去验证它**。

**这是从"讲故事"到"做科研"的分界线。**

### 9.3 报告里必须写的"局限"

**没有局限的结论是不可信的。** 本项目的局限：

| 局限 | 说明 |
|------|------|
| 测试集划分未知 | 论文的随机划分未公开，无法复现完全相同的测试集 |
| 单次运行 | 改进方向未做多种子重复（CNN 那边做了） |
| 特征存在冗余 | 峰度与峭度定义高度重复；信息熵算了未用 |
| 口径不一致 | 论文与 README 对 NAR 的定义冲突，本文双口径报告 |

**★ 主动写局限，反而增加可信度。** 因为审阅者知道"你清楚自己做了什么、没做什么"。

### 9.4 给导师汇报的模板

```
【本周完成】
SVM 基线复现完成，测试集 3084 样本：
- 总体准确率 88.85%，宏平均 F1 88.61%（论文 81.8%）
- 六类指标全面超过论文报告值

【关键发现】
1. 官方差分函数未做类型转换，uint16 数据相减时负值回绕
   （7800-8100 得到 65236）。差分特征占 384 维中的 192 维。
   修复后挖掘类 F1 由 0.673 提升至 0.786。
2. 官方指标循环 range(1,6) 跳过了背景类，与论文 Table 4 不对齐。
3. 论文公式(2) 与仓库 README 对 NAR 的定义不一致，本文双口径报告。

【误差分析】
挖掘类错误率最高（28.5%），其中 17.9% 被误判为背景噪声，
与论文 4.3 节"弱扰动淹没于噪声"的解释一致。

【性能优化】
官方特征提取用纯 Python 循环，全数据集需约 2 小时。
NumPy 向量化后降至 9.6 分钟（12.3 倍），
经 504 次逐位对比验证输出完全一致（最大偏差 0）。

【下一步】
复现 CNN 基线（论文 94%），并做 SVM/CNN 对比分析。
```

---

## 十、完整模板：一份可以直接改用的代码

```python
r"""
svm_pipeline.py —— SVM 分类完整流程模板
换项目时改 CONFIG 区即可。
"""
import os, glob, time, datetime
import numpy as np
import scipy.io as scio
from sklearn import svm, preprocessing
from sklearn.metrics import confusion_matrix

# ============================================================
#  CONFIG —— 换项目只改这里
# ============================================================
DATA_ROOT   = r"...\das_data"
CLASSES     = {"01_background": 0, "02_dig": 1, ...}   # 文件夹 -> 标签
N_CLASS     = 6
FILE_PATTERN = "*.mat"
MAT_KEY     = "data"
OUT_DIR     = r"...\results"

SVM_PARAMS = dict(C=1.0, kernel='rbf', gamma='auto',
                  decision_function_shape='ovo', tol=0.001, max_iter=-1)
# ============================================================


# ---------- 1. 日志 ----------
class Logger(object):
    def __init__(self, filename, stream=__import__('sys').stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='utf-8')
    def write(self, msg):
        self.terminal.write(msg); self.log.write(msg); self.log.flush()
    def flush(self):
        self.log.flush()


# ---------- 2. 读标签 ----------
def read_labels(labelpath):
    with open(labelpath, encoding='utf-8-sig') as f:
        return [ln.strip() for ln in f if ln.strip()]


# ---------- 3. 校验标签顺序（必做！）----------
def check_label_order(split):
    bad = []
    for cls in CLASSES:
        d = os.path.join(DATA_ROOT, split, cls)
        if not os.path.isdir(d): continue
        disk = sorted(os.listdir(d))
        label = [os.path.basename(l.split()[0]) for l in read_labels(
                 os.path.join(DATA_ROOT, split, 'label.txt'))
                 if ('/' + cls + '/') in l]
        if disk != label:
            bad.append(cls)
    return bad


# ---------- 4. 差分（★ 必须转类型）----------
def get_diff_data(data):
    data = data.astype(np.int32)                 # ★ 防止 uint16 回绕
    return data[1:, :] - data[:-1, :]


# ---------- 5. 特征提取 ----------
def extract_features(data):
    """输入一条时序信号，返回 16 维特征"""
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    eps = 1e-10                                   # ★ 防除零

    fft = np.abs(np.fft.fft(data))
    freq = fft[1:int(np.floor(n / 2)) + 1]

    f_max, f_min = data.max(), data.min()
    f_mean, f_var, f_std = data.mean(), data.var(), data.std()
    f_rms = np.sqrt(f_mean ** 2 + f_std ** 2)
    f_arv = np.abs(data).mean()

    adj2 = (data - f_mean) ** 2
    m2, m4 = adj2.sum(), (adj2 ** 2).sum()
    f_kurt = (n*(n+1)*(n-1)*m4)/((n-2)*(n-3)*m2**2) - 3*(n-1)**2/((n-2)*(n-3))

    p = freq / (freq.sum() + eps)
    feats = [
        f_max, f_min, int(f_max) - int(f_min), f_mean,
        np.sum(freq ** 2) / len(freq), f_var, f_std, f_rms,
        f_arv,
        f_rms / (f_arv + eps),                    # 波形因子
        f_max / (f_arv + eps),                    # 脉冲因子
        f_max / (f_rms + eps),                    # 峰值因子
        f_max / (np.sqrt(np.abs(data)).sum() / n) ** 2,   # 裕度因子
        f_kurt,
        (((data ** 2) ** 2).sum() / n) / (f_rms ** 4 + eps),   # 峭度
        -np.sum(np.log2(p + 1e-5) * p),           # 信息熵
    ]
    return np.array([round(v, 3) for v in feats])


def sample_to_vector(raw):
    """(10000,12) -> 384 维向量"""
    diff = get_diff_data(raw)
    f_raw  = np.array([extract_features(raw[:, i])  for i in range(raw.shape[1])])
    f_diff = np.array([extract_features(diff[:, i]) for i in range(diff.shape[1])])
    return np.concatenate((f_raw, f_diff), axis=1).reshape(-1)


# ---------- 6. 构建数据集 ----------
def build_dataset(split):
    names = read_labels(os.path.join(DATA_ROOT, split, 'label.txt'))
    X = np.zeros([len(names), 384])
    y = np.zeros(len(names))
    t0 = time.time()
    print("开始处理 %s：%d 个样本" % (split, len(names)))
    for i, line in enumerate(names):
        parts = line.split(' ')
        raw = scio.loadmat(os.path.join(DATA_ROOT, split) + parts[0])[MAT_KEY]
        X[i, :] = sample_to_vector(raw)
        y[i] = int(parts[1])
        if (i + 1) % 2000 == 0:
            el = time.time() - t0
            print("  %d/%d  已用 %.1f 分钟  预计还需 %.1f 分钟"
                  % (i+1, len(names), el/60,
                     el/(i+1)*(len(names)-i-1)/60))
    print("  完成，用时 %.2f 分钟" % ((time.time()-t0)/60))
    return X, y


# ---------- 7. 指标 ----------
def report_metrics(C, names_cn):
    TP = np.diag(C).astype(float)
    P = TP / C.sum(axis=0)
    R = TP / C.sum(axis=1)
    F = 2*P*R/(P+R)
    print("\n%-12s %10s %10s %10s" % ("事件", "Precision", "Recall", "F1"))
    for i, nm in enumerate(names_cn):
        print("%-12s %10.3f %10.3f %10.3f" % (nm, P[i], R[i], F[i]))
    print("%-12s %10.3f %10.3f %10.3f" % ("宏平均", P.mean(), R.mean(), F.mean()))
    print("\n总体准确率 (micro) = %.4f" % (TP.sum()/C.sum()))
    print("宏平均 F1  (macro) = %.4f   ← 与论文对比用这个" % F.mean())
    return P, R, F


# ---------- 8. 主流程 ----------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    LOG = os.path.join(OUT_DIR, 'svm_result_%s.log'
                       % datetime.datetime.now().strftime('%Y%m%d_%H%M'))
    import sys
    sys.stdout = Logger(LOG, sys.stdout)

    print("=" * 64)
    print(" SVM 完整流程   开始时间:", datetime.datetime.now())
    print(" 数据:", DATA_ROOT)
    print(" 参数:", SVM_PARAMS)
    print("=" * 64)

    # ① 校验
    for sp in ['train', 'test']:
        bad = check_label_order(sp)
        print("[%s] 标签顺序校验: %s" % (sp, '一致' if not bad else '不一致 %s' % bad))

    # ② 构建特征
    X_tr, y_tr = build_dataset('train')
    X_te, y_te = build_dataset('test')

    # ③ 归一化（★ 只在训练集 fit）
    scaler = preprocessing.MinMaxScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xte = scaler.transform(X_te)

    # ④ 训练
    t0 = time.time()
    clf = svm.SVC(**SVM_PARAMS)
    clf.fit(Xtr, y_tr)
    print("\nSVM 训练完成，用时 %.1f 秒" % (time.time() - t0))

    # ⑤ 评估
    y_pred = clf.predict(Xte)
    C = confusion_matrix(y_te, y_pred)
    print("\n混淆矩阵:")
    print(C)

    # ⑥ 指标 + 误差分析
    report_metrics(C, list(CLASSES.keys()))

    print("\n全部完成。日志:", LOG)


if __name__ == '__main__':
    main()
```

**这份模板里每个"★"标记的地方，都是我踩过坑才加的。**

---

## 十一、自测：20 个问题

**原理层**
1. SVM 为什么叫"支持向量机"？
2. 6 类问题，SVM 内部是怎么处理的？训练几个分类器？
3. `kernel='rbf'` 起什么作用？不用它会怎样？
4. **为什么 SVM 必须手工提特征，而 CNN 不用？**

**数据层**
5. `data_profile.py` 检查哪 8 项？
6. 「背景噪声 std 比挖掘还大」这个现象说明什么？
7. `uint16` 差分为什么会得到 65236？怎么修？
8. 为什么读 label.txt 要用 `utf-8-sig` 而不是 `utf-8`？
9. 校验"标签顺序"是为了防什么？

**特征层**
10. 为什么要对原始信号和差分信号**各提一遍**特征？
11. FFT 之后为什么取 `[1 : n/2+1]`？
12. 什么是"无量纲指标"？为什么它们对本任务重要？
13. 为什么 `int64` 算 `x⁴` 会出错？该用什么？
14. 16 个特征里哪一个被官方代码漏掉了？

**预处理层**
15. 不归一化会怎样？RBF 核为什么对量级敏感？
16. 什么是数据泄漏？官方代码在哪一行犯了？
17. 如果数据是"同一段信号切出的片段"，该怎么划 train/test？

**模型与评估层**
18. SVM 训练占整个流程多少时间？说明优化该往哪走？
19. Precision 和 Recall 各自回答什么问题？为什么 F1 用调和平均？
20. **论文的 "average accuracy" 是哪个口径？为什么必须用这个口径对比？**

---

## 附录：SVM 路线 vs CNN 路线对照表

| | SVM 路线 | CNN 路线 |
|---|---------|---------|
| **特征** | 手工提取 384 维 | 网络自动学习 |
| **数据加载** | 一次性读入内存（38 MB） | `Dataset` 按需读取（7.4 GB） |
| **是否用 GPU** | ❌ 不需要 | ✅ 需要（且很关键） |
| **训练代码** | `clf.fit(X, y)` 一行 | 训练循环 + 反向传播 |
| **训练耗时** | 12 秒 | 37 分钟 |
| **瓶颈** | 特征提取（1.5 小时） | 数据加载（26 小时，优化后 8 分钟） |
| **准确率** | 88.85% | 92.90% |
| **可解释性** | ✅ 高（知道哪个特征起作用） | ❌ 低（黑箱） |
| **上手难度** | ✅ 低 | 中 |
| **适合什么时候用** | 数据少、需要可解释、快速验证想法 | 数据多、追求精度 |

**★ 结论：先用 SVM 把整条链子走通，再上 CNN。**
**这条链子（清洗→导入→特征→建模→评估→出图→结论）在两条路线上是一样的。**
