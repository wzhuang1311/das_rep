# 复现过程 Bug 记录

> 格式：位置 / 现象 / 原因 / 修复 / 效果 / 影响
> **这份文档就是你向导师证明"我真的读懂了官方代码"的核心证据。**

---

## Bug 1：`get_diff_data` 的 uint16 差分回绕（🔴 影响结果）

- **位置**：`get_das_data.py` 第 8、10 行
- **现象**：运行时报
  ```
  get_das_data.py:8: RuntimeWarning: invalid value encountered in cast
    data_diff = np.empty([m-1, n]).astype(int)
  ```
- **原因**：有两处问题叠加
  1. **`np.empty` 读未初始化内存**。`np.empty([9999,12])` 只分配内存、不初始化，
     里面是内存里的垃圾数据。`.astype(int)` 去转换这些垃圾值时，
     碰到 NaN/Inf 的位模式就报 `invalid value encountered in cast`。
     （循环会把每个位置都覆盖掉，所以结果不脏，但警告说明写法有问题）
  2. **真正影响结果的是 uint16 回绕**：`scio.loadmat()['data']` 返回的是 `uint16`，
     而这里的差分 `data[i+1,:] - data[i,:]` **没有先转成有符号类型**。
     实测演示：
     ```
     原始 uint16 数据:  [[8100 5000], [7800 5100]]
     uint16 相减:       [65236  100]   ← 7800-8100 应为 -300，实际回绕成 65236
     int32  相减:       [-300   100]   ← 正确
     ```
     背景噪声样本的值域是 7828~8655，差分完全可能出现负值，
     **因此这个回绕在本数据集上必然发生**。

     官方仓库 issue #2 正是反映这个问题，报告修改后 SVM 准确率变为 0.878。

- **修复**：
  ```python
  # 修改前
  data_diff = np.empty([m-1, n]).astype(int)
  for i in range(m-1):
      data_diff[i, :] = data[i+1, :] - data[i, :]

  # 修改后
  data = data.astype(np.int32)                      # 先转有符号类型
  data_diff = np.zeros([m-1, n], dtype=np.int32)    # np.empty -> np.zeros
  for i in range(m-1):
      data_diff[i, :] = data[i+1, :] - data[i, :]
  ```
- **效果**：待补（重跑后记录准确率变化）
- **影响**：**影响结果**。差分特征（占总特征的一半，12×16=192 维）
  在修复前完全不可信，修复后 SVM 准确率预期会明显上升。

---

## Bug 2：`normalize()` 用双层 for 循环（🔴 性能，不影响精度）

- **位置**：`mydataset.py` 第 9-11 行
- **现象**：CNN 训练极慢
- **原因**：对每个样本执行 `10000 × 12 = 120000` 次 Python 层循环运算
- **实测数据**（RTX 4050 / Python 3.13）：

  | 版本 | 单样本耗时 | 1 epoch | 50 epochs |
  |------|-----------|---------|-----------|
  | 官方双层 for | 0.1436 秒 | 29.5 分钟 | **30.8 小时** |
  | NumPy 向量化 | 0.0006 秒 | 7.4 秒 | **约 7 分钟** |
  | | **快 240 倍** | | |

- **修复**：
  ```python
  # 修改前
  for i in range(data.shape[0]):
      for j in range(data.shape[1]):
          data[i][j] = round(((255-0)*(data[i][j]-rawdata_min)/(rawdata_max-rawdata_min))+0)

  # 修改后
  return np.round(255.0 * (data - rawdata_min) / (rawdata_max - rawdata_min))
  ```
- **效果**：函数输出经 `np.array_equal` 逐元素比对，与官方版本**完全一致**
- **影响**：纯性能优化，不影响精度。但决定了这个实验"能不能在 1 天内跑完"

---

## Bug 3：数据路径硬编码（🔴 阻塞，环境相关）

- **位置**：`das_data_svm.py` 第 24 行、`das_data_cnn.py` 第 198 行
- **现象**：`FileNotFoundError` / `xxx does not exist!`
- **原因**：原代码路径为作者 Linux 服务器上的 `/das_data`
- **修复**：
  ```python
  rootpath = r'D:\汕头大学\科研\深度学习\text.demo\das_data'
  ```
  （`r` 前缀必需：路径中 `\t`、`\n` 会被当作转义字符）
- **影响**：环境适配，不影响算法

---

## Bug 4：中文路径编码（🔴 阻塞，环境相关）

- **位置**：`mydataset.py` 第 24 行
- **现象**：`UnicodeDecodeError`
- **原因**：Windows 默认用 GBK 打开文本文件，而路径含中文（`汕头大学`、`科研`）
- **修复**：`file = open(self.names_file, encoding='utf-8')`
- **影响**：环境适配，不影响算法

---

## 待验证的问题（还没动手改，先记录）

| # | 位置 | 问题 | 状态 |
|---|------|------|------|
| 5 | `das_data_svm.py` 第 37 行 | 测试集用 `fit_transform`，数据泄漏 | 待改，需量化影响 |
| 6 | `das_data_svm.py` 第 91 行 | 指标循环 `range(1,6)` 跳过背景类 | 待改 |
| 7 | 论文 vs 代码 | 论文 lr=0.01，代码写死 lr=1e-4 | 已确认，复现以代码为准 |
| 8 | `feature_extraction.py` | 信息熵算了但未纳入返回的 16 维 | 已确认 |
| 9 | `das_data_cnn.py` 第 145 行 | `batches_per_epoch` 用样本数而非 batch 数 | 待验证 |
| 10 | `feature_visualization.py` 第 5 行 | 硬编码作者本机路径 | 待改 |
