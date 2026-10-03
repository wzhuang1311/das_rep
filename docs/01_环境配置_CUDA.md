# 阶段 0：把 CUDA 环境配好（照着敲，别跳步骤）

> 目标：让你的 `torch.cuda.is_available()` 从 `False` 变成 `True`。
> 全程约 30~60 分钟，其中大部分是在下载（约 3 GB）。

---

## 一、先看结论：我要你做什么

| 项目 | 你的现状 | 要变成 |
|------|---------|--------|
| Python | 3.13.7（已装） | 不变 |
| torch | 2.12.0**+cpu**（CPU 版） | **2.12.0+cu130**（CUDA 版） |
| CUDA 显卡 | RTX 4050 Laptop，驱动 581.08 | 不变，驱动够新 |
| VC++ 运行库 | v14.51 已装 | 不变 |
| 安装位置 | 全局 `site-packages` | **新建虚拟环境 `.venv`**（见下方"为什么要 venv"） |

**不用装的东西**（省掉三个坑）：
- ❌ 不用装 Anaconda / Miniconda（你 VSCode 设置里写了 conda，但机器上根本没有，直接忽略那个设置）
- ❌ 不用单独去 NVIDIA 官网下 CUDA Toolkit（PyTorch 的 wheel 里自带 CUDA 运行库）
- ❌ 不用装 cuDNN（同上，已打包在 wheel 里）

---

## 二、为什么是 cu130 而不是 cu128

我去 PyTorch 官方源查过了：**cu128 已经没有 Python 3.13 的轮子了**，cu130 才有。可用版本：

```
torch (2.14.1+cu130)   ← 最新
可用版本: 2.14.1+cu130, 2.14.0+cu130, 2.13.0+cu130, 2.12.1+cu130, 2.12.0+cu130, ...
```

**我建议装 `2.12.0+cu130`**，理由：和你现在全局装的 2.12.0 版本号完全一致，
以后对比"CPU 跑的结果"和"GPU 跑的结果"时，不会因为 torch 版本不同引入额外变量。
（想装最新的 2.14.1 也行，不影响复现，但没必要。）

**你的驱动支持吗？** 驱动 581.08 支持 CUDA 13.0，**支持**。
RTX 4050 是 Ada 架构（compute capability 8.9），CUDA 13 要求 ≥7.5，**支持**。

---

## 三、为什么要用 venv（虚拟环境）

你现在所有包装在**全局** Python 里（1.3 GB）。如果直接 `pip install` 覆盖 torch，
会出现两个问题：
1. 以后别的项目要不同版本的 torch，会互相打架；
2. 万一把全局环境搞坏了，修起来很麻烦。

**venv 就是"给这个项目单独开一个装包的抽屉"**，删掉 `venv` 文件夹就彻底还原，零风险。
这是 Python 官方推荐的隔离方式，比 conda 轻量得多。

---

## 四、开始动手（在 VSCode 终端里敲）

### 步骤 1：打开正确的终端

VSCode 里按 `` Ctrl+` `` 打开终端。**确认终端在 `das_rep` 目录下**，
如果不在，敲：

```powershell
cd "D:\汕头大学\科研\深度学习\text.demo\das_rep"
```

> ⚠️ 你的路径里有中文和空格，所以路径**一定要加双引号**。这是新手第一大坑。

### 步骤 2：创建虚拟环境

```powershell
& "C:\Users\34667\AppData\Local\Programs\Python\Python313\python.exe" -m venv .venv
```

敲完等十几秒，`das_rep` 下会多出一个 `.venv` 文件夹。**没报错就是成功。**

### 步骤 3：激活虚拟环境

```powershell
.\.venv\Scripts\Activate.ps1
```

成功的话，终端提示符前面会多出 **`(.venv)`**。看到它就说明激活了。

> **如果报"禁止运行脚本"**，执行下面这行（只对本窗口有效，关掉就恢复）：
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```
> 然后再执行步骤 3。

### 步骤 4：升级 pip（可选但推荐）

```powershell
python -m pip install --upgrade pip
```

### 步骤 5：装 CUDA 版 PyTorch（这一步最久，3 GB 左右）

```powershell
python -m pip install torch==2.12.0+cu130 torchvision --index-url https://download.pytorch.org/whl/cu130
```

> **国内网速慢的话**，加上清华镜像做"中转"（官方源还是会用到，但常规包走清华更快）：
> ```powershell
> python -m pip install torch==2.12.0+cu130 torchvision --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
> ```

**去泡杯茶。** 如果中途断了，直接重敲同一条命令，pip 会续传已下好的部分。

### 步骤 6：装其余依赖

```powershell
python -m pip install numpy scipy scikit-learn matplotlib pandas
```

### 步骤 7：验证环境（关键一步，别省）

我在 `das_rep\env_check.py` 里给你写好了检查脚本。敲：

```powershell
python env_check.py
```

**期望看到**：每一项都是 `[OK]`，特别是 `cuda.is_available()` 显示 `True`。

> **中文乱码怎么办？** 如果输出变成"�����Լ�"这种乱码，是 Windows 终端编码问题，
> 不影响脚本功能（它照样能检查出问题）。在 VSCode 里这样解决：按 `Ctrl+,` 打开设置，
> 搜 `terminal.integrated.profiles.windows`，或直接在你那个 PowerShell 配置文件里加一行：
> ```powershell
> [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
> ```
> 简单点也可以在运行前敲 `$env:PYTHONIOENCODING='utf-8'`。

### 步骤 8：让 VSCode 认识这个环境（必做，否则后面运行还是用全局 Python）

1. 按 `Ctrl+Shift+P` 打开命令面板
2. 输入 `Python: Select Interpreter`，回车
3. 在列表里选带 **`.venv`** 字样的那个，路径应该长这样：
   `D:\汕头大学\科研\深度学习\text.demo\das_rep\.venv\Scripts\python.exe`
4. 选完后，VSCode 左下角/状态栏会显示这个解释器

> 同时建议把 `.vscode\settings.json` 里那两行 conda 设置改成：
> ```json
> {
>     "python.defaultInterpreterPath": "D:\\汕头大学\\科研\\深度学习\\text.demo\\das_rep\\.venv\\Scripts\\python.exe"
> }
> ```
> （这个文件在 `das_data` 目录下，是数据文件夹带的，改不改都不影响跑代码，改一下更省心。）

---

## 五、每一步的"成功长什么样"（自查清单）

- [ ] `.venv` 文件夹已生成
- [ ] 终端提示符前有 `(.venv)`
- [ ] `pip install` 最后一行是 `Successfully installed torch-2.12.0+cu130 ...`
- [ ] `python env_check.py` 全部 ✅
- [ ] `torch.cuda.is_available()` 返回 `True`
- [ ] `torch.cuda.get_device_name(0)` 显示 `NVIDIA GeForce RTX 4050 Laptop GPU`
- [ ] VSCode 已选择 `.venv` 解释器

---

## 六、出错了怎么办（对照找）

| 报错 | 原因 | 解决 |
|------|------|------|
| `无法加载文件 ... Activate.ps1，因为在此系统上禁止运行脚本` | PowerShell 执行策略 | 执行 `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `ERROR: No matching distribution found for torch==2.12.0+cu130` | Python 版本或平台不匹配 | 敲 `python --version` 确认是 3.13；确认 `(.venv)` 已激活 |
| `python 不是内部或外部命令` | venv 没激活 | 重新执行步骤 3 |
| 下载到一半 `ReadTimeoutError` | 网络断了 | 重敲步骤 5 的命令，pip 会续传 |
| 装完了但 `cuda.is_available()` 还是 `False` | 装成了 CPU 版 | 敲 `python -m pip show torch` 看版本号，**必须带 `+cu130`**；如果显示 `+cpu`，说明装错了，重装步骤 5 |
| `OSError: [WinError 126] 找不到指定的模块` | VC++ 运行库缺失 | 你已装 v14.51，一般不会遇到；如遇到告诉我 |
| C 盘变红/磁盘不够 | venv 在 D 盘，torch 约 3 GB | D 盘还剩 139.7 GB，够用 |

---

## 七、装完之后：CPU 和 GPU 到底差多少（为什么值得折腾）

CNN 阶段每个 epoch 要过 12,335 个 10000×12 的矩阵，计算量不小：
- **CPU**：单 epoch 可能要几分钟到十几分钟，50 epoch 就是好几小时
- **GPU（4050）**：通常快 10~30 倍，50 epoch 十几分钟能跑完

所以这一步值得做。但**如果 30 分钟内搞不定，先别死磕**——
先用 CPU 把 SVM 阶段（阶段 2，不需要 GPU）跑完，回头再修环境。
**不要让环境配置挡住你理解论文的进度**，这是新手最常见的"三天卡在装环境"陷阱。

---

## 八、配置完成后，向导师可以这样汇报

```
【本周完成】
1. 确认官方 baseline 代码框架为 PyTorch + scikit-learn，
   仓库地址 https://github.com/BJTUSensor/Phi-OTDR_dataset_and_codes（最新 commit 2024-06-28）
2. 完成开发环境搭建：新建独立虚拟环境，安装 CUDA 版 PyTorch 2.12.0+cu130，
   已通过验证脚本确认 GPU 可用（RTX 4050，torch.cuda.is_available() = True）
3. 核对数据集：train 12335 + test 3084 = 15419 样本，.mat 内变量名 data，形状 10000×12 uint16

【下一步】
进入 SVM 基线复现：12 通道 × 32 维（原始 16 + 差分 16）特征提取 → MinMaxScaler → RBF-SVM
```

---

## 附：本次环境调研确认的事实（免得你再查一遍）

| 事实 | 值 |
|------|-----|
| 显卡 | NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB |
| 驱动版本 | 581.08 |
| 全局 Python | 3.13.7 @ `C:\Users\34667\AppData\Local\Programs\Python\Python313\` |
| pip | 25.2 |
| setuptools | 81.0.0 |
| VC++ 运行库 | v14.51.36247.00（已装） |
| 全局已有包 | torch 2.12.0+**cpu**、torchvision 0.27.0、sklearn 1.9.1、scipy 1.17.1、numpy 2.3.5、matplotlib 3.10.8、pandas 2.3.3 |
| D 盘剩余空间 | 139.7 GB |
| cu128 轮子 | **无 Python 3.13 版本**（已实测） |
| cu130 轮子 | 有，最新 2.14.1+cu130（已实测） |
