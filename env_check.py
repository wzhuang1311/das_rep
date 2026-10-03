"""
env_check.py —— 环境自检脚本（阶段 0 用，验证 CUDA 环境是否配好）

用法：在已激活 .venv 的终端里执行
    python env_check.py
"""

import sys
import platform

OK = "[OK]"
NG = "[!!]"

problems = []


def check(name, fn, expect=None):
    """跑一个检查函数，打印结果；异常时记录为问题。"""
    try:
        value = fn()
        good = True if expect is None else (value == expect)
        mark = OK if good else NG
        print(f"{mark} {name}: {value}")
        if not good:
            problems.append(f"{name} 实际为 {value}，期望 {expect}")
        return value
    except Exception as e:
        print(f"{NG} {name}: 检查失败 -> {type(e).__name__}: {e}")
        problems.append(f"{name} 检查出错: {e}")
        return None


print("=" * 62)
print(" 环境自检  env_check.py")
print("=" * 62)

# ---------- 1. Python 本身 ----------
print("\n--- 1. Python 解释器 ---")
check("Python 版本", lambda: sys.version.split()[0])
check("解释器路径", lambda: sys.executable)
check("操作系统", lambda: f"{platform.system()} {platform.release()}")

# 判断是否跑在虚拟环境里
in_venv = sys.prefix != sys.base_prefix
mark = OK if in_venv else NG
print(f"{mark} 虚拟环境已激活: {in_venv}")
if not in_venv:
    print("      -> 你可能忘了先执行 .\\.venv\\Scripts\\Activate.ps1")
    problems.append("未在虚拟环境中运行，请先激活 .venv")

# ---------- 2. 各个库的版本 ----------
print("\n--- 2. 依赖库版本 ---")

def lib_version(mod_name):
    def _inner():
        mod = __import__(mod_name)
        return getattr(mod, "__version__", "已安装(无版本号)")
    return _inner

for mod in ["numpy", "scipy", "sklearn", "matplotlib", "pandas"]:
    check(mod, lib_version(mod))

# ---------- 3. PyTorch 与 CUDA（重点） ----------
print("\n--- 3. PyTorch / CUDA（重点检查）---")
import torch

check("torch 版本", lambda: torch.__version__)

ver = torch.__version__
if "+cu" not in ver:
    print(f"{NG} 这不是 CUDA 版 torch！版本号里没有 '+cu'")
    print("      -> 重装：python -m pip install torch==2.12.0+cu130 torchvision "
          "--index-url https://download.pytorch.org/whl/cu130")
    problems.append("torch 不是 CUDA 版")

check("torch 编译时的 CUDA 版本", lambda: torch.version.cuda)
check("cuda.is_available()", lambda: torch.cuda.is_available(), expect=True)
check("GPU 数量", lambda: torch.cuda.device_count())

if torch.cuda.is_available():
    check("GPU 名称", lambda: torch.cuda.get_device_name(0))
    check("计算能力 (SM)", lambda: ".".join(map(str, torch.cuda.get_device_capability(0))))
    props = torch.cuda.get_device_properties(0)
    print(f"{OK} 显存: {props.total_memory / 1024**3:.2f} GB")

    # ---------- 4. 真跑一次 GPU 运算（纸上说的不算，跑通才算） ----------
    print("\n--- 4. 实机 GPU 运算测试 ---")
    try:
        x = torch.randn(2000, 2000, device="cuda")
        y = torch.randn(2000, 2000, device="cuda")
        z = x @ y                      # 矩阵乘法，在 GPU 上算
        torch.cuda.synchronize()       # 等 GPU 真正算完
        print(f"{OK} GPU 矩阵乘法成功，结果形状 {tuple(z.shape)}，"
              f"均值 {z.mean().item():.4f}")
        # 模拟一次真实训练的前向+反向，确认反向传播在 GPU 上也没问题
        w = torch.randn(120, 6, device="cuda", requires_grad=True)
        loss = (torch.randn(100, 120, device="cuda") @ w).pow(2).mean()
        loss.backward()
        print(f"{OK} GPU 反向传播成功，loss={loss.item():.4f}，w.grad 形状 "
              f"{tuple(w.grad.shape)}")
    except Exception as e:
        print(f"{NG} GPU 运算失败: {type(e).__name__}: {e}")
        problems.append(f"GPU 运算失败: {e}")
else:
    print(f"\n{NG} CUDA 不可用，跳过实机测试。")
    print("      可能原因：装的是 CPU 版 torch / 驱动太旧 / 显卡被禁用")

# ---------- 5. 数据集能否读到（提前排雷） ----------
print("\n--- 5. 数据集读取测试 ---")
import os
import glob
import scipy.io as scio

DATA_ROOT = r"D:\汕头大学\科研\深度学习\text.demo\das_data"

check("数据根目录存在", lambda: os.path.isdir(DATA_ROOT), expect=True)

if os.path.isdir(DATA_ROOT):
    for split in ["train", "test"]:
        d = os.path.join(DATA_ROOT, split)
        if os.path.isdir(d):
            n = len(glob.glob(os.path.join(d, "*", "*.mat")))
            print(f"{OK} {split} 目录 .mat 文件数: {n}")
        else:
            print(f"{NG} 找不到 {d}")
            problems.append(f"缺少 {split} 目录")

    # 真的读一个文件进来，确认变量名和形状
    sample = os.path.join(DATA_ROOT, "train", "01_background", "*.mat")
    files = sorted(glob.glob(sample))
    if files:
        try:
            m = scio.loadmat(files[0])
            keys = [k for k in m if not k.startswith("__")]
            data = m["data"]
            print(f"{OK} 样例文件: {os.path.basename(files[0])}")
            print(f"{OK} .mat 内变量名: {keys}")
            print(f"{OK} data.shape = {data.shape}  (期望 (10000, 12))")
            print(f"{OK} data.dtype  = {data.dtype}   (期望 uint16)")
            print(f"{OK} 数值范围    = {data.min()} ~ {data.max()}")
            if data.shape != (10000, 12):
                problems.append(f"data 形状 {data.shape} 不是 (10000, 12)")
        except Exception as e:
            print(f"{NG} 读取 .mat 失败: {e}")
            problems.append(f"读取 .mat 失败: {e}")
    else:
        print(f"{NG} 01_background 下没有找到 .mat 文件")
        problems.append("01_background 下没有 .mat 文件")

# ---------- 总结 ----------
print("\n" + "=" * 62)
if problems:
    print(f" 发现 {len(problems)} 个问题，需要处理：")
    for i, p in enumerate(problems, 1):
        print(f"   {i}. {p}")
    print("\n 把这段完整输出发给 AI 助手，我来帮你定位。")
    print("=" * 62)
    sys.exit(1)
else:
    print(" 全部通过！环境已就绪，可以进入阶段 1（数据可视化）。")
    print("=" * 62)
    sys.exit(0)
