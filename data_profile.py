"""
data_profile.py —— 通用数据体检工具

作用：拿到任何一个新数据集，先用这个脚本"体检"一遍，
      在写任何模型代码之前，搞清楚数据到底长什么样。

用法（在 das_rep 目录下）：
    python data_profile.py

想用到别的项目上？只需要改最下面 CONFIG 区域的几个变量。
"""

import os
import glob
import numpy as np
import scipy.io as scio

# ============================================================
#  CONFIG —— 换项目时只改这里
# ============================================================
DATA_ROOT = r"D:\汕头大学\科研\深度学习\text.demo\das_data"

# 类别文件夹名 -> (标签, 中文名)
CLASSES = {
    "01_background": (0, "背景噪声"),
    "02_dig":        (1, "挖掘"),
    "03_knock":      (2, "敲击"),
    "04_water":      (3, "浇水"),
    "05_shake":      (4, "摇晃"),
    "06_walk":       (5, "走动"),
}

FILE_PATTERN = "*.mat"          # 数据文件后缀
MAT_KEY = "data"                # .mat 文件里存数据的变量名
EXPECTED_SHAPE = (10000, 12)    # 你期望的形状，None 表示不检查
SPLITS = ["train", "test"]
# ============================================================


def section(title):
    print(f"\n{'=' * 64}\n {title}\n{'=' * 64}")


def check_1_files_exist():
    """体检项 1：文件到底在不在"""
    section("1. 文件与目录检查")
    ok = True
    print(f"  数据根目录: {DATA_ROOT}")
    if not os.path.isdir(DATA_ROOT):
        print("  [致命] 根目录不存在！检查 DATA_ROOT 路径")
        return False

    for split in SPLITS:
        d = os.path.join(DATA_ROOT, split)
        exists = os.path.isdir(d)
        print(f"  [{ 'OK' if exists else '致命' }] {split} 目录: {'存在' if exists else '不存在'}")
        ok = ok and exists
        if exists:
            missing = [c for c in CLASSES
                       if not os.path.isdir(os.path.join(d, c))]
            if missing:
                print(f"       [警告] 缺少类别文件夹: {missing}")
                ok = False

    # 标签文件
    for split in SPLITS:
        lp = os.path.join(DATA_ROOT, split, "label.txt")
        e = os.path.isfile(lp)
        print(f"  [{ 'OK' if e else '警告' }] {split}/label.txt: {'存在' if e else '不存在'}")
    return ok


def check_2_amounts():
    """体检项 2：样本数量与标签分布（是否类别不平衡）"""
    section("2. 样本数量与类别分布")
    stats = {}
    for split in SPLITS:
        counts = {}
        for cls, (lb, cname) in CLASSES.items():
            d = os.path.join(DATA_ROOT, split, cls)
            counts[cls] = len(glob.glob(os.path.join(d, FILE_PATTERN))) if os.path.isdir(d) else 0
        total = sum(counts.values())
        stats[split] = (counts, total)
        print(f"\n  --- {split} (共 {total}) ---")
        for cls, (lb, cname) in CLASSES.items():
            n = counts[cls]
            pct = 100.0 * n / total if total else 0
            bar = "#" * int(pct / 2)
            print(f"    label {lb} {cname:<8} {n:>6}  {pct:5.1f}%  {bar}")
        if total:
            mx, mn = max(counts.values()), min(counts.values())
            ratio = mx / mn if mn else float("inf")
            print(f"    最多/最少 = {ratio:.2f} 倍", end="  ")
            if ratio > 3:
                print("→ [警告] 明显不平衡，评估时不能只看 accuracy")
            elif ratio > 1.5:
                print("→ [注意] 轻度不平衡，建议看按类指标")
            else:
                print("→ [OK] 基本均衡")

    # 交叉检查：磁盘文件数 vs label.txt 行数
    section("3. 磁盘文件数 vs label.txt 行数（一致性交叉检查）")
    for split in SPLITS:
        lp = os.path.join(DATA_ROOT, split, "label.txt")
        if not os.path.isfile(lp):
            continue
        with open(lp, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        total = stats[split][1]
        flag = "OK" if len(lines) == total else "警告"
        print(f"  [{flag}] {split}: 磁盘 {total} 个文件, label.txt {len(lines)} 行")
        if len(lines) != total:
            print("        → 有文件没被标注，或有标注找不到文件，务必查清")

    # 训练/测试是否重叠（数据泄漏检查！）
    section("4. 训练集与测试集是否重叠（数据泄漏致命检查）")
    try:
        def names_of(split):
            s = set()
            for cls in CLASSES:
                d = os.path.join(DATA_ROOT, split, cls)
                if os.path.isdir(d):
                    for p in glob.glob(os.path.join(d, FILE_PATTERN)):
                        s.add(os.path.basename(p))
            return s
        tr, te = names_of("train"), names_of("test")
        inter = tr & te
        if inter:
            print(f"  [致命] 发现 {len(inter)} 个同名文件同时出现在 train 和 test！")
            print(f"         例如: {list(inter)[:3]}")
            print("         → 会导致测试准确率虚高，必须删掉重复")
        else:
            print(f"  [OK] 无重叠 (train {len(tr)} 个, test {len(te)} 个)")
            print("       → 官方已按 8:2 划分好，直接用，不要自己重分")
    except Exception as e:
        print(f"  检查失败: {e}")


def check_5_sample_structure(n_sample_per_class=2):
    """体检项 5：样本本身的形状、类型、数值范围"""
    section("5. 单个样本的结构检查")
    problems = []
    for cls, (lb, cname) in CLASSES.items():
        d = os.path.join(DATA_ROOT, "train", cls)
        files = sorted(glob.glob(os.path.join(d, FILE_PATTERN)))
        if not files:
            continue
        print(f"\n  --- {cname} ({cls}) ---")
        for f in files[:n_sample_per_class]:
            name = os.path.basename(f)
            try:
                m = scio.loadmat(f)
                keys = [k for k in m if not k.startswith("__")]
                if MAT_KEY not in m:
                    print(f"    [致命] {name}: 变量名不是 '{MAT_KEY}'，实际是 {keys}")
                    problems.append(f"{name} 变量名不对")
                    continue
                arr = m[MAT_KEY]
                if EXPECTED_SHAPE and arr.shape != EXPECTED_SHAPE:
                    print(f"    [致命] {name}: 形状 {arr.shape} != 期望 {EXPECTED_SHAPE}")
                    problems.append(f"{name} 形状不对")
                    continue
                # 数值健康检查
                nan = int(np.isnan(arr.astype(float)).sum())
                std = arr.std()
                rng = int(arr.max()) - int(arr.min())
                flags = []
                if nan:
                    flags.append(f"含 {nan} 个 NaN")
                if std == 0:
                    flags.append("整个样本恒为常数（坏样本）")
                if rng == 0:
                    flags.append("值域为 0")
                print(f"    {name}")
                print(f"      shape={arr.shape} dtype={arr.dtype} "
                      f"范围={arr.min()}~{arr.max()} std={std:.1f}")
                if flags:
                    print(f"      [警告] {'; '.join(flags)}")
                    problems.append(f"{name}: {'; '.join(flags)}")
            except Exception as e:
                print(f"    [致命] {name}: 读取失败 {type(e).__name__}: {e}")
                problems.append(f"{name} 读取失败")

    # 通道健康度（针对本数据集：12 个空间点）
    if EXPECTED_SHAPE and len(EXPECTED_SHAPE) == 2:
        section("6. 通道健康度检查（有没有'死通道'）")
        print("  说明：对每个空间点算标准差。std≈0 说明该通道没信号（死通道）；")
        print("        各通道 std 差异大说明事件集中在局部——这正是我们要的特征。\n")
        for cls, (lb, cname) in CLASSES.items():
            d = os.path.join(DATA_ROOT, "train", cls)
            files = sorted(glob.glob(os.path.join(d, FILE_PATTERN)))[:20]
            if not files:
                continue
            stds = []
            for f in files:
                arr = scio.loadmat(f)[MAT_KEY]
                stds.append(arr.std(axis=0))     # 每个空间点的 std
            stds = np.array(stds).mean(axis=0)   # 对样本取平均
            dead = int((stds < 1e-6).sum())
            print(f"  {cname:<8} 平均各通道std: {np.round(stds, 1)}")
            if dead:
                print(f"           [警告] 有 {dead} 个死通道")
        print("\n  解读：两端通道 std 小、中间通道 std 大 → 事件集中在光纤局部（符合物理预期）")
    return problems


def main():
    print("#" * 64)
    print("#  数据集体检报告")
    print("#" * 64)
    check_1_files_exist()
    check_2_amounts()
    problems = check_5_sample_structure()

    section("总结")
    if problems:
        print(f"  发现 {len(problems)} 个样本级问题：")
        for p in problems[:10]:
            print(f"    - {p}")
        if len(problems) > 10:
            print(f"    ... 还有 {len(problems)-10} 个")
    else:
        print("  样本级检查全部通过。")
    print("""
  体检完了，接下来问自己三个问题：
    1. 数据量够不够？（每类至少几百个，本数据集最少 1802，够用）
    2. 类别平衡吗？（不平衡就要看按类指标，不能只看 accuracy）
    3. 有没有脏数据？（坏样本、NaN、死通道）
  这三个问题的答案，决定了你后面所有代码怎么写。
""")


if __name__ == "__main__":
    main()
