import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import re
import os
import warnings
import matplotlib.font_manager as fm

# 1. 强制设置中文字体为宋体（SimSun）
plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]  # 英文用Times New Roman，中文用宋体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 2. 手动添加字体（确保Matplotlib能识别）
try:
    # 检查宋体是否存在
    fm.findfont("SimSun")
except:
    # 手动指定字体路径（Windows默认路径）
    font_path = "C:/Windows/Fonts/simsun.ttc"  # 宋体字体路径
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
    else:
        print("警告：未找到宋体字体，中文可能无法正常显示")


def load_smiles_file(file_path):
    """读取SMILES文件（每行一个SMILES），返回SMILES列表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]


def calculate_smiles_lengths(smiles_list):
    """计算SMILES列表中每个分子的长度（字符数）"""
    return [len(smiles) for smiles in smiles_list]


def plot_length_distribution(reactant_lengths, product_lengths, save_path=None):
    """绘制反应物和产物长度的正态分布曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 反应物长度分布
    ax1.hist(reactant_lengths, bins=30, density=True, alpha=0.6, color='blue', label='实际分布')
    mu_r, sigma_r = np.mean(reactant_lengths), np.std(reactant_lengths)
    x_r = np.linspace(min(reactant_lengths), max(reactant_lengths), 100)
    ax1.plot(x_r, norm.pdf(x_r, mu_r, sigma_r), 'r-', lw=2, label=f'正态拟合 (μ={mu_r:.1f}, σ={sigma_r:.1f})')
    ax1.set_title('反应物SMILES长度分布')
    ax1.set_xlabel('长度（字符数）')
    ax1.set_ylabel('概率密度')
    ax1.legend()

    # 产物长度分布
    ax2.hist(product_lengths, bins=30, density=True, alpha=0.6, color='green', label='实际分布')
    mu_p, sigma_p = np.mean(product_lengths), np.std(product_lengths)
    x_p = np.linspace(min(product_lengths), max(product_lengths), 100)
    ax2.plot(x_p, norm.pdf(x_p, mu_p, sigma_p), 'r-', lw=2, label=f'正态拟合 (μ={mu_p:.1f}, σ={sigma_p:.1f})')
    ax2.set_title('产物SMILES长度分布')
    ax2.set_xlabel('长度（字符数）')
    ax2.set_ylabel('概率密度')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"分布曲线已保存至: {save_path}")
    plt.show()


def load_three_part_data(reactants_path, products_path):
    """读取反应物、产物文件，拼接为“反应物》产物”格式"""
    with open(reactants_path, 'r', encoding='utf-8') as f:
        reactants = [line.strip() for line in f if line.strip()]
    with open(products_path, 'r', encoding='utf-8') as f:
        products = [line.strip() for line in f if line.strip()]

    min_len = min(len(reactants), len(products))
    if len(reactants) != len(products):
        print(f"警告：反应物/产物文件行数不匹配，将按最短长度({min_len}条)处理")

    full_reactions = []
    for i in range(min_len):
        r = reactants[i]
        p = products[i]
        full_reaction = f"{r}》{p}"
        full_reactions.append(full_reaction)

    print(f"成功拼接 {len(full_reactions)} 条完整反应式（格式：反应物》产物）")
    return full_reactions


def identify_reaction_type(full_reaction):
    """优化反应类型识别，减少“其他反应”占比"""
    reaction_features = {
        # 1. 偶联反应
        "Suzuki偶联": r'(Br|Cl|I).*》.*[B].*',  # 反应物含卤素，产物含硼
        "Heck反应": r'(Br|Cl|I).*》.*C=C.*',  # 反应物含卤素，产物含烯烃
        "Buchwald-Hartwig胺化": r'(Br|Cl|I).*芳香环.*》.*芳香环-N.*',  # 芳香卤代物变芳香胺
        "Sonogashira偶联": r'(Br|Cl|I).*》.*C#C.*',  # 产物含炔烃
        "Negishi偶联": r'.*\[[Zn]\].*》.*',  # 锌试剂特征

        # 2. 取代反应
        "亲核取代(SN2)": r'(Br|Cl|I).*》.*(O|N|S)-.*',  # 卤素被O/N/S取代
        "芳香亲电取代": r'苯.*》.*苯-[NO2|SO3H|Cl|Br].*',  # 苯环取代
        "Grignard反应": r'.*Mg.*》.*',  # 格氏试剂特征

        # 3. 加成反应
        "Diels-Alder反应": r'C=C.*C=C.*》.*六元环.*',  # 双烯加成
        "Michael加成": r'C=C-C=O.*》.*C-C-C=O.*',  # 共轭加成
        "Wittig反应": r'C=O.*》.*C=C.*',  # 羰基变烯烃

        # 4. 氧化还原
        "氧化反应": r'.*》.*[O]=.*',  # 产物新增氧双键
        "还原反应": r'.*[O]=.*》.*[O]?H.*',  # 氧双键变羟基/氢

        # 5. 其他反应
        "酯化反应": r'.*COOH.*》.*COO-.*',  # 羧酸变酯
        "酰胺化反应": r'.*COOH.*》.*CON-.*',  # 羧酸变酰胺
        "环化反应": r'.*》.*(五元环|六元环).*',  # 成环
        "水解反应": r'.*COO-.*》.*COOH.*',  # 酯水解
        "其他反应": r'.*'  # 最后匹配
    }

    for rxn_type, pattern in reaction_features.items():
        if rxn_type == "其他反应":
            continue
        if re.search(pattern, full_reaction, re.IGNORECASE):
            return rxn_type
    return "其他反应"


def analyze_reaction_types_three_files(reactants_path, products_path, top_n=15, save_path=None):
    """分析反应类型，确保前15种显示"""
    full_reactions = load_three_part_data(reactants_path, products_path)

    print("\n开始识别反应类型...")
    reaction_types = []
    for i, rxn in enumerate(full_reactions):
        if i % 1000 == 0 and i > 0:
            print(f"已处理 {i}/{len(full_reactions)} 条反应")
        rxn_type = identify_reaction_type(rxn)
        reaction_types.append(rxn_type)

    # 统计前15种类型
    type_counts = pd.Series(reaction_types).value_counts().sort_values(ascending=False)
    total = len(reaction_types)
    type_percentages = type_counts / total * 100

    print(f"\n=== 前{top_n}种反应类型统计 ===")
    for i, (rxn_type, count) in enumerate(type_counts.head(top_n).items()):
        percent = type_percentages[rxn_type]
        print(f"{i + 1}. {rxn_type}: {count} 条 ({percent:.2f}%)")

    # 绘制柱状图（浅蓝色填充+边框，刻度向内）
    plt.figure(figsize=(16, 7))  # 加宽适配15种类型
    ax = type_counts.head(top_n).plot(
        kind='bar',
        color='lightblue',  # 浅蓝色填充
        edgecolor='lightblue',  # 浅蓝色边框
        linewidth=1.5
    )

    # 刻度向内
    ax.tick_params(axis='both', direction='in', labelsize=10)
    # 中文标签设置（强制使用宋体）
    plt.title(f'反应类型分布（前{top_n}种）', fontsize=14, fontname='SimSun')
    plt.xlabel('反应类型', fontsize=12, fontname='SimSun')
    plt.ylabel('数量', fontsize=12, fontname='SimSun')
    plt.xticks(rotation=45, ha='right', fontsize=9, fontname='SimSun')

    # 添加数值标签
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f'{height}',
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='bottom',
            fontsize=8, fontname='Times New Roman'
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"反应类型图已保存至: {save_path}")
    plt.show()

    return type_counts.head(top_n)


def main():
    # USPTO-MIT
    # test_s_path = r"D:\t5chem\data\Datasets\USPTO_MIT\MIT_separated_test\test_source.txt"
    # test_t_path = r"D:\t5chem\data\Datasets\USPTO_MIT\MIT_separated_test\test_target.txt"
    # train_s_path = r"D:\t5chem\data\Datasets\USPTO_MIT\MIT_separated_test\train_source.txt"
    # train_t_path = r"D:\t5chem\data\Datasets\USPTO_MIT\MIT_separated_test\train_target.txt"
    # val_s_path = r"D:\t5chem\data\Datasets\USPTO_MIT\MIT_separated_test\val_source.txt"
    # val_t_path = r"D:\t5chem\data\Datasets\USPTO_MIT\MIT_separated_test\val_target.txt"

    # CJHIF
    # test_s_path = r"D:\t5chem\data\CJHIF\test_source.txt"
    # test_t_path = r"D:\t5chem\data\CJHIF\test_target.txt"
    # train_s_path = r"D:\t5chem\data\CJHIF\train_source.txt"
    # train_t_path = r"D:\t5chem\data\CJHIF\train_target.txt"
    # val_s_path = r"D:\t5chem\data\CJHIF\val_source.txt"
    # val_t_path = r"D:\t5chem\data\CJHIF\val_target.txt"

    # USPTO-Schwaller
    test_s_path = r"D:\t5chem\data\Datasets\USPTO-Schwaller\test.source"
    test_t_path = r"D:\t5chem\data\Datasets\USPTO-Schwaller\test.target"
    train_s_path = r"D:\t5chem\data\Datasets\USPTO-Schwaller\train.source"
    train_t_path = r"D:\t5chem\data\Datasets\USPTO-Schwaller\train.target"
    val_s_path = r"D:\t5chem\data\Datasets\USPTO-Schwaller\val.source"
    val_t_path = r"D:\t5chem\data\Datasets\USPTO-Schwaller\val.target"

    # 读取数据
    test_s = load_smiles_file(test_s_path)
    test_t = load_smiles_file(test_t_path)
    train_s = load_smiles_file(train_s_path)
    train_t = load_smiles_file(train_t_path)
    val_s = load_smiles_file(val_s_path)
    val_t = load_smiles_file(val_t_path)

    reactants = test_s + train_s + val_s
    products = test_t + train_t + val_t

    # 长度分析
    reactant_lengths = calculate_smiles_lengths(reactants)
    product_lengths = calculate_smiles_lengths(products)

    plot_length_distribution(
        reactant_lengths,
        product_lengths,
        save_path="length_distribution.png"
    )

    # 反应类型分析（前15种）
    analyze_reaction_types_three_files(
        reactants_path=train_s_path,
        products_path=train_t_path,
        top_n=15,
        save_path="reaction_type_distribution.png"
    )


if __name__ == "__main__":
    main()