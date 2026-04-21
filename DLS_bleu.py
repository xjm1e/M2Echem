# 数据集中不同长度序列对应的BLUE和准确率
# 短序列：0-75
# 中序列：75-150
# 长序列：150以上

import pandas as pd
import sacrebleu
import math

# 读取 CSV 文件
file_path = r'D:\t5chem\checkpoints\USPTO-50k\best_cp-25000\predictions.csv'
df = pd.read_csv(file_path)

# 提取 target 和 prediction_1-5 列
p1_samples = df['prediction_1'].tolist()
p2_samples = df['prediction_2'].tolist()
p3_samples = df['prediction_3'].tolist()
p4_samples = df['prediction_4'].tolist()
p5_samples = df['prediction_5'].tolist()
r_samples = df['target'].tolist()

# 初始化长度分组数据结构
length_groups = {
    'short': {'scores': [], 'top1': 0, 'top3': 0, 'top5': 0, 'count': 0},
    'medium': {'scores': [], 'top1': 0, 'top3': 0, 'top5': 0, 'count': 0},
    'long': {'scores': [], 'top1': 0, 'top3': 0, 'top5': 0, 'count': 0}
}

for i in range(len(r_samples)):
    # 获取预测和参考样本
    prediction1 = str(p1_samples[i])
    prediction2 = str(p2_samples[i])
    prediction3 = str(p3_samples[i])
    prediction4 = str(p4_samples[i])
    prediction5 = str(p5_samples[i])
    reference = r_samples[i]

    # 跳过reference为nan的样本
    if not isinstance(reference, str) or pd.isna(reference):
        continue

    # 确定参考序列长度并分组
    ref_length = len(reference)  # 按空格分割计算长度
    if ref_length < 55:
        group = 'short'
    elif ref_length < 110:
        group = 'medium'
    else:
        group = 'long'

    # 计算BLEU分数
    bleu1 = sacrebleu.sentence_bleu(prediction1, [reference]).score
    bleu2 = sacrebleu.sentence_bleu(prediction2, [reference]).score
    bleu3 = sacrebleu.sentence_bleu(prediction3, [reference]).score
    bleu4 = sacrebleu.sentence_bleu(prediction4, [reference]).score
    bleu5 = sacrebleu.sentence_bleu(prediction5, [reference]).score

    max_bleu = max(bleu1, bleu2, bleu3, bleu4, bleu5)
    length_groups[group]['scores'].append(max_bleu)
    length_groups[group]['count'] += 1

    # 计算准确率
    top1_correct = prediction1 == reference
    top3_correct = top1_correct or (prediction2 == reference) or (prediction3 == reference)
    top5_correct = top3_correct or (prediction4 == reference) or (prediction5 == reference)

    length_groups[group]['top1'] += top1_correct
    length_groups[group]['top3'] += top3_correct
    length_groups[group]['top5'] += top5_correct

# 输出结果
for group_name, group_data in length_groups.items():
    count = group_data['count']
    if count == 0:
        print(f"\n{group_name.capitalize()} Sequences (0-0 tokens):")
        print("  No samples in this length group.")
        continue

    avg_bleu = sum(group_data['scores']) / count if group_data['scores'] else 0
    top1_acc = group_data['top1'] / count
    top3_acc = group_data['top3'] / count
    top5_acc = group_data['top5'] / count

    print(
        f"\n{group_name.capitalize()} Sequences ({'0-55' if group_name == 'short' else '55-110' if group_name == 'medium' else '110+'} tokens):")
    print(f"  Samples: {count}")
    print(f"  Average BLEU: {avg_bleu:.2f}")
    print(f"  Top-1 Accuracy: {top1_acc:.2%}")
    print(f"  Top-3 Accuracy: {top3_acc:.2%}")
    print(f"  Top-5 Accuracy: {top5_acc:.2%}")

# 计算总体统计数据
total_count = sum(g['count'] for g in length_groups.values())
total_avg_bleu = sum(sum(g['scores']) for g in length_groups.values()) / total_count if total_count > 0 else 0
total_top1 = sum(g['top1'] for g in length_groups.values()) / total_count if total_count > 0 else 0
total_top3 = sum(g['top3'] for g in length_groups.values()) / total_count if total_count > 0 else 0
total_top5 = sum(g['top5'] for g in length_groups.values()) / total_count if total_count > 0 else 0

print(f"\nOverall Statistics:")
print(f"  Total Samples: {total_count}")
print(f"  Average BLEU: {total_avg_bleu:.2f}")
print(f"  Top-1 Accuracy: {total_top1:.2%}")
print(f"  Top-3 Accuracy: {total_top3:.2%}")
print(f"  Top-5 Accuracy: {total_top5:.2%}")
