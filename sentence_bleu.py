# top-1 bleu
# import pandas as pd
# import sacrebleu
#
# # 读取 CSV 文件
# file_path = '/home/zhulx/t5chem/checkpoints/MIT_separated_dual_max/best_cp-370000/predictions.csv'
# df = pd.read_csv(file_path)
#
# # 提取 target 和 prediction_1 列
# p_samples = df['prediction_1'].tolist()
# r_samples = df['target'].tolist()
#
# scores = 0
# for i in range(len(r_samples)):
#     prediction = p_samples[i]
#     reference = r_samples[i]
#     if isinstance(prediction, str) and isinstance(reference, str):
#         bleu = sacrebleu.sentence_bleu(prediction, [reference])
#         score = bleu.score
#         scores += score
#
# result = scores / len(r_samples)
# print(f"BLEU 分数的平均值为: {result}")

# # top-5 bleu
import pandas as pd
import sacrebleu
import math

# 读取 CSV 文件
file_path = r'D:\t5chem\checkpoints\USPTO-MIT_RtoP_aug5\best_cp-760000\predictions.csv'
df = pd.read_csv(file_path)

# 提取 target 和 prediction_1 列
p1_samples = df['prediction_1'].tolist()
p2_samples = df['prediction_2'].tolist()
p3_samples = df['prediction_3'].tolist()
p4_samples = df['prediction_4'].tolist()
p5_samples = df['prediction_5'].tolist()
r_samples = df['target'].tolist()
num = len(r_samples)

scores = 0
for i in range(len(r_samples)):
    prediction1 = str(p1_samples[i])
    prediction2 = str(p2_samples[i])
    prediction3 = str(p3_samples[i])
    prediction4 = str(p4_samples[i])
    prediction5 = str(p5_samples[i])
    reference = r_samples[i]

    if isinstance(reference, str):
        if isinstance(prediction1, str):
            bleu1 = sacrebleu.sentence_bleu(prediction1, [reference])
        if isinstance(prediction2, str):
            bleu2 = sacrebleu.sentence_bleu(prediction2, [reference])
        if isinstance(prediction3, str):
            bleu3 = sacrebleu.sentence_bleu(prediction3, [reference])
        if isinstance(prediction4, str):
            bleu4 = sacrebleu.sentence_bleu(prediction4, [reference])
        if isinstance(prediction5, str):
            bleu5 = sacrebleu.sentence_bleu(prediction5, [reference])

        score1 = bleu1.score
        score2 = bleu2.score
        score3 = bleu3.score
        score4 = bleu4.score
        score5 = bleu5.score
        score = max(score1, score2, score3, score4, score5)
        scores += score
    else:
        print("输入的reference为 nan，不执行后续操作。")
        num = num - 1

print("num: ", num)
result = scores / num
print(f"BLEU 分数的平均值为: {result}")


