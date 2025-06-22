# # import pandas as pd
# #
# # data = pd.read_csv('histogram.csv')
# # scores = data['scores']  # Assuming 'score' is the column name for the score
# # labels = data['labels']  # Assuming 'label' is the column name for the actual label
# #
# # thresholds = sorted(scores)
# #
# # from sklearn.metrics import accuracy_score
# #
# # best_threshold = thresholds[0]
# # best_accuracy = 0
# #
# # for threshold in thresholds:
# #     preds = [1 if score > threshold else 0 for score in scores]
# #     accuracy = accuracy_score(labels, preds)
# #     if accuracy > best_accuracy:
# #         best_accuracy = accuracy
# #         best_threshold = threshold
# #
# # print(best_threshold)
# # print(best_accuracy)
#
# # import pandas as pd
# # import numpy as np
# # from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, average_precision_score
# # import matplotlib.pyplot as plt
# #
# # # 1.加载数据
# # data = pd.read_csv('histogram.csv')
# # scores = data.iloc[1:, 1]  # Assuming scores are in the second column
# # labels = data.iloc[1:, 2]  # Assuming labels are in the third column
# #
# # # 2.计算ROC曲线的数据点
# # fpr, tpr, thresholds = roc_curve(labels, scores)
# #
# # # 3.计算AUC值
# # roc_auc = auc(fpr, tpr)
# # print("AUC: ", roc_auc)
# #
# # # 4.找到最优阈值 (最接近左上角的点对应的阈值)
# # optimal_idx = np.argmax(tpr - fpr)
# # optimal_threshold = thresholds[optimal_idx]
# # print("Optimal threshold value: ", optimal_threshold)
# #
# # # 5.计算并输出在最优阈值下的准确率、精确率、召回率、F1分数
# # preds = [1 if score >= optimal_threshold else 0 for score in scores]
# # print("Accuracy: ", accuracy_score(labels, preds))
# # print("AUPRC: ", average_precision_score(labels, preds))
# # print("Precision: ", precision_score(labels, preds))
# # print("Recall: ", recall_score(labels, preds))
# # print("F1 Score: ", f1_score(labels, preds))
# #
# # # 6. 绘制ROC曲线
# # plt.title('Receiver Operating Characteristic')
# # plt.plot(fpr, tpr, label='AUC = %0.2f'% roc_auc)
# # plt.legend(loc='lower right')
# # plt.plot([0,1],[0,1],'r--')
# # plt.xlim([-0.1,1.2])
# # plt.ylim([-0.1,1.2])
# # plt.ylabel('True Positive Rate')
# # plt.xlabel('False Positive Rate')
# # plt.show()
# #
# #
# #
#
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# import numpy as np
#
# # 读取csv数据
# data = pd.read_csv('histogram.csv')
#
# # 从数据中提取scores列
# scores = data['scores']
#
# # 计算均值和标准差
# mu, std = norm.fit(scores)
#
# # 绘制histogram
# plt.hist(scores, bins=25, density=True, alpha=0.6, color='g')
#
# # 绘制PDF
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)
# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
# plt.title(title)
#
# plt.show()

# import matplotlib.pyplot as plt
#
# # 数据设置
# dataset = {'mono': {'train': 1410, 'test_normal': 118, 'test_abnormal': 121},
#            'poly': {'train': 1743, 'test_normal': 180, 'test_abnormal': 134}}
#
# # 创建画布
# fig, axs = plt.subplots(1, 2, figsize=(10,5))
#
# # 设置颜色
# colors = ['#66b3ff', '#99ff99', '#ffcc99']
#
# # 自定义函数，用于在饼图上添加百分比和具体数据
# def make_autopct(values):
#     def my_autopct(pct):
#         total = sum(values)
#         val = int(round(pct*total/100.0))
#         return '{p:.1f}%\n({v:d})'.format(p=pct,v=val)
#     return my_autopct
#
# # 画出mono的数据分布
# axs[0].pie(dataset['mono'].values(), labels=dataset['mono'].keys(),
#            autopct=make_autopct(dataset['mono'].values()), pctdistance=0.85, labeldistance=1.1,
#            shadow=True, startangle=140, colors=colors, textprops={'fontsize': 12})
# axs[0].set_title('Mono Data Distribution', fontsize=12)
#
# # 画出poly的数据分布
# axs[1].pie(dataset['poly'].values(), labels=dataset['poly'].keys(),
#            autopct=make_autopct(dataset['poly'].values()), pctdistance=0.85, labeldistance=1.1,
#            shadow=True, startangle=140, colors=colors, textprops={'fontsize': 12})
# axs[1].set_title('Poly Data Distribution', fontsize=12)
#
# # 显示图形
# plt.show()
