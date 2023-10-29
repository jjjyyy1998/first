import matplotlib.pyplot as plt
import japanize_matplotlib
# ae_6 = [89.51,74.6,98.49,87.44,87.51]
# idnn_6 = [89.08,67.94,99.76,97.94,88.68]
# mobile_p_6 = [93.51,99.43,99.03,92.25,96.06]
# resnet_p_6 = [95.34,99.65,99.72,94.09,97.2]
# ae = [59.18,62.31,92.19,76.72,72.6]
# idnn = [58.17,58.67,94.07,95.59,76.63]
# mobile_p = [70.3,81.42,88.11,82.12,80.49]
# resnet_p = [71.61, 81.81, 90.08, 888, 81.95]
# mobile = [77.52, 80.38, 92.89, 80.23, 82.76]
# resnet = [78.16,87.08, 94.06, 88.74, 87.01]
# resnet_a = [79.29, 87.8, 93.33, 84.4, 86.21]
# unet = [73.18, 82.72, 76.72, 78.76, 77.84]
# unet_c = [77.61, 86.31, 91.47, 99.29, 88.67]
# unet_a = [71.87, 85.74, 76.36, 75.77, 77.44]
# unet_c_a = [83.76, 86.64, 96.27, 75.77, 85.61]
# unet_t_a = [75.47, 83.48, 89.28, 76.71, 81.24]
# unet_c_t_a = [84.38, 87.57, 95.38, 78.12, 86.36]
# unet_1 = [76.5,88.52,96.8,77.84,84.91]
# unrt_01 = [75.43,80.89,91.59,75.1,80.75]
# unet_all = [83.25,88.45,96.05,78.3]
# min6_min6 = [84.38, 87.57, 95.38, 78.12, 86.36]
# min6_normal6 = [88.95, 93.98, 97.19, 75.12, 88.81]
# normal6_min6 = [62.72, 34.45, 70.17, 65.48, 58.21]
# normal6_normal6 = [98.83, 99.53, 99.89, 82.65, 95.23]
cm = plt.cm.get_cmap('tab20').colors
print(cm[0])
ae_6 = {'score':[89.51,74.6,98.49,87.44,87.51],'color_':cm[0],'name':'AE_6dB'}
idnn_6 = {'score':[89.08,67.94,99.76,97.94,88.68],'color_':cm[2],'name':'IDNN_6dB'}
mobile_p_6 = {'score':[93.51,99.43,99.03,92.25,96.06],'color_':cm[4],'name':'MobileNetV2(pre)_6dB'}
resnet_p_6 = {'score':[95.34,99.65,99.72,94.09,97.2],'color_':cm[6],'name':'ResNet34(pre)_6dB'}
low = {'score':[95.34,99.65,99.72,94.09],'color_':cm[0],'name':'環境雑音：小'}
ae_min6 = {'score':[59.18,62.31,92.19,76.72,72.6],'color_':cm[1],'name':'AE_-6dB'}
idnn_min6 = {'score':[58.17,58.67,94.07,95.59,76.63],'color_':cm[3],'name':'IDNN_-6dB'}
mobile_p_min6 = {'score':[70.3,81.42,88.11,82.12,80.49],'color_':cm[5],'name':'MobileNetV2(pre)_-6dB'}
resnet_p_min6 = {'score':[71.61, 81.81, 90.08, 888, 81.95],'color_':cm[7],'name':'ResNet34(pre)_-6dB'}
high = {'score':[71.61, 81.81, 90.08, 88.8],'color_':cm[6],'name':'環境雑音：大'}
ae = {'score':[59.18,62.31,92.19,76.72,72.6],'color_':cm[1],'name':'AE'}
idnn = {'score':[58.17,58.67,94.07,95.59,76.63],'color_':cm[3],'name':'IDNN'}
mobile_p = {'score':[70.3,81.42,88.11,82.12,80.49],'color_':cm[5],'name':'MobileNetV2(pre)'}
resnet_p = {'score':[71.61, 81.81, 90.08, 88.8, 81.95],'color_':cm[7],'name':'ResNet34(pre)'}
mobile = {'score':[77.52, 80.38, 92.89, 80.23, 82.76],'color_':cm[8],'name':'MobileNetV2'}
resnet = {'score':[78.16,87.08, 94.06, 88.74, 87.01],'color_':cm[9],'name':'ResNet34'}
resnet_a = {'score':[79.29, 87.8, 93.33, 84.4, 86.21],'color_':cm[10],'name':'ResNet34 + aug'}
unet = {'score':[73.18, 82.72, 76.72, 78.76, 77.84],'color_':cm[11],'name':'U-Net'}
unet_c = {'score':[77.61, 86.31, 91.47, 99.29, 88.67],'color_':cm[12],'name':'U-Net + clf'}
unet_a = {'score':[71.87, 85.74, 76.36, 75.77, 77.44],'color_':cm[13],'name':'U-Net + aug'}
unet_c_a = {'score':[83.76, 86.64, 96.27, 75.77, 85.61],'color_':cm[14],'name':'U-Net + clf + aug'}
unet_t_a = {'score':[75.47, 83.48, 89.28, 76.71, 81.24],'color_':cm[15],'name':'U-Net + triplet + aug'}
unet_c_t_a = {'score':[84.38, 87.57, 95.38, 78.12, 86.36],'color_':cm[16],'name':'proposed'}
proposed = {'score':[84.38, 87.57, 95.38, 78.12, 86.36],'color_':cm[16],'name':'半分'}
unet_1 = {'score':[76.5,88.52,96.8,77.84,84.91],'color_':cm[17],'name':'alpha = 1.0'}
unet_01 = {'score':[75.43,80.89,91.59,75.1,80.75],'color_':cm[18],'name':'alpha = 0.1'}
unet_all = {'score':[83.25,88.45,96.05,78.3,86.51],'color_':cm[19],'name':'全体'}
min6_min6 = {'score':[84.38, 87.57, 95.38, 78.12, 86.36],'color_':cm[16],'name':'train:雑音大 test:雑音大'}
min6_normal6 = {'score':[88.95, 93.98, 97.19, 75.12, 88.81],'color_':cm[0],'name':'train:雑音大 test:雑音小'}
normal6_min6 = {'score':[62.72, 34.45, 70.17, 65.48, 58.21],'color_':cm[1],'name':'train:雑音小 test:雑音大'}
normal6_normal6 = {'score':[98.83, 99.53, 99.89, 82.65, 95.23],'color_':cm[2],'name':'train:雑音小 test:雑音小'}

import matplotlib.pyplot as plt
import numpy as np


cor = [low,high]
pre = [ae_6,ae_min6,idnn_6,idnn_min6,mobile_p_6,mobile_p_min6,resnet_p_6,resnet_p_min6]
compare = [ae,idnn,mobile_p,resnet_p,unet_c_t_a]
compare2 = [resnet_p,resnet,resnet_a,unet_c,unet_c_t_a]
ablation = [unet,unet_c,unet_a,unet_c_a,unet_t_a,unet_c_t_a]
pseudo = [proposed,unet_1,unet_01]
half = [proposed,unet_all]
snr = [normal6_normal6,normal6_min6,min6_normal6,min6_min6]



height1 = [80, 65, 100, 42, 54]  # 点数1
height2 = [55, 100, 98, 30]  # 点数2

left = np.arange(len(height2)) * 2  # numpyで横軸を設定
labels = ['機械A', '機械B', '機械C', '機械D']
width = 0.8

# plt.rcParams['figure.subplot.bottom'] = 0.8
plt.rcParams['figure.subplot.right'] = 0.8
plt.rcParams["font.size"] = 20
plt.figure(figsize=[15,7.5])

for i,v in enumerate(cor):
    plt.bar(left+width*i, v['score'], color=v['color_'], width=width, align='center',label=v['name'])
plt.xticks(left + (width*i)/2, labels)
plt.ylabel('Avarage AUC')
plt.ylim(50,100)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.rcParams['figure.subplot.right'] = 0.8
plt.savefig('intro')
plt.close()

left = np.arange(len(height1)) * 2  # numpyで横軸を設定

labels = ['fan', 'pump', 'slider', 'valve', 'average']

width = 0.15

# plt.rcParams['figure.subplot.bottom'] = 0.8
plt.rcParams['figure.subplot.right'] = 0.8
plt.rcParams["font.size"] = 14
plt.figure(figsize=[15,3])

for i,v in enumerate(pre):
    plt.bar(left+width*i, v['score'], color=v['color_'], width=width, align='center',label=v['name'])
plt.xticks(left + (width*i)/2, labels)
plt.ylabel('Avarage AUC')
plt.ylim(50,100)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.rcParams['figure.subplot.right'] = 0.8
plt.savefig('pre_exp')
plt.close()

width = 0.3
plt.rcParams['figure.subplot.right'] = 0.8
plt.rcParams["font.size"] = 14
plt.figure(figsize=[15,3])

for i,v in enumerate(ablation):
    plt.bar(left+width*i, v['score'], color=v['color_'], width=width, align='center',label=v['name'])
plt.xticks(left + (width*i)/2, labels)
plt.ylabel('Avarage AUC')
plt.ylim(50,100)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.rcParams['figure.subplot.right'] = 0.8
plt.savefig('ablation')
plt.close()
width = 0.35
plt.rcParams['figure.subplot.right'] = 0.8
plt.rcParams["font.size"] = 14
plt.figure(figsize=[15,3])
for i,v in enumerate(compare):
    plt.bar(left+width*i, v['score'], color=v['color_'], width=width, align='center',label=v['name'])
plt.xticks(left + (width*i)/2, labels)
plt.ylabel('Avarage AUC')
plt.ylim(50,100)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.rcParams['figure.subplot.right'] = 0.8
plt.savefig('compare')
plt.close()
width = 0.35
plt.rcParams['figure.subplot.right'] = 0.8
plt.rcParams["font.size"] = 14
plt.figure(figsize=[15,3])
for i,v in enumerate(compare2):
    plt.bar(left+width*i, v['score'], color=v['color_'], width=width, align='center',label=v['name'])
plt.xticks(left + (width*i)/2, labels)
plt.ylabel('Avarage AUC')
plt.ylim(50,100)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.rcParams['figure.subplot.right'] = 0.8
plt.savefig('compare2')
plt.close()
plt.rcParams['figure.subplot.right'] = 0.8
plt.rcParams["font.size"] = 14
plt.figure(figsize=[15,3])
width = 0.6
for i,v in enumerate(pseudo):
    plt.bar(left+width*i, v['score'], color=v['color_'], width=width, align='center',label=v['name'])
plt.xticks(left + (width*i)/2, labels)
plt.ylabel('Avarage AUC')
plt.ylim(50,100)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.rcParams['figure.subplot.right'] = 0.8
plt.savefig('pseudo')
plt.close()
width = 0.9
plt.rcParams['figure.subplot.right'] = 0.8
plt.rcParams["font.size"] = 14
plt.figure(figsize=[15,3])
for i,v in enumerate(half):
    plt.bar(left+width*i, v['score'], color=v['color_'], width=width, align='center',label=v['name'])
plt.xticks(left + (width*i)/2, labels)
plt.ylabel('Avarage AUC')
plt.ylim(50,100)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.rcParams['figure.subplot.right'] = 0.8
plt.savefig('half')
plt.close()
width = 0.4
plt.rcParams['figure.subplot.right'] = 0.8
plt.rcParams["font.size"] = 14
plt.figure(figsize=[15,3])
for i,v in enumerate(snr):
    plt.bar(left+width*i, v['score'], color=v['color_'], width=width, align='center',label=v['name'])
plt.xticks(left + (width*i)/2, labels)
plt.ylabel('Avarage AUC')
plt.ylim(50,100)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.rcParams['figure.subplot.right'] = 0.8
plt.savefig('snr')
plt.close()
