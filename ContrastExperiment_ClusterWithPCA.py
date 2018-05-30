import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,DBSCAN
from sklearn.decomposition import PCA
import time
from numpy import arange, sin, pi, random

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

np.random.seed((round(time.time())))

# 模式高度异常，总长1100，异常在200-300的位置发生
def gen_pattern_height_outlier():
    t = np.arange(0.0, 10.0, 0.01)
    wave1 = sin(2 * pi * t)
    noise = random.normal(0, 0.01, len(t))
    wave1 = wave1 + noise
    insert = len(t)//5
    wave2 = 4*sin(2*pi*t)
    wave1 = np.concatenate((wave1[:insert],wave2[:100],wave1[insert:]))
    return wave1

# 模式长度异常，总长1200，异常在400-800的位置发生
def gen_pattern_length_outlier():
    t = np.arange(0.0, 4.0, 0.01)
    wave = sin(2 * pi * t)
    noise = random.normal(0, 0.01, len(t))
    wave = wave + noise
    t_rider = arange(0.0, 4, 0.01)
    wave2 = sin(0.5 * pi * t_rider)
    wave1 = np.concatenate((wave,wave2),0)
    wave1 = np.concatenate((wave1,wave))
    return wave1

# 模式均值和标准差异常，总长1000，异常在200-350的位置发生
def gen_pattern_mean_and_std_outlier():
    t = np.arange(0.0, 10.0, 0.01)
    wave1 = sin(2 * pi * t)
    noise = random.normal(0, 0.01, len(t))
    t_rider = arange(0.0, 1.5, 0.01)
    noise_outlier = random.normal(1.5, 0.02, len(t_rider))
    insert = len(t)//5
    noise[insert:insert+len(t_rider)] = noise_outlier
    wave1 = wave1 + noise
    return wave1

# 连接顺序为均值标准差、高度、长度，因此总长度为1000+1100+1200=3300，异常发生的位置分别为：200-350,1200-1300，2500-2900
def get_data():
    wave_len_outlier = gen_pattern_length_outlier()
    wave_height_outlier = gen_pattern_height_outlier()
    wave_mean_and_std_outlier = gen_pattern_mean_and_std_outlier()
    data = np.concatenate((wave_mean_and_std_outlier, wave_height_outlier, wave_len_outlier))
    return data

def reduct_dimension(origin_data):
    pca = PCA(n_components='mle')
    pca.fit(origin_data)
    new_data = pca.transform(origin_data)
    return new_data

def divide_interval(data, interval_size, space=10):
    intervals = []
    start = 0
    end = start + interval_size
    while end <= len(data):
        intervals.append(data[start:end])
        start += space
        end = start + interval_size
    intervals = np.array(intervals)
    print(intervals.shape)
    # intervals = np.array(intervals, (intervals.shape[0],intervals.shape[1]))
    return intervals

def pick_anomaly_and_plot(labels,paramStr,data_len, ground_anomaly_from_to,costTime):
    anomaly_labels = []
    tmp = labels
    if -1 in labels:
        anomaly_labels.append(-1)
        negativeIndex = np.argwhere(labels==-1)
        labels = np.delete(labels, negativeIndex)
    # -1表示非核心点，即噪声，但是-1不能进行counts，所以先去除，计算counts后，再加上
    counts = np.bincount(labels)
    # normal_label = np.argmax(counts)
    counts = np.array(counts)
    counts_mean = counts.mean()
    counts_std = counts.std()
    print(counts)
    for label in labels:
        if counts[label] < counts_mean and label not in anomaly_labels:
            anomaly_labels.append(label)
    print(anomaly_labels)
    labels = tmp

    detect_anomaly_indexes = []

    plt.figure(1)
    plt.plot(np.arange(0, len(data)), data, 'blue', linewidth=1, label="震荡区间")
    for index in arange(intervals_num):
        if labels[index] not in anomaly_labels:
            continue
        start_index = space * index
        end_index = start_index + interval_size
        detect_anomaly_indexes.append([start_index, end_index])
        curFromTo = np.arange(start_index, end_index)
        curValues = np.array(data[start_index: end_index])
        plt.plot(curFromTo, curValues, 'red', linewidth=2.5, label="异常")
    accuracy, precision, recall, F1 = get_accuracy_precision_recall_F1(0, data_len, detect_anomaly_indexes, ground_anomaly_from_to)
    quotaStr = "\naccuracy="+str(accuracy)+",  precision="+str(precision)+"\nrecall="+str(recall)+",  F1="+str(F1)+"\n耗时："+str(costTime)
    plt.title("检测结果\n方法及参数:"+paramStr+quotaStr)
    plt.show()
    print(paramStr+quotaStr+"\n")

# num是一个坐标，from_to是[[a1,b1],[a2,b2]...]的形式，判断是否存在[ai,bi]使得 ai<=num<bi
# 判断一个num是否在from_to的
def is_exist(num, from_to):
    for [start, end] in from_to:
        if start<=num and num < end:
            return True
    return False

# accuracy：准确率，被正确分类的频率
# precision：精确率，TP/TP+FP， 在我们的例子中，“P”即为“异常”，“N”为正常
# recall： 召回率，TP/TP+FN
# F1：精确率和召回率的调和均值
# data_len 指震荡区间的长度
def get_accuracy_precision_recall_F1(start, end, detect_from_to, ground_from_to):
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for index in range(start, end):
        ground_anomaly = is_exist(index, ground_from_to)
        predict_anomaly = is_exist(index, detect_from_to)
        if(ground_anomaly and predict_anomaly):
            TP += 1
        elif ground_anomaly and (not predict_anomaly):
            FN += 1
        elif (not ground_anomaly) and predict_anomaly:
            FP += 1
        elif (not ground_anomaly) and (not predict_anomaly):
            TN += 1
    accuracy = (TP + TN) / (end - start)
    if TP + FP ==0:
        precision = -1
    else:
        precision = TP / (TP + FP)
    if TP + FN ==0:
        recall = -1
    else:
        recall = TP / (TP + FN)
    if recall==-1 or precision == -1:
        F1 = -1
    else:
        F1 = 2 / (1/precision + 1/recall)
    return accuracy, precision, recall, F1

if __name__ == '__main__':
    interval_size = 50
    space = 15
    # 总长度为1000 + 1100 + 1200 = 3300，异常发生的真实位置分别为：200 - 350, 1200 - 1300，2500 - 2900
    ground_anomaly_from_to= [[200,350],[1200,1300],[2500,2900]]

    data = get_data()
    print(data.shape)
    intervals = divide_interval(data, interval_size=interval_size, space=space)
    intervals_new = reduct_dimension(intervals)
    print(intervals_new.shape)
    intervals_num = round((len(data) - interval_size + 1) / space)
    intervals_num = intervals_new.shape[0]

    time_start = time.time()
    kmeans = KMeans(n_clusters=5).fit(intervals_new)
    labels = kmeans.labels_
    param = 'KMeans，cluster = 5'
    costTime = time.time() - time_start
    pick_anomaly_and_plot(labels, param, data_len=len(data), ground_anomaly_from_to=ground_anomaly_from_to,
                          costTime=costTime)

    time_start = time.time()
    kmeans = KMeans(n_clusters=6).fit(intervals_new)
    labels = kmeans.labels_
    param = 'KMeans，cluster = 6'
    costTime = time.time() - time_start
    pick_anomaly_and_plot(labels, param, data_len=len(data), ground_anomaly_from_to=ground_anomaly_from_to,
                          costTime=costTime)

    time_start = time.time()
    kmeans = KMeans(n_clusters=10).fit(intervals_new)
    labels = kmeans.labels_
    param = 'KMeans，cluster = 10'
    costTime = time.time() - time_start
    pick_anomaly_and_plot(labels, param, data_len=len(data), ground_anomaly_from_to=ground_anomaly_from_to,
                          costTime=costTime)

    time_start = time.time()
    kmeans = KMeans(n_clusters=50).fit(intervals_new)
    labels = kmeans.labels_
    param = 'KMeans，cluster = 50'
    costTime = time.time()-time_start
    pick_anomaly_and_plot(labels,param,data_len=len(data),ground_anomaly_from_to=ground_anomaly_from_to,costTime=costTime)

    time_start = time.time()
    dbscan = DBSCAN(eps=1e-1, min_samples=1).fit(intervals_new)
    labels = dbscan.labels_
    param = "DBSCAN, eps=1e-1, min_samples=1"
    costTime = time.time() - time_start
    pick_anomaly_and_plot(labels, param, data_len=len(data), ground_anomaly_from_to=ground_anomaly_from_to,
                          costTime=costTime)


    time_start = time.time()
    dbscan = DBSCAN(eps=1e-1, min_samples=1).fit(intervals_new)
    labels = dbscan.labels_
    param = "DBSCAN, eps=1e-1, min_samples=1"
    costTime = time.time() - time_start
    pick_anomaly_and_plot(labels, param, data_len=len(data), ground_anomaly_from_to=ground_anomaly_from_to,
                          costTime=costTime)

    time_start = time.time()
    dbscan = DBSCAN(eps=1, min_samples=1).fit(intervals_new)
    labels = dbscan.labels_
    param = "DBSCAN, eps=1, min_samples=1"
    costTime = time.time() - time_start
    pick_anomaly_and_plot(labels, param, data_len=len(data), ground_anomaly_from_to=ground_anomaly_from_to,
                          costTime=costTime)

    time_start = time.time()
    dbscan = DBSCAN(eps=1e-1, min_samples=2).fit(intervals_new)
    labels = dbscan.labels_
    costTime = time.time() - time_start
    param = "DBSCAN, eps=1e-1, min_samples=2"
    pick_anomaly_and_plot(labels, param, data_len=len(data), ground_anomaly_from_to=ground_anomaly_from_to,
                          costTime=costTime)
