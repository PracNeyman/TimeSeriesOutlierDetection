import matplotlib.pyplot as plt
import numpy as np
import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Bidirectional
from numpy import arange, sin, pi, random

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

np.random.seed((round(time.time())))

# 指定GPU
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


#获取基础的波形
def get_base_pattern():
    t = np.arange(0.0,0.0,0.01)
    wave = sin(2*pi*t)
    noise = random.normal(0,0.05,len(t))
    wave = wave+noise
    return wave

# 模式高度异常
def gen_pattern_height_outlier():
    t = np.arange(0.0, 10.0, 0.01)
    wave1 = sin(2 * pi * t)
    noise = random.normal(0, 0.01, len(t))
    wave1 = wave1 + noise
    insert = len(t)//5
    wave2 = 4*sin(2*pi*t)
    wave1 = np.concatenate((wave1[:insert],wave2[:100],wave1[insert:]))
    # insert = np.random.randint(round(len(t) / 4), round(len(t) * 3 / 4))
    # for index in arange(25):
    #     if wave1[insert+index] > 0:
    #         wave1[insert+index] = wave1[insert + index] * 3
    # plt.figure(1)
    # plt.title("pattern_height_outlier")
    # plt.plot(wave1)
    # plt.show()
    return wave1

# 模式长度异常
def gen_pattern_length_outlier():
    t = np.arange(0.0, 4.0, 0.01)
    wave = sin(2 * pi * t)
    noise = random.normal(0, 0.01, len(t))
    wave = wave + noise
    t_rider = arange(0.0, 4, 0.01)
    wave2 = sin(0.5 * pi * t_rider)
    wave1 = np.concatenate((wave,wave2),0)
    wave1 = np.concatenate((wave1,wave))
    # plt.figure(1)
    # plt.title("pattern_length_outlier")
    # plt.plot(wave1)
    # plt.show()
    return wave1

# 模式均值和标准差异常
def gen_pattern_mean_and_std_outlier():
    t = np.arange(0.0, 10.0, 0.01)
    wave1 = sin(2 * pi * t)
    noise = random.normal(0, 0.01, len(t))
    t_rider = arange(0.0, 1.5, 0.01)
    noise_outlier = random.normal(1.5, 0.02, len(t_rider))
    # insert = np.random.randint(round(len(t) / 4), round(len(t) / 2))
    insert = len(t)//5
    noise[insert:insert+len(t_rider)] = noise_outlier
    wave1 = wave1 + noise
    # plt.figure(1)
    # plt.title("pattern_mean_and_std_outlier")
    # plt.plot(wave1)
    # plt.show()
    return wave1

# z-score标准化，即0均值标准化，将数据变换为均值为0、标准差为1的标准正态分布
def z_norm(result):
    result_mean = result.mean()
    result_std = result.std()
    tmp = result - result_mean
    tmp /= result_std
    return tmp, result_mean

def build_model(sub_interval_len):
    model = Sequential()
    layers = {'input': 1, 'hidden1': 64, 'hidden2': 256, 'hidden3': 100, 'output': 1}

    model.add(Bidirectional(LSTM(
            input_length= sub_interval_len-1,
            input_dim=layers['input'],
            output_dim=layers['hidden1'],
            return_sequences=True),input_shape=(sub_interval_len-1,1)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(
            layers['hidden2'],
            return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(
            layers['hidden3'],
            return_sequences=False)))
    model.add(Dropout(0.2))

    model.add(Dense(
            output_dim=layers['output']))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print ("Compilation Time : ", time.time() - start)
    return model

#在一个前驱区间中获取训练集或者在一个嫌疑区间中获取测试集
def get_train_or_test(data = None, window_size = None):
    train_or_test  = []
    window_start = 0
    # 对所有数据进行0均值标准化
    data = np.array(data)
    data,data_mean = z_norm(data)
    while(window_start+window_size <= len(data)):
        train_or_test.append(data[window_start:window_start+window_size])
        window_start = window_start+1
    return train_or_test

hava_model = {}


#这里data指的是当前所有数据,suspect_start 指的是嫌疑区间的开始坐标，suspect_end 指的是嫌疑区间的终止坐标,
# sub_interval_len指的是子区间的长度,window_size 是窗口大小，weights是前驱区间的权重，它的长度是前驱区间的个数
def run_network(model=None, data=None,suspect_start=None,suspect_end = None,sub_interval_len = None,window_size = None, weights = None, epochs = 1, batch_size = 64):
    # global_start_time = time.time()
    all_error = np.array([])
    # 异常子区间的开始坐标和结束坐标
    sub_interval_start = 0
    sub_interval_end = sub_interval_len
    while sub_interval_start <= (suspect_end-suspect_start):
        if(sub_interval_start >= (suspect_end-suspect_start)):
            break
        elif (sub_interval_start < (suspect_end-suspect_start)) and (sub_interval_end > (suspect_end-suspect_start)):
            sub_interval_end = (suspect_end-suspect_start)
        sub_interval = (data[sub_interval_start: sub_interval_end])
        print("目前嫌疑区间：[",sub_interval_start,sub_interval_end,"]")
        test = get_train_or_test(sub_interval, window_size)
        x_test = []
        y_test = []
        for sub in test:
            x_test.append(sub[:-1])
            y_test.append(sub[-1])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        y_test = np.array(y_test)
        # suspect_error 存放着嫌疑区间的累计error，每个前驱区间预测的差值都会累计到这里
        suspect_error = np.zeros(len(y_test))
        pre_interval_num = len(weights)
        for pre_interval_index in arange(pre_interval_num):
            pre_interval_start = sub_interval_start - (pre_interval_num - pre_interval_index) * sub_interval_len
            pre_interval_end = sub_interval_end - (pre_interval_num - pre_interval_index) * sub_interval_len
            # 如果前驱区间在数据中没有，那么要到数据的末尾去找
            pre_interval_start = (len(data) + pre_interval_start) % len(data)
            pre_interval_end = (len(data) + pre_interval_end) % len(data)
            if pre_interval_end == 0:
                pre_interval_end = len(data)
            print("开始坐标")
            print(pre_interval_start)
            print("结束坐标")
            print(pre_interval_end)

            if pre_interval_start in hava_model:
                model = hava_model[pre_interval_start]
            else:
                pre_interval = []
                if(pre_interval_end < pre_interval_start):
                    pre_interval.extend(data[pre_interval_start:len(data)])
                    pre_interval.extend(data[:pre_interval_end])
                else:
                    pre_interval = data[pre_interval_start:pre_interval_end]
                train = get_train_or_test(pre_interval,window_size)
                random.shuffle(train)
                x_train = []
                y_train = []
                for sub in train:
                    x_train.append(sub[:-1])
                    y_train.append(sub[-1])
                x_train = np.array(x_train)
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                y_train = np.array(y_train)
                print("开始训练")
                model.fit(
                    x_train, y_train,
                    batch_size=batch_size, epochs=epochs, validation_split=0.05)
                hava_model[pre_interval_start] = model
            print("开始预测")
            y_predict = model.predict(x_test)
            y_predict = np.reshape(y_predict,(y_predict.size,))
            cur_error = (y_predict - y_test) ** 2 * weights[pre_interval_index]
            suspect_error = suspect_error + cur_error
        weight_sum = 0
        for weight in weights:
            weight_sum = weight_sum+ weight
        suspect_error = suspect_error /weight_sum
        all_error = np.concatenate((all_error, np.zeros(window_size-1),suspect_error))
        print("当前区间检测完毕")
        sub_interval_start += sub_interval_len
        sub_interval_end += sub_interval_len
    # print ('Training duration (s) : ', time.time() - global_start_time)
    if len(all_error) < suspect_end - suspect_start:
        print("长度不够")
    return all_error

# 针对不同的点错误，连缀成一个区间的错误
def dots_to_interval(all_error, data, suspect_start, suspect_end, max_concate_len, alpha):
    anomaly_intervals_indexs = []

    MeanError = all_error.mean()
    StdError = all_error.std()
    i = 0
    while i < suspect_end - suspect_start:
        if all_error[i] > MeanError + alpha * StdError:
            maybe_anomaly = []
            for j in range(i + 1, suspect_end - suspect_start):
                maybe_anomaly.append(j)
                if all_error[j] > MeanError + alpha * StdError:
                    # anomaly_intervals_indexs.append([suspect_start + i, suspect_start + j])
                    maybe_anomaly = []
                if len(maybe_anomaly) > max_concate_len or j == suspect_end - suspect_start - 1:
                    anomaly_intervals_indexs.append([suspect_start + i, suspect_start + j])
                    maybe_anomaly = []
                    i = j + 1
                    break
        else:
            i = i + 1
    return anomaly_intervals_indexs

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
    epochs = 1
    batch_size = 64
    window_size = 50
    weights = [2, 2, 3, 3]
    sub_interval_len = 400
    max_concate_len = 40
    alpha = 0.65

    wave_len_outlier = gen_pattern_length_outlier()
    wave_height_outlier = gen_pattern_height_outlier()
    wave_mean_and_std_outlier = gen_pattern_mean_and_std_outlier()
    # data = get_base_pattern()
    # index2 = random.randint(0,len(data))
    # data = np.concatenate((data[:index2],wave_height_outlier,data[index2:]))
    # index3 = random.randint(0,len(data))
    # data = np.concatenate((data[:index3],wave_mean_and_std_outlier,data[index3:]))
    # index1 = random.randint(0,len(data))
    # data = np.concatenate((data[:index1], wave_len_outlier,data[index1:]))
    data = np.concatenate((wave_mean_and_std_outlier, wave_height_outlier, wave_len_outlier))

    time_start = time.time()
    all_error = run_network(
        model=build_model(window_size),data=data,suspect_start=0,suspect_end=len(data),
        sub_interval_len=sub_interval_len, window_size=window_size,weights=weights, epochs = epochs,
    batch_size = batch_size)
    costTime = time.time()-time_start

    # print(len(all_error))
    # print(len(data))

    anomaly_interval_indexs = dots_to_interval(all_error, data, 0, len(data), max_concate_len=max_concate_len, alpha=alpha)

    # 总长度为1000 + 1100 + 1200 = 3300，异常发生的真实位置分别为：200 - 350, 1200 - 1300，2500 - 2900
    ground_anomaly_from_to= [[200,350],[1200,1300],[2500,2900]]

    accuracy, precision, recall, F1 = get_accuracy_precision_recall_F1(0,len(data),anomaly_interval_indexs,ground_anomaly_from_to)
    paramStr = "windowsize="+str(window_size)+",  weights="+str(weights)+",  sub_interval_len="+str(sub_interval_len)\
               +",  max_concate_len="+str(max_concate_len)+",  alpha="+str(alpha)
    quotaStr = "\naccuracy=" + str(accuracy) + ",  precision=" + str(precision) + "\nrecall=" + str(
        recall) + ",  F1=" + str(F1) + "\n耗时：" + str(costTime)
    print(paramStr+quotaStr+"\n")
    plt.figure(1)
    plt.title("LSTM检测结果\n"+paramStr+quotaStr)
    plt.plot(np.arange(0,len(data)), data, 'blue', linewidth=1, label="震荡区间")
    print(anomaly_interval_indexs)
    for [start_index, end_index] in anomaly_interval_indexs:
        curFromTo = np.arange(start_index, end_index)
        curValues = np.array(data[start_index : end_index])
        plt.plot(curFromTo, curValues, 'red', linewidth=2.5, label="异常")
    plt.show()
    print("Over")
    
