import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.stattools import adfuller, coint
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.covariance import EllipticEnvelope
from sklearn import svm

#滑动平均图
def draw_trend(ts, size):
    f = plt.figure()
    #滑动平均
    rol_mean = ts.rolling(window=size).mean()
    #加权滑动平均
    rol_weighted_mean = pd.ewma(ts, span=size)

    ts.plot(color='blue',label='Original')
    rol_mean.plot(color='red',label='Rolling Mean')
    rol_weighted_mean.plot(color='green',label='Weighted rolling mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()



# adf检验平稳性
def testStationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput
# print(testStationarity(ts))

def validation(ts_a,ts_b,ts_c):
    # 单位根检验平稳性
    df = pd.DataFrame()
    df = pd.concat([ts_a, ts_b, ts_c], axis=1).dropna()
    df.columns = ['a', 'b', 'c']

    # 记录每个分量的阶数
    adf_jieshu = {}
    # 需要进行一阶差分的columns
    diff_columns = []
    # 需要进行二阶差分的columns
    diff_diff_columns = []
    # 进行二阶差分后还不平稳的columns
    other_columns = []
    # 原始数据的p值
    adf_test_p = [adfuller(df[column])[1] for column in df.columns]
    for i in range(len(adf_test_p)):
        if (adf_test_p[i] < 0.05):
            adf_jieshu[0] = []
            adf_jieshu[0].append(df.columns[i])
        else:
            diff_columns.append(df.columns[i])
    # 一阶差分后的p值
    adf_test_p = [adfuller(df[column].diff().dropna())[1] for column in diff_columns]
    for i in range(len(adf_test_p)):
        if (adf_test_p[i] < 0.05):
            adf_jieshu[1] = []
            adf_jieshu[1].append(diff_columns[i])
        else:
            diff_diff_columns.append(df.columns[i])
    # 二阶差分后的p值
    adf_test_p = [adfuller(df[column].diff().dropna().diff().dropna())[1] for column in diff_diff_columns]
    for i in range(len(adf_test_p)):
        if (adf_test_p[i] < 0.01):
            adf_jieshu[2] = []
            adf_jieshu[2].append(diff_diff_columns[i])
        else:
            other_columns.append(df.columns[i])

    # 对阶数相同的时间序列进行协整检验
    # 协整检验的结果不满足对称性吗？这里的p值不相等
    coint_test_p_0 = coint(df['a'], df['c'])[1]
    coint_test_p_1 = coint(df['c'], df['a'])[1]
    # coint_test_p_2 = coint([df[column] for column in adf_jieshu[2]])[1]
    # coint_test_p_0 = coint([df[column] for column in adf_jieshu[0]])[1]
    # coint_test_p_1 = coint([df[column] for column in adf_jieshu[1]])[1]
    # coint_test_p_2 = coint([df[column] for column in adf_jieshu[2]])[1]
    # 计算相同阶数的列的
    print(adf_jieshu[0])
    print(coint_test_p_0)
    print(adf_jieshu[1])
    print(coint_test_p_1)
    # print(adf_jieshu[2])
    # print(coint_test_p_2)

def mini_batch_k_means_cluster(df_data,clm_select,plot=True):
    cluster_pred = [MiniBatchKMeans(n_clusters=8, random_state=9).fit_predict(np.array(df_data[['val', clm]])) for clm
                    in clm_select]
    if plot==True:
        plt.figure()
        for i in range(len(cluster_pred)):
            subplot = plt.subplot(3, 1, i + 1)
            subplot.scatter(df_data['val'], df_data[clm_select[i]], c=cluster_pred[i])
            subplot.set_title('TimeSeries  ' + clm_select[i])
        plt.suptitle("Outlier detection With Mini Batch K-means")
        plt.show()
    return cluster_pred

def dbscan_cluster(df_data, clm_select):
    cluster_pred = [DBSCAN(eps=1e-30, min_samples=100).fit_predict(np.array(df_data[['val', clm]])) for clm
                    in clm_select]
    # print(cluster_pred)
    plt.figure()
    for i in range(len(cluster_pred)):
        subplot = plt.subplot(3, 1, i + 1)
        subplot.scatter(df_data['val'], df_data[clm_select[i]], c=cluster_pred[i])
        subplot.set_title('TimeSeries  ' + clm_select[i])
    plt.suptitle("Outlier detection With DBSCAN")
    plt.show()
    return cluster_pred

def isolateForestDetection(clm_select,all_tss, df_data,plot=False):
    rng = np.random.RandomState(42)
    outliers_fraction = 0.25
    if plot:
        plt.figure()
    iforest_pred = {}
    for i in range(len(clm_select)):
        col = clm_select[i]
        j = 1
        iforest_pred[col] = []
        for kind in all_tss[col].keys():
            j += 1
            X = np.array(all_tss[col][kind])
            # 孤立森林
            clf = IsolationForest(contamination=outliers_fraction, random_state=rng)
            clf.fit(X)
            y_pred = clf.predict(X)
            iforest_pred[col].extend(y_pred)
        if plot:
            subplot = plt.subplot(len(clm_select), 1, i + 1)
            subplot.scatter(df_data['val'], df_data[col], c=iforest_pred[col])
            subplot.set_title('Dimension ' + clm_select[i])
    if plot:
        plt.suptitle('Outlier detection with Isolate Forest')
        plt.show()
    return iforest_pred


def calculate_test_statistic(ts):
    zscores = abs(stats.zscore(ts, ddof=1))
    max_idx = np.argmax(zscores)
    return max_idx, zscores[max_idx]


def calculate_critical_value(ts, alpha):
    size = len(ts)
    t_dist = stats.t.ppf(1 - alpha / (2 * size), size - 2)

    numerator = (size - 1) * t_dist
    denominator = np.sqrt(size ** 2 - size * 2 + size * t_dist ** 2)

    return numerator / denominator


def seasonal_esd(ts, seasonality=None, hybrid=False, max_anomalies=100, alpha=0.5):
    ts = np.array(ts)
    seasonal = seasonality or int(0.2 * len(ts))  # Seasonality is 20% of the ts if not given.
    decomp = sm.tsa.seasonal_decompose(ts, freq=seasonal)
    if hybrid:
        mad = np.median(np.abs(ts - np.median(ts)))
        residual = ts - decomp.seasonal - mad
    else:
        residual = ts - decomp.seasonal - np.median(ts)
    outliers = esd(residual, max_anomalies=max_anomalies, alpha=alpha)
    return outliers


def esd(timeseries, max_anomalies=100, alpha=0.5,):
    ts = np.copy(np.array(timeseries))
    test_statistics = []
    total_anomalies = -1
    for curr in range(max_anomalies):
        test_idx, test_val = calculate_test_statistic(ts)
        critical_value = calculate_critical_value(ts, alpha)
        if test_val > critical_value:
            total_anomalies = curr
        test_statistics.append(test_idx)
        ts = np.delete(ts, test_idx)
    anomalous_indices = test_statistics[:total_anomalies + 1]
    return anomalous_indices

def LOFDetection(clm_select, all_tss, df_data,plot=False):
    if plot:
        plt.figure()
    lof_pred = {}
    for i in range(len(clm_select)):
        col = clm_select[i]
        j = 1
        lof_pred[col] = []
        for kind in all_tss[col].keys():
            j += 1
            X = np.array(all_tss[col][kind])
            # LOF 算法
            clf = LocalOutlierFactor(n_neighbors=35,contamination=0.25)
            y_pred = clf.fit_predict(X)
            lof_pred[col].extend(y_pred)
        if plot:
            subplot = plt.subplot(len(clm_select), 1, i + 1)
            subplot.scatter(df_data['val'], df_data[col], c=lof_pred[col])
            subplot.set_title('Dimension ' + clm_select[i])
    if plot:
        plt.suptitle('Outlier detection With LOF')
        plt.show()
    return lof_pred

def SHESDDetection(clm_select, all_tss, df_data,plot=False):
    if plot:
        plt.figure()
    shesd_pred = {}
    for i in range(len(clm_select)):
        col = clm_select[i]
        j = 1
        # shesd_pred[col] = [1] * len()
        err_indices = []
        # 当前 簇的起始位置，随着kind而变化
        cur_index = 0
        for kind in all_tss[col].keys():
            j += 1
            tmp = np.array(all_tss[col][kind])
            X = np.array(tmp[:,1])
            # SHESD 算法
            anomaly_indices = seasonal_esd(X, hybrid=True, max_anomalies=200)
            anomaly_indices = [k + cur_index for k in anomaly_indices]
            err_indices.extend(anomaly_indices)
            cur_index += len(all_tss[col][kind])
        shesd_pred[col] = [1] * cur_index
        for err_indice in err_indices:
            shesd_pred[col][err_indice] = -1
        if plot:
            subplot = plt.subplot(len(clm_select), 1, i + 1)
            subplot.scatter(df_data['val'], df_data[col], c=shesd_pred[col])
            subplot.set_title('Dimension ' + clm_select[i])
    if plot:
        plt.suptitle('Outlier detection With Seasonal Hybrid ESD')
        plt.show()
    return shesd_pred

def SVMDetection(clm_select, all_tss, df_data,plot=False):
    rng = np.random.RandomState(42)
    outliers_fraction = 0.1
    if plot:
        plt.figure()
    svm_pred = {}
    for i in range(len(clm_select)):
        col = clm_select[i]
        j = 1
        svm_pred[col] = []
        for kind in all_tss[col].keys():
            j += 1
            X = np.array(all_tss[col][kind])
            # ONE-class SVM
            clf = svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,kernel="rbf", gamma=0.1)
            clf.fit(X)
            y_pred = clf.predict(X)
            svm_pred[col].extend(y_pred)
        if plot:
            subplot = plt.subplot(len(clm_select), 1, i + 1)
            subplot.scatter(df_data['val'], df_data[col], c=svm_pred[col])
            subplot.set_title('Dimension ' + clm_select[i])
    if plot:
        plt.suptitle('Outlier detection with one class SVM')
        plt.show()
    return svm_pred

def EllipticEnvelopeDetection(clm_select, all_tss, df_data,plot=False):
    rng = np.random.RandomState(42)
    outliers_fraction = 0.6
    if plot:
        plt.figure()
    ee_pred = {}
    for i in range(len(clm_select)):
        col = clm_select[i]
        j = 1
        ee_pred[col] = []
        for kind in all_tss[col].keys():
            j += 1
            X = np.array(all_tss[col][kind])
            # ONE-class SVM
            clf = EllipticEnvelope(contamination=outliers_fraction)
            clf.fit(X)
            y_pred = clf.predict(X)
            ee_pred[col].extend(y_pred)
        if plot:
            subplot = plt.subplot(len(clm_select), 1, i + 1)
            subplot.scatter(df_data['val'], df_data[col], c=ee_pred[col])
            subplot.set_title('Dimension ' + clm_select[i])
    if plot:
        plt.suptitle('Outlier detection with one class EllipticEnvelope')
        plt.show()
    return ee_pred

if __name__ == '__main__':
    # 读取数据为DataFrame格式
    # df_data = pd.read_csv("./smallwindmachine.csv", index_col=0).iloc[0:5000]
    df_data = pd.read_csv("./smallwindmachine.csv", index_col=0)
    df_val = DataFrame({'val':range(len(df_data))},index=df_data.index)
    df_data['val'] = df_val
    clm_select_num=[0,1,8]
    clm_select = [df_data.columns[num] for num in clm_select_num]
    print(clm_select)

    # 使用MINI_BATCH_K_means
    cluster_pred = mini_batch_k_means_cluster(df_data, clm_select,plot=False)
    # 使用dbscan
    # cluster_pred = dbscan_cluster(df_data, clm_select)

    # 所有时间序列的所有划分，格式为{col1:tss,col2:tss},tss为{kind1:ts1,kind2:ts2,...},ts为[index1,value1;index2,value2;...]
    all_tss= {}
    for i in range(len(cluster_pred)):
        col = clm_select[i]
        all_tss[col] = {}
        for j in range(len(df_data[col])):
            curKind = cluster_pred[i][j]
            if not (curKind in all_tss[col].keys()):
                all_tss[col][curKind] = [[df_data['val'][j], df_data[col][j]]]
            else:
                all_tss[col][curKind].append([df_data['val'][j],df_data[col][j]])

    # 运用三种方法检测异常

    # 孤立森林
    iforest_pred = isolateForestDetection(clm_select, all_tss, df_data,plot=True)

    # Seasonal Hybrid ESD
    shesd_pred = SHESDDetection(clm_select, all_tss, df_data,plot=True)

    # Local Outlier Factor
    lof_pred = LOFDetection(clm_select, all_tss, df_data,plot=True)

    # one-class svm
    svm_pred = SVMDetection(clm_select, all_tss, df_data,plot=True)

    # EllipticEnvelope
    ee_pred = EllipticEnvelopeDetection(clm_select, all_tss, df_data,plot=True)

    sum_pred = {}
    plt.figure()
    for i in range(len(clm_select)):
        col = clm_select[i]
        sum_pred[col] = []
        length = min(len(iforest_pred[col]), len(shesd_pred[col]), len(lof_pred[col]), len(svm_pred[col]), len(ee_pred[col]))
        for j in range(length):
            sum_pred[col].append(iforest_pred[col][j] + shesd_pred[col][j] + lof_pred[col][j] + svm_pred[col][j] + ee_pred[col][j])
        subplot = plt.subplot(len(clm_select), 1, i + 1)
        subplot.scatter(df_data['val'], df_data[col], c=sum_pred[col])
        subplot.set_title('Dimension ' + clm_select[i])
    plt.suptitle('Outlier detection after vote')
    plt.show()



    # 下面是对多元时间序列的相关性检验
    # # 计算每个时间序列的划分情况
    # for i in len(clm_select_num):
    #     cluster_detail={}
    #     clm = clm_select[i]
    #     # 当前clm划分的簇数目
    #     cnt=0
    #     for j in len(cluster_pred[i]):
    #         kind = cluster_pred[i][j]
    #         if not (kind in cluster_detail.keys()):
    #             # 第一个表示起始位置，第二个表示终结位置，起始不动，更新终结
    #             cnt = cnt + 1
    #             cluster_detail[clm][kind] = [j,j]
    #         else:
    #             cluster_detail[clm][kind][1] = j
    #     cluster_detail[clm]['cnt'] = cnt

    # 两段时间序列的起止范围差值不超过100的时候，可以进行协整检验
    # MAX_DIFF = 100
    # for clm1 in clm_select:
    #     for clm2 in clm_select:
    #         if (clm1 == clm2 or clm1['cnt']==clm2['cnt']):
    #             break
    #         else:
    #             for
    #
    #     dt = cluster_detail[clm]


    # 选取列
    # ts_a = df_data.iloc[:, clm_select_num[0]]
    # ts_b = df_data.iloc[:, clm_select_num[1]]
    # ts_c = df_data.iloc[:, clm_select_num[2]]
    #
    # print(len(ts_a))
    # index = [i for i in range(len(ts_a))]
    # ts_a_withindex = [index,ts_a.values]
    # ts_b_withindex = [index,ts_b]
    # ts_c_withindex = [index,ts_c]
    # X = np.array(ts_a_withindex,dtype=tuple)
    # print(len(X))
    # print(type(X))

    # 检验平稳性和相关性，单位根检验有些慢，相关性很快
    # validation(ts_a, ts_b, ts_c)




