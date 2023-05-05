#导入相关包
import numpy as np
import matplotlib.pyplot as plt
import random
import netCDF4
import datetime
# import seaborn as sns
from global_land_mask import globe
from scipy import interpolate
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler

#读取数据
data1 = netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/trans_data/u200.nc')
data2 = netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/trans_data/u850.nc')
data3 = netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/trans_data/olr.nc')

data_sst = netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/trans_data/sst.nc')

data_rmm = np.loadtxt('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/pca_RMM_2020_2023.txt')

data_u200 = np.array(data1.variables['u200'][:,:,:])
data_u850 = np.array(data2.variables['u850'][:,:,:])
data_olr = np.array(data3.variables['olr'][:,:,:])

data_sst = np.array(data_sst.variables['sst'][120:,:,:])

data_rmm = np.array(data_rmm[:,1:3])

print(data_u200.shape, data_u850.shape, data_olr.shape, data_rmm.shape)

#读取气候态
seasonal_cycle_u200 = np.load('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/seasonal_cycle/climatology_seasonal_cycle_u200_sample.npy')
seasonal_cycle_u850 = np.load('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/seasonal_cycle/climatology_seasonal_cycle_u850_sample.npy')
seasonal_cycle_olr = np.load('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/seasonal_cycle/climatology_seasonal_cycle_olr_sample.npy')

print(seasonal_cycle_u200.shape,seasonal_cycle_u850.shape,seasonal_cycle_olr.shape)

#减去气候态
data_u200 = data_u200 - seasonal_cycle_u200
data_u850 = data_u850 - seasonal_cycle_u850
data_olr = data_olr - seasonal_cycle_olr
data_olr = - data_olr

#减去120天平均值
def avg_120(test_data,time_length):
    for i in range(120,time_length):
        mean_120 = np.mean(test_data[i-120:i, :, :],axis=0)
        test_data[i,:,:] = test_data[i, :, :]-mean_120[:, :]
    final_data = np.array(test_data[120:time_length, :, :])
    return final_data

#时间长度
time_data = np.array(data1.variables['time'])
time_length = time_data.size
#执行减去120天平均的操作
reduec_120data_u200 = avg_120(data_u200, time_length)
reduec_120data_u850 = avg_120(data_u850, time_length)
reduec_120data_olr = avg_120(data_olr, time_length)

#数据标准化
def mean_std(data_xxx):
    data_mean = np.mean(data_xxx)
    data_std = np.std(data_xxx)
    final_data = data_xxx - data_mean
    final_data = np.true_divide(final_data, data_std)
    return final_data

fina_data_u200 = mean_std(reduec_120data_u200)
fina_data_u850 = mean_std(reduec_120data_u850)
fina_data_olr = mean_std(reduec_120data_olr)
fina_data_sst = mean_std(data_sst)

print(fina_data_u200.shape, fina_data_u850.shape, fina_data_olr.shape, fina_data_sst.shape)

#升维度，再合并
fina_data_u200 = np.expand_dims(fina_data_u200, axis=3)
fina_data_u850 = np.expand_dims(fina_data_u850, axis=3)
fina_data_olr = np.expand_dims(fina_data_olr, axis=3)
fina_data_sst = np.expand_dims(fina_data_sst, axis=3)
#print(data_u200.shape,data_u850.shape,data_olr.shape,data_sst.shape)
#合并数组（26178，13，144，4）
data_combine = np.concatenate((fina_data_olr, fina_data_u850, fina_data_u200, fina_data_sst), axis=3)
print(data_combine.shape)

re_X = []
re_Y = []
#滑窗构造数据集
for i in range(0, 1041, 7):
    re_X.append(data_combine[i:i+7, :, :, :])
    re_Y.append(data_rmm[i+7:i+42, :])
re_X = np.array(re_X)
re_Y = np.array(re_Y)
print(re_X.shape, re_Y.shape)

#以上是我们新下载数据集作为测试集，我们还可以把之前的验证集+测试集放进去
path = '/WdHeDisk/users/zhangnong/MJO/908_test(the best)/data/s2s_dataset_for7_7_35/'
wei = '_for7_7_35_sample.npy'

old_s2s_X = np.load(path + 'X_test' + wei)
old_s2s_Y = np.load(path + 'Y_test' + wei)

old_re_X = np.load(path + 'X_test' + wei)
old_re_Y = np.load(path + 're_Y_test' + wei)

valid_X = np.load(path + 'X_valid' + wei)
valid_Y = np.load(path + 'Y_valid' + wei)


print(old_s2s_X.shape, old_s2s_Y.shape)
print(old_re_X.shape, old_re_Y.shape)
print(valid_X.shape, valid_Y.shape)


#拼接，可以不管顺序
test_X = np.concatenate((re_X, old_s2s_X, old_re_X, valid_X, valid_X), axis = 0)
test_Y = np.concatenate((re_Y, old_s2s_Y, old_re_Y, valid_Y, valid_Y), axis = 0)

print(test_X.shape, test_Y.shape)

np.save('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/Test_Datasets/X_test_for7_7_35_sample.npy', test_X)
np.save('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/Test_Datasets/Y_test_for7_7_35_sample.npy', test_Y)