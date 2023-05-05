#导入相关包
from re import X
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import random
import datetime
import netCDF4
import math
import seaborn as sns
from global_land_mask import globe
from scipy import interpolate
from eofs.standard import Eof
from sklearn.decomposition import PCA

#读取数据/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/trans_data/olr.nc
data1 = netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/trans_data/u200.nc')
data2 = netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/trans_data/u850.nc')
data3 = netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/trans_data/olr.nc')
data_u200 = np.array(data1.variables['u200'][:,:,:])
data_u850 = np.array(data2.variables['u850'][:,:,:])
data_olr = np.array(data3.variables['olr'][:,:,:])

print(data_olr.shape, data_u200.shape, data_u850.shape)

#读取气候态
seasonal_cycle_u200 = np.load('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/seasonal_cycle/climatology_seasonal_cycle_u200_sample.npy')
seasonal_cycle_u850 = np.load('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/seasonal_cycle/climatology_seasonal_cycle_u850_sample.npy')
seasonal_cycle_olr = np.load('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/seasonal_cycle/climatology_seasonal_cycle_olr_sample.npy')
#seasonal_cycle_olr=np.true_divide(seasonal_cycle_olr,3600)
print(seasonal_cycle_u200.shape,seasonal_cycle_u850.shape,seasonal_cycle_olr.shape)

#减去气候态
data_u200 = data_u200 - seasonal_cycle_u200
data_u850 = data_u850 - seasonal_cycle_u850
data_olr = data_olr - seasonal_cycle_olr
data_olr = - data_olr

#返回真实日期
def caluate_time(data_time):
    start_time = datetime.datetime(1900,1,1,0,0)
    fina_time = []
    for i in range(0, len(data_time)):
        transtime = start_time+datetime.timedelta(hours=int(data_time[i]))
        x = transtime.timetuple().tm_year*10000 + transtime.timetuple().tm_mon*100 + transtime.timetuple().tm_mday
        #print(x)
        fina_time.append(x)
    final_time = np.array(fina_time)
    return final_time

#减去120天平均值
def avg_120(test_data, time_length):
    new_data = np.zeros(test_data[120:time_length,:,:].shape)
    for i in range(120,time_length):
        mean_120 = np.mean(test_data[i-120:i,:,:],axis=0)
        new_data[i-120,:,:]=test_data[i,:,:]-mean_120[:,:]
    final_data = np.array(new_data)
    return final_data

def PCs(data_olr, data_u850, data_u200, lamda_vector):
    #先拼接一下(61,144)
    data_combine=np.concatenate((data_olr, data_u850, data_u200), axis = 1)
    #print(data_combine.shape)
    rmm = np.dot(data_combine, lamda_vector)
    eof1 = 51.37508737 
    eof2 = 48.82820937
    lamda1 = math.sqrt(eof1)
    lamda2 = math.sqrt(eof2)
    rmm1 = np.true_divide(rmm[:,0], lamda1)
    rmm2 = np.true_divide(rmm[:,1], lamda2)
    RMM = np.stack((rmm1, rmm2),axis = 1)
    #print(RMM.shape)
    #print(RMM[1,:])
    return RMM

#时间长度
time_data = np.array(data1.variables['time'])
time_length = time_data.size
print(time_length)

#执行减去120天平均的操作
reduec_120data_u200 = avg_120(data_u200, time_length)
reduec_120data_u850 = avg_120(data_u850, time_length)
reduec_120data_olr = avg_120(data_olr, time_length)

#经向平均
mean_u200 = np.mean(reduec_120data_u200, axis=1)
mean_u850 = np.mean(reduec_120data_u850, axis=1)
mean_olr = np.mean(reduec_120data_olr, axis=1)
print(mean_olr.shape,mean_u200.shape,mean_u850.shape)

#除以标准差 u200=5.3536506 u850=1.9444847 olr=10.035512
olr_std = 16.255560461342792
u200_std = 5.437523725114618
u850_std = 1.9760929603145574

fina_olr = np.true_divide(mean_olr,olr_std)
fina_u200 = np.true_divide(mean_u200,u200_std)
fina_u850 = np.true_divide(mean_u850,u850_std)

#eof，经验正交分解，使用再分析数据的特征值和特征向量 eof1=46.385155 eof2=45.028522
lamda_vector = np.loadtxt('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/pca_lamda_vectors.txt')
print(lamda_vector.shape)
RMM = PCs(fina_olr, fina_u850, fina_u200, lamda_vector)
print(RMM.shape)

#数组中插入第一列，时间，变换一下
ttime_data = np.array(time_data[120:time_length])
true_data = caluate_time(ttime_data)
#true_data = true_data.reshape(true_data.shape[0], 1)
true_data = true_data.reshape(-1, 1)
print(true_data.shape)
#true_data=true_data.T
print(true_data[0:209])
true_data = true_data.astype(np.float64)

fina_RMM = np.concatenate((true_data, RMM),axis=1)
#fina_RMM=np.insert(RMM,0,values=true_data,axis=1)
#fina_RMM=np.c_(true_data,RMM)
print(fina_RMM.shape)
#print(fina_RMM[0:209,0])
np.savetxt('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/pca_RMM_2020_2023.txt', fina_RMM, fmt="%d %.8f %.8f")






