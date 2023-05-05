#导入相关包
from calendar import c
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
#读取数据
data1=netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/new_DATA/u200_1950-2022.nc')
data2=netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/new_DATA/u850_1950-2022.nc')
data3=netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/new_DATA/olr_1950-2022.nc')
data_u200=np.array(data1.variables['u200'][:,:,:])
data_u850=np.array(data2.variables['u850'][:,:,:])
data_olr=np.array(data3.variables['olr'][:,:,:])

#计算91～21年每天30年的平均
add=np.array([365,366,365,365,365,366,365,365,365,366,365,365,365,366,365,365,365,366,365,365,365,366,365,365,
             365,366,365,365,365,366])
another_add=np.array([366,365,365,365,366,365,365,365,366,365,365,365,366,365,365,365,366,365,365,365,366,365,365,365,
             366,365,365,365,366,365])
#计算每天的30年平均
def climatology_seasonal_cycle(data): 
    new_data=[]
    for i in range(0,365):
        time=14975+i
        #print("the day:",i+1)
        m_data=[]
        for j in range(0,30):
            if i<59:
                m_data.append(data[time,:,:])
                time=time+add[j]
            else:
                m_data.append(data[time,:,:])
                time=time+another_add[j]
        m_data=np.array(m_data)
        #print("30_data shape:",m_data.shape)
        mean_data=np.mean(m_data,axis=0)
        #print("mean data shape:",mean_data.shape)
        new_data.append(mean_data)
    new_data=np.array(new_data)
    return new_data

seasonal_cycle_u200=climatology_seasonal_cycle(data_u200)
seasonal_cycle_u850=climatology_seasonal_cycle(data_u850)
seasonal_cycle_olr=climatology_seasonal_cycle(data_olr)
print(seasonal_cycle_u200.shape,seasonal_cycle_u850.shape,seasonal_cycle_olr.shape)
#闰年多一天，这一天就8年平均，因为有8个闰年
run_time=np.array([15399 ,16860,18321 ,19782 ,21243 ,22704 ,24165,25626])
def run_season(data):
    x=[]
    for i in (run_time):
        x.append(data[i,:,:])
    x=np.array(x)
    #print(x.shape)
    mean_data=np.mean(x,axis=0)
    #print(mean_data.shape)
    return mean_data
run_season_u200=run_season(data_u200)
run_season_u850=run_season(data_u850)
run_season_olr=run_season(data_olr)

#闰年就多加一天，（366，13，144）
run_seasonal_cycle_u200=np.insert(seasonal_cycle_u200,59,run_season_u200,axis=0)
run_seasonal_cycle_u850=np.insert(seasonal_cycle_u850,59,run_season_u850,axis=0)
run_seasonal_cycle_olr=np.insert(seasonal_cycle_olr,59,run_season_olr,axis=0)
print(run_seasonal_cycle_u200.shape)
print(run_seasonal_cycle_u850.shape)
print(run_seasonal_cycle_olr.shape)
'''
#看是否插入成功
print(seasonal_cycle_u200[58:60,:,:])
print("-------------------------")
print(run_seasonal_cycle_u200[58:61,:,:])
'''
#拼接，闰年用366的数组，其他用365数组，形成26329的样式
#闰年代号
def concet_data(data1,data2):
    run_number=np.array([2  ,6 ,10 ,14 ,18 ,22 ,26 ,30 ,34 ,38 ,42 ,46 ,50 ,54 ,58 ,62 ,66 ,70])
    x=data1
    for i in range(1,72):
        if i in run_number:
            x=np.concatenate((x,data2),axis=0)
        else:
            x=np.concatenate((x,data1),axis=0)
    connect_data=np.concatenate((x,data1[0:31,:,:]),axis=0)   
    print(connect_data.shape)
    return connect_data

fina_u200=concet_data(seasonal_cycle_u200,run_seasonal_cycle_u200)
fina_u850=concet_data(seasonal_cycle_u850,run_seasonal_cycle_u850)
fina_olr=concet_data(seasonal_cycle_olr,run_seasonal_cycle_olr)
#保存
np.save('/WdHeDisk/users/zhangnong/MJO/new_DATA/climatology_seasonal_cycle_u200_sample.npy', fina_u200)
np.save('/WdHeDisk/users/zhangnong/MJO/new_DATA/climatology_seasonal_cycle_u850_sample.npy', fina_u850)
np.save('/WdHeDisk/users/zhangnong/MJO/new_DATA/climatology_seasonal_cycle_olr_sample.npy', fina_olr)


