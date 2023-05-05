#导入相关包
import numpy as np
import matplotlib.pyplot as plt
import random
import netCDF4
import seaborn as sns
from global_land_mask import globe
from scipy import interpolate

#导入数据(1204, 2, 721, 1440)
data_olr_1950_1978 = netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/original_data/remap/rchazhi_u850.nc')
print(data_olr_1950_1978)


#构建新数据
da = netCDF4.Dataset('/WdHeDisk/users/zhangnong/MJO/2023_0423_Test_DataSets/trans_data/u850.nc','w', format='NETCDF4')
da.createDimension('latitude', 13)  # 创建坐标点
da.createDimension('longitude', 144)
da.createDimension('time',1204)

#设置变量
lat_var = da.createVariable("latitude",'f4',('latitude') )  #添加coordinates  'f'为数据类型，不可或缺
lat_var.units = 'degrees_north'
lon_var = da.createVariable("longitude",'f4',('longitude'))  #添加coordinates  'f'为数据类型，不可或缺
lon_var.units = 'degrees_east'
time_var = da.createVariable("time",'i4',('time'))
time_var.units = 'hours since 1900-01-01 00:00:00.0'

#填充数据
lat = np.array(data_olr_1950_1978.variables['lat'])
lon = np.array(data_olr_1950_1978.variables['lon'])
time = np.array(data_olr_1950_1978.variables['time'])

da.variables['latitude'][:] = lat[:]   #填充数据
da.variables['longitude'][:] = lon[:]   #填充数据
da.variables['time'][:] = time[:]

u850 = da.createVariable('u850','f',('time','latitude','longitude'))
u850.units = 'm s**-1'

#单位换算
u_data = np.array(data_olr_1950_1978['u'][:, 0, :, :])
#u_data = np.true_divide(u_data, 3600)
#缺失值
sum1 = np.sum(u_data == -32767)
print("none value:",str(sum1))

u_data[u_data==-32767] = 0

sum2=np.sum(u_data == -32767)
print("none value:",str(sum2))

print(u_data.shape)

da.variables['u850'][:,:,:] = u_data[:, :, :]
da.close()