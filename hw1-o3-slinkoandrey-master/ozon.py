from scipy.io import netcdf
import matplotlib.pyplot as plt
import numpy as np
import json

sochi_lat = 43
sochi_lon = 39
with netcdf.netcdf_file('MSR-2.nc', mmap=False) as f:
    variables = f.variables
file_lat = np.searchsorted(variables['latitude'].data, sochi_lat) 
file_lon = np.searchsorted(variables['longitude'].data, sochi_lon) 
data_all = variables['Average_O3_std'][:, file_lat, file_lon][:]
data_jan = variables['Average_O3_std'][:, file_lat, file_lon][::12] 
data_jul = variables['Average_O3_std'][:, file_lat, file_lon][6::12] 
time_all = variables['time'][:]
time_jan = variables['time'][::12] 
time_jul = variables['time'][6::12] 
plt.plot(time_all, data_all, label = 'Все время') 
plt.plot(time_jan, data_jan, label = 'Январь') 
plt.plot(time_jul, data_jul, label = 'Июль') 
plt.legend() 
plt.grid() 
plt.savefig('ozon.png')
j = {
  "city": "Sochi",
  "coordinates": [sochi_lat, sochi_lon],
  "jan": {
    "min": float(np.min(data_jan)),
    "max": float(np.max(data_jan)),
    "mean": float(np.mean(data_jan))
  },
  "jul": {
    "min": float(np.min(data_jul)),
    "max": float(np.max(data_jul)),
    "mean": float(np.mean(data_jul))
  },
  "all": {
    "min": float(np.min(data_all)),
    "max": float(np.max(data_all)),
    "mean": float(np.mean(data_all))
  }
}

with open('ozon.json', 'w') as f:
    json.dump(j, f)
    
with open('ozon.json', 'r') as f:
    print(f.read())