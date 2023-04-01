import os
import numpy as np
import h5py
from numpy import genfromtxt

root ='./data/TinT/'
ori_filename = os.path.join(root, 'speed.h5')
distance_filename = os.path.join(root, 'distance.csv')
la_filename = os.path.join(root, 'Sensor_LA.csv')
sf_filename = os.path.join(root, 'Sensor_SF.csv')
sd_filename = os.path.join(root, 'Sensor_SD.csv')
la_save_dist_filename = os.path.join(root, 'distance_LA.csv')
sf_save_dist_filename = os.path.join(root, 'distance_SF.csv')
sd_save_dist_filename = os.path.join(root, 'distance_SD.csv')
la_save_senserid_filename = os.path.join(root, 'Sensor_lookup_LA.csv')
sf_save_senserid_filename = os.path.join(root, 'Sensor_lookup_SF.csv')
sd_save_senserid_filename = os.path.join(root, 'Sensor_lookup_SD.csv')
sd_save_data_filename = os.path.join(root, 'sd')
sf_save_data_filename = os.path.join(root, 'sf')
la_save_data_filename = os.path.join(root, 'la')

city_list = [sd_filename, la_filename, sf_filename]
city_save_dist_list = [sd_save_dist_filename, la_save_dist_filename, sf_save_dist_filename]
city_save_lookup_list = [sd_save_senserid_filename,la_save_senserid_filename, sf_save_senserid_filename]
city_save_data_list = [sd_save_data_filename, la_save_data_filename, sf_save_data_filename]

f = h5py.File(ori_filename,'r')
df_data = f['df']
sensor_id = np.asarray(df_data['axis0']).astype(int)
dist_f = open(distance_filename, 'r')
dist_data = genfromtxt(dist_f, delimiter=',')[1:, :]
all_data = np.expand_dims(np.asarray(df_data['block0_values']), axis=2)[:28800, :, :]

for idx in range(len(city_list)):
    print(city_list[idx])
    city_sensorid = genfromtxt(city_list[idx], delimiter=',')[1:, 0]
    city_index = np.where(np.in1d(sensor_id, city_sensorid))[0]
    dist_index = np.asarray([x in city_sensorid for x in dist_data[:, 0]]) & np.asarray(
        [x in city_sensorid for x in dist_data[:, 1]])
    city_dist = dist_data[dist_index]
    city_lookup_table = dict(zip(city_sensorid, range(len(city_sensorid))))
    for i in range(len(city_dist)):
        city_dist[i][0] = int(city_lookup_table[city_dist[i][0]])
        city_dist[i][1] = int(city_lookup_table[city_dist[i][1]])
    np.savetxt(city_save_dist_list[idx], city_dist, delimiter=',', header=str('from,to,cost'))
    np.savetxt(city_save_lookup_list[idx], city_sensorid, delimiter='')

    city_data = all_data[:, city_index, :]
    np.savez(city_save_data_list[idx], city_data)
