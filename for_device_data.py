import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import shutil

import os
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd

def testContectRemoteDatabase():
    # cd /usr/local/cassandra/bin
    # ./cqlsh
    # USE howetech;
    # COPY howetech.device_data  TO '/usr/local/cassandra/device_data_2020.scv';
    # df = pd.read_csv(r'E:/test_opencv/轨迹分析/device_data_20200924.csv', encoding='utf-8', parse_dates=[1], nrows=5)
    # df = pd.read_csv(r'E:/test_opencv/轨迹分析/device_data_20200924.csv', encoding='utf-8', parse_dates=[1],  names=['device_id','upload_time','latitude','longitude','mileage','other_vals','speed'])
    # df['upload_time_1'] = df['upload_time'].dt.strftime('%Y%m') #多了一列年月
    # df.to_csv(r'E:/test_opencv/轨迹分析/device_data.csv', index=False, mode='w', header=True)
    # latitude_list = df.latitude.values.tolist()
    # longitude_list = df.longitude.values.tolist()

    '''
    df = pd.read_csv(r'E:/test_opencv/轨迹分析/device_data.csv', encoding='utf-8',parse_dates=[1],low_memory=False)
    #device_id长度[11,14,15,16]
    gb = df.groupby(['device_id', 'upload_time_1'])
    sub_dataframe_list = []
    for i in gb.indices:
        sub_df = pd.DataFrame(gb.get_group(i))
        sub_dataframe_list.append(sub_df)
    length_sub_dataframe_list = len(sub_dataframe_list)
    print('子dataframe数组长度:'+str(length_sub_dataframe_list))
    i=1
    for sub_dataframe in sub_dataframe_list:
        device_id = sub_dataframe['device_id'].iloc[0]
        upload_time_1 = sub_dataframe['upload_time_1'].iloc[0]
        sub_dataframe = sub_dataframe.sort_values(by=['upload_time'])
        sub_dataframe.to_csv(r'E:/test_opencv/轨迹分析/all_device_data_csv/'+str(device_id)+'_'+str(upload_time_1)+'.csv', index=False, mode='w', header=True)
        print('第'+str(i)+'张图') #第几个sub_dataframe
        fig = plt.figure(figsize=(20, 10))
        m = Basemap(llcrnrlon=77, llcrnrlat=14, urcrnrlon=140, urcrnrlat=51, projection='lcc', lat_1=33, lat_2=45,lon_0=100)
        m.readshapefile(r'E:/test_opencv/gadm36_CHN_shp/gadm36_CHN_1', 'states', drawbounds=True)
        x = sub_dataframe['longitude'].tolist()
        y = sub_dataframe['latitude'].tolist()
        lats = y
        lons = x
        m.drawcountries(color='#ffffff', linewidth=0.5)
        m.fillcontinents(color='#c0c0c0', lake_color='#ffffff')
        x, y = m(lons, lats)
        plt.plot(x, y, 'bo', color='r', markersize=1)
        # plt.show()
        plt.savefig(r'E:/test_opencv/轨迹分析/all_device_data_image/'+str(device_id)+'_'+str(upload_time_1)+'.png')
        plt.close()
        i += 1
    '''

    '''
    #把图片和csv复制到对应的文件夹
    imageDir = r'E:/test_opencv/轨迹分析/all_device_data_image/'
    csvDir = r'E:/test_opencv/轨迹分析/all_device_data_csv/'
    person_img_dir = r'E:/test_opencv/轨迹分析/person_image'
    device_img_dir = r'E:/test_opencv/轨迹分析/device_image'
    person_csv_dir = r'E:/test_opencv/轨迹分析/person_csv'
    device_csv_dir = r'E:/test_opencv/轨迹分析/device_csv'

    device_list = []
    person_list = []
    for i in os.listdir(imageDir):
        name = i.split('_')[0]
        if len(name) == 11:
            person_list.append(i.split('.')[0])
        else: #长度14,15,16
            device_list.append(i.split('.')[0])

    for item in person_list:
        imageName = imageDir + str(item) + '.png'
        csvlName = csvDir + str(item) + '.csv'
        if os.path.isfile(imageName) and os.path.isfile(csvlName):
            shutil.copy2(imageName, person_img_dir)
            shutil.copy2(csvlName, person_csv_dir)

    for item in device_list:
        imageName = imageDir + str(item) + '.png'
        csvlName = csvDir + str(item) + '.csv'
        if os.path.isfile(imageName) and os.path.isfile(csvlName):
            shutil.copy2(imageName, device_img_dir)
            shutil.copy2(csvlName, device_csv_dir)
    '''

    # 查看csv中的异常经纬度
    device_csv_dir = r'E:/test_opencv/轨迹分析/device_csv/'
    device_image_dir = r'E:/test_opencv/轨迹分析/device_image/'
    normal_device_img_dir = r'E:/test_opencv/轨迹分析/normal_device_image'
    normal_device_csv_dir = r'E:/test_opencv/轨迹分析/normal_device_csv'
    abnormal_device_img_dir = r'E:/test_opencv/轨迹分析/abnormal_device_image'
    abnormal_device_csv_dir = r'E:/test_opencv/轨迹分析/abnormal_device_csv'
    normal_device_list = []
    abnormal_device_list = []
    for name in os.listdir(device_csv_dir):
        csv_name = device_csv_dir + os.sep + name
        print(csv_name)
        names = ['device_id', 'upload_time', 'latitude', 'longitude', 'mileage', 'other_vals', 'speed', 'upload_time_1']
        df = pd.read_csv(csv_name, encoding='utf-8', parse_dates=[1], low_memory=False)
        X = df[['latitude', 'longitude']]
        X = X.drop_duplicates()
        # X = df.iloc[:,2:4]
        # convert eps to radians for use by haversine
        kms_per_rad = 6371.0088
        epsilon = 1.5 / kms_per_rad
        # Extract intersection coordinates (latitude, longitude)
        dbsc = (DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(X)))
        # dbsc = (DBSCAN(eps=epsilon, min_samples=1,n_jobs=1).fit(np.radians(X)))
        fac_cluster_labels = dbsc.labels_
        # get the number of clusters
        num_clusters = len(set(dbsc.labels_))
        # turn the clusters into a pandas series,where each element is a cluster of points
        dbsc_clusters = pd.Series([X[fac_cluster_labels == n] for n in range(num_clusters)])
        # get centroid of each cluster
        fac_centroids = dbsc_clusters.map(get_centroid)
        # unzip the list of centroid points (lat, lon) tuples into separate lat and lon lists
        cent_lats, cent_lons = zip(*fac_centroids)
        # from these lats/lons create a new df of one representative point for eac cluster
        centroids_pd = pd.DataFrame({'longitude': cent_lons, 'latitude': cent_lats})
        # Plot the faciity clusters and cluster centroid
        fig, ax = plt.subplots(figsize=[20, 10])
        facility_scatter = ax.scatter(X['longitude'], X['latitude'], c=fac_cluster_labels,
                                       edgecolor='None', alpha=0.7, s=120)
        centroid_scatter = ax.scatter(centroids_pd['longitude'], centroids_pd['latitude'], marker='x', linewidths=2,
                                      c='k', s=50)
        ax.set_title('Facility Clusters & Facility Centroid', fontsize=30)
        ax.set_xlabel('Longitude', fontsize=24)
        ax.set_ylabel('Latitude', fontsize=24)
        ax.legend([facility_scatter, centroid_scatter], ['Facilities', 'Facility Cluster Centroid'], loc='upper right',
                  fontsize=20)
        plt.show()
        plt.close()


        # m1 = df[['latitude', 'longitude']].diff().abs().gt(0.1) #
        # m2 = df[['latitude', 'longitude']].shift().diff().abs().gt(0.1)
        # m = m1 | m2
        # latitude_diff_list = df.index[m['latitude']].tolist()
        # longitude_diff_list = df.index[m['longitude']].tolist()
        # if not latitude_diff_list and not longitude_diff_list: #如果经纬度的list为空，说明是正常数据
        #     normal_device_list.append(name.split('.')[0])
        # else:
        #     abnormal_device_list.append(name.split('.')[0])

    # for item in normal_device_list:
    #     imageName = device_image_dir + str(item) + '.png'
    #     csvlName = device_csv_dir + str(item) + '.csv'
    #     if os.path.isfile(imageName) and os.path.isfile(csvlName):
    #         shutil.copy2(imageName, normal_device_img_dir)
    #         shutil.copy2(csvlName, normal_device_csv_dir)

    # for item in abnormal_device_list:
    #     imageName = device_image_dir + str(item) + '.png'
    #     csvlName = device_csv_dir + str(item) + '.csv'
    #     if os.path.isfile(imageName) and os.path.isfile(csvlName):
    #         shutil.copy2(imageName, abnormal_device_img_dir)
    #         shutil.copy2(csvlName, abnormal_device_csv_dir)


def get_centroid(cluster):
    """calculate the centroid of a cluster of geographic coordinate points
    Args:
      cluster coordinates, nx2 array-like (array, list of lists, etc)
      n is the number of points(latitude, longitude)in the cluster.
    Return:
      geometry centroid of the cluster

    """
    cluster_ary = np.asarray(cluster)
    centroid = cluster_ary.mean(axis=0)
    return centroid



if __name__ == '__main__':
    testContectRemoteDatabase()
