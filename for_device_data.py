import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import shutil
import os


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
        sub_dataframe.to_csv(r'E:/test_opencv/轨迹分析/device_data_sub_dataframe/'+str(device_id)+'_'+str(upload_time_1)+'.csv', index=False, mode='w', header=True)
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
        plt.savefig(r'E:/test_opencv/轨迹分析/device_data_image/'+str(device_id)+'_'+str(upload_time_1)+'.png')
        plt.close()
        i += 1
    '''

    #把图片复制到对应的文件夹
    imageDir = r'E:/test_opencv/轨迹分析/device_data_image/'
    csvDir = r'E:/test_opencv/轨迹分析/device_data_sub_dataframe/'
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

if __name__ == '__main__':
    testContectRemoteDatabase()
