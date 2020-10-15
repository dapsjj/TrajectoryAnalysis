import os
import logging
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
from pyecharts.charts import Geo
from pyecharts.globals import GeoType
from pyecharts import options
from datetime import datetime
import pymysql
import time
import configparser
import csv
import folium
from selenium import webdriver


now_date = datetime.now().strftime('%Y%m%d')
conn = None  # 连接
cur = None  # 游标


def write_log():
    '''获取日志对象'''
    logger = logging.getLogger()
    log_file = now_date + ".log"  # 文件日志
    if not os.path.exists(os.path.join(os.path.dirname(__file__)) + os.sep + 'log'):
        os.makedirs(os.path.join(os.path.dirname(__file__)) + os.sep + 'log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s line:%(lineno)s %(message)s')
    file_handler = logging.FileHandler(os.path.join(os.path.dirname(__file__)) + os.sep + 'log' + os.sep + log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger

def getConn():
    '''声明数据库连接对象'''
    global conn
    global cur
    try:
        conn = pymysql.connect(server, user, password, database)
        cur = conn.cursor()
    except pymysql.Error as ex:
        logger.error("dbException:" + str(ex))
        raise ex
    except Exception as ex:
        logger.error("Call method getConn() error!")
        raise ex


def closeConn():
    '''关闭数据库连接对象'''
    global conn
    global cur
    try:
        cur.close()
        conn.close()
    except pymysql.Error as ex:
        logger.error("dbException:" + str(ex))
        raise ex
    except Exception as ex:
        logger.error("Call method closeConn() error!")
        raise ex
    finally:
        pass


def read_dateConfig_file_set_database():
    '''读dateConfig.ini,设置数据库信息'''
    if os.path.exists(os.path.join(os.path.dirname(__file__), "dateConfig.ini")):
        try:
            conf = configparser.ConfigParser()
            conf.read(os.path.join(os.path.dirname(__file__), "dateConfig.ini"), encoding="utf-8-sig")
            server = conf.get("server", "server")
            user = conf.get("user", "user")
            password = conf.get("password", "password")
            database = conf.get("database", "database")
            return server,user,password,database
        except Exception as ex:
            logger.error("Content in dateConfig.ini about database has error.")
            logger.error("Exception:" + str(ex))
            raise ex
    else:
        logger.error("DateConfig.ini doesn't exist!")


def read_dateConfig_file_set_year():
    '''读dateConfig.ini,获取年份参数'''
    if os.path.exists(os.path.join(os.path.dirname(__file__), "dateConfig.ini")):
        try:
            conf = configparser.ConfigParser()
            conf.read(os.path.join(os.path.dirname(__file__), "dateConfig.ini"), encoding="utf-8-sig")
            year = conf.get("year", "year")
            return year
        except Exception as ex:
            logger.error("Content in dateConfig.ini has error.")
            logger.error("Exception:" + str(ex))
            raise ex
    else:
        logger.error("DateConfig.ini doesn't exist!")


def get_data_from_remain_statistic(para_year):
    '''从howetech.remain_statistic获取指定年份的数据'''
    try:
        sql = " select " \
              " remain_id, " \
              " device_id, " \
              " staff_id, " \
              " staff_name, " \
              " car_id, " \
              " car_num, " \
              " office_id, " \
              " office_name, " \
              " start_time, " \
              " end_time, " \
              " remain_longitude, " \
              " remain_latitude, " \
              " create_time, " \
              " update_time  from remain_statistic where YEAR(create_time) = %s or YEAR(update_time) = %s" \
              % (para_year, para_year)
        cur.execute(sql)
        rows = cur.fetchall()
        if rows:
            remain_statistic_list = [list(row) for row in rows]
            return remain_statistic_list
        else:
            return ""
    except pymysql.Error as ex:
        logger.error("dbException:" + str(ex))
        raise ex
    except Exception as ex:
        logger.error("Call method get_data_from_remain_statistic() error!")
        logger.error("Exception:" + str(ex))
        raise ex


def list_to_csv(para_save_list):
    '''保存csv'''
    title = [['remain_id', 'device_id', 'staff_id','staff_name','car_id','car_num','office_id','office_name','start_time','end_time','remain_longitude','remain_latitude','create_time','update_time']]
    if para_save_list:
        with open(r'E:/test_opencv/车辆经常停留位置/remain_statistic.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(title)
            writer.writerows(para_save_list)


def draw_with_echarts_scatter():
    '''使用pyecharts画散点图'''
    device_csv_dir = r'E:/test_opencv/车辆经常停留位置/all_device_data_csv/'
    pyecharts_device_html_dir = r'E:/test_opencv/车辆经常停留位置/pyecharts_device_html/'
    if not os.path.exists(device_csv_dir):
        os.makedirs(device_csv_dir)
    if not os.path.exists(pyecharts_device_html_dir):
        os.makedirs(pyecharts_device_html_dir)
    for item in os.listdir(device_csv_dir):
        csvlName = device_csv_dir + item
        df = pd.read_csv(csvlName, encoding='utf-8', low_memory=False)
        # X = df.drop_duplicates(subset=['longitude', 'latitude'])
        X = df
        device_id = X['device_id'].iloc[0]  # 取组内第一个device_id用于存csv用
        staff_id = X['staff_id'].iloc[0]  # 取组内第一个staff_id用于存csv用
        staff_name = X['staff_name'].iloc[0]  # 取组内第一个staff_name用于存csv用
        car_id = X['car_id'].iloc[0]  # 取组内第一个car_id用于存csv用
        car_num = X['car_num'].iloc[0]  # 取组内第一个car_num用于存csv用
        create_time_1 = X['create_time_1'].iloc[0]  # 取组内第一个create_time_1用于存csv用
        g = Geo()
        g.add_schema(maptype="china")
        # 给所有点附上标签 'create_time'
        for index, row in X.iterrows():
            g.add_coordinate(row['create_time'], row['remain_longitude'], row['remain_latitude'])
        create_time = X.create_time.values.tolist()
        # 给每个点的值赋为 1
        data_list = [[item, 1] for item in create_time]
        # 画图
        # g.add('', data_list, type_=GeoType.HEATMAP, symbol_size=2)
        g.add('', data_list, type_=GeoType.EFFECT_SCATTER, symbol_size=2)
        g.set_series_opts(label_opts=options.LabelOpts(is_show=False))
        #heatmap
        # g.set_global_opts(visualmap_opts=options.VisualMapOpts(), title_opts=options.TitleOpts(title=staff_name+"_" + str(car_num) + '_' + str(create_time_1) +'_heatmap'  ,pos_left='50%',pos_top='20'))
        #scatter
        g.set_global_opts(title_opts=options.TitleOpts(title=staff_name+"_" + str(car_num) + '_' + str(create_time_1) +'_scatter'  ,pos_left='50%',pos_top='20'))
        # 保存结果到 html
        result = g.render(pyecharts_device_html_dir +  str(staff_name) + '_' +str(car_num) + '_' + str(create_time_1) + '.html')


def draw_with_folium_all_points_and_dbscan_center():
    '''先使用maker聚类中心生成簇心，再使用每辆车每个月的停留点作成maker坐标，再'''
    all_device_csv_dir = r'E:/test_opencv/车辆经常停留位置/all_device_data_csv/' #所有车辆坐标csv
    dbscan_center_coordinates_csv_dir = r'E:/test_opencv/车辆经常停留位置/dbscan_get_center_coordinates_csv/' #中心点csv
    folium_all_points_and_dbscan_center_html_dir = r'E:/test_opencv/车辆经常停留位置/folium_all_points_and_dbscan_center_html/'
    if not os.path.exists(all_device_csv_dir):
        os.makedirs(all_device_csv_dir)
    if not os.path.exists(dbscan_center_coordinates_csv_dir):
        os.makedirs(dbscan_center_coordinates_csv_dir)
    if not os.path.exists(folium_all_points_and_dbscan_center_html_dir):
        os.makedirs(folium_all_points_and_dbscan_center_html_dir)

    for item in os.listdir(all_device_csv_dir):
        '''处理dbscan聚类后中心点坐标'''
        dbscan_center_coordinates_csv_name = dbscan_center_coordinates_csv_dir + item
        df = pd.read_csv(dbscan_center_coordinates_csv_name, encoding='utf-8', low_memory=False)
        length_df = len(df)
        # 计算dataframe经纬度中心坐标
        longitude_center = df['longitude'].mean()
        latitude_center = df['latitude'].mean()
        # X = df.drop_duplicates(subset=['longitude', 'latitude'])
        X = df
        device_id = X['device_id'].iloc[0]  # 取组内第一个device_id用于存csv用
        staff_id = X['staff_id'].iloc[0]  # 取组内第一个staff_id用于存csv用
        staff_name = X['staff_name'].iloc[0]  # 取组内第一个staff_name用于存csv用
        car_id = X['car_id'].iloc[0]  # 取组内第一个car_id用于存csv用
        car_num = X['car_num'].iloc[0]  # 取组内第一个car_num用于存csv用
        year_month = X['year_month'].iloc[0]  # 取组内第一个year_month用于存csv用
        m = folium.Map(location=[latitude_center, longitude_center], zoom_start=10, control_scale=True)
        for index, row in X.iterrows():
            element_count_in_this_cluster = int(row['length'])
            popup = folium.Popup('该中心点周围共有'+str(element_count_in_this_cluster)+'个停留点', show=True, max_width=400)#show=True代表地图加载是显示簇心周围有几个maker
            folium.Circle(location=[row['latitude'], row['longitude']], radius=500, popup=popup,color='red', fill=True,fill_opacity=0.1).add_to(m)  # radius单位是米 #与dbscan半径对应



        '''处理所有坐标'''
        all_device_csv_name = all_device_csv_dir + item
        df = pd.read_csv(all_device_csv_name, encoding='utf-8', low_memory=False)
        length_df = len(df)
        # 计算dataframe经纬度中心坐标
        longitude_center = df['remain_longitude'].mean()
        latitude_center = df['remain_latitude'].mean()
        # X = df.drop_duplicates(subset=['longitude', 'latitude'])
        X = df
        device_id = X['device_id'].iloc[0]  # 取组内第一个device_id用于存csv用
        staff_id = X['staff_id'].iloc[0]  # 取组内第一个staff_id用于存csv用
        staff_name = X['staff_name'].iloc[0]  # 取组内第一个staff_name用于存csv用
        car_id = X['car_id'].iloc[0]  # 取组内第一个car_id用于存csv用
        car_num = X['car_num'].iloc[0]  # 取组内第一个car_num用于存csv用
        create_time_1 = X['create_time_1'].iloc[0]  # 取组内第一个create_time_1用于存csv用
        # m = folium.Map(location=[latitude_center, longitude_center],zoom_start=12,control_scale=True)
        for index, row in X.iterrows():
            # folium.Circle(location=[row['remain_latitude'], row['remain_longitude']],radius=100,color='red',fill=False).add_to(m) #radius单位是米
            folium.Marker(location=[row['remain_latitude'], row['remain_longitude']]).add_to(m) #radius单位是米

        url = folium_all_points_and_dbscan_center_html_dir + str(staff_name) + '_' + str(car_num) + '_' + str(year_month) + '.html'
        m.save(url)
        # save_screen_shot(url) #html截图


def save_screen_shot(para_url):
    '''打开浏览器并且截图'''
    browser = webdriver.Chrome(r"E:/chromedriver_win32/chromedriver.exe")
    # browser.set_window_size(4000, 3000)  # choose a resolution
    browser.get(para_url)
    # You may need to add time.sleep(seconds) here
    time.sleep(5)
    image_name = para_url.split('.')[0]+'.png'
    browser.save_screenshot(image_name)
    browser.close()


def dbscan_get_center_coordinates():
    '''使用每辆车每个月的停留坐标，生成dbscan簇心'''
    device_csv_dir = r'E:/test_opencv/车辆经常停留位置/all_device_data_csv/'
    dbscan_get_center_coordinates_csv_dir = r'E:/test_opencv/车辆经常停留位置/dbscan_get_center_coordinates_csv/'
    if not os.path.exists(device_csv_dir):
        os.makedirs(device_csv_dir)
    if not os.path.exists(dbscan_get_center_coordinates_csv_dir):
        os.makedirs(dbscan_get_center_coordinates_csv_dir)
    for item in os.listdir(device_csv_dir):
        csvlName = device_csv_dir + item
        df = pd.read_csv(csvlName, encoding='utf-8', low_memory=False)
        length_df = len(df)
        # 计算dataframe经纬度中心坐标
        # longitude_center = df['remain_longitude'].mean()
        # latitude_center = df['remain_latitude'].mean()
        # X = df.drop_duplicates(subset=['longitude', 'latitude'])
        X = df[['remain_latitude', 'remain_longitude']]
        device_id = df['device_id'].iloc[0]  # 取组内第一个device_id用于存csv用
        staff_id = df['staff_id'].iloc[0]  # 取组内第一个staff_id用于存csv用
        staff_name = df['staff_name'].iloc[0]  # 取组内第一个staff_name用于存csv用
        car_id = df['car_id'].iloc[0]  # 取组内第一个car_id用于存csv用
        car_num = df['car_num'].iloc[0]  # 取组内第一个car_num用于存csv用
        create_time_1 = df['create_time_1'].iloc[0]  # 取组内第一个create_time_1用于存csv用

        # convert eps to radians for use by haversine
        kms_per_rad = 6371.0088  # mean radius of the earth
        # epsilon = 1.5 / kms_per_rad  # The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately for your data set and distance function. default=0.5
        epsilon = 0.5 / kms_per_rad  # The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately for your data set and distance function. default=0.5
        dbsc = (DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(X)))
        # dbsc = (DBSCAN(eps=epsilon, min_samples=1,n_jobs=1).fit(np.radians(X)))
        fac_cluster_labels = dbsc.labels_
        values, counts = np.unique(fac_cluster_labels, return_counts=True) #获取聚类簇的索引和每个簇对应元素数量
        # a= {k: v for k, v in zip(values, counts)}
        cent_length = counts.tolist()  # 每个簇中元素的长度
        # get the number of clusters
        num_clusters = len(set(dbsc.labels_))
        # turn the clusters into a pandas series,where each element is a cluster of points
        dbsc_clusters = pd.Series([X[fac_cluster_labels == n] for n in range(num_clusters)])
        # get centroid of each cluster
        fac_centroids = dbsc_clusters.map(get_centroid)
        # unzip the list of centroid points (lat, lon) tuples into separate lat and lon lists
        cent_lats, cent_lons = zip(*fac_centroids)
        # from these lats/lons create a new df of one representative point for eac cluster
        centroids_pd = pd.DataFrame({'longitude': cent_lons, 'latitude': cent_lats, 'length':cent_length})
        centroids_pd['device_id'] = device_id
        centroids_pd['staff_id'] = staff_id
        centroids_pd['staff_name'] = staff_name
        centroids_pd['car_id'] = car_id
        centroids_pd['car_num'] = car_num
        centroids_pd['year_month'] = create_time_1
        centroids_pd.to_csv(dbscan_get_center_coordinates_csv_dir  + str(staff_name) + '_' + str(car_num) + '_' + str(create_time_1) + '.csv', index=False, mode='w', header=True)



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

def split_big_csv_to_small_csv():
    '''大csv拆分成小csv'''
    big_csv = r'E:/test_opencv/车辆经常停留位置/remain_statistic.csv'
    device_csv_dir = r'E:/test_opencv/车辆经常停留位置/all_device_data_csv/'
    if not os.path.exists(device_csv_dir):
        os.makedirs(device_csv_dir)
    df = pd.read_csv(big_csv, encoding='utf-8', parse_dates=[12], low_memory=False)
    df['create_time_1'] = df['create_time'].dt.strftime('%Y%m')  # 多了一列年月
    gb = df.groupby(['device_id', 'staff_id', 'car_id', 'create_time_1'])
    sub_dataframe_list = []
    for i in gb.indices:
        sub_df = pd.DataFrame(gb.get_group(i))
        sub_dataframe_list.append(sub_df)
    length_sub_dataframe_list = len(sub_dataframe_list)
    print('子dataframe数组长度:' + str(length_sub_dataframe_list))


    for sub_dataframe in sub_dataframe_list:
        device_id = sub_dataframe['device_id'].iloc[0]#取组内第一个device_id用于存csv用
        staff_id = sub_dataframe['staff_id'].iloc[0]#取组内第一个staff_id用于存csv用
        staff_name = sub_dataframe['staff_name'].iloc[0]#取组内第一个staff_name用于存csv用
        car_id = sub_dataframe['car_id'].iloc[0]#取组内第一个car_id用于存csv用
        car_num = sub_dataframe['car_num'].iloc[0]#取组内第一个car_num用于存csv用
        create_time_1 = sub_dataframe['create_time_1'].iloc[0]#取组内第一个create_time_1用于存csv用
        sub_dataframe = sub_dataframe.sort_values(by=['create_time'])
        sub_dataframe.to_csv(device_csv_dir  + str(staff_name) + '_' +str(car_num) + '_' + str(create_time_1) + '.csv',index=False, mode='w', header=True)



if __name__ == '__main__':
    logger = write_log()  # 获取日志对象
    time_start = datetime.now()
    start = time.time()
    logger.info("Program starts,now time is:" + str(time_start))
    server, user, password, database = read_dateConfig_file_set_database()  # 读取配置文件中的数据库信息
    year = read_dateConfig_file_set_year() #读取配置文件中的year,查询year对应的年份数据
    getConn()  # 数据库连接对象
    remain_statistic_list = get_data_from_remain_statistic(year)
    closeConn() #关闭数据库连接
    list_to_csv(remain_statistic_list)
    split_big_csv_to_small_csv()
    draw_with_echarts_scatter() #使用pyecharts画散点图
    dbscan_get_center_coordinates()#使用所有停留轨迹生成聚类中心坐标csv文件
    draw_with_folium_all_points_and_dbscan_center()
    time_end = datetime.now()
    end = time.time()
    logger.info("Program ends,now time is:" + str(time_end))
    logger.info("Program ran for : %f seconds" % (end - start))


