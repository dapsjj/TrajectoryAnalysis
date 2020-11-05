import os
import logging
import numpy as np
from sklearn.cluster import DBSCAN
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
from cassandra import ConsistencyLevel
from cassandra.cluster import Cluster
# 引入DCAwareRoundRobinPolicy模块，可用来自定义驱动程序的行为
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement
import shutil
from branca.element import Template, MacroElement

template = """
{% macro html(this, kwargs) %}

<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title></title>
  <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

  <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
  <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>

  <script>
  $( function() {
    $( "#maplegend" ).draggable({
                    start: function (event, ui) {
                        $(this).css({
                            right: "auto",
                            top: "auto",
                            bottom: "auto"
                        });
                    }
                });
    });

  </script>
</head>
<body>


<div id='maplegend' class='maplegend' 
    style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
     border-radius:6px; padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>

<div class='legend-title'>图例</div>
<div class='legend-scale'>
  <ul class='legend-labels'>
    <li><span style='background:#FF0000;opacity:0.7;'></span>1月</li>
    <li><span style='background:#00FF00;opacity:0.7;'></span>2月</li>
    <li><span style='background:#0000FF;opacity:0.7;'></span>3月</li>
    <li><span style='background:#FFFF00;opacity:0.7;'></span>4月</li>
    <li><span style='background:#800080;opacity:0.7;'></span>5月</li>
    <li><span style='background:#FFC0CB;opacity:0.7;'></span>6月</li>
    <li><span style='background:#00FFFF;opacity:0.7;'></span>7月</li>
    <li><span style='background:#A52A2A;opacity:0.7;'></span>8月</li>
    <li><span style='background:#FFA500;opacity:0.7;'></span>9月</li>
    <li><span style='background:#DC143C;opacity:0.7;'></span>10月</li>
    <li><span style='background:#228B22;opacity:0.7;'></span>11月</li>
    <li><span style='background:#708090;opacity:0.7;'></span>12月</li>
  </ul>
</div>
</div>

</body>
</html>

<style type='text/css'>
  .maplegend .legend-title {
    text-align: left;
    margin-bottom: 5px;
    font-weight: bold;
    font-size: 90%;
    }
  .maplegend .legend-scale ul {
    margin: 0;
    margin-bottom: 5px;
    padding: 0;
    float: left;
    list-style: none;
    }
  .maplegend .legend-scale ul li {
    font-size: 80%;
    list-style: none;
    margin-left: 0;
    line-height: 18px;
    margin-bottom: 2px;
    }
  .maplegend ul.legend-labels li span {
    display: block;
    float: left;
    height: 16px;
    width: 30px;
    margin-right: 5px;
    margin-left: 0;
    border: 1px solid #999;
    }
  .maplegend .legend-source {
    font-size: 80%;
    color: #777;
    clear: both;
    }
  .maplegend a {
    color: #777;
    }
</style>
{% endmacro %}"""



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


def read_dateConfig_file_set_cassandra():
    '''读dateConfig.ini,设置数据库信息'''
    if os.path.exists(os.path.join(os.path.dirname(__file__), "dateConfig.ini")):
        try:
            conf = configparser.ConfigParser()
            conf.read(os.path.join(os.path.dirname(__file__), "dateConfig.ini"), encoding="utf-8-sig")
            cassandra_ip = conf.get("cassandra_ip", "cassandra_ip")
            cassandra_username = conf.get("cassandra_username", "cassandra_username")
            cassandra_password = conf.get("cassandra_password", "cassandra_password")
            return cassandra_ip,cassandra_username,cassandra_password
        except Exception as ex:
            logger.error("Content in dateConfig.ini about cassandra has error.")
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


def getClockInDataFromCassandra(para_ip,para_username,para_password):
    '''从cassandra获取clock_in的数据，存入csv'''
    try:
        save_path = r'E:/test_opencv/员工打卡分析/clock_in.csv'
        # 配置Cassandra集群的IP，记得改成自己的远程数据库IP
        contact_points = [para_ip]
        # 配置登陆Cassandra集群的账号和密码，记得改成自己知道的账号和密码
        auth_provider = PlainTextAuthProvider(username=para_username, password=para_password)
        # 创建一个Cassandra的cluster
        cluster = Cluster(contact_points=contact_points, auth_provider=auth_provider)
        # 连接并创建一个会话
        session = cluster.connect()
        # 定义一条cql查询语句
        # cql_str = 'select * from howetech.clock_in limit 5;'
        cql_str = 'select * from howetech.clock_in ;' # clock_in表主要字段staff_id clock_time device_id latitude longitude
        simple_statement = SimpleStatement(cql_str, consistency_level=ConsistencyLevel.ONE,fetch_size=1000000)#不加fetch_size默认智能获取5000行
        # 对语句的执行设置超时时间为None
        execute_result = session.execute(simple_statement, timeout=None)
        # 获取执行结果中的原始数据
        result = execute_result._current_rows
        # 关闭连接
        cluster.shutdown()
        # 把结果转成DataFrame格式
        df = pd.DataFrame(result)
        df['clock_time_add_8hour'] = df['clock_time'] + pd.Timedelta(hours=8)
        df['clock_time_year_month'] = df['clock_time_add_8hour'].dt.strftime('%Y%m')  # 多了一列年月
        df.to_csv(save_path, index=False, mode='w', header=True)
    except Exception as ex:
        logger.error("Call getClockInDataFromCassandra() has error.")
        logger.error("Exception:" + str(ex))
        raise ex


def get_data_from_marketer_info():
    '''从howetech.common_car_info获取全表数据'''
    try:
        sql = " select " \
              " t1.device_id, " \
              " t1.car_id, " \
              " t1.marketer_name " \
              " from  marketer_info as t1  "
        cur.execute(sql)
        rows = cur.fetchall()
        if rows:
            marketer_info_list = [list(row) for row in rows]
            return marketer_info_list
        else:
            return ""
    except pymysql.Error as ex:
        logger.error("dbException:" + str(ex))
        raise ex
    except Exception as ex:
        logger.error("Call method get_data_from_marketer_info() error!")
        logger.error("Exception:" + str(ex))
        raise ex

def save_marketer_info_list_to_csv(para_list):
    '''保存csv'''
    title = [['device_id', 'car_id', 'marketer_name']]
    if para_list:
        save_path = r'E:/test_opencv/员工打卡分析/marketer_info.csv'
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(title)
            writer.writerows(para_list)


def merge_clock_in_and_marketer_info():
    '''将clock_in的数据与maketer_info合并,出现在结果中的数据是要分析的数据'''
    df_clock_in = pd.read_csv(r'E:/test_opencv/员工打卡分析/clock_in.csv', encoding='utf-8')
    df_marketer_info = pd.read_csv(r'E:/test_opencv/员工打卡分析/marketer_info.csv', encoding='utf-8')
    merge_clock_in_and_marketer_info_csv = r'E:/test_opencv/员工打卡分析/merge_clock_in_and_marketer_info.csv'
    df_clock_in['device_id'] = df_clock_in['device_id'].astype(str)
    df_marketer_info['device_id'] = df_marketer_info['device_id'].astype(str)
    '''合并clock_in和marketer_info数据'''
    result = pd.merge(df_clock_in, df_marketer_info, how='inner',on='device_id')  # ['device_id', 'upload_time', 'latitude', 'longitude', 'mileage', 'other_vals', 'speed', 'upload_time_add_8hour', 'upload_time_year_month', 'car_id', 'car_type', 'car_num', 'marketer_name']
    result.to_csv(merge_clock_in_and_marketer_info_csv, index=False, mode='w', header=True, encoding='utf-8-sig')


def split_big_csv_to_small_csv():
    '''大csv拆分成小csv'''
    big_csv = r'E:/test_opencv/员工打卡分析/merge_clock_in_and_marketer_info.csv'
    device_csv_dir = r'E:/test_opencv/员工打卡分析/all_device_data_csv/'
    if not os.path.exists(device_csv_dir):
        os.makedirs(device_csv_dir)
    df = pd.read_csv(big_csv, encoding='utf-8', low_memory=False)
    gb = df.groupby(['device_id', 'clock_time_year_month'])
    sub_dataframe_list = []
    for i in gb.indices:
        sub_df = pd.DataFrame(gb.get_group(i))
        sub_dataframe_list.append(sub_df)

    for sub_dataframe in sub_dataframe_list:
        device_id = sub_dataframe['device_id'].iloc[0]#取组内第一个device_id用于存csv用
        marketer_name = sub_dataframe['marketer_name'].iloc[0]#取组内第一个marketer_name用于存csv用
        clock_time_year_month = sub_dataframe['clock_time_year_month'].iloc[0]#取组内第一个clock_time_year_month用于存csv用
        sub_dataframe = sub_dataframe.sort_values(by=['clock_time_add_8hour'])
        sub_dataframe.to_csv(device_csv_dir  + str(marketer_name) + '_' +str(clock_time_year_month) + '.csv',index=False, mode='w', header=True,encoding='utf-8')


def dbscan_get_center_coordinates_by_year_month():
    '''使用每个人每个月的打卡坐标，生成dbscan簇心'''
    device_csv_dir = r'E:/test_opencv/员工打卡分析/all_device_data_csv/'
    dbscan_get_center_coordinates_by_year_month_csv_dir = r'E:/test_opencv/员工打卡分析/dbscan_get_center_coordinates_by_year_month_csv/'
    if not os.path.exists(device_csv_dir):
        os.makedirs(device_csv_dir)
    if not os.path.exists(dbscan_get_center_coordinates_by_year_month_csv_dir):
        os.makedirs(dbscan_get_center_coordinates_by_year_month_csv_dir)
    for item in os.listdir(device_csv_dir):
        csvlName = device_csv_dir + item
        df = pd.read_csv(csvlName, encoding='utf-8', low_memory=False)
        length_df = len(df)
        # 计算dataframe经纬度中心坐标
        # longitude_center = df['longitude'].mean()
        # latitude_center = df['latitude'].mean()
        # X = df.drop_duplicates(subset=['longitude', 'latitude'])
        X = df[['latitude', 'longitude']]
        device_id = df['device_id'].iloc[0]  #取组内第一个device_id用于存csv用
        marketer_name = df['marketer_name'].iloc[0]  #取组内第一个smarketer_name用于存csv用
        clock_time_year_month = df['clock_time_year_month'].iloc[0]  #取组内第一个clock_time_year_month用于存csv用

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
        set_dbscan_labels = set(dbsc.labels_)
        if set_dbscan_labels:
            if -1 in set_dbscan_labels:  # -1为噪音
                set_dbscan_labels.discard(-1)  # 把-1从集合中删除
                del (cent_length[0])  # 删除cent_length中第一个元素，也就是-1
        if not set_dbscan_labels:  # 如果集合为空，说明没有可聚类对象集合中
            continue
        num_clusters = len(set_dbscan_labels)
        # turn the clusters into a pandas series,where each element is a cluster of points
        dbsc_clusters = pd.Series([X[fac_cluster_labels == n] for n in range(num_clusters)])
        # get centroid of each cluster
        fac_centroids = dbsc_clusters.map(get_centroid)
        # unzip the list of centroid points (lat, lon) tuples into separate lat and lon lists
        cent_lats, cent_lons = zip(*fac_centroids)
        # from these lats/lons create a new df of one representative point for eac cluster
        centroids_pd = pd.DataFrame({'longitude': cent_lons, 'latitude': cent_lats, 'length':cent_length})
        centroids_pd['device_id'] = device_id
        centroids_pd['marketer_name'] = marketer_name
        centroids_pd['clock_time_year_month'] = clock_time_year_month
        centroids_pd.to_csv(dbscan_get_center_coordinates_by_year_month_csv_dir  + str(marketer_name) + '_' + str(clock_time_year_month) + '.csv', index=False, mode='w', header=True,encoding='utf-8')


def draw_with_folium_year_month_points_and_dbscan_center_circle_style():
    '''先使用聚类中心画簇心，再使用每人每个月的打卡点作成maker坐标，中心点是圆'''
    all_device_csv_dir = r'E:/test_opencv/员工打卡分析/all_device_data_csv/' #所有车辆坐标csv
    dbscan_center_coordinates_csv_dir = r'E:/test_opencv/员工打卡分析/dbscan_get_center_coordinates_by_year_month_csv/' #中心点csv
    folium_year_month_points_and_dbscan_center_circle_html_dir = r'E:/test_opencv/员工打卡分析/folium_year_month_points_and_dbscan_center_circle_html/'
    if not os.path.exists(all_device_csv_dir):
        os.makedirs(all_device_csv_dir)
    if not os.path.exists(dbscan_center_coordinates_csv_dir):
        os.makedirs(dbscan_center_coordinates_csv_dir)
    if not os.path.exists(folium_year_month_points_and_dbscan_center_circle_html_dir):
        os.makedirs(folium_year_month_points_and_dbscan_center_circle_html_dir)

    for item in os.listdir(dbscan_center_coordinates_csv_dir):
        '''处理dbscan聚类后中心点坐标'''
        dbscan_center_coordinates_csv_name = dbscan_center_coordinates_csv_dir + item
        df = pd.read_csv(dbscan_center_coordinates_csv_name, encoding='utf-8', low_memory=False)
        length_df = len(df)
        # 计算dataframe经纬度中心坐标
        longitude_center = df['longitude'].mean()
        latitude_center = df['latitude'].mean()
        # X = df.drop_duplicates(subset=['longitude', 'latitude'])
        X = df
        # device_id = X['device_id'].iloc[0]  # 取组内第一个device_id用于存csv用
        # marketer_name = X['marketer_name'].iloc[0]  # 取组内第一个marketer_name用于存csv用
        # clock_time_year_month = X['clock_time_year_month'].iloc[0]  # 取组内第一个clock_time_year_month用于存csv用
        m = folium.Map(location=[latitude_center, longitude_center], zoom_start=10, control_scale=True)
        for index, row in X.iterrows():
            element_count_in_this_cluster = int(row['length'])
            popup = folium.Popup('该中心点周围共有'+str(element_count_in_this_cluster)+'打卡点', show=True, max_width=400)#show=True代表地图加载时显示popup
            folium.Circle(location=[row['latitude'], row['longitude']], radius=500, popup=popup,color='red', fill=True,fill_opacity=0.1).add_to(m)  # radius单位是米 #与dbscan半径对应
            # folium.Marker(location=[row['latitude'], row['longitude']], popup=popup, icon=folium.Icon(color='red')).add_to(m) #红色标记


        '''处理所有坐标'''
        all_device_csv_name = all_device_csv_dir + item
        df = pd.read_csv(all_device_csv_name, encoding='utf-8', low_memory=False)
        length_df = len(df)
        # 计算dataframe经纬度中心坐标
        # X = df.drop_duplicates(subset=['longitude', 'latitude'])
        X = df
        device_id = X['device_id'].iloc[0]  # 取组内第一个device_id用于存csv用
        marketer_name = X['marketer_name'].iloc[0]  # 取组内第一个marketer_name用于存csv用
        clock_time_year_month = X['clock_time_year_month'].iloc[0]  # 取组内第一个clock_time_year_month用于存csv用
        # m = folium.Map(location=[latitude_center, longitude_center],zoom_start=12,control_scale=True)
        for index, row in X.iterrows():
            # folium.Circle(location=[row['remain_latitude'], row['remain_longitude']],radius=100,color='red',fill=False).add_to(m) #radius单位是米
            folium.Marker(location=[row['latitude'], row['longitude']]).add_to(m) #radius单位是米

        url = folium_year_month_points_and_dbscan_center_circle_html_dir + str(marketer_name) + '_' + str(clock_time_year_month) + '.html'
        m.save(url)
        # save_screen_shot(url) #html截图


def draw_with_folium_year_month_points_and_dbscan_center_maker_style():
    '''先使用聚类中心画簇心，再使用每人每个月的打卡点作成maker坐标，中心点是maker样式'''
    all_device_csv_dir = r'E:/test_opencv/员工打卡分析/all_device_data_csv/' #所有车辆坐标csv
    dbscan_center_coordinates_csv_dir = r'E:/test_opencv/员工打卡分析/dbscan_get_center_coordinates_by_year_month_csv/' #中心点csv
    folium_year_month_points_and_dbscan_center_maker_html_dir = r'E:/test_opencv/员工打卡分析/folium_year_month_points_and_dbscan_center_maker_html/'
    if not os.path.exists(all_device_csv_dir):
        os.makedirs(all_device_csv_dir)
    if not os.path.exists(dbscan_center_coordinates_csv_dir):
        os.makedirs(dbscan_center_coordinates_csv_dir)
    if not os.path.exists(folium_year_month_points_and_dbscan_center_maker_html_dir):
        os.makedirs(folium_year_month_points_and_dbscan_center_maker_html_dir)

    for item in os.listdir(dbscan_center_coordinates_csv_dir):
        '''处理dbscan聚类后中心点坐标'''
        dbscan_center_coordinates_csv_name = dbscan_center_coordinates_csv_dir + item
        df = pd.read_csv(dbscan_center_coordinates_csv_name, encoding='utf-8', low_memory=False)
        length_df = len(df)
        # 计算dataframe经纬度中心坐标
        longitude_center = df['longitude'].mean()
        latitude_center = df['latitude'].mean()
        # X = df.drop_duplicates(subset=['longitude', 'latitude'])
        X = df
        # device_id = X['device_id'].iloc[0]  # 取组内第一个device_id用于存csv用
        # marketer_name = X['marketer_name'].iloc[0]  # 取组内第一个marketer_name用于存csv用
        # clock_time_year_month = X['clock_time_year_month'].iloc[0]  # 取组内第一个clock_time_year_month用于存csv用
        m = folium.Map(location=[latitude_center, longitude_center], zoom_start=10, control_scale=True)
        for index, row in X.iterrows():
            element_count_in_this_cluster = int(row['length'])
            popup = folium.Popup('该中心点周围共有'+str(element_count_in_this_cluster)+'打卡点', show=True, max_width=400)#show=True代表地图加载时显示popup
            # folium.Circle(location=[row['latitude'], row['longitude']], radius=500, popup=popup,color='red', fill=True,fill_opacity=0.1).add_to(m)  # radius单位是米 #与dbscan半径对应
            folium.Marker(location=[row['latitude'], row['longitude']], popup=popup, icon=folium.Icon(color='red')).add_to(m) #红色标记


        '''处理所有坐标'''
        all_device_csv_name = all_device_csv_dir + item
        df = pd.read_csv(all_device_csv_name, encoding='utf-8', low_memory=False)
        length_df = len(df)
        # 计算dataframe经纬度中心坐标
        # X = df.drop_duplicates(subset=['longitude', 'latitude'])
        X = df
        device_id = X['device_id'].iloc[0]  # 取组内第一个device_id用于存csv用
        marketer_name = X['marketer_name'].iloc[0]  # 取组内第一个marketer_name用于存csv用
        clock_time_year_month = X['clock_time_year_month'].iloc[0]  # 取组内第一个clock_time_year_month用于存csv用
        # m = folium.Map(location=[latitude_center, longitude_center],zoom_start=12,control_scale=True)
        for index, row in X.iterrows():
            # folium.Circle(location=[row['remain_latitude'], row['remain_longitude']],radius=100,color='red',fill=False).add_to(m) #radius单位是米
            folium.Marker(location=[row['latitude'], row['longitude']]).add_to(m) #radius单位是米

        url = folium_year_month_points_and_dbscan_center_maker_html_dir + str(marketer_name) + '_' + str(clock_time_year_month) + '.html'
        m.save(url)
        # save_screen_shot(url) #html截图


def save_device_center_by_year():
    '''按照年把每人的数据放到指定文件夹'''
    device_center_save_by_year_csv_dir = r'E:/test_opencv/员工打卡分析/device_center_save_by_year/'
    dbscan_get_device_center_coordinates_by_year_month_csv_dir = r'E:/test_opencv/员工打卡分析/dbscan_get_center_coordinates_by_year_month_csv/'  # 年月中心点csv
    if not os.path.exists(device_center_save_by_year_csv_dir):
        os.makedirs(device_center_save_by_year_csv_dir)

    for item in os.listdir(dbscan_get_device_center_coordinates_by_year_month_csv_dir):
        csvlName = dbscan_get_device_center_coordinates_by_year_month_csv_dir + item
        year = item.split('.')[0].split('_')[-1][:4]
        person_name = item.split('.')[0].split('_')[0]
        year_person_name = os.path.join(year,person_name)
        target_dir = device_center_save_by_year_csv_dir + year_person_name
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        shutil.copy2(csvlName, target_dir)  # 把人的csv复制到人对应的年份的文件夹


def draw_with_folium_by_device_and_year_dbscan_center_circle_style():
    '''使用每人每个月的聚类中心坐标按照1-12月不同颜色去画圆
    每年、每人生成一个html,html中12种颜色代表12个月
    '''
    #12种颜色分别是红、绿、蓝、黄、紫、粉、青、棕、橙、赤红、森林绿、板岩灰
    twelve_months_color_list = ['#FF0000','#00FF00','#0000FF','#FFFF00','#800080','#FFC0CB','#00FFFF','#A52A2A','#FFA500','#DC143C','#228B22','#708090']
    device_center_save_by_year_csv_dir = r'E:/test_opencv/员工打卡分析/device_center_save_by_year/'
    folium_device_center_save_by_year_circle_html_dir = r'E:/test_opencv/员工打卡分析/folium_device_center_save_by_year_circle_html/'
    if not os.path.exists(folium_device_center_save_by_year_circle_html_dir):
        os.makedirs(folium_device_center_save_by_year_circle_html_dir)
    for root, dirs, files in os.walk(device_center_save_by_year_csv_dir):
        longitude_center_list = []
        latitude_center_list = []
        for file in files:
            if os.path.isfile(os.path.join(root, file)):
                dbscan_center_coordinates_csv_name = os.path.join(root, file)
                '''处理dbscan聚类后中心点坐标'''
                df = pd.read_csv(dbscan_center_coordinates_csv_name, encoding='utf-8', low_memory=False)
                # 计算dataframe经纬度中心坐标
                longitude_center = df['longitude'].mean()
                latitude_center = df['latitude'].mean()
                longitude_center_list.append(longitude_center)
                latitude_center_list.append(latitude_center)

        if longitude_center_list and latitude_center_list:
            longitude_cent = np.mean(longitude_center_list)#年度中心
            latitude_cent = np.mean(latitude_center_list)#年度中心
        else:
            continue

        m = folium.Map(location=[latitude_cent, longitude_cent], zoom_start=10, control_scale=True)
        macro = MacroElement()
        macro._template = Template(template)
        m.get_root().add_child(macro)
        for file in files:
            if os.path.isfile(os.path.join(root, file)):
                dbscan_center_coordinates_csv_name = os.path.join(root, file)
                '''处理dbscan聚类后中心点坐标'''
                df = pd.read_csv(dbscan_center_coordinates_csv_name, encoding='utf-8', low_memory=False)
                X = df
                device_id = X['device_id'].iloc[0]  # 取组内第一个device_id用于存csv用
                marketer_name = X['marketer_name'].iloc[0]  # 取组内第一个marketer_name用于存csv用
                clock_time_year_month = str(X['clock_time_year_month'].iloc[0])  # 取组内第一个clock_time_year_month用于存csv用
                year_month = clock_time_year_month[0:4]+'年'+clock_time_year_month[4:6]+'月'
                month = int(clock_time_year_month[4:6])
                month_index  = month -1
                month_color = twelve_months_color_list[month_index]
                for index, row in X.iterrows():
                    element_count_in_this_cluster = int(row['length'])
                    popup = folium.Popup(year_month, show=True,max_width=400)  # show=True代表地图加载时显示
                    folium.Circle(location=[row['latitude'], row['longitude']], radius=500, popup=popup, color=month_color,fill=True, fill_opacity=0.1).add_to(m)  # radius单位是米 #与dbscan半径对应
                    # folium.Marker(location=[row['latitude'], row['longitude']], popup=popup, icon=folium.Icon(color='red')).add_to(m) #红色标记
        url = folium_device_center_save_by_year_circle_html_dir + str(marketer_name) + '_' + str(clock_time_year_month[0:4]) + '.html'
        m.save(url)


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
    device_csv_dir = r'E:/test_opencv/员工打卡分析/all_device_data_csv/'
    dbscan_get_center_coordinates_csv_dir = r'E:/test_opencv/员工打卡分析/dbscan_get_center_coordinates_csv/'
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
        set_dbscan_labels = set(dbsc.labels_)
        if set_dbscan_labels:
            if -1 in set_dbscan_labels:  # -1为噪音
                set_dbscan_labels.discard(-1)  # 把-1从集合中删除
                del (cent_length[0])  # 删除cent_length中第一个元素，也就是-1
        if not set_dbscan_labels:  # 如果集合为空，说明没有可聚类对象集合中
            continue
        num_clusters = len(set_dbscan_labels)
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
        centroids_pd.to_csv(dbscan_get_center_coordinates_csv_dir  + str(staff_name) + '_' + str(car_num) + '_' + str(create_time_1) + '.csv', index=False, mode='w', header=True,encoding='utf-8')



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
    logger = write_log()  # 获取日志对象
    time_start = datetime.now()
    start = time.time()
    logger.info("Program starts,now time is:" + str(time_start))
    server, user, password, database = read_dateConfig_file_set_database()  # 读取配置文件中的数据库信息
    cassandra_ip,cassandra_username,cassandra_password = read_dateConfig_file_set_cassandra()
    year = read_dateConfig_file_set_year() #读取配置文件中的year,查询year对应的年份数据
    # getConn()  # 数据库连接对象
    # getClockInDataFromCassandra(cassandra_ip,cassandra_username,cassandra_password)
    # marketer_info_list = get_data_from_marketer_info()
    # closeConn() #关闭数据库连接
    # save_marketer_info_list_to_csv(marketer_info_list)
    # merge_clock_in_and_marketer_info()
    # split_big_csv_to_small_csv()
    # dbscan_get_center_coordinates_by_year_month()  # 使用所有打卡轨迹，按照年月生成聚类中心坐标csv文件
    # draw_with_folium_year_month_points_and_dbscan_center_circle_style()#中心点是圆
    # draw_with_folium_year_month_points_and_dbscan_center_maker_style()#中心点是maker
    # save_device_center_by_year()
    draw_with_folium_by_device_and_year_dbscan_center_circle_style()
    time_end = datetime.now()
    end = time.time()
    logger.info("Program ends,now time is:" + str(time_end))
    logger.info("Program ran for : %f seconds" % (end - start))


