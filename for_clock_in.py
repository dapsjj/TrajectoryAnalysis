from cassandra import ConsistencyLevel
from cassandra.cluster import Cluster
# 引入DCAwareRoundRobinPolicy模块，可用来自定义驱动程序的行为
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

cluster = Cluster(contact_points=['192.168.10.117'],port=9042)
session = cluster.connect()


'''连接远程数据库'''
def testContectRemoteDatabase():
    # 配置Cassandra集群的IP，记得改成自己的远程数据库IP
    contact_points = ['192.168.10.117']
    # 配置登陆Cassandra集群的账号和密码，记得改成自己知道的账号和密码
    auth_provider = PlainTextAuthProvider(username='howezy', password='howezy')
    # 创建一个Cassandra的cluster
    cluster = Cluster(contact_points=contact_points, auth_provider=auth_provider)
    # 连接并创建一个会话
    session = cluster.connect()
    # 定义一条cql查询语句
    # cql_str = 'select * from howetech.clock_in limit 5;'
    cql_str = 'select * from howetech.clock_in ;' # clock_in表主要字段staff_id clock_time device_id latitude longitude
    simple_statement = SimpleStatement(cql_str, consistency_level=ConsistencyLevel.ONE,fetch_size=100000)
    # 对语句的执行设置超时时间为None
    execute_result = session.execute(simple_statement, timeout=None)
    # 获取执行结果中的原始数据
    result = execute_result._current_rows
    # 关闭连接
    cluster.shutdown()
    # 把结果转成DataFrame格式
    df = pd.DataFrame(result)
    df['clock_time_1'] = df['clock_time'].dt.strftime('%Y%m') #多了一列年月
    # 把查询结果写入csv
    df.to_csv(r'E:/test_opencv/轨迹分析/clock_in.csv', index=False, mode='w', header=True)
    # latitude_list = df.latitude.values.tolist()
    # longitude_list = df.longitude.values.tolist()
    gb = df.groupby(['staff_id', 'device_id','clock_time_1'])
    sub_dataframe_list = []
    for i in gb.indices:
        sub_df = pd.DataFrame(gb.get_group(i))
        sub_dataframe_list.append(sub_df)

    # m = Basemap(projection='robin', lon_0=0, resolution='l')
    # m = Basemap(llcrnrlon=73, llcrnrlat=18, urcrnrlon=135, urcrnrlat=53)

    for sub_dataframe in sub_dataframe_list:
        staff_id= sub_dataframe['staff_id'].iloc[0]
        device_id= sub_dataframe['device_id'].iloc[0]
        clock_time_1= sub_dataframe['clock_time_1'].iloc[0]
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
        plt.savefig(r'E:/test_opencv/轨迹分析/clock_in_image/'+str(staff_id)+'_'+str(device_id)+'_'+str(clock_time_1)+'.png')
        plt.close()


if __name__ == '__main__':
    testContectRemoteDatabase()
