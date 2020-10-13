'''
#散点图
import pandas as pd
from pyecharts.charts import Geo
from pyecharts import options
from pyecharts.globals import GeoType

df = pd.read_csv(r'E:\test_opencv\轨迹分析\abnormal_device_csv\17195301070_202007.csv',encoding='utf-8')
X = df.drop_duplicates(subset=['longitude', 'latitude'])
g = Geo()
g.add_schema(maptype="china")
# 给所有点附上标签 'upload_time'
for index, row in X.iterrows():
    g.add_coordinate(row['upload_time'], row['longitude'], row['latitude'])
upload_time = X.upload_time.values.tolist()
# 给每个点的值赋为 1
data_list = [[item,1] for item in upload_time]
# 画图
g.add('',data_list, type_=GeoType.EFFECT_SCATTER, symbol_size=2)
g.set_series_opts(label_opts=options.LabelOpts(is_show=False))
g.set_global_opts(title_opts=options.TitleOpts(title="轨迹分布",pos_left='50%',pos_top='20'))
# 保存结果到 html
result = g.render(r'E:\test_opencv\轨迹分析\1.html')
'''

#热力图
import pandas as pd
from pyecharts.charts import Geo
from pyecharts.globals import GeoType
from pyecharts import options as opts

df = pd.read_csv(r'E:\test_opencv\轨迹分析\abnormal_device_csv\17201100972_202005.csv',encoding='utf-8')
X = df.drop_duplicates(subset=['longitude', 'latitude'])
g = Geo()
g.add_schema(maptype="china")
# 给所有点附上标签 'upload_time'
for index, row in X.iterrows():
    g.add_coordinate(row['upload_time'], row['longitude'], row['latitude'])
upload_time = X.upload_time.values.tolist()
# 给每个点的值赋为 1
data_list = [[item,1] for item in upload_time]
# 画图
g.add('',data_list, type_=GeoType.HEATMAP, symbol_size=2)
g.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
g.set_global_opts(visualmap_opts=opts.VisualMapOpts(),title_opts=opts.TitleOpts(title="Geo-HeatMap"))
# 保存结果到 html
result = g.render(r'E:\test_opencv\轨迹分析\1.html')
