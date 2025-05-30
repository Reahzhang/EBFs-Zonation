# -*- coding: utf-8 -*-
import arcpy
from arcpy import env
from arcpy.sa import *
# 设置工作环境和环境变量
arcpy.env.workspace = "E:/0-毕业论文数据/2020/气象数据/pre/resample"
arcpy.env.parallelProcessingFactor = "0"  # 设置并行处理因子为0

# 指定裁剪用的Shapefile
clip_feature = u"E:/workplace/1/sc.shp"

# 获取工作目录下所有的TIF文件
raster_list = arcpy.ListRasters("*", "TIF")

# 检查raster_list是否为None
if raster_list is None:
    print("No TIF files found. Check the workspace path.")
else:
    # 遍历所有TIF文件
    for raster in raster_list:
        # 定义裁剪后的输出路径
        clipped_output = u"E:/0-毕业论文数据/2020/气象数据/pre/clip/Clipped_" + raster
        # 裁剪操作
        arcpy.Clip_management(raster, "#", clipped_output, clip_feature, "0", "ClippingGeometry",
                              "NO_MAINTAIN_EXTENT")
        print("Processed and clipped: {}".format(raster))

    print("Completed processing all TIF files.")

