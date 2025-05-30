# -*- coding: utf-8 -*-
import arcpy
from arcpy import env
from arcpy.sa import *

# 设置工作环境和环境变量
arcpy.env.workspace = "E:/workplace/1"
arcpy.env.parallelProcessingFactor = "0"  # 设置并行处理因子为0

# 指定裁剪用的Shapefile
clip_feature = "E:/workplace/1/sc.shp"

# 获取工作目录下所有的TIF文件
raster_list = arcpy.ListRasters("*", "TIF")

# 遍历所有TIF文件
for raster in raster_list:
    # 定义重采样后的输出路径
    resampled_output = "E:/workplace/1/Resampled_" + raster
    # 重采样为30米分辨率
    arcpy.Resample_management(raster, resampled_output, "0.0002694875", "NEAREST")

    # 定义裁剪后的输出路径
    clipped_output = "E:/workplace/1/Clipped_" + raster
    # 裁剪操作
    arcpy.Clip_management(resampled_output, "#", clipped_output, clip_feature, "0", "ClippingGeometry",
                          "NO_MAINTAIN_EXTENT")

    print("Processed and clipped: {}".format(raster))

print("Completed processing all TIF files.")
