# -*- coding: utf-8 -*-
import arcpy
import os
from arcpy.sa import Con
from arcpy.sa import IsNull
# 检查空间分析许可是否可用
if arcpy.CheckExtension("Spatial") == "Available":
    arcpy.CheckOutExtension("Spatial")
else:
    raise RuntimeError("Spatial Analyst license is not available.")

# 设置工作环境，即包含TIF文件的文件夹路径
arcpy.env.workspace = "E:/Data/Humid/高密度/HEBF"
# 设置输出文件夹
output_folder = ur'E:/Data/Humid/高密度/HEBF'

# 检查并创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 列出文件夹中所有的TIF文件
tif_files = arcpy.ListRasters("*", "TIF")

for tif in tif_files:
    # 获取栅格属性
    raster = arcpy.Raster(tif)
    pixel_type = raster.pixelType

    # 检查像素类型，如果不是浮点型，则转换为浮点型
    if pixel_type not in ('F32', 'F64'):  # F32是32位浮点型，F64是64位浮点型
        # 使用CopyRaster工具转换像素类型
        float_raster_path = os.path.join(output_folder, "float_{}".format(tif))
        arcpy.CopyRaster_management(tif, float_raster_path, pixel_type='32_BIT_FLOAT')
        raster = arcpy.Raster(float_raster_path)
        # 使用IsNull函数检测NoData值，并将NoData和-9999的值替换为0
        raster = Con(IsNull(raster) | (raster == -9999), 0, raster)
    # 执行归一化
    min_val = raster.minimum
    max_val = raster.maximum
    normalized_raster = ((raster - min_val) / (max_val - min_val)) * 100  # 乘以100是为了后面保留两位小数

    # 保存归一化后的结果，保留两位小数
    normalized_raster_path = os.path.join(output_folder, "normalized_{}".format(tif))
    normalized_raster = arcpy.sa.Int(normalized_raster + 0.5) / 100.0  # 加0.5再取整是为了四舍五入
    normalized_raster.save(normalized_raster_path)

print("处理完成")

# 归还空间分析许可
arcpy.CheckInExtension("Spatial")
