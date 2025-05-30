from osgeo import gdal
from osgeo import osr

def resample_and_clip(input_rasters, target_raster):
    # 打开目标栅格影像
    target_ds = gdal.Open(target_raster, gdal.GA_ReadOnly)

    # 获取目标栅格的投影和仿射变换信息
    target_proj = target_ds.GetProjection()
    target_geotransform = target_ds.GetGeoTransform()

    # 获取目标栅格的空间分辨率（像素大小）
    target_pixel_width = target_geotransform[1]
    target_pixel_height = target_geotransform[5]

    # 循环处理输入栅格影像
    for input_raster in input_rasters:
        # 打开输入栅格影像
        input_ds = gdal.Open(input_raster, gdal.GA_ReadOnly)

        # 获取输入栅格的投影和仿射变换信息
        input_proj = input_ds.GetProjection()
        input_geotransform = input_ds.GetGeoTransform()

        output_resampled = gdal.Warp("output_resampled.tif", input_ds,
                                     xRes=target_pixel_width, yRes=target_pixel_height,
                                     dstSRS=target_proj, outputBounds=target_geotransform)

        # 关闭输入数据集
        input_ds = None



input_rasters = ["E:/workplace/7/water.tif", "E:/workplace/7/PH.tif", "E:/workplace/7/texture.tif"]  # 输入栅格影像列表
target_raster = "E:/workplace/7/DEM.tif"  # 目标栅格影像
resample_and_clip(input_rasters, target_raster)

