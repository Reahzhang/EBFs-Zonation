import rasterio
import pandas as pd

def read_tif_in_chunks(tif_path, chunk_size):
    """
    分块读取tif文件中的数据。

    参数:
    tif_path -- tif文件的路径。
    chunk_size -- 每个数据块的大小。

    返回:
    生成器，每次迭代返回一个数据块。
    """
    with rasterio.open(tif_path) as src:
        for ji, window in src.block_windows(1):
            data = src.read(1, window=window)
            yield data.ravel()  # 展平并返回每个块的数据



def read_tif_in_chunks(tif_path, chunk_size):
    with rasterio.open(tif_path) as src:
        for i in range(0, src.width, chunk_size):
            for j in range(0, src.height, chunk_size):
                window = rasterio.windows.Window(i, j, min(chunk_size, src.width - i), min(chunk_size, src.height - j))
                data = src.read(1, window=window)
                yield data.ravel()



def save_chunks_to_csv(tif_path, output_prefix, chunk_size):
    print(f"处理文件: {tif_path}")  # 打印正在处理的文件
    for i, block in enumerate(read_tif_in_chunks(tif_path, chunk_size)):
        print(f"保存块 {i} 至 {output_prefix}_chunk_{i}.csv")  # 打印每个块的信息
        if block.size > 0:  # 检查块是否有数据
            df = pd.DataFrame(block, columns=[output_prefix])
            df.to_csv(f'{output_prefix}_chunk_{i}.csv', index=False)
        else:
            print("警告: 块没有数据")

# 然后再次运行您的脚本

# 示例路径和参数
chunk_size =10240  # 您可以根据需要调整这个大小
feature_paths = [
    'E:/workplace/7/Spre_min.tif',
    'E:/workplace/7/Sevp_min.tif',
    'E:/workplace/7/Wssd_max.tif',
    'E:/workplace/7/DEM.tif',
    'E:/workplace/7/slope.tif',
    'E:/workplace/7/aspect.tif',
    'E:/workplace/7/slst.tif',
    'E:/workplace/7/Texture_30.tif',
    'E:/workplace/7/PH_30.tif',
    'E:/workplace/7/class_30.tif',
    'E:/workplace/7/Water30.tif']

label_path = 'E:/workplace/7/2020_Humid_RF4.tif'  # 标签文件路径

# 保存每个特征的块为CSV
for i, path in enumerate(feature_paths):
    save_chunks_to_csv(path, f'Feature_{i+1}', chunk_size)

# 保存标签块为CSV
save_chunks_to_csv(label_path, 'Label', chunk_size)
