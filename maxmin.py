import pandas as pd
import numpy as np

# 替换为您的Excel文件路径
file_path = 'E:/Data/Humid/高密度/Semi_高密度区.xlsx'

# 读取 Excel 文件
df = pd.read_excel(file_path)

# 将-9999替换为NaN
df.replace(-9999, pd.NA, inplace=True)

# 删除包含NaN的行
df.dropna(inplace=True)

# 保存到新的Excel文件
new_file_path = 'E:/Data/Humid/区域/偏湿性.xlsx'
df.to_excel(new_file_path, index=False)

# 选择除 'EBF' 列外的其他列进行归一化
columns_to_exclude = ['SEBF']
columns_to_normalize = df.columns[~df.columns.isin(columns_to_exclude)]
# 假设df是你的DataFrame

# 确保所有列都是数值类型，尤其是那些要进行四舍五入的列
for col in columns_to_normalize:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 再次删除包含NaN的行，以确保四舍五入操作不会因为NaN值而出错
df.dropna(inplace=True)
# 对这些列进行最大最小值归一化，并四舍五入保留四位小数
df[columns_to_normalize] = df[columns_to_normalize].apply(
    lambda x: np.round((x - x.min()) / (x.max() - x.min()), 4)
)

output_file_path = 'E:/Data/Humid/高密度/Normalized_Semi_高密度区.xlsx'

# 将归一化后的 DataFrame 保存为 Excel 文件，不包含索引
df.to_excel(output_file_path, index=False)

# 打印输出文件的路径和前几行数据进行检查
print("Normalized data (excluding 'EBF' column) saved to:", output_file_path)
# 如果需要查看DataFrame的前几行数据，可以取消此行的注释
# print(df.head())

