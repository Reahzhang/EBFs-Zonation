import pandas as pd
from scipy.stats import pointbiserialr

# 加载 Excel 文件
file_path = ('E:/0-毕业论文数据/图/图/蒸发量/ET.xlsx')  # 替换为您的 Excel 文件路径
df = pd.read_excel(file_path)

# 假设 'wet' 列是二分类变量
binary_column = 'wet'

# 检查并删除包含异常值的行
df = df.replace(-9999, pd.NA)  # 将 -9999 替换为 pandas 可识别的缺失值
df.dropna(inplace=True)  # 删除包含缺失值的行

# 遍历 DataFrame 中的所有列，计算与 'wet' 列的点二列相关系数
for column in df.columns:
    if column != binary_column:
        correlation, p_value = pointbiserialr(df[binary_column], df[column])
        print(f"Correlation between '{binary_column}' and '{column}': {correlation}, P-value: {p_value}")
