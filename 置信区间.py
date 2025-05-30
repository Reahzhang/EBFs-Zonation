import pandas as pd
import numpy as np

# 读取数据
file_path = 'E:/Data/普通方法/Raster.xlsx'
df = pd.read_excel(file_path)

# 定义条件
conditions = ['偏干性常绿阔叶林', '偏湿性常绿阔叶林']

# 定义所需的置信区间百分位数
percentiles = {
    '90%': [5, 95],
    '95%': [2.5, 97.5],
    '99%': [0.5, 99.5]
}

# 遍历条件和百分位数来计算置信区间
for condition in conditions:
    print(f"对于{condition}:")
    # 筛选出符合条件的数据
    filtered_data = df[df['EBF'] == condition]['Raster'].dropna()  # 假设分析的数据在"SVM"列
    if filtered_data.empty:
        print(f"没有找到{condition}的数据。\n")
        continue

    for confidence, (lower, upper) in percentiles.items():
        lower_percentile = np.percentile(filtered_data, lower)
        upper_percentile = np.percentile(filtered_data, upper)
        print(f"{confidence}置信区间为：{lower_percentile}到{upper_percentile}")
    print()  # 在不同条件间添加空行以便于阅读

