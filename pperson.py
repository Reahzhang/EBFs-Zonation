import pandas as pd

# 加载Excel文件
file_path = 'E:/Data/Humid/高密度/Semi_高密度区.xlsx'  # 请替换为您的文件路径
df = pd.read_excel(file_path)


# 计算相关性矩阵
correlation_matrix = df.corr()

# 打印相关性矩阵
print(correlation_matrix)

# 可选：将相关性矩阵保存为新的Excel文件
correlation_matrix.to_excel('E:/Data/Humid/高密度/correlation_Semi_高密度区.xlsx')
