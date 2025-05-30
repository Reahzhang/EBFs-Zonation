import pandas as pd

# 读取Excel文件
file_path = 'E:/Data/Humid/EBBF.xlsx'
df = pd.read_excel(file_path)

# 将-9999替换为NaN
df.replace(-9999, pd.NA, inplace=True)

# 删除包含NaN的行
df.dropna(inplace=True)

# 保存到新的Excel文件
new_file_path = 'E:/Data/Humid/EBBF_clean.xlsx'
df.to_excel(new_file_path, index=False)
