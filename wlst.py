import pandas as pd

# 替换为您的Excel文件路径
file_path = 'E:/workplace/10/normalize_pppoint.xlsx'

# 读取 Excel 文件
df = pd.read_excel(file_path)

# 确保 'wlst' 列存在于 DataFrame 中
if 'wlst' in df.columns:
    # 找出 'wlst' 列的最大值和最小值
    max_value = df['wlst'].max()
    min_value = df['wlst'].min()

    # 打印最大值和最小值
    print(f"The maximum value in 'wlst' column is: {max_value}")
    print(f"The minimum value in 'wlst' column is: {min_value}")
else:
    print("Column 'wlst' does not exist in the DataFrame.")
