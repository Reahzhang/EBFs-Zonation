import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# 加载Excel文件
df = pd.read_excel('E:/Data/Humid/高密度/Normalized_Humid_VIF.xlsx')

# 假设我们要排除名为'EBF'的列
df = df.drop(columns=['HEBF'])

# 为VIF计算添加常数列
df_with_const = add_constant(df)

# 计算每列的VIF值
vif_data = pd.DataFrame()
vif_data["Feature"] = df_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(df_with_const.values, i) for i in range(df_with_const.shape[1])]

# 打印VIF值
print(vif_data)
