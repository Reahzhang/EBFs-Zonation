import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取Excel文件
my_data = pd.read_excel("E:/workplace/10/pearson_correlation_results.xlsx")

# 创建辅助列以分类相关性和显著性
my_data['r.sign'] = my_data['Pearson Correlation'].apply(lambda x: "Positive" if x > 0 else "Negative")
my_data['p.sign'] = my_data['P-value'].apply(lambda x: "P<0.05" if x < 0.05 else "P>=0.05")
my_data['r.abs'] = pd.cut(abs(my_data['Pearson Correlation']), bins=[0, 0.1, 0.3, 0.5, 1],
                         labels=["<0.1", "0.1-0.3", "0.3-0.5", ">0.5"])

# 使用Seaborn创建相关性的图表
plt.figure(figsize=(10, 6))
sns.scatterplot(data=my_data, x='Variable', y='Pearson Correlation', hue='r.sign', size='r.abs',
                palette={"Positive": "#2a6295", "Negative": "#ca6720"}, sizes=(50, 200))

plt.axhline(y=0, linestyle="--", color='gray')  # 添加虚线表示y=0

plt.title("Pearson Correlation")
plt.xlabel("Variable")
plt.ylabel("Correlation")
plt.legend(title="Correlation Sign", loc='lower right', bbox_to_anchor=(1.1, 0.2))
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("correlation_plot.png", bbox_inches='tight')
plt.show()
