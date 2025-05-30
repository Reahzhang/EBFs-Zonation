import pandas as pd
from scipy.stats import pearsonr

# Load the Excel file
df = pd.read_excel('E:/Data/subset/pppoint.xlsx')

# Replace '/path/to/your/excel.xlsx' with the actual path of your Excel file
# Make sure the Excel file has the columns labeled as mentioned

# Calculate correlations
correlations = {}
columns_to_correlate = ['SOC_b200', 'SOC_b100', 'SOC_b60', 'SOC_b30', 'SOC_b10', 'SOC_b0',
                        'SBD_b200', 'SBD_b100', 'SBD_b60', 'SBD_b30', 'SBD_b10', 'SBD_b0']

for column in columns_to_correlate:
    corr, _ = pearsonr(df['wet'], df[column])
    correlations[column] = corr

# Output the correlations
for column, corr in correlations.items():
    print(f"Correlation between wet and {column}: {corr}")

