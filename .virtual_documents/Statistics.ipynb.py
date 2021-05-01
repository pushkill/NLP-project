import pandas as pd


df = pd.read_csv('full_dataset_all_labels.csv')
print(df.shape)
counts = df.groupby(['label']).count()
print(counts.to_latex())
