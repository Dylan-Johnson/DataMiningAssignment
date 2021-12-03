import pandas as pd
name = 'mclaren'
df = pd.read_csv(name+'Clustered.data') # can replace with df = pd.read_table('input.txt') for '\t'
df.to_excel('output'+name+'.xlsx', 'Sheet1', index=False)
