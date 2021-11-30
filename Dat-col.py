import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('sales.csv')
sns.heatmap(data.isnull())
plt.show()

def impute_data(cols):
    mark=cols[0]
    sal=col[1]
    if pd.isnull('Marketing'):
        if sal>10000000:
            return 100000
        elif sal<100000:
            return 0
        else:
            return 1000
        
    else:
        return mark

    
data['Marketing']=data[['Marketing','Sales']].apply(impute_data,axis=1)
sns.heatmap(data.isnull())
plt.show()

data.drop('Employees',axis=1,inplace=True)
data.dropna(inplace= True)
print(data)

data.to_csv('normalized_dat.csv')

