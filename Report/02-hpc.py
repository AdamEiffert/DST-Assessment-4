import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np


df = pd.read_csv("~/zg18997/kddcup1.csv").drop(columns = ['Unnamed: 0'])
num_feature =[]
for col in df:
    if (df[col].dtypes == 'float64') or (df[col].dtypes == 'int64'):
        num_feature.append(col)
df[num_feature] = StandardScaler().fit_transform(X = df[num_feature])
for i in range(df.shape[0]):
    if (df.at[i,'type'] == 'back.') or (df.at[i,'type'] == 'land.') or (df.at[i,'type'] == 'neptune.') or (df.at[i,'type'] == 'pod.') or (df.at[i,'type'] == 'smurf.') or (df.at[i,'type'] == 'teardrop.'):
        df.at[i,'dos'] = 1
        df.at[i,'u2r'] = 0
        df.at[i,'r2l'] = 0
        df.at[i,'probe'] = 0
        df.at[i,'normal'] = 0
    elif (df.at[i,'type'] == 'ftp_write.') or (df.at[i,'type'] == 'guess_passwd.') or (df.at[i,'type'] == 'imap.') or (df.at[i,'type'] == 'multihop.') or (df.at[i,'type'] == 'phf.') or (df.at[i,'type'] == 'spy.') or (df.at[i,'type'] == 'warezclient.') or (df.at[i,'type'] == 'warezmaster.'):
        df.at[i,'dos'] = 0
        df.at[i,'u2r'] = 0
        df.at[i,'r2l'] = 1
        df.at[i,'probe'] = 0
        df.at[i,'normal'] = 0
    elif (df.at[i,'type'] == 'buffer_overflow.') or (df.at[i,'type'] == 'loadmodule.') or (df.at[i,'type'] == 'perl.') or (df.at[i,'type'] == 'rootkit.'):
        df.at[i,'dos'] = 0
        df.at[i,'u2r'] = 1
        df.at[i,'r2l'] = 0
        df.at[i,'probe'] = 0
        df.at[i,'normal'] = 0
    elif (df.at[i,'type'] == 'nmap.') or (df.at[i,'type'] == 'ipsweep.') or (df.at[i,'type'] == 'portsweep.') or (df.at[i,'type'] == 'satan.'):
        df.at[i,'dos'] = 0
        df.at[i,'u2r'] = 0
        df.at[i,'r2l'] = 0
        df.at[i,'probe'] = 1
        df.at[i,'normal'] = 0
    elif (df.at[i,'type'] == 'normal.'):
        df.at[i,'dos'] = 0
        df.at[i,'u2r'] = 0
        df.at[i,'r2l'] = 0
        df.at[i,'probe'] = 0
        df.at[i,'normal'] = 1

df.drop(columns = ['type'], inplace = True)
cat_features = []
for col in df:
    if df[col].dtypes == 'object':
        cat_features.append(col)
for cat_feature in cat_features:
    for feature in df[cat_feature].unique():
        df[cat_feature+'_is_'+str(feature)] = [0]*df.shape[0]
df.drop(columns = ['protocol_type', 'service', 'flag'], inplace = True)
np.random.seed(63)
kf = KFold(n_splits=4,shuffle=True)
kfsplit=kf.split(df)
trainfolds,testfold=next(kfsplit) 
data=df.loc[trainfolds]
testdata=df.loc[testfold]
data.to_csv('~/zg18997/train2.csv')
testdata.to_csv('~/zg18997/test2.csv')
        
        

