import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np
city_list = ['austin_13','austin_14','austin_15','dc_13','dc_14','dc_15','denver_13','denver_14','denver_15']
for city in city_list:
    path = city +'.csv'
    df_initial = pd.read_csv(path)
    df = df_initial[['% below poverty','unemployment rate','% 18-24 enrolled in college or graduate school','household mean income','% 25-34 years bachelors degree or higher','pop estimate']]
    df.rename(columns={'% below poverty': 'pbp', 'unemployment rate': 'ur','% 18-24 enrolled in college or graduate school':'ec','household mean income':'hmi','% 25-34 years bachelors degree or higher':'bd','pop estimate':'pe'}, inplace=True)
    df['pbp'] = pd.to_numeric(df['pbp'], errors='coerce')
    df['ur'] = pd.to_numeric(df['ur'], errors='coerce')
    df['hmi'] = pd.to_numeric(df['hmi'], errors='coerce')
    df['pe'] = df['pe'].astype('float64')
    df = df.fillna(df.mean())
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)
    pca = PCA()
    pca.fit(df_normalized)
    X1=pca.fit_transform(df_normalized)
    X1=X1[:,:2]
    df['PC1']=X1[:,0]
    df['PC2']=X1[:,1]
    df_pca =df [['PC1','PC2']]
    df_norm = (df_pca - df_pca.min()) / (df_pca.max() - df_pca.min())
    df_norm['gdi']=(df_norm['PC1']*(1+df_norm['PC2']))/(2.0)
    df_initial['gdi']=df_norm['gdi']
    df=df_initial[['city','year','geoid','gdi']]
    df.to_csv(path)
