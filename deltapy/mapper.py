from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.cross_decomposition import CCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import minmax_scale
from sklearn import datasets, cluster
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import tensorflow as tf


def pca_feature(df, memory_issues=False,mem_iss_component=False,variance_or_components=0.80,n_components=5 ,drop_cols=None, non_linear=True):
    
  if non_linear:
    pca = KernelPCA(n_components = n_components, kernel='rbf', fit_inverse_transform=True, random_state = 33, remove_zero_eig= True)
  else:
    if memory_issues:
      if not mem_iss_component:
        raise ValueError("If you have memory issues, you have to preselect mem_iss_component")
      pca = IncrementalPCA(mem_iss_component)
    else:
      if variance_or_components>1:
        pca = PCA(n_components=variance_or_components) 
      else: # automted selection based on variance
        pca = PCA(n_components=variance_or_components,svd_solver="full") 
  if drop_cols:
    X_pca = pca.fit_transform(df.drop(drop_cols,axis=1))
    return pd.concat((df[drop_cols],pd.DataFrame(X_pca, columns=["PCA_"+str(i+1) for i in range(X_pca.shape[1])],index=df.index)),axis=1)

  else:
    X_pca = pca.fit_transform(df)
    return pd.DataFrame(X_pca, columns=["PCA_"+str(i+1) for i in range(X_pca.shape[1])],index=df.index)


  return df

# df_out = pca_feature(df_out, variance_or_components=0.9, n_components=8,non_linear=False)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def cross_lag(df, drop=None, lags=1, components=4 ):

  if drop:
    keep = df[drop]
    df = df.drop([drop],axis=1)

  df_2 = df.shift(lags)
  df = df.iloc[lags:,:]
  df_2 = df_2.dropna().reset_index(drop=True)

  cca = CCA(n_components=components)
  cca.fit(df_2, df)

  X_c, df_2 = cca.transform(df_2, df)
  df_2 = pd.DataFrame(df_2, index=df.index)
  df_2 = df.add_prefix('crd_')

  if drop:
    df = pd.concat([keep,df,df_2],axis=1)
  else:
    df = pd.concat([df,df_2],axis=1)
  return df

# df_out = cross_lag(df)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def a_chi(df, drop=None, lags=1, sample_steps=2 ):

  if drop:
    keep = df[drop]
    df = df.drop([drop],axis=1)

  df_2 = df.shift(lags)
  df = df.iloc[lags:,:]
  df_2 = df_2.dropna().reset_index(drop=True)

  chi2sampler = AdditiveChi2Sampler(sample_steps=sample_steps)

  df_2 = chi2sampler.fit_transform(df_2, df["Close"])

  df_2 = pd.DataFrame(df_2, index=df.index)
  df_2 = df.add_prefix('achi_')

  if drop:
    df = pd.concat([keep,df,df_2],axis=1)
  else:
    df = pd.concat([df,df_2],axis=1)
  return df

# df_out = a_chi(df)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def encoder_dataset(df, drop=None, dimesions=20):

  if drop:
    train_scaled = minmax_scale(df.drop(drop,axis=1).values, axis = 0)
  else:
    train_scaled = minmax_scale(df.values, axis = 0)

  # define the number of encoding dimensions
  encoding_dim = dimesions
  # define the number of features
  ncol = train_scaled.shape[1]
  input_dim = tf.keras.Input(shape = (ncol, ))

  # Encoder Layers
  encoded1 = tf.keras.layers.Dense(3000, activation = 'relu')(input_dim)
  encoded2 = tf.keras.layers.Dense(2750, activation = 'relu')(encoded1)
  encoded3 = tf.keras.layers.Dense(2500, activation = 'relu')(encoded2)
  encoded4 = tf.keras.layers.Dense(750, activation = 'relu')(encoded3)
  encoded5 = tf.keras.layers.Dense(500, activation = 'relu')(encoded4)
  encoded6 = tf.keras.layers.Dense(250, activation = 'relu')(encoded5)
  encoded7 = tf.keras.layers.Dense(encoding_dim, activation = 'relu')(encoded6)

  encoder = tf.keras.Model(inputs = input_dim, outputs = encoded7)
  encoded_input = tf.keras.Input(shape = (encoding_dim, ))

  encoded_train = pd.DataFrame(encoder.predict(train_scaled),index=df.index)
  encoded_train = encoded_train.add_prefix('encoded_')
  if drop:
    encoded_train = pd.concat((df[drop],encoded_train),axis=1)

  return encoded_train

# df_out = encoder_dataset(df, ["Close_1"], 15)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def lle_feat(df, drop=None, components=4):

  if drop:
    keep = df[drop]
    df = df.drop(drop, axis=1)

  embedding = LocallyLinearEmbedding(n_components=components)
  em = embedding.fit_transform(df)
  df = pd.DataFrame(em,index=df.index)
  df = df.add_prefix('lle_')
  if drop:
    df = pd.concat((keep,df),axis=1)
  return df

# df_out = lle_feat(df,["Close_1"],4)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def feature_agg(df, drop=None, components=4):

  if drop:
    keep = df[drop]
    df = df.drop(drop, axis=1)

  components = min(df.shape[1]-1,components)
  agglo = cluster.FeatureAgglomeration(n_clusters=components)
  agglo.fit(df)
  df = pd.DataFrame(agglo.transform(df),index=df.index)
  df = df.add_prefix('feagg_')

  if drop:
    return pd.concat((keep,df),axis=1)
  else:
    return df


# df_out = feature_agg(df.fillna(0),["Close_1"],4 )

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def neigh_feat(df, drop, neighbors=6):
  
  if drop:
    keep = df[drop]
    df = df.drop(drop, axis=1)

  components = min(df.shape[0]-1,neighbors)
  neigh = NearestNeighbors(n_neighbors=neighbors)
  neigh.fit(df)
  neigh = neigh.kneighbors()[0]
  df = pd.DataFrame(neigh, index=df.index)
  df = df.add_prefix('neigh_')

  if drop:
    return pd.concat((keep,df),axis=1)
  else:
    return df

  return df

# df_out = neigh_feat(df,["Close_1"],4 )
