import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AR
from timeit import default_timer as timer
from sklearn.tree import DecisionTreeRegressor
from math import sin, cos, sqrt, atan2, radians,ceil
import ta
from gplearn.genetic import SymbolicTransformer
from scipy import linalg
import math

def lowess(df, cols, y, f=2. / 3., iter=3):
    for col in cols:
      n = len(df[col])
      r = int(ceil(f * n))
      h = [np.sort(np.abs(df[col] - df[col][i]))[r] for i in range(n)]
      w = np.clip(np.abs((df[col][:, None] - df[col][None, :]) / h), 0.0, 1.0)
      w = (1 - w ** 3) ** 3
      yest = np.zeros(n)
      delta = np.ones(n)
      for iteration in range(iter):
          for i in range(n):
              weights = delta * w[:, i]
              b = np.array([np.sum(weights * y), np.sum(weights * y * df[col])])
              A = np.array([[np.sum(weights), np.sum(weights * df[col])],
                            [np.sum(weights * df[col]), np.sum(weights * df[col] * df[col])]])
              beta = linalg.solve(A, b)
              yest[i] = beta[0] + beta[1] * df[col][i]

          residuals = y - yest
          s = np.median(np.abs(residuals))
          delta = np.clip(residuals / (6.0 * s), -1, 1)
          delta = (1 - delta ** 2) ** 2
      df[col+"_LOWESS"] = yest

    return df

# df_out = lowess(df.copy(), ["Open","Volume"], df["Close"], f=0.25, iter=3); df_out.head()


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def autoregression(df, drop=None, settings={"autoreg_lag":4}):
    """
    This function calculates the autoregression for each channel.
    :param x: the input signal. Its size is (number of channels, samples).
    :param settings: a dictionary with one attribute, "autoreg_lag", that is the max lag for autoregression.
    :return: the "final_value" is a matrix (number of channels, autoreg_lag) indicating the parameters of
      autoregression for each channel.
    """
    autoreg_lag = settings["autoreg_lag"]
    if drop:
      keep = df[drop]
      df = df.drop([drop],axis=1).values

    n_channels = df.shape[0]
    t = timer()
    channels_regg = np.zeros((n_channels, autoreg_lag + 1))
    for i in range(0, n_channels):
        fitted_model = AR(df.values[i, :]).fit(autoreg_lag)
        # TODO: This is not the same as Matlab's for some reasons!
        # kk = ARMAResults(fitted_model)
        # autore_vals, dummy1, dummy2 = arburg(x[i, :], autoreg_lag) # This looks like Matlab's but slow
        channels_regg[i, 0: len(fitted_model.params)] = np.real(fitted_model.params)

    for i in range(channels_regg.shape[1]):
      df["LAG_"+str(i+1)] = channels_regg[:,i]
    
    if drop:
      df = pd.concat((keep,df),axis=1)

    t = timer() - t
    return df

# df = autoregression(df.fillna(0))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def muldiv(df, feature_list):
  for feat in feature_list:
    for feat_two in feature_list:
      if feat==feat_two:
        continue
      else:
       df[feat+"/"+feat_two] = df[feat]/(df[feat_two]-df[feat_two].min()) #zero division guard
       df[feat+"_X_"+feat_two] = df[feat]*(df[feat_two])

  return df

# df = muldiv(df, ["Close","Open"]) 

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def decision_tree_disc(df, cols, depth=4 ):
  for col in cols:
    df[col +"_m1"] = df[col].shift(1)
    df = df.iloc[1:,:]
    tree_model = DecisionTreeRegressor(max_depth=depth,random_state=0)
    tree_model.fit(df[col +"_m1"].to_frame(), df[col])
    df[col+"_Disc"] = tree_model.predict(df[col +"_m1"].to_frame())
  return df

# df_out = decision_tree_disc(df, ["Close"]); df_out.head()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def quantile_normalize(df, drop):

    if drop:
      keep = df[drop]
      df = df.drop(drop,axis=1)

    #compute rank
    dic = {}
    for col in df:
      dic.update({col : sorted(df[col])})
    sorted_df = pd.DataFrame(dic)
    rank = sorted_df.mean(axis = 1).tolist()
    #sort
    for col in df:
        t = np.searchsorted(np.sort(df[col]), df[col])
        df[col] = [rank[i] for i in t]
    
    if drop:
      df = pd.concat((keep,df),axis=1)
    return df

# df_out = quantile_normalize(df.fillna(0), drop=["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def haversine_distance(row, lon="Open", lat="Close"):
    c_lat,c_long = radians(52.5200), radians(13.4050)
    R = 6373.0
    long = radians(row['Open'])
    lat = radians(row['Close'])
    
    dlon = long - c_long
    dlat = lat - c_lat
    a = sin(dlat / 2)**2 + cos(lat) * cos(c_lat) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c

# df['distance_central'] = df.apply(haversine_distance,axis=1); df.iloc[:,4:]

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def tech(df):
      return ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
  
# df = tech(df)

    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def genetic_feat(df, num_gen=20, num_comp=10):
  function_set = ['add', 'sub', 'mul', 'div',
                  'sqrt', 'log', 'abs', 'neg', 'inv','tan']

  gp = SymbolicTransformer(generations=num_gen, population_size=200,
                          hall_of_fame=100, n_components=num_comp,
                          function_set=function_set,
                          parsimony_coefficient=0.0005,
                          max_samples=0.9, verbose=1,
                          random_state=0, n_jobs=6)

  gen_feats = gp.fit_transform(df.drop("Close_1", axis=1), df["Close_1"]); df.iloc[:,:8]
  gen_feats = pd.DataFrame(gen_feats, columns=["gen_"+str(a) for a in range(gen_feats.shape[1])])
  gen_feats.index = df.index
  return pd.concat((df,gen_feats),axis=1)

# df = genetic_feat(df)