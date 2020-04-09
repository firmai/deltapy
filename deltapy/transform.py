import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import signal, integrate
from pykalman import UnscentedKalmanFilter
from tsaug import *
from fbprophet import Prophet
import pylab as pl
from seasonal.periodogram import periodogram

def infer_seasonality(train,index=0): ##skip the first one, normally
    interval, power = periodogram(train, min_period=4, max_period=None)
    try:
      season = int(pd.DataFrame([interval, power]).T.sort_values(1,ascending=False).iloc[0,index])
    except:
      print("Failed")
    return min(season,5)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def robust_scaler(df, drop=None,quantile_range=(25, 75) ):
    if drop:
      keep = df[drop]
      df = df.drop(drop, axis=1)
    center = np.median(df, axis=0)
    quantiles = np.percentile(df, quantile_range, axis=0)
    scale = quantiles[1] - quantiles[0]
    df = (df - center) / scale
    if drop:
      df = pd.concat((keep,df),axis=1)
    return df

# df = robust_scaler(df, drop=["Close_1"])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def standard_scaler(df,drop ):
    if drop:
      keep = df[drop]
      df = df.drop(drop, axis=1)
    mean = np.mean(df, axis=0)
    scale = np.std(df, axis=0)
    df = (df - mean) / scale  
    if drop:
      df = pd.concat((keep,df),axis=1)
    return df


# df = standard_scaler(df, drop=["Close"])           


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def fast_fracdiff(x, cols, d):
    for col in cols:
      T = len(x[col])
      np2 = int(2 ** np.ceil(np.log2(2 * T - 1)))
      k = np.arange(1, T)
      b = (1,) + tuple(np.cumprod((k - d - 1) / k))
      z = (0,) * (np2 - T)
      z1 = b + z
      z2 = tuple(x[col]) + z
      dx = pl.ifft(pl.fft(z1) * pl.fft(z2))
      x[col+"_frac"] = np.real(dx[0:T])
    return x 
  
# df_out = fast_fracdiff(df.copy(), ["Close","Open"],0.5); df_out.head()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def outlier_detect(data,col,threshold=1,method="IQR"):
      
    if method == "IQR":
      IQR = data[col].quantile(0.75) - data[col].quantile(0.25)
      Lower_fence = data[col].quantile(0.25) - (IQR * threshold)
      Upper_fence = data[col].quantile(0.75) + (IQR * threshold)
    if method == "STD":
      Upper_fence = data[col].mean() + threshold * data[col].std()
      Lower_fence = data[col].mean() - threshold * data[col].std()   
    if method == "OWN":
      Upper_fence = data[col].mean() + threshold * data[col].std()
      Lower_fence = data[col].mean() - threshold * data[col].std() 
    if method =="MAD":
      median = data[col].median()
      median_absolute_deviation = np.median([np.abs(y - median) for y in data[col]])
      modified_z_scores = pd.Series([0.6745 * (y - median) / median_absolute_deviation for y in data[col]])
      outlier_index = np.abs(modified_z_scores) > threshold
      print('Num of outlier detected:',outlier_index.value_counts()[1])
      print('Proportion of outlier detected',outlier_index.value_counts()[1]/len(outlier_index))
      return outlier_index, (median_absolute_deviation, median_absolute_deviation)

    para = (Upper_fence, Lower_fence)
    tmp = pd.concat([data[col]>Upper_fence,data[col]<Lower_fence],axis=1)
    outlier_index = tmp.any(axis=1)
    print('Num of outlier detected:',outlier_index.value_counts()[1])
    print('Proportion of outlier detected',outlier_index.value_counts()[1]/len(outlier_index))
    
    return outlier_index, para

def windsorization(data,col,para,strategy='both'):
    """
    top-coding & bottom coding (capping the maximum of a distribution at an arbitrarily set value,vice versa)
    """

    data_copy = data.copy(deep=True)  
    if strategy == 'both':
        data_copy.loc[data_copy[col]>para[0],col] = para[0]
        data_copy.loc[data_copy[col]<para[1],col] = para[1]
    elif strategy == 'top':
        data_copy.loc[data_copy[col]>para[0],col] = para[0]
    elif strategy == 'bottom':
        data_copy.loc[data_copy[col]<para[1],col] = para[1]  
    return data_copy

# _, para = outlier_detect(df, "Close")
# df = windsorization(df,"Close",para,strategy='both')

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def operations(df,features):
  df_new = df[features]
  df_new = df_new - df_new.min()

  sqr_name = [str(fa)+"_POWER_2" for fa in df_new.columns]
  log_p_name = [str(fa)+"_LOG_p_one_abs" for fa in df_new.columns]
  rec_p_name = [str(fa)+"_RECIP_p_one" for fa in df_new.columns]
  sqrt_name = [str(fa)+"_SQRT_p_one" for fa in df_new.columns]

  df_sqr = pd.DataFrame(np.power(df_new.values, 2),columns=sqr_name, index=df.index)
  df_log = pd.DataFrame(np.log(df_new.add(1).abs().values),columns=log_p_name, index=df.index)
  df_rec = pd.DataFrame(np.reciprocal(df_new.add(1).values),columns=rec_p_name, index=df.index)
  df_sqrt = pd.DataFrame(np.sqrt(df_new.abs().add(1).values),columns=sqrt_name, index=df.index)

  dfs = [df, df_sqr, df_log, df_rec, df_sqrt]

  df=  pd.concat(dfs, axis=1)

  return df

# df = operations(df,["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen

def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals

def triple_exponential_smoothing(df,cols, slen, alpha, beta, gamma, n_preds):
    for col in cols:
      result = []
      seasonals = initial_seasonal_components(df[col], slen)
      for i in range(len(df[col])+n_preds):
          if i == 0: # initial values
              smooth = df[col][0]
              trend = initial_trend(df[col], slen)
              result.append(df[col][0])
              continue
          if i >= len(df[col]): # we are forecasting
              m = i - len(df[col]) + 1
              result.append((smooth + m*trend) + seasonals[i%slen])
          else:
              val = df[col][i]
              last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
              trend = beta * (smooth-last_smooth) + (1-beta)*trend
              seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
              result.append(smooth+trend+seasonals[i%slen])
      df[col+"_TES"] = result
    #print(seasonals)
    return df

# df_out= triple_exponential_smoothing(df.copy(),["Close"], 12, .2,.2,.2,0); df_out.head()


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def naive_dec(df, columns, freq=2):
  for col in columns:
    decomposition = sm.tsa.seasonal_decompose(df[col], model='additive', freq = freq, two_sided=False)
    df[col+"_NDDT" ] = decomposition.trend
    df[col+"_NDDS"] = decomposition.seasonal
    df[col+"_NDDR"] = decomposition.resid
  return df

# df_out = naive_dec(df.copy(), ["Close","Open"]); df_out.head()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def bkb(df, cols):
  for col in cols:
    df[col+"_BPF"] = sm.tsa.filters.bkfilter(df[[col]].values, 2, 10, len(df)-1)
  return df

# df_out = bkb(df.copy(), ["Close"]); df_out.head()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def butter_lowpass(cutoff, fs=20, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
    
def butter_lowpass_filter(df,cols, cutoff, fs=20, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    for col in cols:
      df[col+"_BUTTER"] = signal.lfilter(b, a, df[col])
    return df

# df_out = butter_lowpass_filter(df.copy(),["Close"],4); df_out.head()


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def instantaneous_phases(df,cols):
    for col in cols:
      df[col+"_HILLB"] = np.unwrap(np.angle(signal.hilbert(df[col], axis=0)), axis=0)
    return df

# df_out = instantaneous_phases(df.copy(), ["Close"]); df_out.head()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def kalman_feat(df, cols):
  for col in cols:
    ukf = UnscentedKalmanFilter(lambda x, w: x + np.sin(w), lambda x, v: x + v, observation_covariance=0.1)
    (filtered_state_means, filtered_state_covariances) = ukf.filter(df[col])
    (smoothed_state_means, smoothed_state_covariances) = ukf.smooth(df[col])
    df[col+"_UKFSMOOTH"] = smoothed_state_means.flatten()
    df[col+"_UKFFILTER"] = filtered_state_means.flatten()
  return df 

# df_out = kalman_feat(df.copy(), ["Close"]); df_out.head()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def perd_feat(df, cols):
  for col in cols:
    sig = signal.periodogram(df[col],fs=1, return_onesided=False)
    df[col+"_FREQ"] = sig[0]
    df[col+"_POWER"] = sig[1]
  return df

# df_out = perd_feat(df.copy(),["Close"]); df_out.head()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def fft_feat(df, cols):
  for col in cols:
    fft_df = np.fft.fft(np.asarray(df[col].tolist()))
    fft_df = pd.DataFrame({'fft':fft_df})
    df[col+'_FFTABS'] = fft_df['fft'].apply(lambda x: np.abs(x)).values
    df[col+'_FFTANGLE'] = fft_df['fft'].apply(lambda x: np.angle(x)).values
  return df 

# df_out = fft_feat(df.copy(), ["Close"]); df_out.head()


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def harmonicradar_cw(df, cols, fs,fc):
    for col in cols:
      ttxt = f'CW: {fc} Hz'
      #%% input
      t = df[col]
      tx = np.sin(2*np.pi*fc*t)
      _,Pxx = signal.welch(tx,fs)
      #%% diode
      d = (signal.square(2*np.pi*fc*t))
      d[d<0] = 0.
      #%% output of diode
      rx = tx * d
      df[col+"_HARRAD"] = rx.values
    return df

# df_out = harmonicradar_cw(df.copy(), ["Close"],0.3,0.2); df_out.head()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def saw(df, cols):
    for col in cols:
      df[col+" SAW"] = signal.sawtooth(df[col])
    return df

# df_out = saw(df.copy(),["Close","Open"]); df_out.head()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def modify(df, cols):
  for col in cols:
    series = df[col].values
    df[col+"_magnify"], _ = magnify(series, series)
    df[col+"_affine"], _ = affine(series, series)
    df[col+"_crop"], _ = crop(series, series)
    df[col+"_cross_sum"], _ = cross_sum(series, series)
    df[col+"_resample"], _ = resample(series, series)
    df[col+"_trend"], _ = trend(series, series)

    df[col+"_random_affine"], _ = random_time_warp(series, series)
    df[col+"_random_crop"], _ = random_crop(series, series)
    df[col+"_random_cross_sum"], _ = random_cross_sum(series, series)
    df[col+"_random_sidetrack"], _ = random_sidetrack(series, series)
    df[col+"_random_time_warp"], _ = random_time_warp(series, series)
    df[col+"_random_magnify"], _ = random_magnify(series, series)
    df[col+"_random_jitter"], _ = random_jitter(series, series)
    df[col+"_random_trend"], _ = random_trend(series, series)
  return df

# df_out = modify(df.copy(),["Close"]); df_out.head()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def multiple_rolling(df, windows = [1,2], functions=["mean","std"], columns=None):
  windows = [1+a for a in windows]
  if not columns:
    columns = df.columns.to_list()
  rolling_dfs = (df[columns].rolling(i)                                    # 1. Create window
                  .agg(functions)                                # 1. Aggregate
                  .rename({col: '{0}_{1:d}'.format(col, i)
                                for col in columns}, axis=1)  # 2. Rename columns
                for i in windows)                                # For each window
  df_out = pd.concat((df, *rolling_dfs), axis=1)
  da = df_out.iloc[:,len(df.columns):]
  da = [col[0] + "_" + col[1] for col in  da.columns.to_list()]
  df_out.columns = df.columns.to_list() + da 

  return  df_out                      # 3. Concatenate dataframes

# df = multiple_rolling(df, columns=["Close"]);

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def multiple_lags(df, start=1, end=3,columns=None):
  if not columns:
    columns = df.columns.to_list()
  lags = range(start, end+1)  # Just two lags for demonstration.

  df = df.assign(**{
      '{}_t_{}'.format(col, t): df[col].shift(t)
      for t in lags
      for col in columns
  })
  return df

# df = multiple_lags(df, start=1, end=3, columns=["Close"])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def prophet_feat(df, cols,date, freq,train_size=150):
  def prophet_dataframe(df): 
    df.columns = ['ds','y']
    return df

  def original_dataframe(df, freq, name):
    prophet_pred = pd.DataFrame({"Date" : df['ds'], name : df["yhat"]})
    prophet_pred = prophet_pred.set_index("Date")
    #prophet_pred.index.freq = pd.tseries.frequencies.to_offset(freq)
    return prophet_pred[name].values

  for col in cols:
    model = Prophet(daily_seasonality=True)
    fb = model.fit(prophet_dataframe(df[[date, col]].head(train_size)))
    forecast_len = len(df) - train_size
    future = model.make_future_dataframe(periods=forecast_len,freq=freq)
    future_pred = model.predict(future)
    df[col+"_PROPHET"] = list(original_dataframe(future_pred,freq,col))
  return df

# df_out  = prophet_feat(df.copy().reset_index(),["Close","Open"],"Date", "D"); df_out.head()

