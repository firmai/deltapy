import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import signal, integrate
from pykalman import UnscentedKalmanFilter
from fbprophet import prophet_dataframe
from tsaug import magnify,affine, crop, cross_sum , resample, reverse, trend, random_affine, random_crop, random_cross_sum, random_sidetrack, random_time_warp, random_magnify, random_jitter, random_trend



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

def fast_fracdiff(x, d):
    import pylab as pl
    T = len(x)
    np2 = int(2 ** np.ceil(np.log2(2 * T - 1)))
    k = np.arange(1, T)
    b = (1,) + tuple(np.cumprod((k - d - 1) / k))
    z = (0,) * (np2 - T)
    z1 = b + z
    z2 = tuple(x) + z
    dx = pl.ifft(pl.fft(z1) * pl.fft(z2))
    return np.real(dx[0:T])
  
# df["FRACDIF"] = fast_fracdiff(df["Close"],0.5)

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

def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)

    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    #print(seasonals)
    return result

# df["TES"] = triple_exponential_smoothing(df["Close"], 12, .2,.2,.2,0)




def naive_dec(df, column, freq=4):
  decomposition = sm.tsa.seasonal_decompose(df[column], model='additive', freq = freq, two_sided=False,)
  df["NDDT"] = decomposition.trend
  df["NDDS"] = decomposition.seasonal
  df["NDDR"] = decomposition.resid
  return df

# df = naive_dec(df, "Close")



def bkb(df, column):
  df["BPF"] = sm.tsa.filters.bkfilter(df[[column]].values, 2, 10, len(df)-1)
  return df

# df = bkb(df, "Close")


def butter_lowpass(cutoff, fs=20, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
    
def butter_lowpass_filter(data, cutoff, fs=20, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

# df["BUTTER"] = butter_lowpass_filter(df["Close"],4)




def instantaneous_phases(band_signals, axis=0):
    analytical_signal = signal.hilbert(band_signals, axis=axis)
    return np.unwrap(np.angle(analytical_signal), axis=axis)

# df["HILBANG"] = instantaneous_phases(df["Close"])



def kalman_feat(df, col):
      ukf = UnscentedKalmanFilter(lambda x, w: x + np.sin(w), lambda x, v: x + v, observation_covariance=0.1)
  (filtered_state_means, filtered_state_covariances) = ukf.filter(df[col])
  (smoothed_state_means, smoothed_state_covariances) = ukf.smooth(df[col])
  df["UKFSMOOTH"] = smoothed_state_means.flatten()
  df["UKFFILTER"] = filtered_state_means.flatten()
  return df 

# df = kalman_feat(df, "Close")


def perd_feat(df, column):
  df["FREQ"] = signal.periodogram(df[column],fs=1, return_onesided=False)[0]
  df["POWER"] = signal.periodogram(df[column],fs=1, return_onesided=False)[1]
  return df

# df = perd_feat(df,"Close")

def fft_feat(df, col):
      fft_df = np.fft.fft(np.asarray(df[col].tolist()))
  fft_df = pd.DataFrame({'fft':fft_df})
  df['FFTABS'] = fft_df['fft'].apply(lambda x: np.abs(x)).values
  df['FFTANGLE'] = fft_df['fft'].apply(lambda x: np.angle(x)).values
  return df 

# df = fft_feat(df, "Close")


def harmonicradar_cw(X,fs,fc):
    ttxt = f'CW: {fc} Hz'
    #%% input
    t = X
    tx = np.sin(2*np.pi*fc*t)
    _,Pxx = signal.welch(tx,fs)
    #%% diode
    d = (signal.square(2*np.pi*fc*t))
    d[d<0] = 0.
    #%% output of diode
    rx = tx * d
    return rx

# df["DIODE"] = harmonicradar_cw(df["Close"],0.3,0.2)

def saw(df, column):
      return signal.sawtooth(df[column])

# df["SAW"] = saw(df,"Close")


def modify(df, column):
  df["magnify"], _ = magnify(df[column].values, df[column].values)
  df["affine"], _ = affine(df[column].values, df[column].values)
  df["crop"], _ = crop(df[column].values, df[column].values)
  df["cross_sum"], _ = cross_sum(df[column].values, df[column].values)
  df["resample"], _ = resample(df[column].values, df[column].values)
  df["trend"], _ = trend(df[column].values, df[column].values)

  df["random_affine"], _ = random_time_warp(df[column].values, df[column].values)
  df["random_crop"], _ = random_crop(df[column].values, df[column].values)
  df["random_cross_sum"], _ = random_cross_sum(df[column].values, df[column].values)
  df["random_sidetrack"], _ = random_sidetrack(df[column].values, df[column].values)
  df["random_time_warp"], _ = random_time_warp(df[column].values, df[column].values)
  df["random_magnify"], _ = random_magnify(df[column].values, df[column].values)
  df["random_jitter"], _ = random_jitter(df[column].values, df[column].values)
  df["random_trend"], _ = random_trend(df[column].values, df[column].values)
  return df

# df_out = modify(df,"Close")

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



def prophet_feat(df, col,drops, freq):
      def prophet_dataframe(df): 
    df.columns = ['ds','y']
    return df

  train_size = 150
  def original_dataframe(df, freq, name):
    prophet_pred = pd.DataFrame({"Date" : df['ds'], name : df["yhat"]})
    prophet_pred = prophet_pred.set_index("Date")
    #prophet_pred.index.freq = pd.tseries.frequencies.to_offset(freq)
    return prophet_pred[name].values

  model = Prophet(daily_seasonality=True)
  fb = model.fit(prophet_dataframe(df[drops].head(train_size)))
  forecast_len = len(df) - train_size
  future = model.make_future_dataframe(periods=forecast_len,freq=freq)
  future_pred = model.predict(future)
  df["PROPHET"] = None
  return list(original_dataframe(future_pred,freq,col))


# df["PROPHET"]  = prophet_feat(df.reset_index(),"Close", ["Date","Close"], "D")

