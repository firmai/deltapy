import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, adfuller, pacf
import itertools
from statsmodels.tsa.ar_model import AR
from scipy.signal import cwt, find_peaks_cwt, ricker, welch
from scipy.stats import linregress
import scipy.stats as stats
from scipy.stats import kurtosis as _kurt
from scipy.stats import skew as _skew
import numpy as np
from scipy import signal, integrate
from scipy.interpolate import interp1d
import math
from statsmodels.tools.sm_exceptions import MissingDataError
from numpy.linalg import LinAlgError


def set_property(key, value):
    """
    This method returns a decorator that sets the property key of the function to value
    """
    def decorate_func(func):
        setattr(func, key, value)
        if func.__doc__ and key == "fctype":
            func.__doc__ = func.__doc__ + "\n\n    *This function is of type: " + value + "*\n"
        return func
    return decorate_func


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#-> In Package
def abs_energy(x):

    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.dot(x, x)

# abs_energy(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#-> In Package
def cid_ce(x, normalize):

    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if normalize:
        s = np.std(x)
        if s!=0:
            x = (x - np.mean(x))/s
        else:
            return 0.0

    x = np.diff(x)
    return np.sqrt(np.dot(x, x))

# cid_ce(df["Close"], True)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#-> In Package
def mean_abs_change(x):
    return np.mean(np.abs(np.diff(x)))

# mean_abs_change(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#-> In Package
def _roll(a, shift):
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])

def mean_second_derivative_central(x):

    diff = (_roll(x, 1) - 2 * np.array(x) + _roll(x, -1)) / 2.0
    return np.mean(diff[1:-1])

# mean_second_derivative_central(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#-> In Package
def variance_larger_than_standard_deviation(x):

    y = np.var(x)
    return y > np.sqrt(y)

# variance_larger_than_standard_deviation(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

var_index_param = {"Volume":None, "Open": None}

@set_property("fctype", "combiner")
@set_property("custom", True)
def var_index(time,param=var_index_param):
    final = []
    keys = []
    for key, magnitude in param.items():
      w = 1.0 / np.power(np.subtract(time[1:], time[:-1]), 2)
      w_mean = np.mean(w)

      N = len(time)
      sigma2 = np.var(magnitude)

      S1 = sum(w * (magnitude[1:] - magnitude[:-1]) ** 2)
      S2 = sum(w)

      eta_e = (w_mean * np.power(time[N - 1] -
                time[0], 2) * S1 / (sigma2 * S2 * N ** 2))
      final.append(eta_e)
      keys.append(key)
    return {"Interact__{}".format(k): eta_e for eta_e, k in zip(final,keys) }

# var_index(df["Close"].values)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#-> In Package
def symmetry_looking(x, param=[{"r": 0.2}]):

    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    mean_median_difference = np.abs(np.mean(x) - np.median(x))
    max_min_difference = np.max(x) - np.min(x)
    return [("r_{}".format(r["r"]), mean_median_difference < (r["r"] * max_min_difference))
            for r in param]
            
# symmetry_looking(df["Close"])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-> In Package
def has_duplicate_max(x):
    """
    Checks if the maximum value of x is observed more than once

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: bool
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.sum(x == np.max(x)) >= 2

# has_duplicate_max(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-> In Package

def partial_autocorrelation(x, param=[{"lag": 1}]):

    # Check the difference between demanded lags by param and possible lags to calculate (depends on len(x))
    max_demanded_lag = max([lag["lag"] for lag in param])
    n = len(x)

    # Check if list is too short to make calculations
    if n <= 1:
        pacf_coeffs = [np.nan] * (max_demanded_lag + 1)
    else:
        if (n <= max_demanded_lag):
            max_lag = n - 1
        else:
            max_lag = max_demanded_lag
        pacf_coeffs = list(pacf(x, method="ld", nlags=max_lag))
        pacf_coeffs = pacf_coeffs + [np.nan] * max(0, (max_demanded_lag - max_lag))

    return [("lag_{}".format(lag["lag"]), pacf_coeffs[lag["lag"]]) for lag in param]

# partial_autocorrelation(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-> In Package
def augmented_dickey_fuller(x, param=[{"attr": "teststat"}]):

    res = None
    try:
        res = adfuller(x)
    except LinAlgError:
        res = np.NaN, np.NaN, np.NaN
    except ValueError: # occurs if sample size is too small
        res = np.NaN, np.NaN, np.NaN
    except MissingDataError: # is thrown for e.g. inf or nan in the data
        res = np.NaN, np.NaN, np.NaN

    return [('attr_"{}"'.format(config["attr"]),
                  res[0] if config["attr"] == "teststat"
             else res[1] if config["attr"] == "pvalue"
             else res[2] if config["attr"] == "usedlag" else np.NaN)
            for config in param]

# augmented_dickey_fuller(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@set_property("fctype", "simple")
@set_property("custom", True)
def gskew(x):
    interpolation="nearest"
    median_mag = np.median(x)
    F_3_value = np.percentile(x, 3, interpolation=interpolation)
    F_97_value = np.percentile(x, 97, interpolation=interpolation)

    skew = (np.median(x[x <= F_3_value]) +
            np.median(x[x >= F_97_value]) - 2 * median_mag)

    return skew

# gskew(df["Close"])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

stestson_param = {"weight":100., "alpha":2., "beta":2., "tol":1.e-6, "nmax":20}

@set_property("fctype", "combiner")
@set_property("custom", True)
def stetson_mean(x, param=stestson_param):
    
    weight= stestson_param["weight"]
    alpha= stestson_param["alpha"]
    beta = stestson_param["beta"]
    tol= stestson_param["tol"]
    nmax= stestson_param["nmax"]
    
    
    mu = np.median(x)
    for i in range(nmax):
        resid = x - mu
        resid_err = np.abs(resid) * np.sqrt(weight)
        weight1 = weight / (1. + (resid_err / alpha)**beta)
        weight1 /= weight1.mean()
        diff = np.mean(x * weight1) - mu
        mu += diff
        if (np.abs(diff) < tol*np.abs(mu) or np.abs(diff) < tol):
            break

    return mu

# stetson_mean(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-> In Package
def length(x):
    return len(x)
    
# length(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#-> In Package
def count_above_mean(x):
    m = np.mean(x)
    return np.where(x > m)[0].size

# count_above_mean(df["Close"])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-> In Package


def get_length_sequences_where(x):

    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]

def longest_strike_below_mean(x):

    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.max(get_length_sequences_where(x <= np.mean(x))) if x.size > 0 else 0

# longest_strike_below_mean(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

woz_param = [{"consecutiveStar": n} for n in [2, 4]]

@set_property("fctype", "combiner")
@set_property("custom", True)
def wozniak(magnitude, param=woz_param):

    iters = []
    for consecutiveStar in [stars["consecutiveStar"] for stars in param]:
      N = len(magnitude)
      if N < consecutiveStar:
          return 0
      sigma = np.std(magnitude)
      m = np.mean(magnitude)
      count = 0

      for i in range(N - consecutiveStar + 1):
          flag = 0
          for j in range(consecutiveStar):
              if(magnitude[i + j] > m + 2 * sigma or
                  magnitude[i + j] < m - 2 * sigma):
                  flag = 1
              else:
                  flag = 0
                  break
          if flag:
              count = count + 1
      iters.append(count * 1.0 / (N - consecutiveStar + 1))

    return [("consecutiveStar_{}".format(config["consecutiveStar"]), iters[en] )  for en, config in enumerate(param)]

# wozniak(df["Close"])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-> In Package
def last_location_of_maximum(x):

    x = np.asarray(x)
    return 1.0 - np.argmax(x[::-1]) / len(x) if len(x) > 0 else np.NaN

# last_location_of_maximum(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-> In Package
def fft_coefficient(x, param = [{"coeff": 10, "attr": "real"}]):

    assert min([config["coeff"] for config in param]) >= 0, "Coefficients must be positive or zero."
    assert set([config["attr"] for config in param]) <= set(["imag", "real", "abs", "angle"]), \
        'Attribute must be "real", "imag", "angle" or "abs"'

    fft = np.fft.rfft(x)

    def complex_agg(x, agg):
        if agg == "real":
            return x.real
        elif agg == "imag":
            return x.imag
        elif agg == "abs":
            return np.abs(x)
        elif agg == "angle":
            return np.angle(x, deg=True)

    res = [complex_agg(fft[config["coeff"]], config["attr"]) if config["coeff"] < len(fft)
           else np.NaN for config in param]
    index = [('coeff_{}__attr_"{}"'.format(config["coeff"], config["attr"]),res[0]) for config in param]
    return index

# fft_coefficient(df["Close"])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-> In Package


def ar_coefficient(x, param=[{"coeff": 5, "k": 5}]):

    calculated_ar_params = {}

    x_as_list = list(x)
    calculated_AR = AR(x_as_list)

    res = {}

    for parameter_combination in param:
        k = parameter_combination["k"]
        p = parameter_combination["coeff"]

        column_name = "k_{}__coeff_{}".format(k, p)

        if k not in calculated_ar_params:
            try:
                calculated_ar_params[k] = calculated_AR.fit(maxlag=k, solver="mle").params
            except (LinAlgError, ValueError):
                calculated_ar_params[k] = [np.NaN]*k

        mod = calculated_ar_params[k]

        if p <= k:
            try:
                res[column_name] = mod[p]
            except IndexError:
                res[column_name] = 0
        else:
            res[column_name] = np.NaN

    return [(key, value) for key, value in res.items()]

# ar_coefficient(df["Close"])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-> In Package
def index_mass_quantile(x, param=[{"q": 0.3}]):

    x = np.asarray(x)
    abs_x = np.abs(x)
    s = sum(abs_x)

    if s == 0:
        # all values in x are zero or it has length 0
        return [("q_{}".format(config["q"]), np.NaN) for config in param]
    else:
        # at least one value is not zero
        mass_centralized = np.cumsum(abs_x) / s
        return [("q_{}".format(config["q"]), (np.argmax(mass_centralized >= config["q"])+1)/len(x)) for config in param]

# index_mass_quantile(df["Close"])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



cwt_param = [ka for ka in [2,6,9]]

@set_property("fctype", "combiner")
@set_property("custom", True)
def number_cwt_peaks(x, param=cwt_param):

    return [("CWTPeak_{}".format(n), len(find_peaks_cwt(vector=x, widths=np.array(list(range(1, n + 1))), wavelet=ricker))) for n in param]

# number_cwt_peaks(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#-> In Package
def spkt_welch_density(x, param=[{"coeff": 5}]):
    freq, pxx = welch(x, nperseg=min(len(x), 256))
    coeff = [config["coeff"] for config in param]
    indices = ["coeff_{}".format(i) for i in coeff]

    if len(pxx) <= np.max(coeff):  # There are fewer data points in the time series than requested coefficients

        # filter coefficients that are not contained in pxx
        reduced_coeff = [coefficient for coefficient in coeff if len(pxx) > coefficient]
        not_calculated_coefficients = [coefficient for coefficient in coeff
                                       if coefficient not in reduced_coeff]

        # Fill up the rest of the requested coefficients with np.NaNs
        return zip(indices, list(pxx[reduced_coeff]) + [np.NaN] * len(not_calculated_coefficients))
    else:
        return pxx[coeff].ravel()[0]

# spkt_welch_density(df["Close"])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#-> In Package
def linear_trend_timewise(x, param= [{"attr": "pvalue"}]):

    ix = x.index

    # Get differences between each timestamp and the first timestamp in seconds.
    # Then convert to hours and reshape for linear regression
    times_seconds = (ix - ix[0]).total_seconds()
    times_hours = np.asarray(times_seconds / float(3600))

    linReg = linregress(times_hours, x.values)

    return [("attr_\"{}\"".format(config["attr"]), getattr(linReg, config["attr"]))
            for config in param]

# linear_trend_timewise(df["Close"])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-> In Package
def c3(x, lag=3):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    n = x.size
    if 2 * lag >= n:
        return 0
    else:
        return np.mean((_roll(x, 2 * -lag) * _roll(x, -lag) * x)[0:(n - 2 * lag)])

# c3(df["Close"])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-> In Package
def binned_entropy(x, max_bins=10):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    hist, bin_edges = np.histogram(x, bins=max_bins)
    probs = hist / x.size
    return - np.sum(p * np.math.log(p) for p in probs if p != 0)

# binned_entropy(df["Close"])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


svd_param = [{"Tau": ta, "DE": de}
                      for ta in [4] 
                      for de in [3,6]]
                      
def _embed_seq(X,Tau,D):
  N =len(X)
  if D * Tau > N:
      print("Cannot build such a matrix, because D * Tau > N")
      exit()
  if Tau<1:
      print("Tau has to be at least 1")
      exit()
  Y= np.zeros((N - (D - 1) * Tau, D))

  for i in range(0, N - (D - 1) * Tau):
      for j in range(0, D):
          Y[i][j] = X[i + j * Tau]
  return Y                     

@set_property("fctype", "combiner")
@set_property("custom", True)
def svd_entropy(epochs, param=svd_param):
    axis=0
    
    final = []
    for par in param:

      def svd_entropy_1d(X, Tau, DE):
          Y = _embed_seq(X, Tau, DE)
          W = np.linalg.svd(Y, compute_uv=0)
          W /= sum(W)  # normalize singular values
          return -1 * np.sum(W * np.log(W))

      Tau = par["Tau"]
      DE = par["DE"]

      final.append(np.apply_along_axis(svd_entropy_1d, axis, epochs, Tau, DE).ravel()[0])


    return [("Tau_\"{}\"__De_{}\"".format(par["Tau"], par["DE"]), final[en]) for en, par in enumerate(param)]

# svd_entropy(df["Close"].values)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def _hjorth_mobility(epochs):
    diff = np.diff(epochs, axis=0)
    sigma0 = np.std(epochs, axis=0)
    sigma1 = np.std(diff, axis=0)
    return np.divide(sigma1, sigma0)

@set_property("fctype", "simple")
@set_property("custom", True)
def hjorth_complexity(epochs):
    diff1 = np.diff(epochs, axis=0)
    diff2 = np.diff(diff1, axis=0)
    sigma1 = np.std(diff1, axis=0)
    sigma2 = np.std(diff2, axis=0)
    return np.divide(np.divide(sigma2, sigma1), _hjorth_mobility(epochs))

# hjorth_complexity(df["Close"])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-> In Package
def _estimate_friedrich_coefficients(x, m, r):
    assert m > 0, "Order of polynomial need to be positive integer, found {}".format(m)
    df = pd.DataFrame({'signal': x[:-1], 'delta': np.diff(x)})
    try:
        df['quantiles'] = pd.qcut(df.signal, r)
    except ValueError:
        return [np.NaN] * (m + 1)

    quantiles = df.groupby('quantiles')

    result = pd.DataFrame({'x_mean': quantiles.signal.mean(), 'y_mean': quantiles.delta.mean()})
    result.dropna(inplace=True)

    try:
        return np.polyfit(result.x_mean, result.y_mean, deg=m)
    except (np.linalg.LinAlgError, ValueError):
        return [np.NaN] * (m + 1)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def max_langevin_fixed_point(x, r=3, m=30):
    coeff = _estimate_friedrich_coefficients(x, m, r)

    try:
        max_fixed_point = np.max(np.real(np.roots(coeff)))
    except (np.linalg.LinAlgError, ValueError):
        return np.nan

    return max_fixed_point

# max_langevin_fixed_point(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

will_param = [ka for ka in [0.2,3]]

@set_property("fctype", "combiner")
@set_property("custom", True)
def willison_amplitude(X, param=will_param):
  return [("Thresh_{}".format(n),np.sum(np.abs(np.diff(X)) >= n)) for n in param]

# willison_amplitude(df["Close"])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

perc_param = [{"base":ba, "exponent":exp} for ba in [3,5] for exp in [-0.1,-0.2]]

@set_property("fctype", "combiner")
@set_property("custom", True)
def percent_amplitude(x, param =perc_param):
    final = []
    for par in param:
      linear_scale_data = par["base"] ** (par["exponent"] * x)
      y_max = np.max(linear_scale_data)
      y_min = np.min(linear_scale_data)
      y_med = np.median(linear_scale_data)
      final.append(max(abs((y_max - y_med) / y_med), abs((y_med - y_min) / y_med)))

    return [("Base_{}__Exp{}".format(pa["base"],pa["exponent"]),fin) for fin, pa in zip(final,param)]

# percent_amplitude(df["Close"])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-> fixes required



cad_param = [0.1,1000, -234]

@set_property("fctype", "combiner")
@set_property("custom", True)
def cad_prob(cads, param=cad_param):
    return [("time_{}".format(time), stats.percentileofscore(cads, float(time) / (24.0 * 60.0)) / 100.0) for time in param]
    
# cad_prob(df["Close"])


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

zero_param = [0.01, 8]

@set_property("fctype", "combiner")
@set_property("custom", True)
def zero_crossing_derivative(epochs, param=zero_param):
    diff = np.diff(epochs)
    norm = diff-diff.mean()
    return [("e_{}".format(e), np.apply_along_axis(lambda epoch: np.sum(((epoch[:-5] <= e) & (epoch[5:] > e))), 0, norm).ravel()[0]) for e in param]

# zero_crossing_derivative(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@set_property("fctype", "simple")
@set_property("custom", True)
def detrended_fluctuation_analysis(epochs):
    def dfa_1d(X, Ave=None, L=None):
        X = np.array(X)

        if Ave is None:
            Ave = np.mean(X)

        Y = np.cumsum(X)
        Y -= Ave

        if L is None:
            L = np.floor(len(X) * 1 / (
                    2 ** np.array(list(range(1, int(np.log2(len(X))) - 4))))
                            )
            
        F = np.zeros(len(L))  # F(n) of different given box length n

        for i in range(0, len(L)):
            n = int(L[i])  # for each box length L[i]
            if n == 0:
                print("time series is too short while the box length is too big")
                print("abort")
                exit()
            for j in range(0, len(X), n):  # for each box
                if j + n < len(X):
                    c = list(range(j, j + n))
                    # coordinates of time in the box
                    c = np.vstack([c, np.ones(n)]).T
                    # the value of data in the box
                    y = Y[j:j + n]
                    # add residue in this box
                    F[i] += np.linalg.lstsq(c, y, rcond=None)[1]
            F[i] /= ((len(X) / n) * n)
        F = np.sqrt(F)

        stacked = np.vstack([np.log(L), np.ones(len(L))])
        stacked_t = stacked.T
        Alpha = np.linalg.lstsq(stacked_t, np.log(F), rcond=None)

        return Alpha[0][0]

    return np.apply_along_axis(dfa_1d, 0, epochs).ravel()[0]

# detrended_fluctuation_analysis(df["Close"])
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def _embed_seq(X, Tau, D):
    
    shape = (X.size - Tau * (D - 1), D)
    strides = (X.itemsize, Tau * X.itemsize)
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

fisher_param = [{"Tau":ta, "DE":de} for ta in [3,15] for de in [10,5]]

@set_property("fctype", "combiner")
@set_property("custom", True)
def fisher_information(epochs, param=fisher_param):
    def fisher_info_1d(a, tau, de):
        # taken from pyeeg improvements

        mat = _embed_seq(a, tau, de)
        W = np.linalg.svd(mat, compute_uv=False)
        W /= sum(W)  # normalize singular values
        FI_v = (W[1:] - W[:-1]) ** 2 / W[:-1]
        return np.sum(FI_v)

    return [("Tau_{}__DE_{}".format(par["Tau"], par["DE"]),np.apply_along_axis(fisher_info_1d, 0, epochs, par["Tau"], par["DE"]).ravel()[0]) for par in param]

# fisher_information(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

hig_param = [{"Kmax": 3},{"Kmax": 5}]

@set_property("fctype", "combiner")
@set_property("custom", True)
def higuchi_fractal_dimension(epochs, param=hig_param):
    def hfd_1d(X, Kmax):
        
        L = []
        x = []
        N = len(X)
        for k in range(1, Kmax):
            Lk = []
            for m in range(0, k):
                Lmk = 0
                for i in range(1, int(np.floor((N - m) / k))):
                    Lmk += abs(X[m + i * k] - X[m + i * k - k])
                Lmk = Lmk * (N - 1) / np.floor((N - m) / float(k)) / k
                Lk.append(Lmk)
            L.append(np.log(np.mean(Lk)))
            x.append([np.log(float(1) / k), 1])

        (p, r1, r2, s) = np.linalg.lstsq(x, L, rcond=None)
        return p[0]
    
    return [("Kmax_{}".format(config["Kmax"]), np.apply_along_axis(hfd_1d, 0, epochs, config["Kmax"]).ravel()[0] ) for  config in param]
    
# higuchi_fractal_dimension(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@set_property("fctype", "simple")
@set_property("custom", True)
def petrosian_fractal_dimension(epochs):
    def pfd_1d(X, D=None):
        # taken from pyeeg
        """Compute Petrosian Fractal Dimension of a time series from either two
        cases below:
            1. X, the time series of type list (default)
            2. D, the first order differential sequence of X (if D is provided,
               recommended to speed up)
        In case 1, D is computed using Numpy's difference function.
        To speed up, it is recommended to compute D before calling this function
        because D may also be used by other functions whereas computing it here
        again will slow down.
        """
        if D is None:
            D = np.diff(X)
            D = D.tolist()
        N_delta = 0  # number of sign changes in derivative of the signal
        for i in range(1, len(D)):
            if D[i] * D[i - 1] < 0:
                N_delta += 1
        n = len(X)
        return np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * N_delta))
    return np.apply_along_axis(pfd_1d, 0, epochs).ravel()[0]

# petrosian_fractal_dimension(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@set_property("fctype", "simple")
@set_property("custom", True)
def hurst_exponent(epochs):
    def hurst_1d(X):

        X = np.array(X)
        N = X.size
        T = np.arange(1, N + 1)
        Y = np.cumsum(X)
        Ave_T = Y / T

        S_T = np.zeros(N)
        R_T = np.zeros(N)
        for i in range(N):
            S_T[i] = np.std(X[:i + 1])
            X_T = Y - T * Ave_T[i]
            R_T[i] = np.ptp(X_T[:i + 1])

        for i in range(1, len(S_T)):
            if np.diff(S_T)[i - 1] != 0:
                break
        for j in range(1, len(R_T)):
            if np.diff(R_T)[j - 1] != 0:
                break
        k = max(i, j)
        assert k < 10, "rethink it!"

        R_S = R_T[k:] / S_T[k:]
        R_S = np.log(R_S)

        n = np.log(T)[k:]
        A = np.column_stack((n, np.ones(n.size)))
        [m, c] = np.linalg.lstsq(A, R_S, rcond=None)[0]
        H = m
        return H
    return np.apply_along_axis(hurst_1d, 0, epochs).ravel()[0]

# hurst_exponent(df["Close"])
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def _embed_seq(X, Tau, D):
    shape = (X.size - Tau * (D - 1), D)
    strides = (X.itemsize, Tau * X.itemsize)
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

lyaup_param = [{"Tau":4, "n":3, "T":10, "fs":9},{"Tau":8, "n":7, "T":15, "fs":6}]

@set_property("fctype", "combiner")
@set_property("custom", True)
def largest_lyauponov_exponent(epochs, param=lyaup_param):
    def LLE_1d(x, tau, n, T, fs):

        Em = _embed_seq(x, tau, n)
        M = len(Em)
        A = np.tile(Em, (len(Em), 1, 1))
        B = np.transpose(A, [1, 0, 2])
        square_dists = (A - B) ** 2  # square_dists[i,j,k] = (Em[i][k]-Em[j][k])^2
        D = np.sqrt(square_dists[:, :, :].sum(axis=2))  # D[i,j] = ||Em[i]-Em[j]||_2

        # Exclude elements within T of the diagonal
        band = np.tri(D.shape[0], k=T) - np.tri(D.shape[0], k=-T - 1)
        band[band == 1] = np.inf
        neighbors = (D + band).argmin(axis=0)  # nearest neighbors more than T steps away

        # in_bounds[i,j] = (i+j <= M-1 and i+neighbors[j] <= M-1)
        inc = np.tile(np.arange(M), (M, 1))
        row_inds = (np.tile(np.arange(M), (M, 1)).T + inc)
        col_inds = (np.tile(neighbors, (M, 1)) + inc.T)
        in_bounds = np.logical_and(row_inds <= M - 1, col_inds <= M - 1)
        # Uncomment for old (miscounted) version
        # in_bounds = numpy.logical_and(row_inds < M - 1, col_inds < M - 1)
        row_inds[~in_bounds] = 0
        col_inds[~in_bounds] = 0

        # neighbor_dists[i,j] = ||Em[i+j]-Em[i+neighbors[j]]||_2
        neighbor_dists = np.ma.MaskedArray(D[row_inds, col_inds], ~in_bounds)
        J = (~neighbor_dists.mask).sum(axis=1)  # number of in-bounds indices by row
        # Set invalid (zero) values to 1; log(1) = 0 so sum is unchanged

        neighbor_dists[neighbor_dists == 0] = 1

        # !!! this fixes the divide by zero in log error !!!
        neighbor_dists.data[neighbor_dists.data == 0] = 1

        d_ij = np.sum(np.log(neighbor_dists.data), axis=1)
        mean_d = d_ij[J > 0] / J[J > 0]

        x = np.arange(len(mean_d))
        X = np.vstack((x, np.ones(len(mean_d)))).T
        [m, c] = np.linalg.lstsq(X, mean_d, rcond=None)[0]
        Lexp = fs * m
        return Lexp

    return [("Tau_{}__n_{}__T_{}__fs_{}".format(par["Tau"], par["n"], par["T"], par["fs"]), np.apply_along_axis(LLE_1d, 0, epochs, par["Tau"], par["n"], par["T"], par["fs"]).ravel()[0]) for par in param]
  
# largest_lyauponov_exponent(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


whelch_param = [100,200]

@set_property("fctype", "combiner")
@set_property("custom", True)
def whelch_method(data, param=whelch_param):

  final = []
  for Fs in param:
    f, pxx = signal.welch(data, fs=Fs, nperseg=1024)
    d = {'psd': pxx, 'freqs': f}
    df = pd.DataFrame(data=d)
    dfs = df.sort_values(['psd'], ascending=False)
    rows = dfs.iloc[:10]
    final.append(rows['freqs'].mean())
  
  return [("Fs_{}".format(pa),fin) for pa, fin in zip(param,final)]

# whelch_method(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-> Basically same as above
freq_param = [{"fs":50, "sel":15},{"fs":200, "sel":20}]

@set_property("fctype", "combiner")
@set_property("custom", True)
def find_freq(serie, param=freq_param):

    final = []
    for par in param:
      fft0 = np.fft.rfft(serie*np.hanning(len(serie)))
      freqs = np.fft.rfftfreq(len(serie), d=1.0/par["fs"])
      fftmod = np.array([np.sqrt(fft0[i].real**2 + fft0[i].imag**2) for i in range(0, len(fft0))])
      d = {'fft': fftmod, 'freq': freqs}
      df = pd.DataFrame(d)
      hop = df.sort_values(['fft'], ascending=False)
      rows = hop.iloc[:par["sel"]]
      final.append(rows['freq'].mean())

    return [("Fs_{}__sel{}".format(pa["fs"],pa["sel"]),fin) for pa, fin in zip(param,final)]

# find_freq(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-> In Package

def flux_perc(magnitude):
    sorted_data = np.sort(magnitude)
    lc_length = len(sorted_data)

    F_60_index = int(math.ceil(0.60 * lc_length))
    F_40_index = int(math.ceil(0.40 * lc_length))
    F_5_index = int(math.ceil(0.05 * lc_length))
    F_95_index = int(math.ceil(0.95 * lc_length))

    F_40_60 = sorted_data[F_60_index] - sorted_data[F_40_index]
    F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
    F_mid20 = F_40_60 / F_5_95

    return {"FluxPercentileRatioMid20": F_mid20}

# flux_perc(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@set_property("fctype", "simple")
@set_property("custom", True)
def range_cum_s(magnitude):
    sigma = np.std(magnitude)
    N = len(magnitude)
    m = np.mean(magnitude)
    s = np.cumsum(magnitude - m) * 1.0 / (N * sigma)
    R = np.max(s) - np.min(s)
    return {"Rcs": R}

# range_cum_s(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


struct_param = {"Volume":None, "Open": None}

@set_property("fctype", "combiner")
@set_property("custom", True)
def structure_func(time, param=struct_param):

      dict_final = {}
      for key, magnitude in param.items():
        dict_final[key] = []
        Nsf, Np = 100, 100
        sf1, sf2, sf3 = np.zeros(Nsf), np.zeros(Nsf), np.zeros(Nsf)
        f = interp1d(time, magnitude)

        time_int = np.linspace(np.min(time), np.max(time), Np)
        mag_int = f(time_int)

        for tau in np.arange(1, Nsf):
            sf1[tau - 1] = np.mean(
                np.power(np.abs(mag_int[0:Np - tau] - mag_int[tau:Np]), 1.0))
            sf2[tau - 1] = np.mean(
                np.abs(np.power(
                    np.abs(mag_int[0:Np - tau] - mag_int[tau:Np]), 2.0)))
            sf3[tau - 1] = np.mean(
                np.abs(np.power(
                    np.abs(mag_int[0:Np - tau] - mag_int[tau:Np]), 3.0)))
        sf1_log = np.log10(np.trim_zeros(sf1))
        sf2_log = np.log10(np.trim_zeros(sf2))
        sf3_log = np.log10(np.trim_zeros(sf3))

        if len(sf1_log) and len(sf2_log):
            m_21, b_21 = np.polyfit(sf1_log, sf2_log, 1)
        else:

            m_21 = np.nan

        if len(sf1_log) and len(sf3_log):
            m_31, b_31 = np.polyfit(sf1_log, sf3_log, 1)
        else:

            m_31 = np.nan

        if len(sf2_log) and len(sf3_log):
            m_32, b_32 = np.polyfit(sf2_log, sf3_log, 1)
        else:

            m_32 = np.nan
        dict_final[key].append(m_21)
        dict_final[key].append(m_31)
        dict_final[key].append(m_32)

      return [("StructureFunction_{}__m_{}".format(key, name), li)  for key, lis in dict_final.items() for name, li in zip([21,31,32], lis)]

# structure_func(df["Close"], struct_param)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#-> In Package
def kurtosis(x):

    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.kurtosis(x)

# kurtosis(df["Close"])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@set_property("fctype", "simple")
@set_property("custom", True)
def stetson_k(x):
    """A robust kurtosis statistic."""
    n = len(x)
    x0 = stetson_mean(x, 1./20**2)
    delta_x = np.sqrt(n / (n - 1.)) * (x - x0) / 20
    ta = 1. / 0.798 * np.mean(np.abs(delta_x)) / np.sqrt(np.mean(delta_x**2))
    return ta
  
# stetson_k(df["Close"])