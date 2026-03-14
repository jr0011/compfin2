import openmeteo_requests
import requests_cache
import pandas as pd 
import numpy as np 
from retry_requests import retry 
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

os.makedirs("results", exist_ok=True)

def pull_data():
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/era5"

    params = {
        "latitude": 52.37,
        "longitude": 4.89,
        "start_date": "2016-01-01",
        "end_date": "2026-02-01",
        "hourly": "temperature_2m",
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Amsterdam",
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    return response 

def hourly_data(response):
    """
    Returns pandas DataFrame with hourly temperature data.
    """

    hourly = response.Hourly()
    hourly_temp = hourly.Variables(0).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit='s', utc=True),
            end = pd.to_datetime(hourly.TimeEnd(), unit='s', utc=True),
            freq = pd.Timedelta(seconds=hourly.Interval()),
            inclusive = 'left'),
        "temperature_2m": hourly_temp
    }

    hourly_dataframe = pd.DataFrame(data=hourly_data)

    return hourly_dataframe

def daily_data(response):
    """
    Returns pandas DataFrame with daily mean temperature data.
    """
    daily = response.Daily()
    daily_temp_mean = daily.Variables(0).ValuesAsNumpy()

    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
            ),
        "temperature_2m_mean": daily_temp_mean
        }
    daily_dataframe = pd.DataFrame(data=daily_data)

    return daily_dataframe

def clean_daily_data(daily_dataframe):
    """
    Converts to ISO format. 
    Drops duplicates, keeping exactly one row per calendar day.
    Linear interpolation/flagging missing values. 
    """
    daily_dataframe["date"] = pd.to_datetime(daily_dataframe["date"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    daily_dataframe = daily_dataframe.drop_duplicates(subset=["date"], keep="first")
    
    daily_dataframe["temperature_2m_mean"] = (
        daily_dataframe["temperature_2m_mean"]
        .interpolate(method="linear", limit=2)
    )

    daily_dataframe["missing"] = daily_dataframe["temperature_2m_mean"].isnull()

    return daily_dataframe

def train_test_split(daily_dataframe_cleaned):
    """
    Splits the cleaned daily dataframe into training and testing sets.
    """
    split_date = "2025-01-01T00:00:00"
    train_df = daily_dataframe_cleaned[daily_dataframe_cleaned["date"] < split_date]
    test_df = daily_dataframe_cleaned[daily_dataframe_cleaned["date"] >= split_date]

    return train_df, test_df

#Part 2: Stochastic model of temperature 

#Task 1: Deterministic seasonal fit 

def fit_seasonal_mean(t, y, w=2*np.pi/365.25):
    """
    Fit linearized seasonal mean using OLS. Return optimized params.
    """
    X = np.column_stack([np.ones(len(t)), t, np.cos(w*t), np.sin(w*t)])
    params = np.linalg.lstsq(X, y, rcond=None)[0]
    return params

def compute_seasonal_mean(t, params, w=2*np.pi/365.25):
    a, b, a1, b1 = params
    u = a + b*t + a1*np.cos(w*t) + b1*np.sin(w*t)
    return u

def seasonal_amplitude_phase(a1, b1):
    """
    Compute amplitude and phase from a1 and b1.
    """
    alpha = np.sqrt(a1**2 + b1**2)
    theta = np.arctan2(b1, a1)
    return alpha, theta

#Task 2: Residual dynamics 

def fit_ar1_residuals(des_residuals): #fitting deseasonalized residuals 
    """
    Fit AR(1) model on deseasonalized series.
    Returns phi, kappa, innovation std, AIC, BIC, residuals.
    """
    X_curr = des_residuals[1:]
    X_lag = des_residuals[:-1]
    X_lag = sm.add_constant(X_lag)  
    model = sm.OLS(X_curr, X_lag).fit()
    
    phi = model.params[1]  # AR(1) coefficient
    kappa = -np.log(phi)
    sigma_e = np.std(model.resid, ddof=1)
    aic = model.aic
    bic = model.bic
    epsilon = model.resid
    
    return phi, kappa, sigma_e, aic, bic, epsilon

#Task 3: Seasonal volatility 

def plot_rolling_volatility(epsilon, rolling_vol, dates):
    """
    Plots innovations (epsilon) and rolling volatility on the training set.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(dates, epsilon, color='gray', linewidth=0.5, alpha=0.6, label='innovations (ε)')
    ax.plot(dates, rolling_vol, color='#1338BE', linewidth=1.5, label='rolling volatility (31-day std)')

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)

    ax.set_title('Rolling Volatility - Train Set', fontsize=13)
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (°C)')
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/rolling_volatility.png", dpi=150)
    plt.show()

def main():
    #Pull raw data, print some metadata
    response = pull_data()

    print(f"Elevation: {response.Elevation()} m")
    print(f"Timezone: {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"UTC Offset (s): {response.UtcOffsetSeconds()}")

    #Print all attributes and methods of the object - dir()
    #print(dir(response))
    #print(help(response))

    hourly_dataframe = hourly_data(response)
    daily_dataframe = daily_data(response)

    print("\nHourly data:")
    print(hourly_dataframe.head())

    print("\nDaily data:")
    print(daily_dataframe.head())

    #Cleaned daily data 
    daily_dataframe_cleaned = clean_daily_data(daily_dataframe)
    daily_dataframe_cleaned.to_csv("daily_data_cleaned.csv", index=False)

    #Train/test split
    train_df, test_df = train_test_split(daily_dataframe_cleaned)

    #Prepare train data to fit, extract parameters
    t_train = np.arange(len(train_df))
    y_train = train_df["temperature_2m_mean"].values
    beta_hat = fit_seasonal_mean(t_train, y_train)

    #Compute seasonal mean for training set with the optimized parameters, and
    u_train = compute_seasonal_mean(t_train, beta_hat)
    # compute the residuals between train prediction and actual data points (y_train)
    residuals_train = y_train - u_train

    #Predict seasonal mean for t_test set values, and compute residuals with actual test data points 
    t_test = np.arange(len(train_df), len(train_df) + len(test_df))
    y_test = test_df["temperature_2m_mean"].values
    u_test = compute_seasonal_mean(t_test, beta_hat)
    #residuals_test = y_test - u_test

    #Save the seasonal mean (u) for the full sample 
    t_full = np.arange(len(daily_dataframe_cleaned))
    y_full = daily_dataframe_cleaned["temperature_2m_mean"].values
    u_full = compute_seasonal_mean(t_full, beta_hat)
    residuals_full = y_full - u_full

    #Amplitude and phase
    alpha, theta = seasonal_amplitude_phase(beta_hat[2], beta_hat[3])

    print("Fitted mean on the full dataset:", u_full)
    print("Fitted parameters:", beta_hat)
    print("Amplitude:", alpha, "Phase:", theta)

    #Task 2: Residual dynamics on training set 

    #Use the residuals of the training set: deseasonalized: y_train - u_train 

    phi, kappa, sigma_e, aic, bic, epsilon = fit_ar1_residuals(residuals_train)
    print("AR(1) coefficient (phi) (don't have to report):", phi)
    print("Mean reversion rate (kappa) (don't have to report):", kappa)
    print("Innovation std (sigma_e):", sigma_e)
    print("AIC:", aic)
    print("BIC:", bic)

    #Task 3: Seasonal volatility 
    #Compute the fitted residual innovations (new, unexpected information that could not have been predicted and arrived today - residuals)
    print("epsilon", epsilon)

    #Rolling standard deviation (30 days)
    train_df['date'] = pd.to_datetime(train_df['date'])
    rolling_vol = pd.Series(epsilon).rolling(window=31, center=True).std()
    plot_rolling_volatility(epsilon, rolling_vol, train_df['date'].iloc[1:].values)
    
if __name__ == "__main__":
    main()