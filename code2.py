import openmeteo_requests
import requests_cache
import pandas as pd 
import numpy as np 
from retry_requests import retry 
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from matplotlib.patches import Patch
import calendar

os.makedirs("results", exist_ok=True)

#Part 2.3: Data preparation

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

#Part 2.4: Stochastic model of temperature 

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

def fit_seasonal_volatility(epsilon, t_train, w=2*np.pi/365.25):
    rolling_vol = pd.Series(epsilon).rolling(window=31, center=True).std()
    mask = ~np.isnan(rolling_vol)
    t_clean = t_train[1:][mask]
    v_clean = rolling_vol[mask].values
    X = np.column_stack([np.ones(len(t_clean)), np.cos(w * t_clean), np.sin(w * t_clean)])
    params = np.linalg.lstsq(X, v_clean, rcond=None)[0]
    return params

def compute_seasonal_vol(t, params, w=2*np.pi/365.25):
    a, b, c = params
    return np.clip(a + b*np.cos(w*t) + c*np.sin(w*t), 1e-4, None)

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

#Part 2.5: Pricing of weather derivatives
    
def calculate_cat(temperatures):
    return np.sum(temperatures)

def calculate_hdd(temperatures, threshold = 18):
    return np.sum(np.maximum(threshold-temperatures,0))

def calculate_cat_payoff(temperatures, N, K):
    return N * max(calculate_cat(temperatures) - K, 0)

def calculate_hdd_payoff(temperatures, N, K):
    return N * max(calculate_hdd(temperatures) - K, 0)

def asian_option_payoff(day_degree_index, cap, floor, alpha, beta, strike, call_strike, put_strike, ):
    """
    Accumulated day_degree_index: CAT or HDD summed over the contract period.
    K, K1, K2: strike value minimimumal, then K1 - strike value for call leg, and K2 - strike for put leg of the collar.
    Alpha, Beta: dollar amount per degree day
    """
    call_option_cap = min(alpha*max(day_degree_index - strike,0), cap)
    put_option_floor = min(alpha*max(strike - day_degree_index,0), floor)
    collar = min(alpha * max(day_degree_index - call_strike, 0), cap) - min(beta * max(put_strike - day_degree_index,0), floor)
    return call_option_cap, put_option_floor, collar

def simulate_mc_paths(M, n_days, phi, sigma_t, mu_t, X0=0):
    paths = np.zeros((M, n_days))
    residuals = np.zeros((M, n_days))

    residuals[:, 0] = X0
    for t in range(1, n_days):
        z = np.random.normal(0, 1, M)

        residuals[:, t] = (phi * residuals[:, t-1] + sigma_t[t] * z)

    paths = mu_t[:n_days] + residuals

    return paths

def compute_index_payoffs(paths, N, cat_strike=None, hdd_strike=None):
    cat_payoffs = []
    hdd_payoffs = []

    for path in paths:
        if cat_strike is not None:
            index = calculate_cat(path)
            cat_payoffs.append(N * max(index - cat_strike, 0))

        if hdd_strike is not None:
            index = calculate_hdd(path)
            hdd_payoffs.append(N * max(index - hdd_strike, 0))

    return np.array(cat_payoffs), np.array(hdd_payoffs)

def monte_carlo_price(payoffs, r, tau):
    price = np.exp(-r * tau) * np.mean(payoffs)
    return price

def plot_simulated_paths(paths, mu_t, n_plot=30):
    """
    Plot a subset of simulated temperature paths together with the deterministic mean.
    """
    M, n_days = paths.shape

    # randomly select paths
    idx = np.random.choice(M, size=n_plot, replace=False)

    plt.figure(figsize=(12,6))

    for i in idx:
        plt.plot(paths[i], alpha=0.35, linewidth=0.5)

    # deterministic mean
    plt.plot(mu_t[:n_days], color="black", linewidth=1.5, label="Deterministic mean")

    plt.title("Simulated Temperature Paths")
    plt.xlabel("Day of Contract")
    plt.ylabel("Temperature (°C)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/simulated_temperature_paths.png", dpi=150)
    plt.show()

#Path 2.6: Request for Quote price - simulate many MC paths, and compute prices for each contract in the quote 
    
def quote_price_engine(contracts_df, M, phi, sigma_t_full, beta_hat, r, X0, t0):
    """
    Prices of all contracts in the contract specifications using Monte Carlo simulation.
    Returns a DataFrame with contract_id and Monte Carlo price.
    """

    prices = []

    for idx, row in contracts_df.iterrows():
        contract_id = row['contract_id']
        contract_type = row['contract_type']
        start_date = pd.to_datetime(row['start_date'])
        end_date = pd.to_datetime(row['end_date'])
        strike = row['strike']
        N = row['notional']
        
        #number of days in contract
        n_days = (end_date - start_date).days + 1
        #create future t index
        t_future = np.arange(t0, t0 + n_days)
        mu_future = compute_seasonal_mean(t_future, beta_hat)
        sigma_future = compute_seasonal_vol(t_future, sigma_t_full)

        paths = simulate_mc_paths(M, n_days, phi, sigma_future, mu_future, X0)

        #compute payoffs depending on contract type
        if 'CALL' in contract_type:
            if 'CAT' in contract_type:
                payoffs = np.array([calculate_cat_payoff(path, N, strike) for path in paths])
            elif 'HDD' in contract_type:
                payoffs = np.array([calculate_hdd_payoff(path, N, strike) for path in paths])
            else:
                raise ValueError(f"Unknown call contract type: {contract_type}")
        elif 'FUT' in contract_type:
            # futures: payoff is just the total index multiplied by notional
            if 'CAT' in contract_type:
                payoffs = np.array([N * calculate_cat(path) for path in paths])
            elif 'HDD' in contract_type:
                payoffs = np.array([N * calculate_hdd(path) for path in paths])
            else:
                raise ValueError(f"Unknown future contract type: {contract_type}")
        else:
            raise ValueError(f"Unknown contract type: {contract_type}")

        # discount payoffs to valuation date
        tau_days = (pd.to_datetime(row['valuation_date']) - start_date).days
        tau = tau_days / 365.0  # convert to years
        discounted_price = np.exp(-r * tau) * np.mean(payoffs)

        prices.append({
            'contract_id': contract_id,
            'price': discounted_price,
            'payoffs': payoffs  # store for plotting
        })

    return pd.DataFrame(prices)

def plot_pricing_summary_table(priced_contracts):
    """
    Saves a clean HTML table (readable with 150+ rows) and also returns the summary DataFrame.
    """
    rows = []
    for _, row in priced_contracts.iterrows():
        p = row['payoffs']
        ctype = 'CAT' if 'CAT' in row['contract_id'] else 'HDD'
        rows.append({
            'Contract':       row['contract_id'],
            'Type':           ctype,
            'MC Price':       round(float(row['price']), 2),
            'Mean Payoff':    round(float(np.mean(p)), 2),
            'Std Dev':        round(float(np.std(p)), 2),
            '95th Pct':       round(float(np.percentile(p, 95)), 2),
            'P(payoff > 0)':  f"{(p > 0).mean()*100:.1f}%"
        })

    summary_df = pd.DataFrame(rows)

    html_rows = ""
    for i, r in enumerate(rows):
        bg      = "#f5f9ff" if r['Type'] == 'CAT' else "#fffaf0"
        stripe  = "" if i % 2 == 0 else "filter:brightness(0.96)"
        type_color = "#0C447C" if r['Type'] == 'CAT' else "#633806"
        type_bg    = "#E6F1FB" if r['Type'] == 'CAT' else "#FAEEDA"
        html_rows += f"""
        <tr style="background:{bg};{stripe}">
            <td>{r['Contract']}</td>
            <td><span style="background:{type_bg};color:{type_color};padding:2px 8px;
                border-radius:4px;font-size:11px;font-weight:600">{r['Type']}</span></td>
            <td>€{r['MC Price']:,.2f}</td>
            <td>€{r['Mean Payoff']:,.2f}</td>
            <td>€{r['Std Dev']:,.2f}</td>
            <td>€{r['95th Pct']:,.2f}</td>
            <td>{r['P(payoff > 0)']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Contract Pricing Summary</title>
<style>
  body {{ font-family: -apple-system, sans-serif; padding: 24px; background: #fafafa; }}
  h2   {{ font-size: 18px; font-weight: 500; margin-bottom: 16px; color: #222; }}
  table {{ border-collapse: collapse; width: 100%; background: white;
           border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  th   {{ background: #2C2C2A; color: white; font-size: 12px; font-weight: 500;
          padding: 10px 12px; text-align: left; }}
  td   {{ padding: 8px 12px; font-size: 12px; color: #333;
          border-bottom: 0.5px solid #e8e8e8; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(0,0,0,0.03) !important; }}
</style>
</head>
<body>
<h2>Contract Pricing Summary — {len(rows)} contracts</h2>
<table>
  <thead>
    <tr>
      <th>Contract</th><th>Type</th><th>MC Price</th>
      <th>Mean Payoff</th><th>Std Dev</th><th>95th Pct</th><th>P(payoff &gt; 0)</th>
    </tr>
  </thead>
  <tbody>{html_rows}</tbody>
</table>
</body>
</html>"""

    path = "results/pricing_summary_table.html"
    with open(path, "w") as f:
        f.write(html)
    print(f"Summary table saved to {path} — open in browser to view.")

    # also print to terminal
    print(summary_df.to_string(index=False))
    return summary_df

def plot_payoff_boxplots(priced_contracts):
    ids = priced_contracts['contract_id'].tolist()
    all_payoffs = [row['payoffs'] for _, row in priced_contracts.iterrows()]
    types = ['CAT' if 'CAT' in cid else 'HDD' for cid in ids]
    colors = ['#378ADD' if t == 'CAT' else '#BA7517' for t in types]

    fig, ax = plt.subplots(figsize=(22, 7))

    bp = ax.boxplot(
        all_payoffs,
        patch_artist=True,
        medianprops=dict(color='white', linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=dict(marker='o', markersize=2, alpha=0.3, linestyle='none'),
        widths=0.7
    )

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    for flier, color in zip(bp['fliers'], colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    ax.set_xticks(range(1, len(ids) + 1))
    ax.set_xticklabels(ids, rotation=90, ha='right', fontsize=5)
    ax.set_title('Payoff Distribution by Contract', fontsize=13)
    ax.set_xlabel('Contract')
    ax.set_ylabel('Payoff (€)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(handles=[Patch(facecolor='#378ADD', alpha=0.75, label='CAT'),
                       Patch(facecolor='#BA7517', alpha=0.75, label='HDD')], fontsize=10)

    plt.tight_layout()
    plt.savefig("results/payoff_boxplots.png", dpi=150)
    plt.show()

def evaluate_model(phi, u_test, y_test, residuals_train, vol_params, t_test):
    """
    Computes RMSE_T and RMSE_sigma on the test set.
    """
    residuals_test = y_test - u_test

    # one-day-ahead forecast: mu(u) + phi * X(u-1)
    # X(u-1) is the lagged actual residual, starting from last training residual
    X_lag = np.concatenate([[residuals_train[-1]], residuals_test[:-1]])
    T_hat = u_test + phi * X_lag
    rmse_T = np.sqrt(np.mean((T_hat - y_test) ** 2))

    # empirical volatility on test set: rolling std of test innovations
    epsilon_test = residuals_test[1:] - phi * residuals_test[:-1]
    empirical_vol = pd.Series(epsilon_test).rolling(window=31, center=True).std().bfill().ffill().values

    sigma_fitted = compute_seasonal_vol(t_test[1:], vol_params)

    rmse_sigma = np.sqrt(np.mean((sigma_fitted - empirical_vol) ** 2))

    print(f"RMSE_T: {rmse_T:.4f} °C")
    print(f"RMSE_sigma: {rmse_sigma:.4f} °C")

    return rmse_T, rmse_sigma

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

    #Part 2.5: Pricing of Weather Derivatives 

    vol_params = fit_seasonal_volatility(epsilon, t_train)

    # Visualize simulated temperature paths
    M = 30
    n_days = 365 * 2
    t_future = np.arange(len(daily_dataframe_cleaned),
                         len(daily_dataframe_cleaned) + n_days)
    mu_future = compute_seasonal_mean(t_future, beta_hat)
    sigma_future = compute_seasonal_vol(t_future, vol_params)
    X0 = residuals_train[-1]

    paths = simulate_mc_paths(M, n_days, phi, sigma_future, mu_future, X0)
    plot_simulated_paths(paths, mu_future)

    # Part 2.6: Request for Quote price
    contracts_df = pd.read_csv("contract_specifications.csv")
    M_mc = 10000
    r = contracts_df['rate'].iloc[0]

    priced_contracts = quote_price_engine(
        contracts_df,
        M=M_mc,
        phi=phi,
        sigma_t_full=vol_params,
        beta_hat=beta_hat,
        r=r,
        X0=X0,
        t0 = len(daily_dataframe_cleaned)
    )

    # Print prices
    print(priced_contracts[['contract_id', 'price']])

    # Plots
    plot_payoff_boxplots(priced_contracts)
    summary = plot_pricing_summary_table(priced_contracts)
    print(summary.to_string(index=False))

    #Part 2.7: Evaluation Protocol 

    evaluate_model(
        phi=phi,
        u_test=u_test,
        y_test=y_test,
        residuals_train=residuals_train,
        vol_params=vol_params,
        t_test=t_test
    )
    
if __name__ == "__main__":
    main()