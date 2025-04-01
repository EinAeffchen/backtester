import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from statsmodels.tsa.stattools import coint
from pykalman import KalmanFilter
from sklearn.preprocessing import StandardScaler


# Step 1: Download Stock Data
def get_stock_data(tickers, start="2023-01-01", end="2024-01-01"):
    data = yf.download(tickers, start=start, end=end)
    data = data["Close"]
    return data


# Step 2: Hierarchical Clustering for Pair Selection
def cluster_pairs(data, num_clusters=5):
    returns = data.pct_change().dropna()
    scaler = StandardScaler()
    normalized_returns = scaler.fit_transform(returns.T)
    linkage_matrix = linkage(normalized_returns, method="ward")
    clusters = fcluster(linkage_matrix, num_clusters, criterion="maxclust")
    cluster_dict = {}
    for i, ticker in enumerate(data.columns):
        cluster_dict.setdefault(clusters[i], []).append(ticker)
    return cluster_dict


# Step 3: Cointegration Test to Select Best Pair
def find_cointegrated_pair(cluster, data):
    best_pval = 1
    best_pair = None
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            pval = coint(data[cluster[i]], data[cluster[j]])[1]
            if pval < best_pval:
                best_pval = pval
                best_pair = (cluster[i], cluster[j])
    return best_pair if best_pval < 0.05 else None


# Step 4: Kalman Filter to Estimate Spread
def kalman_filter_spread(series1, series2):
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=2)
    kf = kf.em(np.column_stack([series1, series2]), n_iter=10)
    state_means, _ = kf.filter(np.column_stack([series1, series2]))
    spread = series1 - state_means[:, 0] * series2
    return spread


# Step 5: Generate Trading Signals
def generate_signals(spread, z_threshold=1.5):
    mean = spread.mean()
    std = spread.std()
    z_score = (spread - mean) / std
    signals = pd.DataFrame(index=spread.index)
    signals["long"] = z_score < -z_threshold
    signals["short"] = z_score > z_threshold
    signals["exit"] = abs(z_score) < 0.5
    return signals


# Step 6: Backtest Strategy
def backtest(data, pair, signals, capital=10000):
    stock1, stock2 = pair
    positions = pd.DataFrame(index=data.index, columns=[stock1, stock2])
    capital_per_trade = capital / 2
    stock1_price, stock2_price = data[stock1], data[stock2]

    for date, row in signals.iterrows():
        if row["long"]:
            positions.loc[date] = [
                capital_per_trade / stock1_price[date],
                -capital_per_trade / stock2_price[date],
            ]
        elif row["short"]:
            positions.loc[date] = [
                -capital_per_trade / stock1_price[date],
                capital_per_trade / stock2_price[date],
            ]
        elif row["exit"]:
            positions.loc[date] = [0, 0]

    portfolio = (positions * data).sum(axis=1)
    portfolio.ffill(inplace=True)
    return portfolio


# Running the Strategy
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "^GSPC", "ES=F", "^VIX"]  # Example stocks
data = get_stock_data(tickers)
clusters = cluster_pairs(data)

# Select the best pair from the first cluster
best_pair = find_cointegrated_pair(clusters[list(clusters.keys())[0]], data)
if best_pair:
    spread = kalman_filter_spread(data[best_pair[0]], data[best_pair[1]])
    signals = generate_signals(spread)
    portfolio = backtest(data, best_pair, signals)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio, label="Portfolio Value")
    plt.title(f"Pair Trading Strategy: {best_pair[0]} & {best_pair[1]}")
    plt.legend()
    plt.show()
else:
    print("No cointegrated pair found.")
