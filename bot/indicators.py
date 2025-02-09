import pandas as pd


class Indicators:
    def calculate_moving_avg_std(
        data: pd.DataFrame, column: str, window: int = 30
    ):
        data[f"{window}_MA"] = (
            data[column].rolling(window=window, min_periods=1).mean()
        )
        data[f"{window}_STD"] = (
            data[column].rolling(window=window, min_periods=1).std()
        )
        window = 200
        data[f"{window}_MA"] = (
            data[column].rolling(window=window, min_periods=1).mean()
        )
        window = 50
        data[f"{window}_MA"] = (
            data[column].rolling(window=window, min_periods=1).mean()
        )
        return data

    def identify_divergence(
        stock_data: pd.DataFrame,
        vix_data: pd.DataFrame,
    ):
        stock_data["Divergence"] = (
            stock_data["Close"].pct_change() <= 0
        ) & ~(  # S&P 500 correction
            vix_data["Close"].pct_change() <= 0
        )  # No VIX spike
        return stock_data

    def calculate_volume_indicators(data: pd.DataFrame, window: int = 30):
        window = window * 3
        data[f"vol_{window}_MA"] = (
            data["Volume"].rolling(window=window, min_periods=1).mean()
        )
        data["Volume_Deviation"] = (
            (data["Volume"] - data[f"vol_{window}_MA"])
            / data[f"vol_{window}_MA"]
        ) * 100

        return data