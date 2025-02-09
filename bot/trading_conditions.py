import pandas as pd


class TradeConditions:
    @staticmethod
    def over_lower_bb(stock_data: pd.DataFrame):
        return stock_data["Close"] >= stock_data["Lower_BB"]

    @staticmethod
    def under_lower_bb(stock_data: pd.DataFrame):
        return stock_data["Close"] < stock_data["Lower_BB"]

    @staticmethod
    def rsi_trending_up(stock_data: pd.DataFrame):
        average = stock_data["rsi"].rolling(window=30).mean()
        return (stock_data["rsi"] >= 30) & (stock_data["rsi"] > average)

    @staticmethod
    def crossing_50_through_200(stock_data: pd.DataFrame):
        # bearish signal
        return (stock_data["50_MA"].shift() > stock_data["200_MA"].shift()) & (
            stock_data["50_MA"] <= stock_data["200_MA"]
        )

    @staticmethod
    def crossing_200_through_50(stock_data: pd.DataFrame):
        # bullish signal
        return (stock_data["50_MA"].shift() > stock_data["200_MA"].shift()) & (
            stock_data["50_MA"] <= stock_data["200_MA"]
        )

    @staticmethod
    def overbought_rsi_bb(stock_data: pd.DataFrame):
        return (stock_data["Close"] > stock_data["Upper_BB"]) & (
            stock_data["rsi"] > 70
        )

    @staticmethod
    def oversold_rsi_bb(stock_data: pd.DataFrame):
        return (stock_data["Center_BB_Div"] <= -50) & (stock_data["rsi"] < 35)

    @staticmethod
    def low_below_center_bb(stock_data: pd.DataFrame):
        return stock_data["Low"] < stock_data["Center_BB"]

    @staticmethod
    def low_out_of_bb(stock_data: pd.DataFrame):
        return stock_data["Low"] < stock_data["Lower_BB"]

    @staticmethod
    def new_month_high(stock_data: pd.DataFrame, window: int = 30):
        # new maximum, but following value is lower
        rolling_max = stock_data["Close"].rolling(window=window).max()
        return (stock_data["Close"].shift() > rolling_max.shift(2)) & (
            stock_data["Close"] < stock_data["Close"].shift()
        )

    @staticmethod
    def over_window_ma(stock_data: pd.DataFrame, window):
        return stock_data["Close"] > stock_data[f"{window}_MA"]

    @staticmethod
    def bb_reversal(stock_data: pd.DataFrame):
        return (
            (stock_data["Open"].shift() < stock_data["Lower_BB"].shift())
            | (stock_data["Close"].shift() < stock_data["Lower_BB"].shift())
        ) & (stock_data["Close"] > stock_data["Lower_BB"])

    @staticmethod
    def new_month_low(stock_data: pd.DataFrame, window: int = 30):
        # new minimum, but following value is higher
        rolling_min = stock_data["Close"].rolling(window=window).min()
        return (stock_data["Close"].shift() < rolling_min.shift(2)) & (
            stock_data["Close"] > stock_data["Close"].shift()
        )

    @staticmethod
    def low_vix_and_falling(stock_data: pd.DataFrame, vix_data: pd.DataFrame):
        stock_avg = stock_data["Close"].rolling(window=14).mean()
        return (stock_data["Close"] < stock_avg) & (vix_data["Close"] < 17)

    @staticmethod
    def close_roughly_equal_low(
        stock_data: pd.DataFrame, vix_data: pd.DataFrame
    ):
        candle_breadth = stock_data["High"] - stock_data["Low"]
        lower_stick = stock_data["Close"] - stock_data["Low"]
        return (
            (lower_stick.shift() < 1)
            | (lower_stick.shift() / candle_breadth.shift() < 0.1)
        ) & (vix_data["Close"].shift() < vix_data["Close"])

    @staticmethod
    def crossing_200ma_down(stock_data: pd.DataFrame):
        volume_avg = (
            stock_data["Volume"].rolling(window=30).mean()
        )  # 30-day moving average
        return (
            (stock_data["Close"].shift() > stock_data["200_MA"].shift())
            & (stock_data["Close"] < stock_data["200_MA"])
            & (stock_data["Volume"] > volume_avg)
        )

    @staticmethod
    def above_200ma(stock_data: pd.DataFrame):
        return stock_data["Close"] > stock_data["200_MA"]

    @staticmethod
    def above_avg_volume(stock_data: pd.DataFrame):
        avg_volume = stock_data["Volume"].rolling(window=50).mean()
        return stock_data["Volume"] > avg_volume
