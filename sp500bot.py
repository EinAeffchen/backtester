import time
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pytz
import yfinance as yf
from discord import File, SyncWebhook
from pandas.tseries.holiday import USFederalHolidayCalendar
from ta.momentum import rsi
from ta.volatility import BollingerBands

p = Path(__file__).parent


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


class SP500Trader:

    def plot_entry_and_exit_points(stock_data: pd.DataFrame):
        fig = go.Figure()

        # Add closing price line
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data["Close"],
                mode="lines",
                name="Closing Price",
                line=dict(color="blue"),
            )
        )

        # Add long entry markers
        long_entry_indices = stock_data.index[stock_data["Long_Entry"]]
        long_entry_prices = stock_data["Close"][stock_data["Long_Entry"]]
        long_exit_indices = stock_data.index[
            stock_data["Long_Exit"]
        ]  # Get indices where Long_Entry is True
        long_exit_prices = stock_data["Close"][
            stock_data["Long_Exit"]
        ]  # Get corresponding Close prices

        fig.add_trace(
            go.Scatter(
                x=long_entry_indices,
                y=long_entry_prices,
                mode="markers",
                name="Long Entry Signal",
                marker=dict(color="green", size=10, symbol="circle"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=long_exit_indices,
                y=long_exit_prices,
                mode="markers",
                name="Long Exit Signal",
                marker=dict(color="red", size=10, symbol="circle"),
            )
        )

        segments = []
        current_segment = []

        for i in range(len(stock_data)):
            if stock_data["Position"].iloc[i] == 1:
                current_segment.append(
                    (stock_data.index[i], stock_data["Close"].iloc[i])
                )
            else:
                if current_segment:  # If we have a segment, store it
                    segments.append(current_segment)
                    current_segment = []

        # If the last segment didn't get added
        if current_segment:
            segments.append(current_segment)

        # Add each segment as a separate trace to avoid unwanted connections
        for segment in segments:
            x_vals, y_vals = zip(*segment)
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    name="Holding Period",
                    line=dict(
                        color="purple", width=3
                    ),  # Purple line for holding periods
                    opacity=0.8,
                )
            )
            # Update layout for better viewing
            fig.update_layout(
                title="Stock Closing Price with Long Entry Signals",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_dark",  # You can change this to "plotly_white" for a light theme
                hovermode="x unified",
                height=700,  # Increase height for better visualization
            )

        # Show the interactive plot
        fig.show()

    def fetch_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        production: bool = False,
    ) -> pd.DataFrame:
        """Fetch historical data for a given ticker."""
        if (
            not production
            and p.is_file()
            and not date.fromtimestamp(p.stat().st_mtime) < date.today()
        ):
            data = pd.read_csv(p, index_col=0)
        else:
            data = yf.download(
                ticker, start=start_date, end=end_date, prepost=True
            )
            data.columns = data.columns.droplevel(-1)
        return data

    def calculate_strategy_return(
        self, stock_data: pd.DataFrame, leverage: int = 1
    ):
        stock_data[f"Strategy_Return_lev_{leverage}"] = (
            stock_data["Position"].shift()
            * stock_data["Daily_Return"]
            * leverage
        )
        return stock_data

    def create_graph(
        self,
        leveraged_returns: dict[int, pd.Series],
        cumulative_market_return: pd.Series,
    ):
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_market_return, label="Market Return")
        for i, strategy_return in leveraged_returns.items():
            plt.plot(
                strategy_return,
                label=f"lev: {i}",
            )
        plt.title("Ivan Scherman's Strategy Backtest with Leverages 1,3 & 5")
        plt.grid(True)
        plt.legend()

        plt_path = p / "todays_plot.png"
        if plt_path.is_file():
            plt_path.unlink()
        plt.savefig(plt_path)
        plt.close()
        return plt_path

    def calculate_indicators(self):
        """Apply Ivan Scherman's strategy and generate entry/exit markers."""
        sp500_data = self.sp500_data.copy()
        vix_data = self.vix_data.copy()
        sp500_data, vix_data = sp500_data.align(vix_data, join="inner", axis=0)

        # Calculate 30-day moving average and standard deviation for S&P 500
        sp500_data["20_MA"] = sp500_data["Close"].rolling(window=20).mean()
        sp500_data["20_STD"] = sp500_data["Close"].rolling(window=20).std()

        bb = BollingerBands(sp500_data["Close"], window=20)
        sp500_data["bb_upper_break"] = bb.bollinger_hband_indicator()
        sp500_data["bb_lower_break"] = bb.bollinger_lband_indicator()
        # Calculate Bollinger Bands
        sp500_data["Upper_BB"] = sp500_data["20_MA"] + 2 * sp500_data["20_STD"]
        sp500_data["Lower_BB"] = sp500_data["20_MA"] - 2 * sp500_data["20_STD"]
        sp500_data["Lower_BB2"] = bb.bollinger_lband()
        sp500_data["Upper_BB2"] = bb.bollinger_hband()

        # Identify divergences between S&P 500 and VIX
        sp500_data["Divergence"] = (
            sp500_data["Close"].pct_change() < 0
        ) & (  # S&P 500 correction
            vix_data["Close"].pct_change() <= 0
        )  # No VIX spike

        # Define entry and exit points
        sp500_data["Long_Entry"] = sp500_data["Divergence"] & (
            sp500_data["Close"] >= sp500_data["Lower_BB"]
        )
        sp500_data["Long_Exit"] = sp500_data["Close"] > sp500_data["Upper_BB"]
        sp500_data["Position"] = 0
        sp500_data["rsi"] = rsi(close=sp500_data["Close"])
        pos_idx = sp500_data.columns.get_loc("Position")
        for i in range(1, len(sp500_data)):
            if sp500_data["Long_Entry"].iloc[i]:
                sp500_data.iat[i, pos_idx] = 1
            elif (
                sp500_data["Long_Exit"].iloc[i]
                and sp500_data["Position"].iloc[i - 1] == 1
            ):
                sp500_data.iat[i, pos_idx] = 0
            else:
                sp500_data.iat[i, pos_idx] = sp500_data["Position"].iloc[i - 1]
        sp500_data["Daily_Return"] = sp500_data["Close"].pct_change()

        leverages = [1, 3, 5]
        leveraged_returns = {}
        leveraged_returns_final = {}
        for i in leverages:
            sp500_data = self.calculate_strategy_return(sp500_data, i)
            leveraged_returns[i] = (
                1 + sp500_data[f"Strategy_Return_lev_{i}"]
            ).cumprod()
            leveraged_returns_final[i] = leveraged_returns[i].iloc[-1]

        cumulative_market_return = (1 + sp500_data["Daily_Return"]).cumprod()

        buy = sp500_data["Long_Entry"].iloc[-1]
        sell = sp500_data["Long_Exit"].iloc[-1]
        currently_invested = sp500_data["Position"].iloc[-1]

        plt_path = self.create_graph(
            leveraged_returns, cumulative_market_return
        )
        if buy and not currently_invested:
            action = "buy"
        elif sell:
            action = "sell"
        else:
            action = "hold"
        return {
            "Timeframe": f"{len(sp500_data)} trading days",
            "Trading start": sp500_data.index[0],
            "market_return": cumulative_market_return.iloc[-1],
            "strategy_returns": leveraged_returns_final,
            "action": action,
            "currently_holding": "yes" if currently_invested else "no",
            "plt_path": plt_path,
        }

    def analyze_vix(self) -> dict:
        """
        Analyze the VIX data to extract daily insights.

        Args:
            vix_data (pd.DataFrame): DataFrame with 'Date' and 'Close' (VIX closing value).

        Returns:
            dict: Key metrics and insights for a daily status update.
        """

        # Add key metrics
        self.vix_data["1D_Change"] = self.vix_data[
            "Close"
        ].diff()  # Daily change
        self.vix_data["1D_Percent_Change"] = (
            self.vix_data["Close"].pct_change() * 100
        )  # Daily percent change
        self.vix_data["7D_Avg"] = (
            self.vix_data["Close"].rolling(window=7).mean()
        )  # 7-day moving average
        self.vix_data["30D_Avg"] = (
            self.vix_data["Close"].rolling(window=30).mean()
        )  # 30-day moving average
        self.vix_data["30D_STD"] = (
            self.vix_data["Close"].rolling(window=30).std()
        )  # 30-day standard deviation

        # Calculate latest metrics
        latest_date = self.vix_data.index[-1]
        latest_close = self.vix_data["Close"].iloc[-1]
        one_day_change = self.vix_data["1D_Change"].iloc[-1]
        one_day_percent_change = self.vix_data["1D_Percent_Change"].iloc[-1]
        seven_day_avg = self.vix_data["7D_Avg"].iloc[-1]
        thirty_day_avg = self.vix_data["30D_Avg"].iloc[-1]
        thirty_day_std = self.vix_data["30D_STD"].iloc[-1]

        # Calculate thresholds
        upper_threshold = thirty_day_avg + 2 * thirty_day_std
        lower_threshold = thirty_day_avg - 2 * thirty_day_std

        # Determine trends
        trend = (
            "Rising"
            if latest_close > thirty_day_avg and one_day_change > 0
            else (
                "Falling"
                if latest_close < thirty_day_avg and one_day_change < 0
                else "Stable"
            )
        )
        recommendation = (
            "Stable market: Buy growth stocks, high-beta equities & risk-on assets"
            if latest_close < 15
            else (
                "Normal vola: balance portfolio be aware of potential vola spikes"
                if latest_close > 15 and latest_close <= 25
                else (
                    "Potential correction: Reduce equities shift to bonds, gold & cash"
                    if latest_close > 25 and latest_close <= 35
                    else "Market panic: Keep cash and stay safe or look for opportunities in blue-chip stocks"
                )
            )
        )

        # Insights
        insights = {
            "Date": latest_date.date(),
            "Latest VIX Close": round(latest_close, 2),
            "1-Day Change": round(one_day_change, 2),
            "1-Day % Change": f"{round(one_day_percent_change, 2)}%",
            "7-Day Moving Avg": round(seven_day_avg, 2),
            "30-Day Moving Avg": round(thirty_day_avg, 2),
            "30-Day Std Dev": round(thirty_day_std, 2),
            "Upper Threshold (30D)": round(upper_threshold, 2),
            "Lower Threshold (30D)": round(lower_threshold, 2),
            "Trend": trend,
            "Recommendation": recommendation,
            "Volatility Status": (
                "High Volatility"
                if latest_close > upper_threshold
                else (
                    "Low Volatility"
                    if latest_close < lower_threshold
                    else "Normal Volatility"
                )
            ),
        }

        return insights

    def __init__(self):
        # Parameters
        self.sp500_ticker = "^GSPC"  # S&P 500
        self.vix_ticker = "^VIX"  # VIX
        end_date = date.today() + timedelta(days=1)
        start_date = end_date - timedelta(days=250)
        self.cache = Path(__file__).parent / ".cache"
        self.cache.mkdir(parents=True, exist_ok=True)

        # Fetch data
        self.vix_data = self.fetch_data(self.vix_ticker, start_date, end_date)
        print(self.vix_data.iloc[-1])
        self.sp500_data = self.fetch_data(
            self.sp500_ticker, start_date, end_date
        )
        print(self.sp500_data.iloc[-1])


def create_report(webhook: str):
    trader = SP500Trader()
    vix_insights = trader.analyze_vix()
    sp500_with_markers = trader.calculate_indicators()
    wh = SyncWebhook.from_url(webhook)
    format_string = f"""
# Analysis for {vix_insights["Date"]}

## S&P500 trading recommendation for today
**action**: {sp500_with_markers["action"]}
**currently invested**: {sp500_with_markers["currently_holding"]}
-# Simulation duration: {sp500_with_markers["Timeframe"]}
-# market return: {sp500_with_markers["market_return"]}
-# strategy return (leverage 1): {sp500_with_markers["strategy_returns"][1]}
-# strategy return (leverage 3): {sp500_with_markers["strategy_returns"][3]}
-# strategy return (leverage 5): {sp500_with_markers["strategy_returns"][5]}

## Vix based analysis:
**recommendation**: {vix_insights["Recommendation"]}
**Volatility status**: {vix_insights["Volatility Status"]}
**Trend**: {vix_insights["Trend"]}
**Key Metrics**: 
-# 'Latest VIX Close': {vix_insights["Latest VIX Close"]},
-# '1-Day Change': {vix_insights["1-Day Change"]},
-# '1-Day % Change': {vix_insights["1-Day % Change"]},
-# '7-Day Moving Avg': {vix_insights["7-Day Moving Avg"]}
-# '30-Day Moving Avg': {vix_insights["30-Day Moving Avg"]}
-# '30-Day Std Dev': {vix_insights["30-Day Std Dev"]},
-# 'Upper Threshold (30D)':{vix_insights["Upper Threshold (30D)"]},
-# 'Lower Threshold (30D)': {vix_insights["Lower Threshold (30D)"]},
    """
    wh.send(format_string)
    wh.send(file=File(p / "todays_plot.png"))


if __name__ == "__main__":
    cal = USFederalHolidayCalendar()
    today = date.today()
    webhook = "https://discord.com/api/webhooks/1332677643471945770/IjF1ISXAyQKOq_CA1rd7Fu63LXF8ayNPTBDujf8lazUKsDgh_QZDRmUsYoPzVu6Oa9y7"
    holidays = cal.holidays(
        start=today, end=f"{today.year}-12-31"
    ).to_pydatetime()
    print("Starting Bot...")
    create_report(webhook)
    while True:
        nye_time = datetime.now(pytz.timezone("US/Eastern"))
        print("Current time in NYE: ", nye_time)
        time.sleep(30)
        if (nye_time.hour == 9 and nye_time.minute == 35) or (
            nye_time.hour == 16 and nye_time.minute == 0
        ):
            if nye_time.date() in holidays or today.weekday >= 5:
                wh = SyncWebhook.from_url(webhook)
                wh.send("Today is a holiday/weekend. No update!")
            else:
                create_report(webhook)
