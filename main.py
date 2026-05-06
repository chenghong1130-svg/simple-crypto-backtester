import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client


# --- 第一部分：数据获取模块 ---
def get_binance_data(symbol="BTCUSDT", interval="1h", start_str="1 Jan, 2024"):
    """从币安直接获取历史行情数据"""
    client = Client()  # 无需 API Key 即可获取公开行情
    print(f"正在从 Binance 获取 {symbol} 的 {interval} 数据...")

    # 获取 K 线数据
    klines = client.get_historical_klines(symbol, interval, start_str)

    # 整理成 DataFrame
    cols = ['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close_Time', 'Quote_Volume', 'Trades_Count', 'Taker_Base', 'Taker_Quote', 'Ignore']
    df = pd.DataFrame(klines, columns=cols)

    # 格式转换
    df['Open_Time'] = pd.to_datetime(df['Open_Time'], unit='ms') + pd.Timedelta(hours=8)  # 转为北京时间
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    df = df[['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.set_index('Open_Time', inplace=True)
    return df


# --- 第二部分：核心回测引擎模块 ---
class QuantEngine:
    def __init__(self, df, initial_capital=10000, commission=0.001, slippage=0.0005):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.commission = commission  # 默认 0.1% 手续费
        self.slippage = slippage  # 默认 0.05% 滑点

    def run_strategy(self, short_ma=20, long_ma=50):
        """双均线策略逻辑"""
        df = self.df
        df['MA_S'] = df['Close'].rolling(window=short_ma).mean()
        df['MA_L'] = df['Close'].rolling(window=long_ma).mean()

        # 生成信号：1 买入持仓，0 空仓
        df['Signal'] = np.where(df['MA_S'] > df['MA_L'], 1, 0)
        df['Trade_Action'] = df['Signal'].diff()  # 信号变化点

        cash = self.initial_capital
        position = 0
        equity_curve = []

        # 逐笔模拟交易
        for i in range(len(df)):
            price = df['Close'].iloc[i]
            action = df['Trade_Action'].iloc[i]

            # 买入逻辑
            if action == 1:
                real_price = price * (1 + self.slippage)
                can_buy = cash / real_price
                fee = can_buy * real_price * self.commission
                position = (cash - fee) / real_price
                cash = 0

            # 卖出逻辑
            elif action == -1 and position > 0:
                real_price = price * (1 - self.slippage)
                sell_val = position * real_price
                fee = sell_val * self.commission
                cash = sell_val - fee
                position = 0

            # 记录当前总资产
            current_total = cash + (position * price)
            equity_curve.append(current_total)

        df['Equity'] = equity_curve
        return df


# --- 第三部分：评估指标模块 ---
def show_report(df):
    """计算并展示回测报告"""
    returns = df['Equity'].pct_change().dropna()
    total_return = (df['Equity'].iloc[-1] / df['Equity'].iloc[0]) - 1

    # 最大回撤计算
    rolling_max = df['Equity'].cummax()
    drawdown = (df['Equity'] - rolling_max) / rolling_max
    max_dd = drawdown.min()

    print("\n" + "=" * 30)
    print(f"最终资产: {df['Equity'].iloc[-1]:.2f} USDT")
    print(f"累计收益率: {total_return * 100:.2f}%")
    print(f"最大回撤: {max_dd * 100:.2f}%")
    print("=" * 30)

    # 简单的可视化绘图
    plt.figure(figsize=(12, 6))
    plt.plot(df['Equity'], label='Strategy Equity Curve', color='blue')
    plt.title('BTC Strategy Backtest Result')
    plt.legend()
    plt.grid(True)
    plt.show()


# --- 第四部分：主程序入口 ---
if __name__ == "__main__":
    # 1. 下载数据
    btc_data = get_binance_data(symbol="BTCUSDT", interval="1h", start_str="1 Jan, 2024")

    # 2. 运行回测
    engine = QuantEngine(btc_data, initial_capital=10000)
    results = engine.run_strategy(short_ma=20, long_ma=50)

    # 3. 输出报告
    show_report(results)