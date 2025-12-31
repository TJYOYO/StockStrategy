#!/usr/bin/env python3
"""
美股趋势上涨策略脚本
功能：
1. 针对美股
2. 根据趋势上涨策略
3. 提供买和卖股票的时机
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import json
import os
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import warnings
warnings.filterwarnings('ignore')

class StockTrendStrategy:
    """美股趋势上涨策略类"""
    
    def __init__(self, symbol="AAPL", period="6mo", interval="1d"):
        """
        初始化策略
        
        Args:
            symbol: 股票代码 (默认: AAPL)
            period: 数据周期 (默认: 6个月)
            interval: 数据间隔 (默认: 1天)
        """
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.data = None
        self.signals = None
        self.indicators = {}
        
    def fetch_data(self):
        """获取股票数据"""
        print(f"正在获取 {self.symbol} 的数据...")
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period, interval=self.interval)
           
            datasFromYF = yf.download(self.symbol, period=self.period, interval=self.interval, progress=False) 
            if datasFromYF.empty:
                raise ValueError(f"无法获取 {self.symbol} 的数据000")
           
            if self.data.empty:
                raise ValueError(f"无法获取 {self.symbol} 的数据")
                
            print(f"成功获取 {len(self.data)} 条数据")
            print(f"数据时间范围: {self.data.index[0]} 到 {self.data.index[-1]}")
            return True
        except Exception as e:
            print(f"获取数据失败: {e}")
            return False
    
    def calculate_indicators(self):
        """计算技术指标"""
        if self.data is None or self.data.empty:
            print("请先获取数据")
            return False
            
        print("计算技术指标...")
        
        # 移动平均线
        self.data['SMA_20'] = SMAIndicator(close=self.data['Close'], window=20).sma_indicator()
        self.data['SMA_50'] = SMAIndicator(close=self.data['Close'], window=50).sma_indicator()
        self.data['EMA_12'] = EMAIndicator(close=self.data['Close'], window=12).ema_indicator()
        self.data['EMA_26'] = EMAIndicator(close=self.data['Close'], window=26).ema_indicator()
        
        # MACD
        macd = MACD(close=self.data['Close'])
        self.data['MACD'] = macd.macd()
        self.data['MACD_signal'] = macd.macd_signal()
        self.data['MACD_diff'] = macd.macd_diff()
        
        # RSI
        self.data['RSI'] = RSIIndicator(close=self.data['Close'], window=14).rsi()
        
        # 布林带
        bb = BollingerBands(close=self.data['Close'], window=20, window_dev=2)
        self.data['BB_upper'] = bb.bollinger_hband()
        self.data['BB_middle'] = bb.bollinger_mavg()
        self.data['BB_lower'] = bb.bollinger_lband()
        
        # 价格变化
        self.data['Price_Change'] = self.data['Close'].pct_change() * 100
        self.data['Volume_Change'] = self.data['Volume'].pct_change() * 100
        
        print("技术指标计算完成")
        return True
    
    def generate_signals(self):
        """生成买卖信号"""
        if self.data is None or self.data.empty:
            print("请先获取数据并计算指标")
            return False
            
        print("生成买卖信号...")
        
        # 初始化信号列
        self.data['Signal'] = 0  # 0: 持有, 1: 买入, -1: 卖出
        self.data['Signal_Strength'] = 0.0
        
        signals = []
        
        for i in range(50, len(self.data)):
            current_price = self.data['Close'].iloc[i]
            sma_20 = self.data['SMA_20'].iloc[i]
            sma_50 = self.data['SMA_50'].iloc[i]
            ema_12 = self.data['EMA_12'].iloc[i]
            ema_26 = self.data['EMA_26'].iloc[i]
            macd = self.data['MACD'].iloc[i]
            macd_signal = self.data['MACD_signal'].iloc[i]
            rsi = self.data['RSI'].iloc[i]
            bb_lower = self.data['BB_lower'].iloc[i]
            bb_upper = self.data['BB_upper'].iloc[i]
            volume = self.data['Volume'].iloc[i]
            avg_volume = self.data['Volume'].iloc[i-20:i].mean()
            
            signal_strength = 0
            buy_signals = 0
            sell_signals = 0
            
            # 买入信号条件
            # 1. 价格在20日均线之上且20日均线在50日均线之上（上升趋势）
            if current_price > sma_20 and sma_20 > sma_50:
                buy_signals += 1
                signal_strength += 0.3
                
            # 2. MACD金叉
            if macd > macd_signal and self.data['MACD'].iloc[i-1] <= self.data['MACD_signal'].iloc[i-1]:
                buy_signals += 1
                signal_strength += 0.2
                
            # 3. RSI从超卖区域回升
            if rsi < 30 and self.data['RSI'].iloc[i-1] < rsi:
                buy_signals += 1
                signal_strength += 0.2
                
            # 4. 价格触及布林带下轨
            if current_price <= bb_lower * 1.02:
                buy_signals += 1
                signal_strength += 0.15
                
            # 5. 成交量放大
            if volume > avg_volume * 1.2:
                buy_signals += 1
                signal_strength += 0.15
                
            # 卖出信号条件
            # 1. 价格在20日均线之下且20日均线在50日均线之下（下降趋势）
            if current_price < sma_20 and sma_20 < sma_50:
                sell_signals += 1
                signal_strength -= 0.3
                
            # 2. MACD死叉
            if macd < macd_signal and self.data['MACD'].iloc[i-1] >= self.data['MACD_signal'].iloc[i-1]:
                sell_signals += 1
                signal_strength -= 0.2
                
            # 3. RSI从超买区域回落
            if rsi > 70 and self.data['RSI'].iloc[i-1] > rsi:
                sell_signals += 1
                signal_strength -= 0.2
                
            # 4. 价格触及布林带上轨
            if current_price >= bb_upper * 0.98:
                sell_signals += 1
                signal_strength -= 0.15
                
            # 5. 成交量萎缩
            if volume < avg_volume * 0.8:
                sell_signals += 1
                signal_strength -= 0.15
            
            # 确定最终信号
            if buy_signals >= 3 and signal_strength > 0.5:
                self.data.loc[self.data.index[i], 'Signal'] = 1
                self.data.loc[self.data.index[i], 'Signal_Strength'] = signal_strength
                signals.append({
                    'date': self.data.index[i],
                    'type': 'BUY',
                    'price': current_price,
                    'strength': signal_strength,
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals
                })
            elif sell_signals >= 3 and signal_strength < -0.5:
                self.data.loc[self.data.index[i], 'Signal'] = -1
                self.data.loc[self.data.index[i], 'Signal_Strength'] = signal_strength
                signals.append({
                    'date': self.data.index[i],
                    'type': 'SELL',
                    'price': current_price,
                    'strength': signal_strength,
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals
                })
        
        self.signals = signals
        print(f"生成 {len(signals)} 个交易信号")
        return True
    
    def analyze_performance(self):
        """分析策略表现"""
        if self.signals is None or len(self.signals) == 0:
            print("没有交易信号可分析")
            return None
            
        print("分析策略表现...")
        
        buy_signals = [s for s in self.signals if s['type'] == 'BUY']
        sell_signals = [s for s in self.signals if s['type'] == 'SELL']
        
        # 模拟交易
        trades = []
        position = None
        for signal in self.signals:
            if signal['type'] == 'BUY' and position is None:
                position = signal
            elif signal['type'] == 'SELL' and position is not None:
                profit_pct = ((signal['price'] - position['price']) / position['price']) * 100
                trade = {
                    'buy_date': position['date'],
                    'buy_price': position['price'],
                    'sell_date': signal['date'],
                    'sell_price': signal['price'],
                    'profit_pct': profit_pct,
                    'holding_days': (signal['date'] - position['date']).days
                }
                trades.append(trade)
                position = None
        
        # 计算统计指标
        if trades:
            profits = [t['profit_pct'] for t in trades]
            avg_profit = np.mean(profits)
            win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
            max_profit = max(profits) if profits else 0
            min_profit = min(profits) if profits else 0
            
            performance = {
                'total_trades': len(trades),
                'win_rate': win_rate,
                'avg_profit_pct': avg_profit,
                'max_profit_pct': max_profit,
                'min_profit_pct': min_profit,
                'total_buy_signals': len(buy_signals),
                'total_sell_signals': len(sell_signals),
                'trades': trades
            }
        else:
            performance = {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit_pct': 0,
                'total_buy_signals': len(buy_signals),
                'total_sell_signals': len(sell_signals),
                'message': '没有完整的买卖交易对'
            }
        
        return performance
    
    def plot_results(self, save_path=None):
        """绘制结果图表"""
        if self.data is None or self.data.empty:
            print("没有数据可绘制")
            return
            
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 价格和移动平均线
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['Close'], label='收盘价', linewidth=2)
        ax1.plot(self.data.index, self.data['SMA_20'], label='20日SMA', alpha=0.7)
        ax1.plot(self.data.index, self.data['SMA_50'], label='50日SMA', alpha=0.7)
        
        # 标记买卖信号
        buy_dates = [s['date'] for s in self.signals if s['type'] == 'BUY'] if self.signals else []
        sell_dates = [s['date'] for s in self.signals if s['type'] == 'SELL'] if self.signals else []
        
        if buy_dates:
            buy_prices = [self.data.loc[date, 'Close'] for date in buy_dates]
            ax1.scatter(buy_dates, buy_prices, color='green', s=100, marker='^', label='买入信号', zorder=5)
        
        if sell_dates:
            sell_prices = [self.data.loc[date, 'Close'] for date in sell_dates]
            ax1.scatter(sell_dates, sell_prices, color='red', s=100, marker='v', label='卖出信号', zorder=5)
        
        ax1.set_title(f'{self.symbol} - 价格走势和交易信号')
        ax1.set_ylabel('价格 ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MACD
        ax2 = axes[1]
        ax2.plot(self.data.index, self.data['MACD'], label='MACD', linewidth=2)
        ax2.plot(self.data.index, self.data['MACD_signal'], label='信号线', linewidth=2)
        ax2.fill_between(self.data.index, 0, self.data['MACD_diff'], alpha=0.3, label='MACD差值')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('MACD指标')
        ax2.set_ylabel('MACD')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # RSI
        ax3 = axes[2]
        ax3.plot(self.data.index, self.data['RSI'], label='RSI', linewidth=2, color='purple')
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='超买线 (70)')
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='超卖线 (30)')
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
        ax3.set_title('RSI指标')
        ax3.set_ylabel('RSI')
        ax3.set_xlabel('日期')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def run_strategy(self):
        """运行完整策略"""
        print(f"\n{'='*60}")
        print(f"运行美股趋势上涨策略 - {self.symbol}")
        print(f"{'='*60}")
        
        # 获取数据
        if not self.fetch_data():
            return False
            
        # 计算指标
        if not self.calculate_indicators():
            return False
            
        # 生成信号
        if not self.generate_signals():
            return False
            
        # 分析表现
        performance = self.analyze_performance()
        
        # 输出结果
        self.print_results(performance)
        
        return True
    
    def print_results(self, performance):
        """打印结果"""
        print(f"\n{'='*60}")
        print("策略分析结果")
        print(f"{'='*60}")
        
        if performance:
            print(f"股票代码: {self.symbol}")
            print(f"分析周期: {self.period}")
            print(f"数据间隔: {self.interval}")
            print(f"总交易次数: {performance.get('total_trades', 0)}")
            print(f"胜率: {performance.get('win_rate', 0):.2f}%")
            print(f"平均收益率: {performance.get('avg_profit_pct', 0):.2f}%")
            
            if 'max_profit_pct' in performance:
                print(f"最大收益: {performance['max_profit_pct']:.2f}%")
                print(f"最小收益: {performance['min_profit_pct']:.2f}%")
            
            print(f"\n买入信号总数: {performance.get('total_buy_signals', 0)}")
            print(f"卖出信号总数: {performance.get('total_sell_signals', 0)}")
            
            if self.signals:
                print(f"\n最近的交易信号:")
                for i, signal in enumerate(self.signals[-5:]):  # 显示最近5个信号
                    print(f"  {signal['date'].strftime('%Y-%m-%d')}: {signal['type']} @ ${signal['price']:.2f} "
                          f"(强度: {signal['strength']:.2f})")
        
        print(f"\n建议:")
        if self.signals:
            last_signal = self.signals[-1]
            if last_signal['type'] == 'BUY':
                print(f"  当前建议: 买入 {self.symbol}")
                print(f"  理由: 检测到{last_signal['buy_signals']}个买入信号条件满足")
            elif last_signal['type'] == 'SELL':
                print(f"  当前建议: 卖出 {self.symbol}")
                print(f"  理由: 检测到{last_signal['sell_signals']}个卖出信号条件满足")
        else:
            print(f"  当前建议: 持有观望")
            print(f"  理由: 未检测到明确的买卖信号")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='美股趋势上涨策略脚本')
    parser.add_argument('--symbol', type=str, default='AAPL', help='股票代码 (默认: AAPL)')
    parser.add_argument('--period', type=str, default='6mo', help='数据周期 (默认: 6mo)')
    parser.add_argument('--interval', type=str, default='1d', help='数据间隔 (默认: 1d)')
    parser.add_argument('--plot', action='store_true', help='显示图表')
    parser.add_argument('--save-plot', type=str, help='保存图表到文件')
    parser.add_argument('--export', type=str, help='导出结果到JSON文件')
    
    args = parser.parse_args()
    
    # 创建策略实例
    strategy = StockTrendStrategy(
        symbol=args.symbol,
        period=args.period,
        interval=args.interval
    )
    
    # 运行策略
    success = strategy.run_strategy()
    
    if success:
        # 显示图表
        if args.plot or args.save_plot:
            strategy.plot_results(save_path=args.save_plot)
        
        # 导出结果
        if args.export:
            export_results(strategy, args.export)
    else:
        print("策略运行失败")
        
def export_results(strategy, filename):
    """导出结果到JSON文件"""
    results = {
        'symbol': strategy.symbol,
        'period': strategy.period,
        'interval': strategy.interval,
        'data_points': len(strategy.data) if strategy.data is not None else 0,
        'signals': strategy.signals if strategy.signals is not None else [],
        'performance': strategy.analyze_performance()
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, default=str, indent=2, ensure_ascii=False)
    
    print(f"结果已导出到: {filename}")

if __name__ == "__main__":
    main()
