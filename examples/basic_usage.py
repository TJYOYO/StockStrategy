#!/usr/bin/env python3
"""
基本使用示例
"""

import sys
import os
sys.path.append('..')

from stock_strategy import StockTrendStrategy

def main():
    """基本使用示例"""
    print("美股趋势上涨策略 - 基本使用示例")
    print("=" * 50)
    
    # 示例1: 分析苹果股票
    print("\n1. 分析苹果股票 (AAPL):")
    strategy1 = StockTrendStrategy(symbol="AAPL", period="3mo")
    strategy1.run_strategy()
    
    # 示例2: 分析微软股票并显示图表
    print("\n" + "=" * 50)
    print("\n2. 分析微软股票 (MSFT):")
    strategy2 = StockTrendStrategy(symbol="MSFT", period="6mo")
    if strategy2.run_strategy():
        print("\n显示微软股票分析图表...")
        strategy2.plot_results()
    
    # 示例3: 自定义参数
    print("\n" + "=" * 50)
    print("\n3. 分析特斯拉股票 (TSLA) 使用1年数据:")
    strategy3 = StockTrendStrategy(symbol="TSLA", period="1y")
    strategy3.run_strategy()
    
    print("\n" + "=" * 50)
    print("示例完成！")

if __name__ == "__main__":
    main()
