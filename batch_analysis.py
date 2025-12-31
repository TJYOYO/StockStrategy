#!/usr/bin/env python3
"""
批量分析多个美股的脚本
"""

import json
import os
from datetime import datetime
from stock_strategy import StockTrendStrategy

def load_config():
    """加载配置文件"""
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("配置文件 config.json 不存在，使用默认配置")
        return {
            "default_symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "default_period": "6mo",
            "default_interval": "1d"
        }

def analyze_stock(symbol, period, interval):
    """分析单个股票"""
    print(f"\n{'='*60}")
    print(f"开始分析: {symbol}")
    print(f"{'='*60}")
    
    strategy = StockTrendStrategy(symbol=symbol, period=period, interval=interval)
    success = strategy.run_strategy()
    
    if success and strategy.signals:
        # 获取最近信号
        last_signal = strategy.signals[-1]
        return {
            'symbol': symbol,
            'success': True,
            'last_signal': last_signal['type'],
            'last_price': last_signal['price'],
            'signal_strength': last_signal['strength'],
            'signal_date': last_signal['date'].strftime('%Y-%m-%d'),
            'total_signals': len(strategy.signals),
            'buy_signals': len([s for s in strategy.signals if s['type'] == 'BUY']),
            'sell_signals': len([s for s in strategy.signals if s['type'] == 'SELL'])
        }
    else:
        return {
            'symbol': symbol,
            'success': False,
            'error': '分析失败或无信号'
        }

def main():
    """主函数"""
    config = load_config()
    
    symbols = config.get('default_symbols', ['AAPL', 'MSFT', 'GOOGL'])
    period = config.get('default_period', '6mo')
    interval = config.get('default_interval', '1d')
    
    print(f"批量分析 {len(symbols)} 只美股")
    print(f"时间周期: {period}, 数据间隔: {interval}")
    print(f"股票列表: {', '.join(symbols)}")
    
    results = []
    
    for symbol in symbols:
        result = analyze_stock(symbol, period, interval)
        results.append(result)
    
    # 输出汇总结果
    print(f"\n{'='*60}")
    print("批量分析汇总结果")
    print(f"{'='*60}")
    
    successful = [r for r in results if r['success']]
    buy_recommendations = [r for r in successful if r['last_signal'] == 'BUY']
    sell_recommendations = [r for r in successful if r['last_signal'] == 'SELL']
    
    print(f"成功分析: {len(successful)}/{len(symbols)} 只股票")
    print(f"买入建议: {len(buy_recommendations)} 只")
    print(f"卖出建议: {len(sell_recommendations)} 只")
    
    if buy_recommendations:
        print(f"\n推荐买入的股票:")
        for stock in sorted(buy_recommendations, key=lambda x: x['signal_strength'], reverse=True):
            print(f"  {stock['symbol']}: ${stock['last_price']:.2f} "
                  f"(信号强度: {stock['signal_strength']:.2f}, 日期: {stock['signal_date']})")
    
    if sell_recommendations:
        print(f"\n推荐卖出的股票:")
        for stock in sorted(sell_recommendations, key=lambda x: x['signal_strength']):
            print(f"  {stock['symbol']}: ${stock['last_price']:.2f} "
                  f"(信号强度: {stock['signal_strength']:.2f}, 日期: {stock['signal_date']})")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"batch_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'symbols': symbols,
                'period': period,
                'interval': interval
            },
            'results': results,
            'summary': {
                'total_stocks': len(symbols),
                'successful_analysis': len(successful),
                'buy_recommendations': len(buy_recommendations),
                'sell_recommendations': len(sell_recommendations)
            }
        }, f, default=str, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
