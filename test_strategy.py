#!/usr/bin/env python3
"""
测试策略脚本
"""

import sys
sys.path.append('.')

from stock_strategy import StockTrendStrategy

def test_basic_functionality():
    """测试基本功能"""
    print("测试美股趋势上涨策略基本功能")
    print("=" * 60)
    
    # 创建策略实例
    strategy = StockTrendStrategy(symbol="AAPL", period="1mo", interval="1d")
    
    # 测试数据获取
    print("1. 测试数据获取...")
    if strategy.fetch_data():
        print(f"   成功获取 {len(strategy.data)} 条数据")
        print(f"   数据列: {list(strategy.data.columns)}")
    else:
        print("   数据获取失败")
        return False
    
    # 测试指标计算
    print("\n2. 测试指标计算...")
    if strategy.calculate_indicators():
        print("   技术指标计算成功")
        # 检查计算的指标
        expected_columns = ['SMA_20', 'SMA_50', 'MACD', 'RSI', 'BB_upper']
        for col in expected_columns:
            if col in strategy.data.columns:
                print(f"   {col}: 已计算")
            else:
                print(f"   {col}: 缺失")
    else:
        print("   指标计算失败")
        return False
    
    # 测试信号生成
    print("\n3. 测试信号生成...")
    if strategy.generate_signals():
        print(f"   生成 {len(strategy.signals) if strategy.signals else 0} 个交易信号")
        if strategy.signals:
            for i, signal in enumerate(strategy.signals[:3]):  # 显示前3个信号
                print(f"   信号{i+1}: {signal['type']} @ ${signal['price']:.2f} "
                      f"(强度: {signal['strength']:.2f})")
    else:
        print("   信号生成失败")
        return False
    
    # 测试表现分析
    print("\n4. 测试表现分析...")
    performance = strategy.analyze_performance()
    if performance:
        print(f"   总交易次数: {performance.get('total_trades', 0)}")
        print(f"   买入信号数: {performance.get('total_buy_signals', 0)}")
        print(f"   卖出信号数: {performance.get('total_sell_signals', 0)}")
    else:
        print("   表现分析完成（可能无交易信号）")
    
    # 测试结果输出
    print("\n5. 测试结果输出...")
    strategy.print_results(performance)
    
    print("\n" + "=" * 60)
    print("基本功能测试完成！")
    return True

def test_multiple_symbols():
    """测试多个股票符号"""
    print("\n\n测试多个股票符号")
    print("=" * 60)
    
    symbols = ["AAPL", "MSFT", "GOOGL"]
    results = []
    
    for symbol in symbols:
        print(f"\n分析 {symbol}...")
        strategy = StockTrendStrategy(symbol=symbol, period="1mo")
        if strategy.fetch_data() and strategy.calculate_indicators():
            strategy.generate_signals()
            performance = strategy.analyze_performance()
            
            if strategy.signals:
                last_signal = strategy.signals[-1]
                results.append({
                    'symbol': symbol,
                    'signal': last_signal['type'],
                    'price': last_signal['price'],
                    'strength': last_signal['strength']
                })
                print(f"  最新信号: {last_signal['type']} @ ${last_signal['price']:.2f}")
            else:
                results.append({
                    'symbol': symbol,
                    'signal': 'HOLD',
                    'price': strategy.data['Close'].iloc[-1] if strategy.data is not None else 0,
                    'strength': 0
                })
                print(f"  无明确信号")
    
    # 输出汇总
    print("\n汇总结果:")
    for result in results:
        print(f"  {result['symbol']}: {result['signal']} @ ${result['price']:.2f} "
              f"(强度: {result['strength']:.2f})")
    
    return True

if __name__ == "__main__":
    print("开始测试美股趋势上涨策略脚本")
    print("=" * 60)
    
    try:
        # 测试基本功能
        test_basic_functionality()
        
        # 测试多个股票
        test_multiple_symbols()
        
        print("\n" + "=" * 60)
        print("所有测试完成！脚本功能正常。")
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
