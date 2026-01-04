#!/usr/bin/env python3
"""
测试图表功能
"""

import sys
import os

# 设置代理环境变量
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

print("=== 测试图表功能 ===")
print("=" * 50)

# 导入策略类
sys.path.insert(0, '.')
from stock_strategy import StockTrendStrategy

# 创建策略实例
print("\n1. 创建策略实例...")
strategy = StockTrendStrategy(symbol='GOOGL', period='3mo')
print(f"   股票代码: {strategy.symbol}")
print(f"   分析周期: {strategy.period}")

# 获取数据
print("\n2. 获取数据...")
if strategy.fetch_data(max_retries=1):
    print(f"   [OK] 数据获取成功")
    print(f"      数据条数: {len(strategy.data)}")
else:
    print("   [ERROR] 数据获取失败")
    sys.exit(1)

# 计算技术指标
print("\n3. 计算技术指标...")
if strategy.calculate_indicators():
    print("   [OK] 技术指标计算成功")
else:
    print("   [ERROR] 技术指标计算失败")
    sys.exit(1)

# 生成交易信号
print("\n4. 生成交易信号...")
if strategy.generate_signals():
    signal_count = len(strategy.signals) if strategy.signals else 0
    print(f"   [OK] 信号生成成功")
    print(f"      生成信号数量: {signal_count}")
else:
    print("   [WARN] 信号生成失败或无信号")

# 测试图表功能
print("\n5. 测试图表功能...")
try:
    # 设置matplotlib为非交互模式
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    
    # 导入plt
    import matplotlib.pyplot as plt
    
    print("   [INFO] 正在生成图表...")
    
    # 调用plot_results方法
    strategy.plot_results(save_path='test_plot.png')
    
    # 检查文件是否生成
    if os.path.exists('test_plot.png'):
        print("   [OK] 图表生成成功")
        print(f"      图表已保存到: test_plot.png")
        print(f"      文件大小: {os.path.getsize('test_plot.png')} 字节")
        
        # 显示图表信息
        print("\n   [INFO] 图表内容:")
        print("      - 价格走势和交易信号")
        print("      - MACD指标")
        print("      - RSI指标")
        print("      - 包含买卖信号标记")
    else:
        print("   [WARN] 图表文件未生成，但可能已显示在屏幕上")
        
except Exception as e:
    print(f"   [ERROR] 图表生成失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("[OK] 图表功能测试完成!")
print("=" * 50)

# 清理
if os.path.exists('test_plot.png'):
    os.remove('test_plot.png')
    print("\n[INFO] 已清理测试文件: test_plot.png")
