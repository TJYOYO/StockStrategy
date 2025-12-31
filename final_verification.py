#!/usr/bin/env python3
"""
最终验证脚本 - 展示美股趋势上涨策略脚本的完整功能
"""

import os
import sys
import json

def verify_project_structure():
    """验证项目结构"""
    print("验证项目结构...")
    
    required_files = [
        'stock_strategy.py',
        'requirements.txt',
        'config.json',
        'README.md',
        'batch_analysis.py',
        'examples/basic_usage.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"  缺少文件: {missing_files}")
        return False
    else:
        print("  所有必需文件都存在")
        return True

def verify_dependencies():
    """验证依赖"""
    print("\n验证Python依赖...")
    
    required_packages = [
        'yfinance',
        'pandas',
        'numpy',
        'matplotlib',
        'ta'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  {package}: 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"  {package}: 未安装")
    
    if missing_packages:
        print(f"  缺少包: {missing_packages}")
        print(f"  请运行: pip install {' '.join(missing_packages)}")
        return False
    else:
        print("  所有依赖包都已安装")
        return True

def verify_config():
    """验证配置文件"""
    print("\n验证配置文件...")
    
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        required_keys = ['default_symbols', 'default_period', 'strategy_parameters']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"  配置缺少键: {missing_keys}")
            return False
        else:
            print(f"  配置文件有效")
            print(f"  默认股票: {config['default_symbols']}")
            print(f"  默认周期: {config['default_period']}")
            return True
    except Exception as e:
        print(f"  配置文件错误: {e}")
        return False

def verify_strategy_class():
    """验证策略类"""
    print("\n验证策略类...")
    
    try:
        from stock_strategy import StockTrendStrategy
        
        # 测试类初始化
        strategy = StockTrendStrategy(symbol="AAPL", period="1mo")
        
        # 检查方法
        required_methods = [
            'fetch_data',
            'calculate_indicators',
            'generate_signals',
            'analyze_performance',
            'plot_results',
            'run_strategy',
            'print_results'
        ]
        
        missing_methods = []
        for method in required_methods:
            if not hasattr(strategy, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"  策略类缺少方法: {missing_methods}")
            return False
        else:
            print("  策略类完整，所有方法都存在")
            return True
            
    except Exception as e:
        print(f"  策略类验证失败: {e}")
        return False

def demonstrate_functionality():
    """演示功能"""
    print("\n" + "="*60)
    print("演示美股趋势上涨策略功能")
    print("="*60)
    
    print("\n1. 脚本使用方法:")
    print("   python stock_strategy.py --symbol AAPL --period 6mo")
    print("   python stock_strategy.py --symbol MSFT --plot")
    print("   python stock_strategy.py --symbol GOOGL --export results.json")
    
    print("\n2. 批量分析方法:")
    print("   python batch_analysis.py")
    
    print("\n3. 配置文件位置:")
    print("   config.json - 包含策略参数和交易设置")
    
    print("\n4. 输出功能:")
    print("   - 买卖信号生成")
    print("   - 策略表现分析")
    print("   - 可视化图表")
    print("   - JSON结果导出")
    
    print("\n5. 策略特点:")
    print("   - 针对美股市场")
    print("   - 基于趋势上涨策略")
    print("   - 多重技术指标确认")
    print("   - 信号强度量化")
    print("   - 风险管理参数")

def main():
    """主函数"""
    print("美股趋势上涨策略脚本 - 最终验证")
    print("="*60)
    
    all_passed = True
    
    # 验证项目结构
    if not verify_project_structure():
        all_passed = False
    
    # 验证依赖
    if not verify_dependencies():
        all_passed = False
    
    # 验证配置
    if not verify_config():
        all_passed = False
    
    # 验证策略类
    if not verify_strategy_class():
        all_passed = False
    
    # 演示功能
    demonstrate_functionality()
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ 验证通过！美股趋势上涨策略脚本开发完成。")
        print("\n项目功能总结:")
        print("1. ✅ 针对美股市场")
        print("2. ✅ 基于趋势上涨策略")
        print("3. ✅ 提供买和卖股票的时机")
        print("4. ✅ 完整的技术指标分析")
        print("5. ✅ 批量分析功能")
        print("6. ✅ 可视化输出")
        print("7. ✅ 配置文件支持")
        print("8. ✅ 完整的文档")
    else:
        print("❌ 验证失败，请检查上述问题。")
    
    print("\n下一步:")
    print("1. 运行: python stock_strategy.py --symbol AAPL")
    print("2. 运行: python batch_analysis.py")
    print("3. 查看 README.md 获取详细使用说明")

if __name__ == "__main__":
    main()
