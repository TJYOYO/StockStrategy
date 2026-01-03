#!/usr/bin/env python3
"""
简单测试代理功能
"""

import os
import sys
import json

print("=== 简单代理测试 ===")

# 读取配置
try:
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    yf_config = config.get('yfinance_settings', {})
    proxy_settings = yf_config.get('proxy', {})
    use_proxy = proxy_settings.get('enabled', False)
    proxy_url = proxy_settings.get('url', '')
    
    print(f"1. 代理状态: 启用={use_proxy}, URL={proxy_url}")
    
    if use_proxy and proxy_url:
        # 设置环境变量
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        print(f"2. 已设置环境变量")
        
        # 测试基本网络
        print("3. 测试网络连接...")
        try:
            import requests
            test_url = 'http://httpbin.org/ip'
            print(f"   请求: {test_url}")
            response = requests.get(test_url, timeout=5)
            print(f"   成功! 状态码: {response.status_code}")
            print(f"   响应: {response.text[:100]}")
        except Exception as e:
            print(f"   网络测试失败: {e}")
            
        # 测试yfinance简单调用
        print("4. 测试yfinance...")
        try:
            import yfinance as yf
            print("   下载AAPL数据...")
            # 使用非常短的超时时间
            data = yf.download('AAPL', period='1d', progress=False, timeout=5)
            if data is not None and not data.empty:
                print(f"   成功! 数据形状: {data.shape}")
                print(f"   列: {list(data.columns)}")
            else:
                print("   失败: 数据为空")
        except Exception as e:
            print(f"   yfinance失败: {type(e).__name__}: {e}")
    else:
        print("代理未启用")
        
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
