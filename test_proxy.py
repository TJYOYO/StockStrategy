#!/usr/bin/env python3
"""
测试代理配置
"""

import os
import json
import sys

print("=== 代理配置测试 ===")

# 读取配置文件
try:
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    yf_config = config.get('yfinance_settings', {})
    proxy_settings = yf_config.get('proxy', {})
    use_proxy = proxy_settings.get('enabled', False)
    proxy_url = proxy_settings.get('url', '')
    
    print(f"1. 代理配置状态:")
    print(f"   启用: {use_proxy}")
    print(f"   URL: {proxy_url}")
    
    if use_proxy and proxy_url:
        print("\n2. 设置环境变量...")
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        print(f"   HTTP_PROXY = {os.environ.get('HTTP_PROXY')}")
        print(f"   HTTPS_PROXY = {os.environ.get('HTTPS_PROXY')}")
        
        print("\n3. 测试网络连接...")
        try:
            import requests
            # 测试代理是否工作
            test_url = 'http://httpbin.org/ip'
            print(f"   请求测试URL: {test_url}")
            response = requests.get(test_url, timeout=10)
            print(f"   响应状态码: {response.status_code}")
            print(f"   响应内容: {response.text[:150]}...")
        except Exception as e:
            print(f"   网络测试失败: {e}")
            print(f"   错误类型: {type(e).__name__}")
            
        print("\n4. 测试yfinance...")
        try:
            import yfinance as yf
            print("   尝试获取AAPL数据...")
            data = yf.download('AAPL', period='1d', progress=False)
            if not data.empty:
                print(f"   成功! 获取{len(data)}条数据")
                # 安全地获取收盘价
                close_price = data['Close'].iloc[-1]
                if hasattr(close_price, '__float__'):
                    print(f"   最新收盘价: ${float(close_price):.2f}")
                else:
                    print(f"   最新收盘价: {close_price}")
            else:
                print("   失败: 数据为空")
        except Exception as e:
            print(f"   yfinance测试失败: {e}")
            print(f"   错误类型: {type(e).__name__}")
    else:
        print("\n代理未启用，跳过测试")
        
except Exception as e:
    print(f"配置文件读取失败: {e}")
    sys.exit(1)
