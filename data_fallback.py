#!/usr/bin/env python3
"""
数据回退模块 - 当yfinance API限制时提供模拟数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import time
import random
import json

class DataFallback:
    """数据回退类，提供多种数据获取方式"""
    
    @staticmethod
    def generate_mock_data(symbol, period='6mo', interval='1d'):
        """生成模拟股票数据用于演示

        Args:
            symbol: 股票代码
            period: 时间周期
            interval: 数据间隔

        Returns:
            pandas.DataFrame: 模拟数据
        """
        # 根据周期确定数据点数量
        if 'd' in period:
            days = int(period.replace('d', ''))
        elif 'mo' in period:
            days = int(period.replace('mo', '')) * 30
        elif 'y' in period:
            days = int(period.replace('y', '')) * 365
        else:
            days = 180  # 默认6个月

        # 生成日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # 根据间隔调整日期范围
        if interval == '1d':
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            # 过滤掉周末
            dates = dates[dates.weekday < 5]  # 0-4 代表周一到周五
        elif interval == '1wk':
            dates = pd.date_range(start=start_date, end=end_date, freq='W')
        elif interval == '1mo':
            dates = pd.date_range(start=start_date, end=end_date, freq='M')
        else:
            dates = pd.date_range(start=start_date, end=end_date, freq=interval)
            # 过滤掉周末
            if interval.endswith('d'):
                dates = dates[dates.weekday < 5]

        # 基础价格（根据股票代码不同而不同）
        base_price = {
            'AAPL': 180.0, 'MSFT': 400.0, 'GOOGL': 150.0,
            'AMZN': 170.0, 'TSLA': 250.0, 'NVDA': 120.0,
            'META': 500.0, 'NFLX': 650.0
        }.get(symbol, 100.0)

        # 生成价格序列（带有趋势、波动和更真实的市场行为）
        np.random.seed(hash(symbol) % 10000)  # 根据股票代码设置随机种子
        n = len(dates)

        if n == 0:
            # 如果日期范围太小，生成最小数据集
            dates = pd.date_range(start=start_date, periods=30, freq='D')
            dates = dates[dates.weekday < 5]  # 只保留工作日
            n = len(dates)

        # 生成更真实的价格走势，包含趋势、均值回归和随机波动
        # 使用几何布朗运动模型
        dt = 1.0 / 252  # 假设年交易日为252天
        mu = 0.08  # 年化收益率
        sigma = 0.2  # 年化波动率

        # 生成对数收益率
        returns = np.random.normal((mu - 0.5 * sigma ** 2) * dt, sigma * np.sqrt(dt), n)

        # 生成价格序列
        prices = [base_price]
        for i in range(1, n):
            new_price = prices[-1] * np.exp(returns[i])
            prices.append(new_price)

        prices = np.array(prices)

        # 生成OHLC数据
        # 开盘价在前一日收盘价附近
        opens = [base_price]
        for i in range(1, n):
            # 开盘价是前一日收盘价附近的小幅波动
            change = np.random.normal(0, 0.005)  # 0.5%的平均波动
            opens.append(prices[i-1] * (1 + change))

        opens = np.array(opens)

        # 高价和低价基于波动率生成
        daily_volatility = np.abs(np.random.normal(0.02, 0.01, n))  # 日波动率
        highs = np.maximum(opens, prices) * (1 + daily_volatility)
        lows = np.minimum(opens, prices) * (1 - daily_volatility)

        # 确保高低价格合理
        for i in range(n):
            max_price = max(opens[i], prices[i])
            min_price = min(opens[i], prices[i])
            highs[i] = max(max_price, highs[i])
            lows[i] = min(min_price, lows[i])

        # 生成成交量（与价格波动相关，更符合实际）
        price_changes = np.abs(np.diff(prices, prepend=prices[0])) / prices
        base_volume = {
            'AAPL': 50_000_000, 'MSFT': 20_000_000, 'GOOGL': 15_000_000,
            'AMZN': 8_000_000, 'TSLA': 30_000_000, 'NVDA': 25_000_000,
            'META': 10_000_000, 'NFLX': 5_000_000
        }.get(symbol, 10_000_000)

        volumes = base_volume * (0.5 + 0.5 * price_changes)  # 成交量与价格波动正相关
        volumes = np.maximum(volumes, base_volume * 0.1)  # 设置最小成交量

        # 创建DataFrame
        data = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes,
            'Dividends': np.zeros(n),
            'Stock Splits': np.zeros(n)
        }, index=dates[:len(opens)])  # 确保索引长度与数据长度匹配

        # 确保数据按时间排序
        data = data.sort_index()

        return data
    
    @staticmethod
    def try_yfinance_with_proxy(symbol, period='6mo', interval='1d', max_retries=None):
        """尝试使用yfinance获取数据，带有代理和重试机制

        Args:
            symbol: 股票代码
            period: 时间周期
            interval: 数据间隔
            max_retries: 最大重试次数，如果为None则从配置文件读取

        Returns:
            pandas.DataFrame or None: 获取的数据，失败返回None
        """
        # 从配置文件读取设置
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            yf_config = config.get('yfinance_settings', {})

            if max_retries is None:
                max_retries = yf_config.get('max_retries', 3)

            retry_delay_min = yf_config.get('retry_delay_min', 1.0)
            retry_delay_max = yf_config.get('retry_delay_max', 3.0)
            rate_limit_delay_multiplier = yf_config.get('rate_limit_delay_multiplier', 5.0)
            user_agents = yf_config.get('user_agents', [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0"
            ])
            proxy_settings = yf_config.get('proxy', {})
            use_proxy = proxy_settings.get('enabled', False)
            proxy_url = proxy_settings.get('url', '')
        except:
            # 如果配置文件读取失败，使用默认值
            max_retries = max_retries or 3
            retry_delay_min = 1.0
            retry_delay_max = 3.0
            rate_limit_delay_multiplier = 5.0
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0"
            ]
            use_proxy = False
            proxy_url = ''

        for attempt in range(max_retries):
            try:
                # 增加延迟
                if attempt > 0:
                    wait_time = random.uniform(retry_delay_min, retry_delay_max) * (attempt + 1) * 2  # 增加延迟时间
                    print(f"  等待 {wait_time:.2f} 秒后重试...")
                    time.sleep(wait_time)

                # 添加代理配置（如果启用）
                if use_proxy and proxy_url:
                    # 设置环境变量代理 - yfinance会自动使用这些环境变量
                    import os
                    os.environ['HTTP_PROXY'] = proxy_url
                    os.environ['HTTPS_PROXY'] = proxy_url
                    print(f"  使用代理: {proxy_url}")
                    
                    # yfinance 1.0+ 使用 curl_cffi，会自动读取环境变量中的代理设置
                    # 我们不传递自定义session，让yfinance使用其默认的curl_cffi session
                    import yfinance as yf
                    
                    # 设置User-Agent（通过修改yfinance的utils）
                    try:
                        from yfinance import utils
                        user_agent = random.choice(user_agents)
                        utils.session.headers.update({'User-Agent': user_agent})
                    except:
                        pass  # 如果设置失败，继续执行
                    
                    data = yf.download(
                        symbol,
                        period=period,
                        interval=interval,
                        progress=False,
                        timeout=30
                        # 不传递session参数，让yfinance使用默认的curl_cffi
                    )
                else:
                    # 不使用代理
                    import yfinance as yf
                    
                    # 设置User-Agent
                    try:
                        from yfinance import utils
                        user_agent = random.choice(user_agents)
                        utils.session.headers.update({'User-Agent': user_agent})
                    except:
                        pass
                    
                    data = yf.download(
                        symbol,
                        period=period,
                        interval=interval,
                        progress=False,
                        timeout=30
                    )

                if not data.empty:
                    return data

            except Exception as e:
                # 检查是否是yfinance的速率限制错误
                if 'YFRateLimitError' in str(type(e)) or 'Too Many Requests' in str(e) or 'rate limited' in str(e).lower() or '429' in str(e):
                    print(f"  检测到速率限制错误，等待更长时间后重试...")
                    # 增加更长的等待时间来处理速率限制
                    wait_time = random.uniform(retry_delay_min * rate_limit_delay_multiplier * 2,
                                              retry_delay_max * rate_limit_delay_multiplier * 3) * (attempt + 1)
                    print(f"  等待 {wait_time:.2f} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                print(f"  尝试 {attempt + 1} 失败: {e}")
                # 如果是其他错误，也等待一段时间再重试
                if attempt < max_retries - 1:
                    wait_time = random.uniform(retry_delay_min * 1.5, retry_delay_max * 2) * (attempt + 1)
                    print(f"  等待 {wait_time:.2f} 秒后重试...")
                    time.sleep(wait_time)
                continue

        return None
    
    @staticmethod
    def get_stock_data(symbol, period='6mo', interval='1d', use_fallback=True, max_retries=None):
        """获取股票数据，支持回退到模拟数据

        Args:
            symbol: 股票代码
            period: 时间周期
            interval: 数据间隔
            use_fallback: 是否使用回退数据
            max_retries: 最大重试次数，如果为None则从配置文件读取

        Returns:
            pandas.DataFrame: 股票数据
        """
        print(f"尝试获取 {symbol} 的数据...")

        # 首先尝试从缓存加载数据
        data = DataFallback.load_data_from_cache(symbol, period)
        if data is not None:
            print(f"  从缓存加载 {len(data)} 条数据")
            return data

        # 尝试yfinance
        data = DataFallback.try_yfinance_with_proxy(symbol, period, interval, max_retries)

        if data is not None and not data.empty:
            print(f"  成功从yfinance获取 {len(data)} 条数据")
            # 保存到缓存
            DataFallback.save_data_to_cache(data, symbol, period)
            return data

        # 如果yfinance失败且允许回退
        if use_fallback:
            print(f"  yfinance API限制，使用模拟数据进行演示...")
            print(f"  注意: 模拟数据仅用于功能演示，非真实市场数据")
            data = DataFallback.generate_mock_data(symbol, period, interval)
            print(f"  生成 {len(data)} 条模拟数据")
            return data
        else:
            raise ValueError(f"无法获取 {symbol} 的数据，且未启用回退模式")
    
    @staticmethod
    def save_data_to_cache(data, symbol, period):
        """保存数据到缓存文件"""
        import os
        cache_dir = 'data_cache'
        os.makedirs(cache_dir, exist_ok=True)
        
        filename = f"{cache_dir}/{symbol}_{period}_{datetime.now().strftime('%Y%m%d')}.csv"
        data.to_csv(filename)
        print(f"数据已缓存到: {filename}")
        return filename
    
    @staticmethod
    def load_data_from_cache(symbol, period, max_days_old=1):
        """从缓存加载数据"""
        import os
        from datetime import datetime, timedelta
        
        cache_dir = 'data_cache'
        if not os.path.exists(cache_dir):
            return None
        
        # 查找最新的缓存文件
        cache_files = []
        for file in os.listdir(cache_dir):
            if file.startswith(f"{symbol}_{period}_"):
                try:
                    file_date_str = file.split('_')[2].replace('.csv', '')
                    file_date = datetime.strptime(file_date_str, '%Y%m%d')
                    if (datetime.now() - file_date).days <= max_days_old:
                        cache_files.append((file_date, file))
                except:
                    continue
        
        if cache_files:
            # 使用最新的文件
            cache_files.sort(reverse=True)
            latest_file = cache_files[0][1]
            filepath = os.path.join(cache_dir, latest_file)
            
            try:
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                print(f"从缓存加载数据: {latest_file}")
                return data
            except:
                pass
        
        return None


# 使用示例
if __name__ == "__main__":
    # 测试数据获取
    symbols = ['AAPL', 'MSFT', 'TSLA']
    
    for symbol in symbols:
        print(f"\n获取 {symbol} 数据:")
        try:
            data = DataFallback.get_stock_data(symbol, period='1mo')
            print(f"  数据形状: {data.shape}")
            print(f"  最新收盘价: ${data['Close'].iloc[-1]:.2f}")
        except Exception as e:
            print(f"  错误: {e}")
