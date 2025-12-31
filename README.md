# 美股趋势上涨策略脚本

## 功能概述

本脚本针对美股市场，基于趋势上涨策略，提供买和卖股票的时机。主要功能包括：

1. **美股数据获取**：通过 Yahoo Finance API 获取实时美股数据
2. **技术指标计算**：计算移动平均线、MACD、RSI、布林带等关键技术指标
3. **买卖信号生成**：基于多重条件生成买入和卖出信号
4. **策略表现分析**：模拟交易并分析策略表现
5. **可视化展示**：生成价格走势图和技术指标图表

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 基本使用

```bash
# 分析苹果股票 (AAPL)
python stock_strategy.py

# 分析特定股票
python stock_strategy.py --symbol MSFT

# 使用不同时间周期
python stock_strategy.py --symbol GOOGL --period 1y

# 显示图表
python stock_strategy.py --symbol AAPL --plot

# 保存图表
python stock_strategy.py --symbol AAPL --save-plot output.png

# 导出结果
python stock_strategy.py --symbol AAPL --export results.json
```

### 高级选项

```bash
# 分析多个股票
python batch_analysis.py

# 自定义策略参数
python custom_strategy.py
```

## 策略原理

### 买入信号条件

1. **上升趋势确认**：价格在20日均线之上，且20日均线在50日均线之上
2. **MACD金叉**：MACD线上穿信号线
3. **RSI超卖回升**：RSI从超卖区域（<30）回升
4. **布林带下轨支撑**：价格触及或接近布林带下轨
5. **成交量放大**：成交量显著高于20日平均成交量

### 卖出信号条件

1. **下降趋势确认**：价格在20日均线之下，且20日均线在50日均线之下
2. **MACD死叉**：MACD线下穿信号线
3. **RSI超买回落**：RSI从超买区域（>70）回落
4. **布林带上轨阻力**：价格触及或接近布林带上轨
5. **成交量萎缩**：成交量显著低于20日平均成交量

### 信号强度计算

每个信号条件都有相应的权重，当满足至少3个条件且总强度超过0.5时，生成交易信号。

## 配置文件

`config.json` 包含可配置的参数：

```json
{
  "default_symbols": ["AAPL", "MSFT", "GOOGL", ...],
  "default_period": "6mo",
  "default_interval": "1d",
  "strategy_parameters": {
    "sma_short": 20,
    "sma_long": 50,
    ...
  },
  "trading_settings": {
    "initial_capital": 10000,
    "commission_rate": 0.001,
    ...
  }
}
```

## 输出示例

```
============================================================
运行美股趋势上涨策略 - AAPL
============================================================
正在获取 AAPL 的数据...
成功获取 126 条数据
数据时间范围: 2024-07-01 到 2024-12-30
计算技术指标...
技术指标计算完成
生成买卖信号...
生成 8 个交易信号
分析策略表现...

============================================================
策略分析结果
============================================================
股票代码: AAPL
分析周期: 6mo
数据间隔: 1d
总交易次数: 3
胜率: 66.67%
平均收益率: 5.23%
最大收益: 8.45%
最小收益: -2.15%

买入信号总数: 4
卖出信号总数: 4

最近的交易信号:
  2024-12-15: BUY @ $195.23 (强度: 0.65)
  2024-12-10: SELL @ $192.45 (强度: -0.58)

建议:
  当前建议: 买入 AAPL
  理由: 检测到4个买入信号条件满足
```

## 文件结构

```
StockStrategy/
├── stock_strategy.py      # 主策略脚本
├── requirements.txt       # Python依赖包
├── config.json           # 配置文件
├── README.md            # 使用说明
├── batch_analysis.py    # 批量分析脚本（可选）
└── examples/           # 示例目录
    ├── basic_usage.py
    └── custom_strategy.py
```

## 注意事项

1. **数据延迟**：Yahoo Finance 数据有15分钟延迟
2. **历史回测**：策略基于历史数据，过去表现不代表未来结果
3. **风险管理**：实际交易请结合风险管理
4. **参数优化**：可根据市场情况调整策略参数

## 许可证

MIT License
