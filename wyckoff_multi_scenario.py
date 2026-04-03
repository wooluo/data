#!/usr/bin/env python3
"""
威科夫多场景选股框架 V1
根据市场环境自动切换策略

市场判断：
- 牛市：沪深300 > MA60 + MA20 > MA60（多头排列）
- 熊市：沪深300 < MA60 + MA20 < MA60（空头排列）
- 震荡：沪深300在MA20和MA60之间反复穿越

用法：
  python3 wyckoff_multi_scenario.py                # 自动判断+选股
  python3 wyckoff_multi_scenario.py --scenario bull  # 强制牛市策略
  python3 wyckoff_multi_scenario.py --scenario bear  # 强制熊市策略
  python3 wyckoff_multi_scenario.py --scenario range # 强制震荡策略
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import numpy as np

# 清除代理
for v in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','all_proxy','no_proxy']:
    os.environ.pop(v, None)

import requests


# ============================================================
# 数据获取
# ============================================================

def get_sina_kline(code: str, days: int = 120) -> Optional[List[Dict]]:
    symbol = f"sh{code}" if code.startswith('6') else f"sz{code}"
    url = f"https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?symbol={symbol}&scale=240&datalen={days}"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if not data:
            return None
        return [{'date': d['day'], 'open': float(d['open']), 'high': float(d['high']),
                 'low': float(d['low']), 'close': float(d['close']), 'volume': float(d['volume'])}
                for d in data]
    except:
        return None


def get_index_kline(code: str = 'sh000300', days: int = 120) -> Optional[List[Dict]]:
    """获取指数K线"""
    return get_sina_kline(code.replace('sh','').replace('sz','').replace('000300','000300'), days)


def get_index_data() -> Optional[List[Dict]]:
    """获取沪深300 K线"""
    url = "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?symbol=sh000300&scale=240&datalen=120"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if not data:
            return None
        return [{'date': d['day'], 'open': float(d['open']), 'high': float(d['high']),
                 'low': float(d['low']), 'close': float(d['close']), 'volume': float(d['volume'])}
                for d in data]
    except:
        return None


def get_stock_list() -> List[Dict]:
    """获取全部A股"""
    stocks = []
    for page in range(1, 50):
        url = f"https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page={page}&num=100&sort=changepercent&asc=0&node=hs_a"
        try:
            resp = requests.get(url, timeout=15)
            data = resp.json()
            if not data:
                break
            for item in data:
                name = item.get('name', '')
                if 'ST' in name or '*' in name:
                    continue
                stocks.append({
                    'code': item['code'], 'name': name,
                    'price': float(item.get('trade', 0) or 0),
                    'change_pct': float(item.get('changepercent', 0) or 0),
                    'mktcap': float(item.get('mktcap', 0) or 0) / 10000,  # 亿
                    'turnover': float(item.get('turnoverratio', 0) or 0),
                })
            if len(data) < 100:
                break
        except:
            break
    return stocks


# ============================================================
# 市场环境判断
# ============================================================

@dataclass
class MarketRegime:
    """市场环境"""
    scenario: str           # bull/bear/range
    index_price: float
    ma20: float
    ma60: float
    index_vs_ma20_pct: float  # 距MA20百分比
    trend_strength: float     # 趋势强度（MA20和MA60的距离）
    volatility: float         # 近20日波动率
    description: str

    def __str__(self):
        icons = {'bull': '🐂', 'bear': '🐻', 'range': '↔️'}
        return f"{icons.get(self.scenario, '?')} {self.description}"


def judge_market(index_data: List[Dict]) -> MarketRegime:
    """判断市场环境"""
    if not index_data or len(index_data) < 60:
        return MarketRegime('range', 0, 0, 0, 0, 0, 0, "数据不足")

    closes = np.array([d['close'] for d in index_data])
    current = closes[-1]
    ma20 = np.mean(closes[-20:])
    ma60 = np.mean(closes[-60:])

    # 距MA20
    vs_ma20 = (current - ma20) / ma20 * 100

    # 趋势强度：MA20和MA60的距离
    trend = (ma20 - ma60) / ma60 * 100

    # 波动率
    returns = np.diff(closes[-20:]) / closes[-21:-1]
    volatility = np.std(returns) * 100

    # 判断逻辑
    if ma20 > ma60 and current > ma20 and trend > 2:
        scenario = 'bull'
        desc = f"牛市（MA20>{ma60:.0f}，价格站上MA20）"
    elif ma20 < ma60 and current < ma20 and trend < -2:
        scenario = 'bear'
        desc = f"熊市（MA20<{ma60:.0f}，价格跌破MA20）"
    else:
        scenario = 'range'
        if ma20 > ma60:
            desc = f"震荡偏多（MA20>MA60，价格在MA20附近）"
        else:
            desc = f"震荡偏空（MA20<MA60，价格在MA20附近）"

    return MarketRegime(scenario, current, ma20, ma60, vs_ma20, trend, volatility, desc)


# ============================================================
# 策略定义
# ============================================================

@dataclass
class BullStrategy:
    """牛市策略：追强势SOS突破"""
    name = "牛市-强势突破"
    min_change_pct: float = 5.0       # 牛市要求更高涨幅
    min_volume_ratio: float = 2.5     # 放量突破
    max_volume_ratio: float = 8.0
    need_ma20_above: bool = True      # 必须站上MA20
    need_ma60_above: bool = False     # 不强制MA60
    prefer_continuous: bool = True    # 偏好连续突破
    min_market_cap: float = 100.0     # 牛市偏好中大市值
    max_5day_gain: float = 15.0       # 排除短期暴涨
    atr_range: Tuple[float, float] = (1.0, 5.0)
    score_weights: Dict = field(default_factory=lambda: {
        'change': 15, 'volume': 20, 'relative_strength': 25,
        'structure': 20, 'momentum': 20
    })

    # 牛市核心：追龙头、持股为主
    description = """
    牛市特征：资金充裕，板块轮动快，强者恒强
    选股逻辑：
    1. 涨幅≥5%，量比≥2.5，放量突破确认
    2. 偏好板块龙头（市值>100亿）
    3. 站上MA20，最好站上MA60
    4. 排除短期暴涨（5日涨幅<15%）
    5. 关注连续突破（2天内涨>4%+连续放量）
    操作策略：
    - 买入后持股，移动止盈从5%放宽到8%
    - 止盈目标放大到3-5倍ATR
    - 不轻易止损，给趋势更多空间
    """


@dataclass
class BearStrategy:
    """熊市策略：只做Spring超跌反弹"""
    name = "熊市-Spring抄底"
    min_change_pct: float = 3.0       # 熊市涨幅要求降低
    min_volume_ratio: float = 2.0     # 但仍需放量
    max_volume_ratio: float = 6.0
    need_ma20_above: bool = False     # 不要求站上MA20
    need_ma60_above: bool = False
    prefer_spring: bool = True        # 只做Spring
    min_bottom_days: int = 5          # 底部至少5天
    max_5day_gain: float = 10.0       # 排除短期反弹
    atr_range: Tuple[float, float] = (1.0, 4.0)
    score_weights: Dict = field(default_factory=lambda: {
        'spring_quality': 30, 'volume': 20, 'bottom_depth': 25,
        'relative_strength': 15, 'risk_reward': 10
    })

    description = """
    熊市特征：整体下跌，资金流出，反弹是出货机会
    选股逻辑：
    1. 只做Spring（破底翻）形态
    2. 底部至少震荡5天，恐慌抛售后缩量企稳
    3. 放量收回MA20下方区域即可
    4. 严格止损（入场价-3%），不抱幻想
    5. 快进快出，持仓不超过5天
    操作策略：
    - 仓位减半（单只≤10%）
    - 止损从-5%收紧到-3%
    - 止盈从+10%降低到+5-7%
    - 不做加仓，一次到位
    - 连续2次止损后暂停3天
    """


@dataclass
class RangeStrategy:
    """震荡市策略：支撑阻力高抛低吸"""
    name = "震荡市-区间操作"
    min_change_pct: float = 3.5
    min_volume_ratio: float = 2.0
    max_volume_ratio: float = 6.0
    need_ma20_above: bool = False
    need_ma60_above: bool = False
    prefer_range_bound: bool = True   # 偏好在震荡区间内
    range_lookback: int = 30          # 区间回看30天
    min_range_days: int = 10          # 区间至少10天
    max_5day_gain: float = 12.0
    atr_range: Tuple[float, float] = (0.8, 4.0)
    score_weights: Dict = field(default_factory=lambda: {
        'range_position': 25, 'volume': 20, 'support_test': 25,
        'change_quality': 15, 'volatility': 15
    })

    description = """
    震荡市特征：价格在区间内反复，趋势不明
    选股逻辑：
    1. 在30天震荡区间内，接近支撑位放量反弹
    2. 区间至少存在10天
    3. 支撑位有2次以上测试确认
    4. 突破区间上沿也可能是方向选择
    操作策略：
    - 区间下沿买入，上沿卖出
    - 止损设在区间下沿下方2%
    - 如果放量突破区间上沿，跟进做多
    - 持仓不超过10天
    - 不追涨，只在支撑位附近操作
    """


# ============================================================
# 信号检测
# ============================================================

def detect_bull_signal(closes, highs, lows, volumes, idx, strategy) -> Optional[Dict]:
    """牛市信号：SOS放量突破"""
    if idx < 30:
        return None

    change_pct = (closes[idx] - closes[idx-1]) / closes[idx-1] * 100
    if change_pct < strategy.min_change_pct:
        return None

    avg_vol = np.mean(volumes[max(0,idx-20):idx])
    if avg_vol == 0:
        return None
    vol_ratio = volumes[idx] / avg_vol
    if vol_ratio < strategy.min_volume_ratio or vol_ratio > strategy.max_volume_ratio:
        return None

    ma20 = np.mean(closes[max(0,idx-20):idx])
    if strategy.need_ma20_above and closes[idx] < ma20:
        return None

    # ATR
    tr_list = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
               for i in range(max(1,idx-20), idx+1)]
    atr = np.median(tr_list) if tr_list else 1
    atr_pct = atr / closes[idx] * 100
    if not (strategy.atr_range[0] <= atr_pct <= strategy.atr_range[1]):
        return None

    # 5日涨幅过滤
    if idx >= 5:
        gain_5d = (closes[idx] - closes[idx-5]) / closes[idx-5] * 100
        if gain_5d > strategy.max_5day_gain:
            return None

    # 威科夫阶段
    range_high = np.max(highs[max(0,idx-30):idx])
    range_low = np.min(lows[max(0,idx-30):idx])
    range_size = range_high - range_low
    if range_size == 0:
        return None
    position = (closes[idx] - range_low) / range_size

    phase = 'Unknown'
    if position > 0.8 and vol_ratio >= 2.0:
        phase = 'SOS'
    elif lows[idx] < range_low * 1.02 and closes[idx] > closes[idx-1] and vol_ratio >= 1.5:
        phase = 'Spring'

    if phase == 'Unknown':
        return None

    # 评分
    score = 30
    if 2.5 <= vol_ratio <= 4: score += 25
    elif vol_ratio >= 2: score += 15
    if change_pct >= 9.9: score += 20
    elif change_pct >= 7: score += 15
    elif change_pct >= 5: score += 10
    if phase == 'SOS': score += 15
    elif phase == 'Spring': score += 10
    if position > 0.9: score += 10  # 接近突破

    return {
        'change_pct': round(change_pct, 2),
        'volume_ratio': round(vol_ratio, 1),
        'atr_pct': round(atr_pct, 2),
        'phase': phase,
        'score': min(100, score),
        'position': round(position, 2),
        'ma20': round(ma20, 2),
    }


def detect_bear_signal(closes, highs, lows, volumes, idx, strategy) -> Optional[Dict]:
    """熊市信号：Spring抄底"""
    if idx < 30:
        return None

    # 熊市涨幅要求低
    change_pct = (closes[idx] - closes[idx-1]) / closes[idx-1] * 100
    if change_pct < strategy.min_change_pct:
        return None

    avg_vol = np.mean(volumes[max(0,idx-20):idx])
    if avg_vol == 0:
        return None
    vol_ratio = volumes[idx] / avg_vol
    if vol_ratio < strategy.min_volume_ratio or vol_ratio > strategy.max_volume_ratio:
        return None

    # ATR
    tr_list = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
               for i in range(max(1,idx-20), idx+1)]
    atr = np.median(tr_list) if tr_list else 1
    atr_pct = atr / closes[idx] * 100
    if not (strategy.atr_range[0] <= atr_pct <= strategy.atr_range[1]):
        return None

    # 关键：必须是Spring形态
    range_high = np.max(highs[max(0,idx-30):idx])
    range_low = np.min(lows[max(0,idx-30):idx])
    range_size = range_high - range_low
    if range_size == 0:
        return None

    # Spring：今天最低点接近或跌破区间下沿，但收盘高于昨天
    is_spring = (lows[idx] <= range_low * 1.03 and closes[idx] > closes[idx-1] and vol_ratio >= 2.0)
    if not is_spring:
        return None

    # 底部深度：最低点距今天数
    recent_lows = lows[max(0,idx-30):idx]
    min_idx = np.argmin(recent_lows)
    if min_idx > len(recent_lows) - strategy.min_bottom_days:
        return None

    # 5日涨幅过滤
    if idx >= 5:
        gain_5d = (closes[idx] - closes[idx-5]) / closes[idx-5] * 100
        if gain_5d > strategy.max_5day_gain:
            return None

    # Spring质量评分
    position = (closes[idx] - range_low) / range_size
    score = 30
    # 收回幅度越大越好
    if closes[idx] > (range_low + range_size * 0.3):
        score += 25  # 收回到区间中上部
    elif closes[idx] > (range_low + range_size * 0.1):
        score += 15
    # 量比
    if 2 <= vol_ratio <= 4: score += 20
    elif vol_ratio >= 2: score += 10
    # 底部深度
    if min_idx < len(recent_lows) * 0.3:
        score += 15  # 底部很早，积累充分
    # 涨幅
    if change_pct >= 5: score += 10

    return {
        'change_pct': round(change_pct, 2),
        'volume_ratio': round(vol_ratio, 1),
        'atr_pct': round(atr_pct, 2),
        'phase': 'Spring',
        'score': min(100, score),
        'position': round(position, 2),
        'bottom_days_ago': len(recent_lows) - min_idx,
    }


def detect_range_signal(closes, highs, lows, volumes, idx, strategy) -> Optional[Dict]:
    """震荡市信号：支撑位反弹"""
    if idx < strategy.range_lookback:
        return None

    change_pct = (closes[idx] - closes[idx-1]) / closes[idx-1] * 100
    if change_pct < strategy.min_change_pct:
        return None

    avg_vol = np.mean(volumes[max(0,idx-20):idx])
    if avg_vol == 0:
        return None
    vol_ratio = volumes[idx] / avg_vol
    if vol_ratio < strategy.min_volume_ratio or vol_ratio > strategy.max_volume_ratio:
        return None

    # 震荡区间
    lookback = min(idx, strategy.range_lookback)
    range_high = np.max(highs[idx-lookback:idx])
    range_low = np.min(lows[idx-lookback:idx])
    range_size = range_high - range_low
    if range_size == 0:
        return None

    # 区间波动率检查（不能太大）
    range_pct = range_size / np.mean(closes[idx-lookback:idx]) * 100
    if range_pct > 30:  # 区间超过30%不算震荡
        return None

    position = (closes[idx] - range_low) / range_size

    # 支撑位反弹：在区间下部放量上涨
    is_support_bounce = (position < 0.4 and change_pct > 0 and vol_ratio >= 2.0)
    # 突破上沿：可能方向选择
    is_breakout = (position > 0.9 and vol_ratio >= 2.5 and closes[idx] > range_high)

    if not is_support_bounce and not is_breakout:
        return None

    # 支撑测试次数
    support_tests = 0
    for i in range(max(1, idx-lookback), idx):
        if lows[i] <= range_low * 1.05 and closes[i] > closes[i-1]:
            support_tests += 1

    # ATR
    tr_list = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
               for i in range(max(1,idx-20), idx+1)]
    atr = np.median(tr_list) if tr_list else 1
    atr_pct = atr / closes[idx] * 100
    if not (strategy.atr_range[0] <= atr_pct <= strategy.atr_range[1]):
        return None

    # 5日涨幅过滤
    if idx >= 5:
        gain_5d = (closes[idx] - closes[idx-5]) / closes[idx-5] * 100
        if gain_5d > strategy.max_5day_gain:
            return None

    signal_type = '突破上沿' if is_breakout else '支撑反弹'
    score = 30
    if is_breakout: score += 20
    if support_tests >= 2: score += 15
    if 2 <= vol_ratio <= 4: score += 15
    elif vol_ratio >= 2: score += 8
    if change_pct >= 5: score += 10
    if position < 0.2: score += 10  # 接近下沿更好

    return {
        'change_pct': round(change_pct, 2),
        'volume_ratio': round(vol_ratio, 1),
        'atr_pct': round(atr_pct, 2),
        'phase': signal_type,
        'score': min(100, score),
        'position': round(position, 2),
        'support_tests': support_tests,
        'range_pct': round(range_pct, 1),
    }


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='威科夫多场景选股')
    parser.add_argument('--scenario', choices=['bull', 'bear', 'range', 'auto'], default='auto',
                       help='市场场景 (auto=自动判断)')
    args = parser.parse_args()

    print("=" * 70)
    print("  威科夫多场景选股系统 V1")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # 1. 判断市场环境
    print("\n📊 判断市场环境...")
    index_data = get_index_data()
    regime = judge_market(index_data) if index_data else MarketRegime('range', 0, 0, 0, 0, 0, 0, "数据不足")
    print(f"  {regime}")
    print(f"  沪深300: {regime.index_price:.2f}")
    print(f"  MA20: {regime.ma20:.2f}, MA60: {regime.ma60:.2f}")
    print(f"  距MA20: {regime.index_vs_ma20_pct:+.2f}%")
    print(f"  趋势强度: {regime.trend_strength:+.2f}%")
    print(f"  波动率: {regime.volatility:.2f}%")

    scenario = args.scenario if args.scenario != 'auto' else regime.scenario

    # 2. 选择策略
    strategies = {
        'bull': BullStrategy(),
        'bear': BearStrategy(),
        'range': RangeStrategy(),
    }
    strategy = strategies[scenario]
    detectors = {
        'bull': detect_bull_signal,
        'bear': detect_bear_signal,
        'range': detect_range_signal,
    }
    detect_fn = detectors[scenario]

    print(f"\n📋 使用策略: {strategy.name}")
    print(f"  {scenario.upper()}策略核心:")
    for line in strategy.description.strip().split('\n')[1:8]:
        print(f"  {line.strip()}")

    # 3. 获取股票列表
    print(f"\n🔍 获取股票列表...")
    stocks = get_stock_list()
    print(f"  共{len(stocks)}只股票")

    # 4. 扫描信号
    print(f"\n🚀 开始扫描...")
    signals = []
    tested = 0
    for stock in stocks:
        tested += 1
        if tested % 50 == 0:
            print(f"\r  进度: {tested}/{len(stocks)}, 信号: {len(signals)}", end="", flush=True)

        klines = get_sina_kline(stock['code'], 120)
        if not klines or len(klines) < 60:
            time.sleep(0.2)
            continue

        closes = np.array([k['close'] for k in klines])
        highs = np.array([k['high'] for k in klines])
        lows = np.array([k['low'] for k in klines])
        volumes = np.array([k['volume'] for k in klines])

        # 在最近10天内检测信号
        for idx in range(max(40, len(klines)-10), len(klines)):
            sig = detect_fn(closes, highs, lows, volumes, idx, strategy)
            if sig:
                sig['code'] = stock['code']
                sig['name'] = stock['name']
                sig['price'] = float(closes[idx])
                sig['date'] = klines[idx]['date']
                signals.append(sig)
                break  # 每只股票只取最新一个信号

        time.sleep(0.3)

    print(f"\r  完成: 测试{tested}只, 信号{len(signals)}个    ")

    # 5. 排序输出
    signals.sort(key=lambda x: -x['score'])

    print(f"\n{'='*70}")
    print(f"  {strategy.name} 扫描结果（{scenario.upper()}环境）")
    print(f"{'='*70}")
    print(f"  {'代码':8s} {'名称':10s} {'现价':>8s} {'涨幅':>8s} {'量比':>6s} {'阶段':10s} {'得分':>6s}")
    print(f"  {'-'*60}")

    for sig in signals[:30]:
        print(f"  {sig['code']:8s} {sig['name']:10s} {sig['price']:>8.2f} {sig['change_pct']:>+7.2f}% {sig['volume_ratio']:>6.1f} {sig['phase']:10s} {sig['score']:>6.0f}")

    if not signals:
        print("  未发现符合条件的信号")

    # 6. 保存结果
    out_path = os.path.expanduser(f'~/.openclaw/workspace/data/multi_scenario_{scenario}_{datetime.now().strftime("%Y%m%d")}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'scenario': scenario,
            'regime': asdict(regime),
            'strategy': strategy.name,
            'total_tested': tested,
            'signals': signals,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_path}")


if __name__ == '__main__':
    main()
