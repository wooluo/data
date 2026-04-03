#!/usr/bin/env python3
"""
威科夫策略回测框架 V1
用于验证和优化扫描参数

用法:
  python3 wyckoff_backtest_engine.py                    # 默认参数回测
  python3 wyckoff_backtest_engine.py --stock 300293     # 单股详细分析
  python3 wyckoff_backtest_engine.py --optimize          # 参数优化模式
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

# ---- 数据获取 ----
def get_sina_kline(code: str, days: int = 750) -> Optional[List[Dict]]:
    """新浪K线"""
    import requests
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


def get_stock_list_sina() -> List[Dict]:
    """获取全部A股列表"""
    import requests
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
                })
            if len(data) < 100:
                break
        except:
            break
    return stocks


# ---- 威科夫信号检测（纯函数，方便回测） ----
@dataclass
class SignalParams:
    """信号检测参数"""
    min_change_pct: float = 5.0
    min_volume_ratio: float = 2.0
    max_volume_ratio: float = 10.0
    lookback_days: int = 30       # 底部结构回看天数
    min_bottom_days: int = 3      # 最低点距今天数
    atr_period: int = 20          # ATR计算周期
    atr_min_pct: float = 0.8      # ATR最低百分比
    atr_max_pct: float = 6.0      # ATR最高百分比
    stop_loss_atr_mult: float = 2.0   # 止损ATR倍数
    stop_loss_max_pct: float = 0.05   # 止损最大百分比
    take_profit_atr_mult: float = 3.0 # 止盈ATR倍数
    take_profit_min_pct: float = 0.10  # 止盈最小百分比
    trailing_start_pct: float = 0.05   # 移动止盈启动%
    trailing_drop_pct: float = 0.04    # 移动止盈回撤%
    max_hold_days: int = 30             # 最大持仓天数
    cooldown_days: int = 5              # 止损后冷却期


@dataclass
class Trade:
    """交易记录"""
    code: str
    name: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    exit_reason: str
    hold_days: int
    pnl_pct: float
    atr_pct: float
    volume_ratio: float
    wyckoff_phase: str
    signal_score: float


def detect_signal(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                  volumes: np.ndarray, idx: int, params: SignalParams) -> Optional[Dict]:
    """
    在idx位置检测威科夫信号
    返回信号信息或None
    """
    if idx < params.lookback_days + 5:
        return None

    # 当日涨幅
    change_pct = (closes[idx] - closes[idx-1]) / closes[idx-1] * 100
    if change_pct < params.min_change_pct:
        return None

    # 量比
    avg_vol = np.mean(volumes[max(0, idx-20):idx])
    if avg_vol == 0:
        return None
    vol_ratio = volumes[idx] / avg_vol
    if vol_ratio < params.min_volume_ratio or vol_ratio > params.max_volume_ratio:
        return None

    # ATR
    tr_list = []
    for i in range(max(1, idx - params.atr_period), idx + 1):
        h, l, pc = highs[i], lows[i], closes[i-1]
        tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(tr_list) < 5:
        return None
    atr = np.median(tr_list)
    atr_pct = atr / closes[idx] * 100
    if atr_pct < params.atr_min_pct or atr_pct > params.atr_max_pct:
        return None

    # 底部结构：最低点在min_bottom_days天前
    recent_lows = lows[max(0, idx - params.lookback_days):idx]
    min_low = np.min(recent_lows)
    min_idx = np.argmin(recent_lows)
    if min_idx > len(recent_lows) - params.min_bottom_days:
        return None  # 最低点太近，还在下跌

    # MA20突破
    ma20 = np.mean(closes[max(0, idx-20):idx])
    if closes[idx] < ma20:
        return None

    # 威科夫阶段识别
    range_high = np.max(highs[max(0, idx - params.lookback_days):idx])
    range_low = min_low
    range_size = range_high - range_low
    if range_size == 0:
        return None
    position = (closes[idx] - range_low) / range_size

    phase = 'Unknown'
    if position > 0.8 and vol_ratio >= 2.0:
        phase = 'SOS'
    elif lows[idx] < range_low * 1.02 and closes[idx] > closes[idx-1] and vol_ratio >= 1.5:
        phase = 'Spring'
    elif 0.1 < position < 0.4 and closes[idx] > ma20 * 0.98:
        phase = 'LPS'

    if phase == 'Unknown':
        return None

    # 评分
    score = 40
    if 2 <= vol_ratio <= 4: score += 25
    elif 4 < vol_ratio <= 6: score += 20
    elif vol_ratio >= 2: score += 12
    if change_pct >= 9.9: score += 20
    elif change_pct >= 7: score += 15
    elif change_pct >= 5: score += 10
    if phase == 'SOS': score += 10
    elif phase == 'Spring': score += 8

    return {
        'change_pct': round(change_pct, 2),
        'volume_ratio': round(vol_ratio, 1),
        'atr_pct': round(atr_pct, 2),
        'atr': round(atr, 4),
        'ma20': round(ma20, 2),
        'phase': phase,
        'score': min(100, score),
        'range_low': round(range_low, 2),
        'range_high': round(range_high, 2),
    }


def simulate_trade(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                   entry_idx: int, signal: Dict, params: SignalParams) -> Optional[Trade]:
    """模拟交易"""
    entry_price = closes[entry_idx]
    stop_loss = entry_price * (1 - max(params.stop_loss_atr_mult * signal['atr_pct'] / 100, params.stop_loss_max_pct))
    take_profit = entry_price * (1 + max(params.take_profit_atr_mult * signal['atr_pct'] / 100, params.take_profit_min_pct))

    highest = entry_price
    for day in range(1, params.max_hold_days + 1):
        if entry_idx + day >= len(closes):
            break

        h, l, c = highs[entry_idx + day], lows[entry_idx + day], closes[entry_idx + day]
        highest = max(highest, h)

        # 移动止盈
        if (highest - entry_price) / entry_price >= params.trailing_start_pct:
            trailing_stop = highest * (1 - params.trailing_drop_pct)
            if l <= trailing_stop:
                return Trade(
                    entry_price=entry_price,
                    exit_price=trailing_stop, exit_reason='移动止盈',
                    hold_days=day, pnl_pct=(trailing_stop - entry_price) / entry_price * 100,
                    atr_pct=signal['atr_pct'], volume_ratio=signal['volume_ratio'],
                    wyckoff_phase=signal['phase'], signal_score=signal['score'],
                    code='', name='', entry_date='', exit_date=''
                )

        # 止损
        if l <= stop_loss:
            return Trade(
                    entry_price=entry_price,
                    exit_price=stop_loss, exit_reason='止损',
                hold_days=day, pnl_pct=(stop_loss - entry_price) / entry_price * 100,
                atr_pct=signal['atr_pct'], volume_ratio=signal['volume_ratio'],
                wyckoff_phase=signal['phase'], signal_score=signal['score'],
                code='', name='', entry_date='', exit_date=''
            )

        # 止盈
        if h >= take_profit:
            return Trade(
                    entry_price=entry_price,
                    exit_price=take_profit, exit_reason='止盈',
                hold_days=day, pnl_pct=(take_profit - entry_price) / entry_price * 100,
                atr_pct=signal['atr_pct'], volume_ratio=signal['volume_ratio'],
                wyckoff_phase=signal['phase'], signal_score=signal['score'],
                code='', name='', entry_date='', exit_date=''
            )

    # 到期
    return Trade(
                    entry_price=entry_price,
                    exit_price=c, exit_reason='到期',
        hold_days=params.max_hold_days, pnl_pct=(c - entry_price) / entry_price * 100,
        atr_pct=signal['atr_pct'], volume_ratio=signal['volume_ratio'],
        wyckoff_phase=signal['phase'], signal_score=signal['score'],
        code='', name='', entry_date='', exit_date=''
    )


def backtest_stock(klines: List[Dict], code: str, name: str,
                   params: SignalParams) -> List[Trade]:
    """对单只股票回测"""
    if not klines or len(klines) < 60:
        return []

    closes = np.array([k['close'] for k in klines])
    highs = np.array([k['high'] for k in klines])
    lows = np.array([k['low'] for k in klines])
    volumes = np.array([k['volume'] for k in klines])
    dates = [k['date'] for k in klines]

    trades = []
    last_exit_idx = -params.cooldown_days  # 冷却期

    for idx in range(40, len(closes) - params.max_hold_days):
        if idx - last_exit_idx < params.cooldown_days:
            continue

        signal = detect_signal(closes, highs, lows, volumes, idx, params)
        if signal is None:
            continue

        trade = simulate_trade(closes, highs, lows, idx, signal, params)
        if trade:
            trade.code = code
            trade.name = name
            trade.entry_date = dates[idx]
            trade.exit_date = dates[min(idx + trade.hold_days, len(dates) - 1)]
            trades.append(trade)
            last_exit_idx = idx + trade.hold_days

    return trades


def print_backtest_report(trades: List[Trade], title: str = ""):
    """打印回测报告"""
    if not trades:
        print("  无交易记录")
        return

    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]
    total_pnl = sum(t.pnl_pct for t in trades)

    print(f"\n{'='*70}")
    if title:
        print(f"  {title}")
    print(f"{'='*70}")
    print(f"  总交易: {len(trades)}次")
    print(f"  胜率: {len(wins)/len(trades)*100:.1f}% ({len(wins)}胜/{len(losses)}负)")
    print(f"  总收益: {total_pnl:.1f}%")
    print(f"  平均收益: {total_pnl/len(trades):.1f}%")
    if wins:
        print(f"  平均盈利: {sum(t.pnl_pct for t in wins)/len(wins):.1f}%")
    if losses:
        print(f"  平均亏损: {sum(t.pnl_pct for t in losses)/len(losses):.1f}%")

    # 出场原因
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print(f"\n  出场原因:")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = c / len(trades) * 100
        avg_pnl = np.mean([t.pnl_pct for t in trades if t.exit_reason == r])
        print(f"    {r}: {c}次 ({pct:.0f}%), 平均{avg_pnl:.1f}%")

    # 威科夫阶段统计
    phases = {}
    for t in trades:
        phases[t.wyckoff_phase] = phases.get(t.wyckoff_phase, [])
        phases[t.wyckoff_phase].append(t.pnl_pct)
    print(f"\n  阶段表现:")
    for phase, pnls in sorted(phases.items(), key=lambda x: -np.mean(x[1])):
        w = len([p for p in pnls if p > 0])
        print(f"    {phase}: {len(pnls)}次, 胜率{w/len(pnls)*100:.0f}%, 平均{np.mean(pnls):.1f}%")


def optimize_params():
    """参数优化：遍历关键参数组合"""
    print("🧪 威科夫参数优化模式")
    print("=" * 70)

    # 获取测试股票（从知识库已知的好标的）
    test_stocks = [
        ('002475', '立讯精密'), ('300573', '兴齐眼药'), ('300702', '天宇股份'),
        ('002472', '双环传动'), ('601888', '中国中免'), ('601688', '华泰证券'),
        ('300750', '宁德时代'), ('002703', '浙江世宝'), ('300293', '蓝英装备'),
    ]

    # 参数组合
    param_sets = [
        {"name": "宽松", "min_change_pct": 3.0, "min_volume_ratio": 1.5, "atr_min_pct": 0.5, "atr_max_pct": 8.0},
        {"name": "默认", "min_change_pct": 5.0, "min_volume_ratio": 2.0, "atr_min_pct": 0.8, "atr_max_pct": 6.0},
        {"name": "严格", "min_change_pct": 7.0, "min_volume_ratio": 3.0, "atr_min_pct": 1.0, "atr_max_pct": 5.0},
        {"name": "高量比", "min_change_pct": 5.0, "min_volume_ratio": 3.0, "atr_min_pct": 0.8, "atr_max_pct": 6.0},
        {"name": "Spring专用", "min_change_pct": 3.0, "min_volume_ratio": 1.5, "atr_min_pct": 0.5, "atr_max_pct": 6.0},
        {"name": "SOS专用", "min_change_pct": 5.0, "min_volume_ratio": 2.5, "atr_min_pct": 0.8, "atr_max_pct": 5.0},
    ]

    results = []
    for ps in param_sets:
        params = SignalParams(**{k: v for k, v in ps.items() if k != 'name'})
        all_trades = []
        for code, name in test_stocks:
            print(f"\r  测试 {ps['name']}: {name}({code})", end="", flush=True)
            klines = get_sina_kline(code, 750)
            if klines:
                trades = backtest_stock(klines, code, name, params)
                all_trades.extend(trades)

        if all_trades:
            wins = len([t for t in all_trades if t.pnl_pct > 0])
            total = len(all_trades)
            total_pnl = sum(t.pnl_pct for t in all_trades)
            avg_pnl = total_pnl / total
            win_rate = wins / total * 100
            results.append({
                'name': ps['name'],
                'trades': total,
                'win_rate': win_rate,
                'total_pnl': round(total_pnl, 1),
                'avg_pnl': round(avg_pnl, 1),
            })
        else:
            results.append({'name': ps['name'], 'trades': 0, 'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0})

    print("\n\n" + "=" * 70)
    print("📊 参数优化结果（按总收益排序）")
    print("=" * 70)
    print(f"{'策略':12s} {'交易':>6s} {'胜率':>8s} {'总收益':>10s} {'平均':>8s}")
    print("-" * 50)
    for r in sorted(results, key=lambda x: -x['total_pnl']):
        if r['trades'] > 0:
            print(f"{r['name']:12s} {r['trades']:>6d} {r['win_rate']:>7.1f}% {r['total_pnl']:>9.1f}% {r['avg_pnl']:>7.1f}%")
        else:
            print(f"{r['name']:12s} {r['trades']:>6d} {'N/A':>8s} {'N/A':>10s} {'N/A':>8s}")

    return results


def main():
    parser = argparse.ArgumentParser(description='威科夫策略回测框架')
    parser.add_argument('--stock', help='单股分析 (代码)')
    parser.add_argument('--optimize', action='store_true', help='参数优化模式')
    parser.add_argument('--days', type=int, default=750, help='回测天数')
    args = parser.parse_args()

    params = SignalParams()

    if args.optimize:
        results = optimize_params()
        # 保存结果
        out_path = os.path.expanduser('~/.openclaw/workspace/data/backtest_optimization.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {out_path}")
        return

    if args.stock:
        # 单股详细分析
        code = args.stock
        name = code
        print(f"🔍 {code} 威科夫信号回测分析")
        klines = get_sina_kline(code, args.days)
        if not klines:
            print("无法获取K线数据")
            return

        print(f"  数据范围: {klines[0]['date']} ~ {klines[-1]['date']} ({len(klines)}天)")
        trades = backtest_stock(klines, code, name, params)
        print_backtest_report(trades, f"{code} 威科夫回测")

        # 打印每笔交易
        if trades:
            print(f"\n  {'#':>3s} {'入场日':>12s} {'出场日':>12s} {'入场价':>8s} {'出场价':>8s} {'收益%':>8s} {'天数':>4s} {'原因':>8s} {'阶段':>8s}")
            print("  " + "-" * 85)
            for i, t in enumerate(trades):
                print(f"  {i+1:>3d} {t.entry_date:>12s} {t.exit_date:>12s} {t.entry_price:>8.2f} {t.exit_price:>8.2f} {t.pnl_pct:>+7.1f}% {t.hold_days:>4d} {t.exit_reason:>8s} {t.wyckoff_phase:>8s}")
        return

    # 默认：全市场回测
    print("🌍 威科夫策略全市场回测")
    print("=" * 70)
    print(f"  参数: 涨幅>={params.min_change_pct}%, 量比>={params.min_volume_ratio}")
    print(f"  止损: max({params.stop_loss_atr_mult}xATR, {params.stop_loss_max_pct*100}%)")
    print(f"  止盈: max({params.take_profit_atr_mult}xATR, {params.take_profit_min_pct*100}%)")
    print(f"  移动止盈: 盈利>{params.trailing_start_pct*100}%后回撤{params.trailing_drop_pct*100}%出场")
    print()

    stocks = get_stock_list_sina()
    print(f"  获取到 {len(stocks)} 只股票，开始回测...")

    all_trades = []
    tested = 0
    for stock in stocks:
        tested += 1
        if tested % 50 == 0:
            print(f"\r  进度: {tested}/{len(stocks)}, 信号: {len(all_trades)}", end="", flush=True)
        klines = get_sina_kline(stock['code'], args.days)
        if klines:
            trades = backtest_stock(klines, stock['code'], stock['name'], params)
            all_trades.extend(trades)
        time.sleep(0.3)

    print(f"\r  完成: 测试{tested}只, 信号{len(all_trades)}个    ")

    print_backtest_report(all_trades, "全市场回测结果")

    # 按股票汇总
    stock_stats = {}
    for t in all_trades:
        key = f"{t.code} {t.name}"
        if key not in stock_stats:
            stock_stats[key] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
        stock_stats[key]['trades'] += 1
        if t.pnl_pct > 0:
            stock_stats[key]['wins'] += 1
        stock_stats[key]['total_pnl'] += t.pnl_pct

    # 过滤：至少2次信号且胜率>50%
    good = {k: v for k, v in stock_stats.items() if v['trades'] >= 2 and v['wins']/v['trades'] > 0.5}
    print(f"\n📊 通过筛选的股票（≥2次信号, 胜率>50%）: {len(good)}只")
    print(f"{'股票':20s} {'信号':>6s} {'胜率':>8s} {'总收益':>10s}")
    print("-" * 50)
    for code, s in sorted(good.items(), key=lambda x: -x[1]['total_pnl'])[:20]:
        wr = s['wins'] / s['trades'] * 100
        print(f"{code:20s} {s['trades']:>6d} {wr:>7.1f}% {s['total_pnl']:>9.1f}%")

    # 保存结果
    out_path = os.path.expanduser('~/.openclaw/workspace/data/backtest_engine_results.json')
    result_data = {
        'timestamp': datetime.now().isoformat(),
        'params': asdict(params),
        'total_trades': len(all_trades),
        'stocks_tested': tested,
        'trades': [asdict(t) for t in all_trades[:100]],  # 保存前100条
    }
    with open(out_path, 'w') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == '__main__':
    main()
