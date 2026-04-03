#!/usr/bin/env python3
"""
威科夫SOS信号全市场扫描器 v8
多数据源支持 (Ashare + 东方财富备用)
- 自动故障切换
- 趋势确认
- 威科夫结构识别
- 量价分析
"""

import hashlib
import json
import logging
import os
import pickle
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 清除代理设置
for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'all_proxy', 'no_proxy']:
    os.environ.pop(proxy_var, None)

# 尝试导入Ashare
try:
    from Ashare import get_price as ashare_get_price
    ASHARE_AVAILABLE = True
except ImportError:
    ASHARE_AVAILABLE = False


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """配置日志系统"""
    logger = logging.getLogger('wyckoff_scanner')
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


@dataclass
class ScannerConfig:
    """扫描器配置 V9 - 专业优化版"""
    max_workers: int = 2
    request_delay: float = 0.8
    request_timeout: int = 15
    request_retries: int = 3

    # 基础过滤
    min_change_pct: float = 5.0
    min_price: float = 3.0
    max_price: float = 100.0

    # 动态量比阈值（3-8范围）
    min_volume_ratio: float = 3.0
    max_volume_ratio: float = 8.0  # 超过8视为异常

    # 量价健康度：涨幅需 >= 量比 × health_factor
    volume_price_health_factor: float = 1.5

    # 弱市阈值
    weak_market_threshold: float = -0.5  # 大盘跌0.5%以上视为弱市
    weak_market_min_change: float = 7.0  # 弱市要求涨幅>7%

    # 行业强度
    industry_top_percent: float = 30.0  # 只保留行业前30%

    # 结构确认
    structure_confirm_days: int = 5

    cache_enabled: bool = True
    cache_ttl_hours: int = 8
    log_level: str = "INFO"

    data_sources: List[str] = field(default_factory=lambda: ['ashare', 'sina', 'em'])

    def __post_init__(self):
        self.cache_dir = Path.home() / ".openclaw/workspace/cache"


@dataclass
class SOSSignal:
    """SOS信号"""
    code: str
    name: str
    price: float
    change_pct: float
    volume: float
    amount: float
    signal_type: str
    structure: str
    volume_ratio: float
    quality: str
    score: float
    data_source: str = ""


class DataCache:
    """数据缓存"""

    def __init__(self, cache_dir: Path, ttl_hours: int = 8):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self._memory_cache: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        cache_key = hashlib.md5(key.encode()).hexdigest()

        if cache_key in self._memory_cache:
            timestamp, data = self._memory_cache[cache_key]
            if (time.time() - timestamp) <= self.ttl_seconds:
                return data
            del self._memory_cache[cache_key]

        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    timestamp, data = pickle.load(f)
                if (time.time() - timestamp) <= self.ttl_seconds:
                    self._memory_cache[cache_key] = (timestamp, data)
                    return data
                cache_path.unlink()
            except:
                cache_path.unlink(missing_ok=True)
        return None

    def set(self, key: str, data: Any) -> None:
        cache_key = hashlib.md5(key.encode()).hexdigest()
        timestamp = time.time()
        self._memory_cache[cache_key] = (timestamp, data)

        cache_path = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump((timestamp, data), f)
        except:
            pass

    def get_stats(self) -> Dict:
        disk_files = list(self.cache_dir.glob("*.pkl"))
        disk_size = sum(f.stat().st_size for f in disk_files) if disk_files else 1
        return {
            'memory': len(self._memory_cache),
            'disk': len(disk_files),
            'size_mb': round(disk_size / 1048576, 2)
        }


class MultiSourceAPI:
    """多数据源API"""

    # 新浪API
    SINA_LIST_URL = "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
    SINA_QUOTE_URL = "https://hq.sinajs.cn/list="

    # 东方财富API
    EM_LIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"
    EM_QUOTE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

    def __init__(self, config: ScannerConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._session = None
        self._current_source = 'sina'

    def _get_session(self) -> requests.Session:
        if self._session is None:
            session = requests.Session()
            session.trust_env = False

            retry = Retry(total=self.config.request_retries, backoff_factor=1)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("https://", adapter)
            session.mount("http://", adapter)

            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://finance.sina.com.cn/',
            })
            self._session = session
        return self._session

    def _request(self, url: str, source: str = 'sina') -> Optional[requests.Response]:
        session = self._get_session()
        time.sleep(self.config.request_delay + random.uniform(0, 0.3))

        try:
            resp = session.get(url, timeout=self.config.request_timeout)
            return resp
        except Exception as e:
            self.logger.warning(f"[{source}] 请求失败: {str(e)[:40]}")
            return None

    def get_stock_list(self, min_change: float = 3.0) -> Tuple[List[Dict], str]:
        """获取股票列表，返回(数据, 数据源)"""

        # 尝试新浪API
        stocks = self._get_stock_list_sina(min_change)
        if stocks:
            return stocks, 'sina'

        # 备用：东方财富API
        stocks = self._get_stock_list_em(min_change)
        if stocks:
            return stocks, 'em'

        return [], ''

    def _get_stock_list_sina(self, min_change: float) -> List[Dict]:
        """新浪涨幅榜"""
        stocks = []
        page = 1

        while len(stocks) < 500:
            url = f"{self.SINA_LIST_URL}?page={page}&num=100&sort=changepercent&asc=0&node=hs_a"
            resp = self._request(url, 'sina')
            if resp is None:
                break

            try:
                data = resp.json()
                if not data:
                    break

                for item in data:
                    name = item.get('name', '')
                    code = item.get('code', '')

                    if 'ST' in name or 'st' in name or '*' in name:
                        continue

                    try:
                        change_pct = float(item.get('changepercent', 0) or 0)
                        price = float(item.get('trade', 0) or 0)
                        pe = float(item.get('per', 0) or 0)
                    except:
                        continue

                    if change_pct < min_change:
                        continue
                    if price < self.config.min_price or price > self.config.max_price:
                        continue

                    stocks.append({
                        'code': code,
                        'name': name,
                        'symbol': item.get('symbol', ''),
                        'price': price,
                        'change_pct': change_pct,
                        'volume': float(item.get('volume', 0) or 0),
                        'amount': float(item.get('amount', 0) or 0),
                        'high': float(item.get('high', 0) or 0),
                        'low': float(item.get('low', 0) or 0),
                        'settlement': float(item.get('settlement', 0) or 0),
                        'per': pe,
                        'nmc': float(item.get('nmc', 0) or 0),
                    })

                if len(data) < 100:
                    break
                page += 1

            except:
                break

        self.logger.info(f"[新浪] 获取到 {len(stocks)} 只股票")
        return stocks

    def _get_stock_list_em(self, min_change: float) -> List[Dict]:
        """东方财富涨幅榜"""
        stocks = []

        for page in range(1, 6):
            url = (f"{self.EM_LIST_URL}?pn={page}&pz=100&po=1&np=1&fltt=2&invt=2&fid=f3&"
                  f"fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23&"
                  f"fields=f2,f3,f6,f9,f12,f14,f15,f16,f17,f20")

            resp = self._request(url, 'em')
            if resp is None:
                continue

            try:
                data = resp.json()
                if not data or not data.get('data') or not data['data'].get('diff'):
                    break

                for item in data['data']['diff']:
                    name = item.get('f14', '')
                    code = item.get('f12', '')

                    if 'ST' in name or 'st' in name:
                        continue

                    try:
                        change_pct = float(item.get('f3', 0) or 0)
                        price = float(item.get('f2', 0) or 0)
                        pe = float(item.get('f9', 0) or 0)
                    except:
                        continue

                    if change_pct < min_change:
                        continue
                    if price < self.config.min_price or price > self.config.max_price:
                        continue

                    market = 1 if code.startswith('6') else 0
                    symbol = f"{'sh' if market == 1 else 'sz'}{code}"

                    stocks.append({
                        'code': code,
                        'name': name,
                        'symbol': symbol,
                        'price': price,
                        'change_pct': change_pct,
                        'volume': float(item.get('f17', 0) or 0),
                        'amount': float(item.get('f20', 0) or 0),
                        'high': float(item.get('f15', 0) or 0),
                        'low': float(item.get('f16', 0) or 0),
                        'per': pe,
                    })

            except:
                continue

        self.logger.info(f"[东方财富] 获取到 {len(stocks)} 只股票")
        return stocks

    def get_kline_data(self, code: str, days: int = 120) -> Tuple[Optional[List[Dict]], str]:
        """获取K线数据，返回(数据, 数据源)"""

        # 优先使用Ashare
        if ASHARE_AVAILABLE:
            klines = self._get_kline_ashare(code, days)
            if klines:
                return klines, 'ashare'

        # 备用：新浪
        klines = self._get_kline_sina(code, days)
        if klines:
            return klines, 'sina'

        return None, ''

    def _get_kline_ashare(self, code: str, days: int) -> Optional[List[Dict]]:
        """使用Ashare获取K线"""
        try:
            # 转换代码格式
            if code.startswith('6'):
                xcode = f"{code}.XSHG"
            else:
                xcode = f"{code}.XSHE"

            df = ashare_get_price(xcode, frequency='1d', count=days)
            if df is None or df.empty():
                return None

            klines = []
            for idx, row in df.iterrows():
                klines.append({
                    'date': str(idx.date()),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                })

            return klines
        except Exception as e:
            self.logger.debug(f"[Ashare] K线获取失败 {code}: {str(e)[:30]}")
            return None

    def _get_kline_sina(self, code: str, days: int) -> Optional[List[Dict]]:
        """新浪K线"""
        # 转换代码格式
        if code.startswith('6'):
            symbol = f"sh{code}"
        else:
            symbol = f"sz{code}"

        url = f"https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?symbol={symbol}&scale=240&datalen={days}"
        resp = self._request(url, 'sina')
        if resp is None:
            return None

        try:
            data = resp.json()
            if not data:
                return None

            klines = []
            for item in data:
                klines.append({
                    'date': item.get('day', ''),
                    'open': float(item.get('open', 0)),
                    'high': float(item.get('high', 0)),
                    'low': float(item.get('low', 0)),
                    'close': float(item.get('close', 0)),
                    'volume': float(item.get('volume', 0)),
                })

            return klines
        except:
            return None

    def get_index_strength(self) -> float:
        """获取沪深300强度"""
        url = f"{self.SINA_QUOTE_URL}sh000300"
        resp = self._request(url, 'sina')
        if resp is None:
            return 0.0

        try:
            line = resp.text.strip()
            if '=' in line:
                parts = line.split('=')[1].split(',')
                if len(parts) > 3:
                    current = float(parts[3])
                    prev = float(parts[2])
                    if prev > 0:
                        return round((current - prev) / prev * 100, 2)
        except:
            pass
        return 0.0

    def get_industry_strength(self, stock_code: str) -> float:
        """获取股票所属行业强度"""
        # 简化版：根据股票代码查询行业涨幅
        # 实际应用中可以接入行业数据API
        try:
            # 获取行业板块数据
            url = "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page=1&num=50&sort=changepercent&asc=0&node=gn_a"
            resp = self._request(url, 'sina')
            if resp is None:
                return 0.0

            data = resp.json()
            if not data:
                return 0.0

            # 简化处理：返回平均行业涨幅作为参考
            total_change = 0
            count = 0
            for item in data[:10]:  # 取前10个行业平均
                try:
                    total_change += float(item.get('changepercent', 0) or 0)
                    count += 1
                except:
                    pass

            return round(total_change / count, 2) if count > 0 else 0.0
        except:
            return 0.0


class WyckoffAnalyzer:
    """威科夫分析器 V9 - 专业优化版"""

    def __init__(self, config: ScannerConfig):
        self.config = config

    def analyze(self, klines: List[Dict], index_strength: float, industry_strength: float = 0.0) -> Optional[Dict]:
        """分析K线数据 - 专业版"""
        if len(klines) < 60:
            return None

        closes = np.array([k['close'] for k in klines])
        volumes = np.array([k['volume'] for k in klines])
        highs = np.array([k['high'] for k in klines])
        lows = np.array([k['low'] for k in klines])

        # 量比计算（3-8范围）
        today_volume = volumes[-1]
        avg_volume_20 = np.mean(volumes[-20:])
        volume_ratio = today_volume / avg_volume_20 if avg_volume_20 > 0 else 1

        # 异常量比过滤（超过8倍视为异常）
        if volume_ratio > self.config.max_volume_ratio:
            return None

        # 涨幅计算
        prev_close = closes[-2]
        change_pct = (closes[-1] - prev_close) / prev_close * 100

        # === 量价健康度校验 ===
        # 涨幅需 >= 量比 × health_factor
        min_required_change = volume_ratio * self.config.volume_price_health_factor
        if change_pct < min_required_change:
            return None  # 量价背离，过滤

        # 市场环境判断
        is_weak_market = index_strength < self.config.weak_market_threshold

        # 弱市要求更高涨幅
        if is_weak_market and change_pct < self.config.weak_market_min_change:
            return None

        # === 威科夫阶段识别 ===
        wyckoff_phase = self._identify_wyckoff_phase(closes, highs, lows, volumes)

        # 结构确认
        has_support_test = self._check_support_structure(closes, lows, volumes)

        # SOS信号判断
        is_sos = False
        signal_reason = []

        # 1. SOS（需求主导信号）：放量突破前高
        if volume_ratio >= self.config.min_volume_ratio:
            if closes[-1] > highs[-2] and wyckoff_phase in ['SOS', 'Spring', 'LPS']:
                is_sos = True
                signal_reason.append(f"SOS放量突破({wyckoff_phase})")

        # 2. Spring（破底翻）：下探后强势回升
        if wyckoff_phase == 'Spring' and volume_ratio >= 2.0:
            if closes[-1] > closes[-2]:
                is_sos = True
                signal_reason.append("Spring破底翻")

        # 3. 弱市强势 + 行业强势
        if is_weak_market and change_pct > 5:
            if industry_strength > 0:  # 行业也是涨的
                if volume_ratio >= 2.0 and has_support_test:
                    is_sos = True
                    signal_reason.append(f"弱市强势+行业强({industry_strength:.1f}%)")

        if not is_sos:
            return None

        # === 专业评分系统 V9 ===
        score = 40  # 基础分提高

        # 量比评分（3-5最佳，>5略逊）
        if 3 <= volume_ratio <= 4:
            score += 25  # 最佳量比区间
        elif 4 < volume_ratio <= 5:
            score += 20
        elif 5 < volume_ratio <= 6:
            score += 15
        elif volume_ratio >= 2.5:
            score += 12

        # 涨幅评分（涨停更高权重）
        if change_pct >= 9.9:
            score += 20  # 涨停
        elif change_pct >= 7:
            score += 15
        elif change_pct >= 5:
            score += 10

        # 相对强度（大盘跌个股涨）- 核心加分项
        if is_weak_market and change_pct > 7:
            score += 15
        elif is_weak_market and change_pct > 5:
            score += 10

        # 威科夫阶段加分
        if wyckoff_phase == 'SOS':
            score += 10
        elif wyckoff_phase == 'Spring':
            score += 8

        # 结构确认加分
        if has_support_test:
            score += 5

        # === 分级标准 ===
        # A级：分数>=80 + 量比3-5 + 涨幅>=7% + 结构确认
        quality = 'C'
        if (score >= 80 and 3 <= volume_ratio <= 5 and change_pct >= 7 and has_support_test):
            quality = 'A'
        elif score >= 65 and volume_ratio >= 3 and change_pct >= 5:
            quality = 'B'

        return {
            'is_sos': True,
            'volume_ratio': round(volume_ratio, 1),
            'change_pct': round(change_pct, 2),
            'score': min(100, score),
            'quality': quality,
            'signal_reason': ', '.join(signal_reason),
            'wyckoff_phase': wyckoff_phase,
            'support_confirmed': has_support_test,
            'health_check': change_pct >= min_required_change,
        }

    def _identify_wyckoff_phase(self, closes: np.ndarray, highs: np.ndarray,
                                lows: np.ndarray, volumes: np.ndarray) -> str:
        """识别威科夫阶段"""
        n = len(closes)
        if n < 30:
            return 'Unknown'

        # 计算区间
        range_high = np.max(highs[-30:])
        range_low = np.min(lows[-30:])
        range_size = range_high - range_low

        if range_size == 0:
            return 'Unknown'

        # 当前位置
        current_close = closes[-1]
        position_in_range = (current_close - range_low) / range_size

        # 20日均线
        sma20 = np.mean(closes[-20:])

        # SOS: 放量突破区间上沿
        if position_in_range > 0.8 and volumes[-1] > np.mean(volumes[-20:]) * 2:
            return 'SOS'

        # Spring: 下探后回升（破底翻）
        if lows[-1] < range_low * 1.02 and closes[-1] > closes[-2]:
            if volumes[-1] > np.mean(volumes[-20:]) * 1.5:
                return 'Spring'

        # LPS: 最后支撑点（接近区间下沿但未破）
        if 0.1 < position_in_range < 0.3:
            if closes[-1] > sma20 * 0.98:
                return 'LPS'

        # AR: 自动反弹
        if position_in_range > 0.5 and closes[-1] > closes[-2] > closes[-3]:
            return 'AR'

        return 'Accumulation'

    def _check_support_structure(self, closes: np.ndarray, lows: np.ndarray, volumes: np.ndarray) -> bool:
        """检测前期支撑结构"""
        if len(closes) < 20:
            return False

        # 检查过去5-20天内是否有支撑测试
        recent_lows = lows[-20:-2]
        recent_closes = closes[-20:-2]
        recent_volumes = volumes[-20:-2]

        # 寻找低位支撑测试（下探后回升）
        support_tests = 0
        for i in range(1, len(recent_lows)):
            # 当天最低价接近前低，但收盘价高于前收（支撑有效）
            if recent_lows[i] <= recent_lows[i-1] * 1.02:
                if recent_closes[i] > recent_closes[i-1]:
                    support_tests += 1

        return support_tests >= 2  # 至少2次支撑测试


class WyckoffScanner:
    """威科夫扫描器"""

    def __init__(self, config: Optional[ScannerConfig] = None):
        self.config = config or ScannerConfig()
        self.logger = setup_logging(self.config.log_level)
        self.api = MultiSourceAPI(self.config, self.logger)
        self.cache = DataCache(self.config.cache_dir, self.config.cache_ttl_hours)

    def run(self) -> List[SOSSignal]:
        """执行扫描"""
        start = time.time()

        self.logger.info("=" * 60)
        self.logger.info(f"威科夫SOS扫描 V8 (多数据源)")
        self.logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Ashare: {'可用' if ASHARE_AVAILABLE else '不可用'}")
        self.logger.info("=" * 60)

        # 大盘强度
        index_strength = self.api.get_index_strength()
        self.logger.info(f"大盘强度(沪深300): {index_strength:+.2f}%")

        # 获取股票列表
        stocks, list_source = self.api.get_stock_list(self.config.min_change_pct)
        if not stocks:
            self.logger.warning("获取股票列表失败")
            return []

        self.logger.info(f"数据源: {list_source}, 共筛选出 {len(stocks)} 只候选股票")
        self.logger.info("开始分析K线...")

        # 分析
        results = []
        count = 0
        source_stats = {'ashare': 0, 'sina': 0, 'em': 0}

        for stock in stocks:
            klines, kline_source = self.api.get_kline_data(stock['code'], 120)

            if klines and len(klines) >= 60:
                source_stats[kline_source] = source_stats.get(kline_source, 0) + 1

                analyzer = WyckoffAnalyzer(self.config)
                # 获取行业强度（简化处理，实际可缓存）
                industry_strength = 0.0  # self.api.get_industry_strength(stock['code'])
                analysis = analyzer.analyze(klines, index_strength, industry_strength)

                if analysis and analysis.get('is_sos'):
                    phase = analysis.get('wyckoff_phase', '')
                    signal = SOSSignal(
                        code=stock['code'],
                        name=stock['name'],
                        price=stock['price'],
                        change_pct=stock['change_pct'],
                        volume=stock['volume'],
                        amount=stock['amount'],
                        signal_type=analysis.get('signal_reason', 'SOS信号'),
                        structure=f"{phase}+结构确认" if analysis.get('support_confirmed') else phase,
                        volume_ratio=analysis['volume_ratio'],
                        quality=analysis['quality'],
                        score=analysis['score'],
                        data_source=kline_source,
                    )
                    results.append(signal)
                    support_mark = "✓" if analysis.get('support_confirmed') else ""
                    health_mark = "♥" if analysis.get('health_check') else ""
                    self.logger.info(f"  ★ {signal.name}({signal.code}) {signal.change_pct:+.1f}% VR{signal.volume_ratio:.1f} [{signal.quality}]{support_mark}{health_mark}({phase})")

            count += 1
            if count % 50 == 0:
                self.logger.info(f"  进度: {count}/{len(stocks)}，发现 {len(results)} 个信号")

        elapsed = time.time() - start
        self.logger.info(f"扫描完成: {len(stocks)}只，{len(results)}个信号，耗时{elapsed:.1f}秒")
        self.logger.info(f"数据源统计: {source_stats}")

        # 输出结果
        self._output_results(results)

        return results

    def _output_results(self, results: List[SOSSignal]):
        """输出结果"""
        if not results:
            self.logger.warning("无SOS信号")
            return

        results.sort(key=lambda x: x.score, reverse=True)

        print("\n" + "=" * 100)
        print(f"{'代码':^8} {'名称':^10} {'现价':^8} {'涨幅%':^8} {'信号':^10} {'量比':^6} {'质量':^6} {'数据源':^8} {'总分':^6}")
        print("-" * 100)

        for r in results[:25]:
            print(f"{r.code:^8} {r.name:^10} {r.price:>8.2f} {r.change_pct:>6.2f}% {r.signal_type:^10} {r.volume_ratio:>6.1f} {r.quality:^6} {r.data_source:^8} {r.score:>6.1f}")

        # 质量分布
        dist = {'A': 0, 'B': 0, 'C': 0}
        for r in results:
            dist[r.quality] = dist.get(r.quality, 0) + 1
        print(f"\n信号质量: A级({dist['A']}) B级({dist['B']}) C级({dist['C']})")

        # 保存
        output_dir = Path.home() / ".openclaw/workspace/data"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "sos_scan_results.json"

        output_data = []
        for r in results:
            output_data.append({
                'code': r.code,
                'name': r.name,
                'price': r.price,
                'change_pct': r.change_pct,
                'signal_type': r.signal_type,
                'volume_ratio': r.volume_ratio,
                'quality': r.quality,
                'score': r.score,
                'data_source': r.data_source,
                'timestamp': datetime.now().isoformat(),
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"结果已保存: {output_file}")

        stats = self.cache.get_stats()
        self.logger.info(f"缓存: 内存{stats['memory']}条, 磁盘{stats['disk']}条")


def main():
    config = ScannerConfig()
    scanner = WyckoffScanner(config)
    scanner.run()


if __name__ == '__main__':
    main()
