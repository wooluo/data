#!/usr/bin/env python3
"""
威科夫SOS信号全市场扫描器 V5
使用东方财富API， curl + ThreadPoolExecutor 实现并发

策略增强：
- 威科夫累积结构识别 (PS/SC/AR/ST)
- OBV趋势确认
- 量价背离检测
- 假突破过滤
- 大盘强度过滤
"""

import hashlib
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

# 清除所有代理环境变量
for proxy_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy']:
    os.environ.pop(proxy_var, None)


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """配置日志系统"""
    logger = logging.getLogger('wyckoff_scanner')
    logger.setLevel(log_level)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    log_dir = Path.home() / ".openclaw/workspace/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        log_dir / f"scanner_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


class WyckoffPhase(Enum):
    """威科夫累积阶段"""
    NONE = "无结构"
    PS = "初级支撑"
    SC = "恐慌抛售"
    AR = "自动反弹"
    ST = "二次测试"
    SOS = "强势信号"
    LPS = "最后支撑点"


class SignalQuality(Enum):
    """信号质量等级"""
    A = "A级(优质)"
    B = "B级(良好)"
    C = "C级(一般)"


@dataclass
class ScannerConfig:
    """扫描器配置"""
    # 筛选条件
    min_change_pct: float = 2.0
    max_pe: float = 200.0
    min_price: float = 3.0
    max_price: float = 300.0

    # SOS检测参数
    min_volume_ratio: float = 1.5
    atr_min: float = 0.8
    atr_max: float = 6.0

    # 威科夫结构参数
    sc_lookback: int = 30
    sc_volume_ratio: float = 2.0
    ar_bounce_pct: float = 0.5
    st_shrink_ratio: float = 0.7

    # 量价确认参数
    obv_ma_period: int = 10
    divergence_lookback: int = 5

    # 大盘过滤
    index_strength_threshold: float = 0.3

    # 网络配置
    max_workers: int = 5
    request_timeout: int = 15
    request_delay: float = 0.15

    # 缓存配置
    cache_enabled: bool = True
    cache_ttl_hours: int = 4
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".openclaw/workspace/cache")

    # 输出配置
    output_dir: Path = field(default_factory=lambda: Path.home() / ".openclaw/workspace/data")

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class StockInfo:
    """股票信息"""
    code: str
    name: str
    price: float
    change_pct: float
    high: float
    low: float
    volume: float
    amount: float
    pe: float
    turnover_rate: float
    market: int


@dataclass
class WyckoffStructure:
    """威科夫结构"""
    phase: WyckoffPhase
    sc_date: Optional[str] = None
    sc_low: Optional[float] = None
    sc_volume_ratio: Optional[float] = None
    ar_high: Optional[float] = None
    ar_bounce_pct: Optional[float] = None
    st_confirmed: bool = False
    structure_score: float = 0.0


@dataclass
class VolumePriceAnalysis:
    """量价分析结果"""
    obv_trend: str = "neutral"
    obv_divergence: bool = False
    volume_trend: str = "neutral"
    price_trend: str = "neutral"
    vp_score: float = 0.0


@dataclass
class SOSSignal:
    """SOS信号"""
    code: str
    name: str
    price: float
    change_pct: float
    amount: float
    pe: float
    signal_date: str
    signal_type: str
    resistance: float
    stop_loss: float
    volume_ratio: float
    atr_pct: float
    strength: float
    quality: str = "C级(一般)"
    wyckoff_phase: str = "无结构"
    structure_score: float = 0.0
    vp_score: float = 0.0
    obv_trend: str = "neutral"
    has_divergence: bool = False
    index_strength: float = 0.0
    total_score: float = 0.0


class DataCache:
    """数据缓存管理器"""

    def __init__(self, cache_dir: Path, ttl_hours: int = 4):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self._memory_cache: Dict[str, Tuple[float, Any]] = {}

    def _get_cache_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.pkl"

    def _is_expired(self, timestamp: float) -> bool:
        return (time.time() - timestamp) > self.ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        cache_key = self._get_cache_key(key)

        if cache_key in self._memory_cache:
            timestamp, data = self._memory_cache[cache_key]
            if not self._is_expired(timestamp):
                return data
            else:
                del self._memory_cache[cache_key]

        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    timestamp, data = pickle.load(f)
                if not self._is_expired(timestamp):
                    self._memory_cache[cache_key] = (timestamp, data)
                    return data
                else:
                    cache_path.unlink()
            except (pickle.PickleError, EOFError):
                cache_path.unlink(missing_ok=True)

        return None

    def set(self, key: str, data: Any) -> None:
        cache_key = self._get_cache_key(key)
        timestamp = time.time()
        self._memory_cache[cache_key] = (timestamp, data)

        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump((timestamp, data), f)
        except (pickle.PickleError, IOError):
            pass

    def clear_expired(self) -> int:
        cleared = 0
        expired_keys = [k for k, (ts, _) in self._memory_cache.items() if self._is_expired(ts)]
        for k in expired_keys:
            del self._memory_cache[k]
            cleared += 1

        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    timestamp, _ = pickle.load(f)
                if self._is_expired(timestamp):
                    cache_file.unlink()
                    cleared += 1
            except (pickle.PickleError, EOFError):
                cache_file.unlink()
                cleared += 1

        return cleared

    def get_stats(self) -> Dict[str, Any]:
        disk_files = list(self.cache_dir.glob("*.pkl"))
        disk_size = sum(f.stat().st_size for f in disk_files) if disk_files else 0
        return {
            'memory_entries': len(self._memory_cache),
            'disk_entries': len(disk_files),
            'disk_size_mb': round(disk_size / (1024 * 1024), 2)
        }


class VectorizedIndicators:
    """向量化技术指标计算"""

    @staticmethod
    def extract_arrays(klines: List[Dict[str, Any]]) -> Tuple[np.ndarray, ...]:
        if not klines:
            return None, None, None, None, None

        n = len(klines)
        opens = np.empty(n, dtype=np.float64)
        highs = np.empty(n, dtype=np.float64)
        lows = np.empty(n, dtype=np.float64)
        closes = np.empty(n, dtype=np.float64)
        volumes = np.empty(n, dtype=np.float64)

        for i, k in enumerate(klines):
            opens[i] = k['open']
            highs[i] = k['high']
            lows[i] = k['low']
            closes[i] = k['close']
            volumes[i] = k['volume']

        return opens, highs, lows, closes, volumes

    @staticmethod
    def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                      period: int = 20) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        prev_closes = closes[:-1]
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - prev_closes),
                np.abs(lows[1:] - prev_closes)
            )
        )
        if len(tr) < period:
            return None
        return float(np.mean(tr[-period:]))

    @staticmethod
    def calculate_obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        obv = np.zeros(len(closes), dtype=np.float64)
        obv[0] = volumes[0]
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif closes[i] < closes[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]
        return obv

    @staticmethod
    def detect_divergence(prices: np.ndarray, indicator: np.ndarray,
                          lookback: int = 5) -> bool:
        if len(prices) < lookback or len(indicator) < lookback:
            return False
        recent_prices = prices[-lookback:]
        recent_indicator = indicator[-lookback:]
        price_min_idx = np.argmin(recent_prices)
        if price_min_idx == lookback - 1:
            for i in range(lookback - 1):
                if recent_prices[i] < recent_prices[-1] and recent_indicator[i] > recent_indicator[-1]:
                    return True
        return False

    @staticmethod
    def calculate_trend(data: np.ndarray, lookback: int = 10) -> str:
        if len(data) < lookback:
            return "neutral"
        recent = data[-lookback:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        mean_val = np.mean(recent)
        if mean_val == 0:
            return "neutral"
        normalized_slope = slope / mean_val * 100
        if normalized_slope > 1:
            return "up"
        elif normalized_slope < -1:
            return "down"
        else:
            return "neutral"

    @staticmethod
    def calculate_avg_volume(volumes: np.ndarray, period: int = 20) -> float:
        if len(volumes) < period:
            return float(np.mean(volumes))
        return float(np.mean(volumes[-period:]))

    @staticmethod
    def calculate_resistance(highs: np.ndarray, period: int = 20,
                             ratio: float = 0.96) -> float:
        if len(highs) < period:
            return float(np.max(highs) * ratio)
        return float(np.max(highs[-period:]) * ratio)


class WyckoffAnalyzer:
    """威科夫结构分析器"""

    def __init__(self, config: ScannerConfig):
        self.config = config

    def analyze_structure(self, klines: List[Dict[str, Any]],
                          highs: np.ndarray, lows: np.ndarray,
                          closes: np.ndarray, volumes: np.ndarray) -> WyckoffStructure:
        structure = WyckoffStructure(phase=WyckoffPhase.NONE)
        if len(klines) < self.config.sc_lookback:
            return structure

        lookback = min(self.config.sc_lookback, len(klines))
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        recent_volumes = volumes[-lookback:]
        recent_closes = closes[-lookback:]
        avg_vol = np.mean(recent_volumes)

        sc_idx, sc_score = self._detect_sc(recent_highs, recent_lows,
                                            recent_volumes, recent_closes, avg_vol)
        if sc_idx is None:
            return structure

        structure.sc_date = klines[-lookback + sc_idx]['date']
        structure.sc_low = float(recent_lows[sc_idx])
        structure.sc_volume_ratio = float(recent_volumes[sc_idx] / avg_vol) if avg_vol > 0 else 0

        ar_idx, ar_score = self._detect_ar(recent_highs, recent_lows,
                                            recent_closes, sc_idx)
        if ar_idx is None:
            structure.phase = WyckoffPhase.PS
            structure.structure_score = 20
            return structure

        structure.ar_high = float(recent_highs[ar_idx])
        ar_range = structure.ar_high - structure.sc_low
        structure.ar_bounce_pct = (ar_range / structure.sc_low * 100) if structure.sc_low > 0 else 0

        st_confirmed, st_score = self._detect_st(recent_lows, recent_volumes,
                                                  sc_idx, ar_idx, structure.sc_low, avg_vol)
        structure.st_confirmed = st_confirmed

        current_idx = lookback - 1
        current_close = float(recent_closes[-1])
        current_high = float(recent_highs[-1])

        if st_confirmed:
            if current_high >= structure.ar_high * 0.98:
                structure.phase = WyckoffPhase.SOS
                structure.structure_score = 80 + st_score
            elif current_close > structure.sc_low * 1.02:
                structure.phase = WyckoffPhase.LPS
                structure.structure_score = 60 + st_score
            else:
                structure.phase = WyckoffPhase.ST
                structure.structure_score = 50 + st_score
        else:
            if current_close > structure.sc_low * 1.05:
                structure.phase = WyckoffPhase.AR
                structure.structure_score = 40
            else:
                structure.phase = WyckoffPhase.SC
                structure.structure_score = 30

        return structure

    def _detect_sc(self, highs: np.ndarray, lows: np.ndarray,
                   volumes: np.ndarray, closes: np.ndarray,
                   avg_vol: float) -> Tuple[Optional[int], float]:
        sc_candidates = []
        for i in range(len(lows)):
            vol_ratio = volumes[i] / avg_vol if avg_vol > 0 else 0
            lower_shadow = (highs[i] - lows[i]) / closes[i] * 100 if closes[i] > 0 else 0
            if vol_ratio >= self.config.sc_volume_ratio:
                score = vol_ratio * 10
                if lower_shadow > 3:
                    score += 20
                sc_candidates.append((i, score, lows[i]))

        if not sc_candidates:
            return None, 0

        min_low = np.min(lows)
        best_sc = None
        best_score = 0

        for idx, score, low in sc_candidates:
            distance = abs(low - min_low) / min_low * 100 if min_low > 0 else 100
            adjusted_score = score - distance
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_sc = idx

        return best_sc, min(best_score, 50)

    def _detect_ar(self, highs: np.ndarray, lows: np.ndarray,
                   closes: np.ndarray, sc_idx: int) -> Tuple[Optional[int], float]:
        if sc_idx >= len(highs) - 2:
            return None, 0
        sc_low = lows[sc_idx]
        for i in range(sc_idx + 1, len(highs)):
            bounce_pct = (highs[i] - sc_low) / sc_low * 100 if sc_low > 0 else 0
            if bounce_pct >= self.config.ar_bounce_pct:
                return i, min(bounce_pct * 5, 30)
        return None, 0

    def _detect_st(self, lows: np.ndarray, volumes: np.ndarray,
                   sc_idx: int, ar_idx: int, sc_low: float,
                   avg_vol: float) -> Tuple[bool, float]:
        if ar_idx >= len(lows) - 1:
            return False, 0
        for i in range(ar_idx + 1, len(lows)):
            low_distance = abs(lows[i] - sc_low) / sc_low * 100 if sc_low > 0 else 100
            vol_ratio = volumes[i] / avg_vol if avg_vol > 0 else 0
            if low_distance < 3 and vol_ratio < self.config.st_shrink_ratio:
                return True, 20
            if low_distance < 5 and vol_ratio < 1.0:
                return True, 10
        return False, 0


class VolumePriceAnalyzer:
    """量价分析器"""

    def __init__(self, config: ScannerConfig):
        self.config = config

    def analyze(self, closes: np.ndarray, volumes: np.ndarray) -> VolumePriceAnalysis:
        analysis = VolumePriceAnalysis()
        if len(closes) < 20:
            return analysis

        obv = VectorizedIndicators.calculate_obv(closes, volumes)
        analysis.obv_trend = VectorizedIndicators.calculate_trend(
            obv, self.config.obv_ma_period
        )

        analysis.obv_divergence = VectorizedIndicators.detect_divergence(
            closes, obv, self.config.divergence_lookback
        )

        analysis.volume_trend = VectorizedIndicators.calculate_trend(volumes[-20:], 10)
        analysis.price_trend = VectorizedIndicators.calculate_trend(closes[-20:], 10)

        score = 0
        if analysis.obv_trend == "up":
            score += 30
        elif analysis.obv_trend == "neutral":
            score += 10

        if not analysis.obv_divergence:
            score += 20
        else:
            score -= 10

        if analysis.volume_trend == "up" and analysis.price_trend == "up":
            score += 30
        elif analysis.volume_trend == "up":
            score += 15

        analysis.vp_score = max(0, min(100, score))
        return analysis


class WyckoffScanner:
    """威科夫扫描器"""

    def __init__(self, config: Optional[ScannerConfig] = None):
        self.config = config or ScannerConfig()
        self.logger = setup_logging()
        self.cache = DataCache(self.config.cache_dir, self.config.cache_ttl_hours)
        self.wyckoff_analyzer = WyckoffAnalyzer(self.config)
        self.vp_analyzer = VolumePriceAnalyzer(self.config)
        self._cache_hits = 0
        self._cache_misses = 0
        self._index_strength: float = 0.0

    def close(self):
        """关闭资源"""
        stats = self.cache.get_stats()
        self.logger.info(f"缓存统计: 内存{stats['memory_entries']}条, "
                         f"磁盘{stats['disk_entries']}条, "
                         f"大小{stats['disk_size_mb']}MB")
        self.logger.info(f"缓存命中: {self._cache_hits}, 未命中: {self._cache_misses}")

    def _api_get(self, url: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """HTTP GET请求（使用curl）"""
        if use_cache and self.config.cache_enabled:
            cached = self.cache.get(url)
            if cached is not None:
                self._cache_hits += 1
                return cached

        self._cache_misses += 1

        # 使用完整的请求头（包含完整Cookie）
        cmd = [
            'curl', '-s', '-4', '--noproxy', '*',
            '--connect-timeout', '10',
            '--max-time', str(self.config.request_timeout),
            '-H', 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            '-H', 'Accept-Language: zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
            '-H', 'Cache-Control: no-cache',
            '-H', 'Connection: keep-alive',
            '-H', 'Pragma: no-cache',
            '-H', 'Sec-Fetch-Dest: document',
            '-H', 'Sec-Fetch-Mode: navigate',
            '-H', 'Sec-Fetch-Site: none',
            '-H', 'Sec-Fetch-User: ?1',
            '-H', 'Upgrade-Insecure-Requests: 1',
            '-H', 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36',
            '-H', 'sec-ch-ua: "Chromium";v="146", "Not-A.Brand";v="24", "Google Chrome";v="146"',
            '-H', 'sec-ch-ua-mobile: ?0',
            '-H', 'sec-ch-ua-platform: "macOS"',
            '-b', 'qgqp_b_id=a6b7fdc2e7b5a497eecbd5479ac66387; st_nvi=aYp2wao6rDuXgKbmQtXl06598; nid18=0eb72bca22228cd120cafad64c39bd30; nid18_create_time=1769561614569; gviem=ruYA8TSFZ9EYgNC1gOYZyfe35; gviem_create_time=1769561614569; fullscreengg=1; fullscreengg2=1; st_si=93488908361421; st_pvi=50071006060836; st_sp=2025-09-29%2013%3A55%3A42; st_inirUrl=https%3A%2F%2Fwww.baidu.com%2F; st_sn=61; st_psi=20260327165916989-113104312931-4149617526; st_asi=delete',
            '--compressed',
            url
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=self.config.request_timeout + 5)
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                if use_cache and self.config.cache_enabled:
                    self.cache.set(url, data)
                return data
            else:
                err_msg = result.stderr[:50] if result.stderr else 'empty response'
                self.logger.debug(f"curl返回空: {err_msg}")
        except subprocess.TimeoutExpired:
            self.logger.warning(f"请求超时: {url[:60]}...")
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析错误: {str(e)[:50]}")
        except Exception as e:
            self.logger.warning(f"请求错误: {str(e)[:50]}")
        return None

    def get_index_strength(self) -> float:
        """获取大盘强度"""
        if self._index_strength != 0.0:
            return self._index_strength
        url = "https://push2.eastmoney.com/api/qt/ulist.np/get?fltt=2&secids=1.000300&fields=f3"
        data = self._api_get(url, use_cache=False)
        if data and data.get('data') and data['data'].get('diff'):
            change_pct = data['data']['diff'][0].get('f3', 0) or 0
            self._index_strength = float(change_pct)
        return self._index_strength

    def get_stock_list(self) -> List[StockInfo]:
        """获取股票列表"""
        all_stocks: List[StockInfo] = []

        for page in range(1, 12):
            url = (f"https://push2.eastmoney.com/api/qt/clist/get?"
                   f"pn={page}&pz=500&po=1&np=1&fltt=2&invt=2&fid=f3&"
                   f"fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23&"
                   f"fields=f2,f3,f6,f9,f12,f14,f15,f16,f17,f20,f39")

            data = self._api_get(url, use_cache=False)
            if not data or not data.get('data') or not data['data'].get('diff'):
                break

            stocks = []
            items = data['data']['diff']

            for item in items:
                name = item.get('f14', '')
                try:
                    price = float(item.get('f2', 0) or 0)
                    change_pct = float(item.get('f3', 0) or 0)
                    pe = float(item.get('f9', 0) or 0)
                except (ValueError, TypeError):
                    continue

                if 'ST' in name or 'st' in name or '*' in name:
                    continue
                if price <= 0:
                    continue
                if change_pct < self.config.min_change_pct:
                    continue
                if price < self.config.min_price or price > self.config.max_price:
                    continue

                code = item.get('f12', '')
                market = 1 if code.startswith('6') else 0

                stocks.append(StockInfo(
                    code=code,
                    name=name,
                    price=price,
                    change_pct=change_pct,
                    high=float(item.get('f15', 0) or 0),
                    low=float(item.get('f16', 0) or 0),
                    volume=float(item.get('f17', 0) or 0),
                    amount=float(item.get('f20', 0) or 0),
                    pe=pe,
                    turnover_rate=float(item.get('f39', 0) or 0),
                    market=market
                ))

            if stocks:
                self.logger.info(f"第{page}页: +{len(stocks)}只，累计{len(all_stocks) + len(stocks)}只")
            all_stocks.extend(stocks)
            time.sleep(0.1)

        self.logger.info(f"共筛选出 {len(all_stocks)} 只候选股票")
        return all_stocks

    def get_kline_data(self, stock: StockInfo, days: int = 60) -> Optional[List[Dict[str, Any]]]:
        """获取K线数据"""
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days + 30)).strftime('%Y%m%d')

        url = (f"https://push2his.eastmoney.com/api/qt/stock/kline/get?"
               f"secid={stock.market}.{stock.code}&fields1=f1,f2,f3,f4,f5,f6&"
               f"fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&"
               f"klt=101&fqt=1&beg={start_date}&end={end_date}&lmt={days}")

        data = self._api_get(url, use_cache=True)
        if not data or not data.get('data') or not data['data'].get('klines'):
            return None

        klines = []
        for line in data['data']['klines']:
            parts = line.split(',')
            if len(parts) >= 11:
                try:
                    klines.append({
                        'date': parts[0],
                        'open': float(parts[1]),
                        'close': float(parts[2]),
                        'high': float(parts[3]),
                        'low': float(parts[4]),
                        'volume': float(parts[5]),
                        'amount': float(parts[6]),
                        'amplitude': float(parts[7]),
                        'change_pct': float(parts[8]),
                        'change_amount': float(parts[9]),
                        'turnover_rate': float(parts[10])
                    })
                except (ValueError, IndexError):
                    continue

        return klines

    def calculate_signal_quality(self, structure: WyckoffStructure,
                                  vp_analysis: VolumePriceAnalysis,
                                  index_strength: float,
                                  volume_ratio: float) -> Tuple[SignalQuality, float]:
        """计算信号质量"""
        total_score = 0

        total_score += structure.structure_score * 0.4
        total_score += vp_analysis.vp_score * 0.3

        if index_strength >= self.config.index_strength_threshold:
            total_score += 100 * 0.15
        elif index_strength >= 0:
            total_score += 50 * 0.15
        else:
            total_score += max(0, 50 + index_strength * 5) * 0.15

        vol_score = min(100, volume_ratio * 30)
        total_score += vol_score * 0.15

        if total_score >= 70:
            quality = SignalQuality.A
        elif total_score >= 50:
            quality = SignalQuality.B
        else:
            quality = SignalQuality.C

        return quality, total_score

    def detect_sos(self, klines: List[Dict[str, Any]], stock: StockInfo,
                   index_strength: float) -> Optional[Dict[str, Any]]:
        """检测SOS信号"""
        if not klines or len(klines) < 30:
            return None

        pe = stock.pe
        if isinstance(pe, (int, float)) and (pe < 0 or pe > self.config.max_pe):
            return {'reason': f'PE={pe}'}

        opens, highs, lows, closes, volumes = VectorizedIndicators.extract_arrays(klines)
        if highs is None:
            return None

        # 威科夫结构分析
        structure = self.wyckoff_analyzer.analyze_structure(klines, highs, lows, closes, volumes)

        # 量价分析
        vp_analysis = self.vp_analyzer.analyze(closes, volumes)

        # ATR
        atr = VectorizedIndicators.calculate_atr(highs, lows, closes, 20)
        if atr is None:
            return {'reason': 'ATR不足'}

        current_price = float(closes[-1])
        atr_pct = (atr / current_price) * 100
        if atr_pct < self.config.atr_min or atr_pct > self.config.atr_max:
            return {'reason': f'ATR={atr_pct:.1f}%'}

        # 底部结构验证
        recent_lows = lows[-30:]
        min_low = float(np.min(recent_lows))
        min_idx = int(np.argmin(recent_lows))
        min_days_ago = 29 - min_idx

        if min_days_ago < 3:
            return {'reason': f'最低点仅{min_days_ago}天前'}

        # 破位检查
        if min_idx < 29:
            subsequent_lows = recent_lows[min_idx + 1:]
            if np.any(subsequent_lows < min_low * 0.98):
                return {'reason': '破位'}

        # 阻力位和均量
        resistance = VectorizedIndicators.calculate_resistance(highs, 20, 0.96)
        avg_vol = VectorizedIndicators.calculate_avg_volume(volumes, 20)

        # 检查SOS信号
        for i in range(3):
            idx = -3 + i
            change_pct = klines[idx]['change_pct']
            vol = float(volumes[idx])
            close = float(closes[idx])

            if (change_pct > 2.5 and
                vol > avg_vol * self.config.min_volume_ratio and
                close > resistance):

                signal_type = '单日突破'
                if i > 0:
                    prev_change = klines[idx - 1]['change_pct']
                    prev_vol = float(volumes[idx - 1])
                    if prev_change > 1.5 and prev_vol > avg_vol * 1.3:
                        signal_type = '连续突破'

                vol_ratio = vol / avg_vol

                quality, total_score = self.calculate_signal_quality(
                    structure, vp_analysis, index_strength, vol_ratio
                )

                return {
                    'signal_date': klines[idx]['date'],
                    'signal_type': signal_type,
                    'change_pct': change_pct,
                    'volume_ratio': round(vol_ratio, 1),
                    'resistance': round(resistance, 2),
                    'stop_loss': round(min_low * 0.97, 2),
                    'atr_pct': round(atr_pct, 2),
                    'min_low': min_low,
                    'strength': round(change_pct * vol_ratio, 1),
                    'quality': quality.value,
                    'wyckoff_phase': structure.phase.value,
                    'structure_score': round(structure.structure_score, 1),
                    'vp_score': round(vp_analysis.vp_score, 1),
                    'obv_trend': vp_analysis.obv_trend,
                    'has_divergence': vp_analysis.obv_divergence,
                    'index_strength': round(index_strength, 2),
                    'total_score': round(total_score, 1)
                }

        return None

    def analyze_stock(self, stock: StockInfo, index_strength: float) -> Optional[SOSSignal]:
        """分析单只股票"""
        time.sleep(self.config.request_delay)
        try:
            klines = self.get_kline_data(stock)
            if klines and len(klines) >= 30:
                signal = self.detect_sos(klines, stock, index_strength)
                if signal and 'signal_date' in signal:
                    return SOSSignal(
                        code=stock.code,
                        name=stock.name,
                        price=stock.price,
                        change_pct=stock.change_pct,
                        amount=stock.amount,
                        pe=stock.pe,
                        signal_date=signal['signal_date'],
                        signal_type=signal['signal_type'],
                        resistance=signal['resistance'],
                        stop_loss=signal['stop_loss'],
                        volume_ratio=signal['volume_ratio'],
                        atr_pct=signal['atr_pct'],
                        strength=signal['strength'],
                        quality=signal['quality'],
                        wyckoff_phase=signal['wyckoff_phase'],
                        structure_score=signal['structure_score'],
                        vp_score=signal['vp_score'],
                        obv_trend=signal['obv_trend'],
                        has_divergence=signal['has_divergence'],
                        index_strength=signal['index_strength'],
                        total_score=signal['total_score']
                    )
        except Exception as e:
            self.logger.error(f"分析 {stock.code} {stock.name} 时出错: {str(e)}")

        return None

    def scan(self) -> List[SOSSignal]:
        """执行扫描"""
        start_time = time.time()

        cleared = self.cache.clear_expired()
        if cleared > 0:
            self.logger.info(f"已清理 {cleared} 个过期缓存")

        self.logger.info("=" * 60)
        self.logger.info("威科夫SOS扫描 V5 (curl + ThreadPoolExecutor)")
        self.logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"并发数: {self.config.max_workers}")
        self.logger.info("=" * 60)

        # 获取大盘强度
        index_strength = self.get_index_strength()
        self.logger.info(f"大盘强度(沪深300): {index_strength:+.2f}%")

        stocks = self.get_stock_list()
        if not stocks:
            self.logger.warning("无候选股票")
            return []

        self.logger.info(f"开始并发分析K线...")

        results: List[SOSSignal] = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self.analyze_stock, stock, index_strength): stock
                for stock in stocks
            }

            completed = 0
            for future in as_completed(futures):
                try:
                    signal = future.result()
                    if signal:
                        results.append(signal)
                        self.logger.info(
                            f"  ★ {signal.name}({signal.code}) {signal.signal_type} "
                            f"+{signal.change_pct:.1f}% 量比{signal.volume_ratio} "
                            f"[{signal.quality}]"
                        )
                except Exception as e:
                    self.logger.error(f"任务执行错误: {str(e)}")

                completed += 1
                if completed % 50 == 0:
                    self.logger.info(f"  进度: {completed}/{len(stocks)}，发现 {len(results)} 个信号")

        # 按总分排序
        results.sort(key=lambda x: x.total_score, reverse=True)
        elapsed = time.time() - start_time
        self.logger.info(f"扫描完成: {len(stocks)}只候选，{len(results)}个SOS信号，耗时{elapsed:.1f}秒")

        return results

    def save_results(self, results: List[SOSSignal]) -> Path:
        """保存结果"""
        output_file = self.config.output_dir / "sos_scan_results.json"

        output = {
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_signals': len(results),
            'signals': [asdict(s) for s in results]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        self.logger.info(f"结果已保存: {output_file}")
        return output_file

    def print_results(self, results: List[SOSSignal]):
        """打印结果"""
        if not results:
            return

        print(f"\n{'=' * 100}")
        print(f"{'代码':<8} {'名称':<10} {'现价':<8} {'涨幅%':<8} {'信号':<8} "
              f"{'量比':<6} {'结构':<12} {'质量':<12} {'总分':<6}")
        print("-" * 100)

        for r in results[:25]:
            print(f"{r.code:<8} {r.name:<10} {r.price:<8.2f} {r.change_pct:<8.2f} "
                  f"{r.signal_type:<8} {r.volume_ratio:<6.1f} {r.wyckoff_phase:<12} "
                  f"{r.quality:<12} {r.total_score:<6.1f}")

        # 统计
        a_count = sum(1 for r in results if r.quality == SignalQuality.A.value)
        b_count = sum(1 for r in results if r.quality == SignalQuality.B.value)
        c_count = sum(1 for r in results if r.quality == SignalQuality.C.value)

        print(f"\n信号质量分布: A级({a_count}) B级({b_count}) C级({c_count})")


def main():
    """主函数"""
    config = ScannerConfig()
    scanner = WyckoffScanner(config)

    try:
        results = scanner.scan()
        scanner.save_results(results)
        scanner.print_results(results)
    finally:
        scanner.close()


if __name__ == "__main__":
    main()
