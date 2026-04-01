import json
import time
import requests
from datetime import datetime

# 股票配置
STOCKS = [
    {
        "name": "风范股份",
        "code": "601700",
        "secid": "1.601700",  # 沪股
        "market": "sh"
    },
    {
        "name": "浙江世宝",
        "code": "002703",
        "secid": "0.002703",  # 深股
        "market": "sz"
    }
]

# 请求头
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://quote.eastmoney.com/",
    "Origin": "https://quote.eastmoney.com",
}


def get_kline_data_eastmoney(stock, retry=3):
    """使用东方财富API获取K线数据"""
    end_date = datetime.now().strftime("%Y%m%d")

    params = {
        "secid": stock["secid"],
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": 101,
        "fqt": 1,
        "beg": "20100101",
        "end": end_date,
        "lmt": 5000,
        "ut": "b2884a393a59ad64002292a3e90d46a5",
    }

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

    for attempt in range(retry):
        try:
            response = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            if data.get("data") and data["data"].get("klines"):
                return parse_klines(data, stock)
            else:
                print(f"  尝试 {attempt + 1}: 无数据返回")
                if attempt < retry - 1:
                    time.sleep(2)
        except Exception as e:
            print(f"  尝试 {attempt + 1} 失败: {e}")
            if attempt < retry - 1:
                time.sleep(3)
    return None


def get_kline_data_sina(stock, retry=3):
    """使用新浪API获取K线数据（备用）"""
    # 新浪财经API
    symbol = f"sh{stock['code']}" if stock['market'] == 'sh' else f"sz{stock['code']}"
    url = f"https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"

    params = {
        "symbol": symbol,
        "scale": 240,  # 日K
        "ma": "no",
        "datalen": 5000
    }

    for attempt in range(retry):
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": HEADERS["User-Agent"]},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            if data and isinstance(data, list):
                klines = []
                for item in data:
                    klines.append({
                        "date": item.get("day", ""),
                        "open": float(item.get("open", 0)),
                        "close": float(item.get("close", 0)),
                        "high": float(item.get("high", 0)),
                        "low": float(item.get("low", 0)),
                        "volume": float(item.get("volume", 0)),
                    })
                return {
                    "name": stock["name"],
                    "code": stock["code"],
                    "market": stock["market"],
                    "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "klines": klines
                }
            else:
                print(f"  新浪尝试 {attempt + 1}: 无数据返回")
                if attempt < retry - 1:
                    time.sleep(2)
        except Exception as e:
            print(f"  新浪尝试 {attempt + 1} 失败: {e}")
            if attempt < retry - 1:
                time.sleep(3)
    return None


def parse_klines(data, stock):
    """解析东方财富K线数据"""
    klines = []
    for line in data["data"]["klines"]:
        parts = line.split(",")
        klines.append({
            "date": parts[0],
            "open": float(parts[1]),
            "close": float(parts[2]),
            "high": float(parts[3]),
            "low": float(parts[4]),
            "volume": float(parts[5]),
            "amount": float(parts[6]),
            "amplitude": float(parts[7]),
            "change_pct": float(parts[8]),
            "change": float(parts[9]),
            "turnover": float(parts[10])
        })
    return {
        "name": stock["name"],
        "code": stock["code"],
        "market": stock["market"],
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "klines": klines
    }


def get_kline_data(stock):
    """获取K线数据，依次尝试多个数据源"""
    print(f"  尝试东方财富API...")
    data = get_kline_data_eastmoney(stock)
    if data:
        return data

    print(f"  尝试新浪财经API...")
    data = get_kline_data_sina(stock)
    if data:
        return data

    return None


def main():
    """主函数"""
    results = []
    for stock in STOCKS:
        print(f"正在获取 {stock['name']} 数据...")
        data = get_kline_data(stock)
        if data:
            filename = f"{stock['name']}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  成功: 获取 {len(data['klines'])} 条数据")
            results.append(stock['name'])
        else:
            print(f"  失败: 所有数据源均无法获取")
        time.sleep(1)

    if results:
        print(f"更新完成: {', '.join(results)}")
    else:
        print("无数据更新")


if __name__ == "__main__":
    main()
