import json
import time
import requests
from datetime import datetime
from urllib.parse import urlencode

# 股票配置
STOCKS = [
    {
        "name": "风范股份",
        "code": "601700",
        "secid": "1.601700",
        "market": "sh"
    },
    {
        "name": "浙江世宝",
        "code": "002703",
        "secid": "0.002703",
        "market": "sz"
    }
]

# 创建全局session
session = requests.Session()

# 请求头
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def init_session():
    """初始化session，访问页面获取cookie"""
    try:
        # 访问东方财富首页获取cookie
        session.get("https://www.eastmoney.com/", headers=HEADERS, timeout=30)
        time.sleep(0.5)
        # 访问行情页面
        session.get("https://quote.eastmoney.com/", headers=HEADERS, timeout=30)
        time.sleep(0.5)
        return True
    except Exception as e:
        print(f"初始化session失败: {e}")
        return False


def get_kline_data(stock, retry=3):
    """获取K线数据"""
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
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "_": str(int(time.time() * 1000))
    }

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

    api_headers = HEADERS.copy()
    api_headers["Referer"] = "https://quote.eastmoney.com/"
    api_headers["Accept"] = "*/*"

    for attempt in range(retry):
        try:
            response = session.get(
                url,
                params=params,
                headers=api_headers,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            if data.get("data") and data["data"].get("klines"):
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
            else:
                print(f"  尝试 {attempt + 1}: 无数据返回")
                if attempt < retry - 1:
                    time.sleep(2)
        except Exception as e:
            print(f"  尝试 {attempt + 1} 失败: {e}")
            if attempt < retry - 1:
                # 重新初始化session
                init_session()
                time.sleep(3)
    return None


def main():
    """主函数"""
    # 初始化session获取cookie
    print("初始化连接...")
    init_session()

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
            print(f"  失败: 无法获取数据")
        time.sleep(1)

    if results:
        print(f"更新完成: {', '.join(results)}")
    else:
        print("无数据更新")


if __name__ == "__main__":
    main()
