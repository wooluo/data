import json
import time
import requests
from datetime import datetime

# 股票配置
STOCKS = [
    {
        "name": "风范股份",
        "secid": "1.601700",  # 沪股
        "market": "sh"
    },
    {
        "name": "浙江世宝",
        "secid": "0.002703",  # 深股
        "market": "sz"
    }
]

# API配置
BASE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
FIELDS1 = "f1,f2,f3,f4,f5,f6"
FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
KLT = 101  # K线类型：101=日K
FQT = 1    # 复权类型：1=前复权
BEG = "20100101"
END = datetime.now().strftime("%Y%m%d")
LMT = 5000  # 获取数据条数

# 请求头
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0",
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://quote.eastmoney.com/",
    "Origin": "https://quote.eastmoney.com",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "Sec-Ch-Ua": '"Chromium";v="130", "Microsoft Edge";v="130", "Not?A_Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
}


def get_kline_data(stock, retry=3):
    """获取单只股票的K线数据"""
    params = {
        "secid": stock["secid"],
        "fields1": FIELDS1,
        "fields2": FIELDS2,
        "klt": KLT,
        "fqt": FQT,
        "beg": BEG,
        "end": END,
        "lmt": LMT,
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "cb": "",
        "_": str(int(time.time() * 1000))
    }

    for attempt in range(retry):
        try:
            response = requests.get(
                BASE_URL,
                params=params,
                headers=HEADERS,
                timeout=60,
                verify=True
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
                    "code": stock["secid"].split(".")[1],
                    "market": stock["market"],
                    "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "klines": klines
                }
            else:
                print(f"获取 {stock['name']} 数据失败: 无数据返回")
                if attempt < retry - 1:
                    time.sleep(2)
                    continue
                return None
        except Exception as e:
            print(f"获取 {stock['name']} 数据出错 (尝试 {attempt + 1}/{retry}): {e}")
            if attempt < retry - 1:
                time.sleep(3)
            else:
                return None
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
            print(f"{stock['name']}: 获取 {len(data['klines'])} 条数据")
            results.append(stock['name'])
        time.sleep(1)  # 请求间隔

    if results:
        print(f"更新完成: {', '.join(results)}")
    else:
        print("无数据更新")


if __name__ == "__main__":
    main()
