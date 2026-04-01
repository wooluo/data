import json
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


def get_kline_data(stock):
    """获取单只股票的K线数据"""
    params = {
        "secid": stock["secid"],
        "fields1": FIELDS1,
        "fields2": FIELDS2,
        "klt": KLT,
        "fqt": FQT,
        "beg": BEG,
        "end": END,
        "lmt": LMT
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        "Referer": "https://quote.eastmoney.com/",
    }

    try:
        response = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("data") and data["data"].get("klines"):
            klines = []
            for line in data["data"]["klines"]:
                # 数据格式: 日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
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
            return None
    except Exception as e:
        print(f"获取 {stock['name']} 数据出错: {e}")
        return None


def main():
    """主函数"""
    results = []
    for stock in STOCKS:
        print(f"正在获取 {stock['name']} 数据...")
        data = get_kline_data(stock)
        if data:
            # 保存单只股票数据
            filename = f"{stock['name']}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"{stock['name']}: 获取 {len(data['klines'])} 条数据")
            results.append(stock['name'])

    if results:
        print(f"更新完成: {', '.join(results)}")
    else:
        print("无数据更新")


if __name__ == "__main__":
    main()
