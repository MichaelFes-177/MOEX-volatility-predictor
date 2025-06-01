import pandas as pd
import requests
import pathlib


# Define the function to fetch daily candles from MOEX ISS
def candles(sec: str, start: str, end: str) -> pd.DataFrame:
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{sec}/candles.json"
    params = {"from": start, "till": end, "interval": 24, "start": 0}
    keep = ["end", "open", "high", "low", "close", "value"]
    frames = []

    while True:
        js = requests.get(url, params=params, timeout=20).json()["candles"]
        if not js["data"]:
            break
        cols = [c.lower() for c in js["columns"]]
        df_part = pd.DataFrame(js["data"], columns=cols)[keep]
        frames.append(df_part)
        params["start"] += len(df_part)

    df = pd.concat(frames, ignore_index=True)
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date").sort_index()

    df["ret1"] = df["close"].pct_change()
    df["sma5"] = df["close"].rolling(5).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["vol_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std(ddof=0)
    df["atr14"] = (df["high"] - df["low"]).rolling(14).mean() / df["close"]

    return df.dropna()


# Fetch candles from 2014-01-01 to today (2025-05-30 for MOEX trading data)
START = "2014-01-01"
END = "2025-05-31"
security = "MOEX"

# Execute the fetch and generate the candles DataFrame
df_candles = candles(security, START, END)

# Define output path and save as compressed CSV
output_path = pathlib.Path("candles.csv.gz")
df_candles.to_csv(output_path, index=True, encoding="utf-8-sig", compression="gzip")
