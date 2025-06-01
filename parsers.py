# update_investing_news.py
import re, csv, time, pathlib, contextlib
import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

CSV_FILE = pathlib.Path("investing_news.csv")
BASE_URL = "https://ru.investing.com/equities/moskovskaya-birzha-oao-news"
DATE_RE  = re.compile(r"(\d{2}.\d{2}.\d{4}, \d{2}:\d{2})")
HEADERS  = ["title", "url", "date"]

# ----------------------------------------------------------------------
# 1. читаем CSV «как текст» → самая свежая дата-время
# ----------------------------------------------------------------------
def latest_timestamp(path: pathlib.Path) -> pd.Timestamp:
    if not path.exists():
        return pd.Timestamp("1900-01-01")
    newest = pd.Timestamp("1900-01-01")
    with path.open(encoding="utf-8-sig") as fh:
        next(fh)                         # skip header
        for line in fh:
            m = DATE_RE.search(line)
            if m:
                ts = pd.to_datetime(m.group(1), format="%d.%m.%Y, %H:%M")
                if ts > newest:
                    newest = ts
    return newest

LAST_TS = latest_timestamp(CSV_FILE)
print("Последняя запись в CSV:", LAST_TS)

# ----------------------------------------------------------------------
# 2. контекст-менеджер браузера
# ----------------------------------------------------------------------
@contextlib.contextmanager
def get_driver():
    drv = uc.Chrome()
    drv.implicitly_wait(3)
    try:
        yield drv
    finally:
        with contextlib.suppress(Exception):
            drv.quit()

def accept_cookies(drv):
    try:
        drv.find_element(
            By.XPATH,
            "//button[contains(translate(.,'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
            "'abcdefghijklmnopqrstuvwxyz'),'accept all')]"
        ).click()
        time.sleep(.5)
    except NoSuchElementException:
        pass

# ----------------------------------------------------------------------
# 3. скрапинг
# ----------------------------------------------------------------------
new_rows, stop, page = [], False, 1
with get_driver() as driver:
    while not stop and page <= 50:                    # safety-лимит
        url = BASE_URL if page == 1 else f"{BASE_URL}/{page}"
        driver.get(url); accept_cookies(driver)
        print(f"[page {page}]")

        cards = driver.find_elements(By.CSS_SELECTOR,
                                     'a[data-test="article-title-link"]')
        titles = [c.text.strip() for c in cards]
        hrefs  = [c.get_attribute("href") for c in cards]
        hrefs  = [
            h if h and h.startswith("http") else "https://ru.investing.com" + h
            for h in hrefs
        ]

        for title, href in zip(titles, hrefs):
            driver.get(href); time.sleep(1.1)

            body_text = driver.find_element(By.TAG_NAME, "body").text
            m = DATE_RE.search(body_text)
            if not m:
                continue
            raw = m.group(1)                          # «DD.MM.YYYY, HH:MM»
            ts  = pd.to_datetime(raw, format="%d.%m.%Y, %H:%M")

            if ts <= LAST_TS:                         # дошли до старых
                stop = True
                break

            new_rows.append([title, href, raw])
            print("  +", ts, title[:70])
        page += 1

# ----------------------------------------------------------------------
# 4. дозаписываем CSV
# ----------------------------------------------------------------------
if not new_rows:
    print("Свежих новостей нет – CSV не изменён.")
else:
    mode = "a" if CSV_FILE.exists() else "w"
    with CSV_FILE.open(mode, encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if mode == "w":
            w.writerow(HEADERS)
        w.writerows(new_rows)
    print(f"✔  Добавлено {len(new_rows)} строк → {CSV_FILE}")
# ——— ПОСЛЕ append новых строк — сортируем весь CSV по дате ———
import datetime
import re

# та же регулярка, что и при сборе:
DATE_RE = re.compile(r"(\d{2}\.\d{2}\.\d{4}, \d{2}:\d{2})")

# читаем файл «как есть», сохраняя переводы строк
text = CSV_FILE.read_text(encoding="utf-8-sig")
lines = text.splitlines(keepends=True)
if len(lines) <= 1:
    print("⚠  В файле нет записей кроме заголовка — ничего не сортируем.")
else:
    header, body = lines[0], lines[1:]

    def extract_dt(line):
        m = DATE_RE.search(line)
        if not m:
            return datetime.datetime.min
        return datetime.datetime.strptime(m.group(1), "%d.%m.%Y, %H:%M")

    # сортируем body по дате (новейшие в начало)
    body.sort(key=extract_dt, reverse=True)

    # перезаписываем CSV в том же формате
    with CSV_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        f.write(header)
        f.writelines(body)

    print(f"✔  CSV отсортирован — теперь {len(body)} строк.")

#!/usr/bin/env python3
import re
import time
import pathlib
import datetime as dt
import requests
import csv
import pandas as pd

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F

# --------- КОНФИГ ----------
CSV_FILE        = pathlib.Path("top_per_day_news.csv")
BASE_URL        = "https://www.mk.ru/news/{y}/{m}/{d}/"
IMPORTANT_CLASS = "news-listing__item_hot"
KEY_NEWS = [
    "Россия объявила мобилизацию", "вторжение в Украину", "санкции против России",
    "отключение от SWIFT", "падение рубля", "ракетный обстрел",
    "рост цен на нефть", "кризис на границе", "запрет экспорта",
    "заявление ЦБ РФ", "чрезвычайное положение", "война",
]
FMT      = "%Y-%m-%d %H:%M:%S"
TIME_RE  = re.compile(r"(\d{2}):(\d{2})")

def last_ts(path):
    if not path.exists():
        return dt.datetime.min
    df_old = pd.read_csv(path, usecols=["datetime"], parse_dates=["datetime"])
    if df_old.empty:
        return dt.datetime.min
    return df_old["datetime"].max().to_pydatetime()

def fetch_hot_for(d, page=1):
    url = BASE_URL.format(y=d.year, m=d.month, d=d.day)
    if page>1: url += f"page{page}.html"
    r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"})
    if r.status_code!=200: return []
    soup = BeautifulSoup(r.text, "html.parser")
    out=[]
    for li in soup.select("li.news-listing__item"):
        if IMPORTANT_CLASS not in li.get("class",[]): continue
        a = li.select_one("a.news-listing__item-link")
        t = li.select_one("span.news-listing__item-time")
        h = li.select_one("h3.news-listing__item-title")
        if not (a and t and h): continue
        href = a["href"]
        if not href.startswith("http"): href="https://www.mk.ru"+href
        m = TIME_RE.search(t.text.strip())
        if not m: continue
        hh,mm=map(int,m.groups())
        ts=dt.datetime.combine(d, dt.time(hh,mm))
        out.append((ts, h.text.strip(), href))
    return out

def main():
    LAST = last_ts(CSV_FILE)
    print("CSV до:", LAST)

    # 1) Собираем все новые строки
    rows=[]
    today=dt.date.today(); cur=today
    while cur>=LAST.date():
        p=1
        while True:
            hits = fetch_hot_for(cur, p)
            if not hits: break
            for ts,title,href in hits:
                if ts<=LAST: break
                rows.append((ts, title, href))
            else:
                p+=1; continue
            break
        cur-=dt.timedelta(days=1); time.sleep(0.2)

    if not rows:
        print("Новых нет"); return

    # 2) SBERT + ручная нормализация
    model   = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    key_e   = model.encode(KEY_NEWS,  convert_to_tensor=True)
    news_e  = model.encode([r[1] for r in rows], convert_to_tensor=True, batch_size=64)
    key_e   = F.normalize(key_e, p=2, dim=1)
    news_e  = F.normalize(news_e, p=2, dim=1)
    sims    = util.cos_sim(news_e, key_e).max(dim=1).values.cpu().numpy()

    # 3) Формируем DataFrame новых
    new_df = pd.DataFrame({
        "datetime": [r[0].strftime(FMT) for r in rows],
        "title":     [r[1] for r in rows],
        "url":       [r[2] for r in rows],
        "sim":       sims
    })

    # 4) Загружаем старый CSV (если есть) и конкатим
    if CSV_FILE.exists():
        old_df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df

    # 5) Гарантируем корректные типы
    df["sim"]      = pd.to_numeric(df["sim"], errors="coerce").fillna(0.0)
    df["datetime"] = pd.to_datetime(df["datetime"], format=FMT)

    # 6) Оставляем по 1 записи в день с max(sim), при равных — самый свежий datetime
    df["date"] = df["datetime"].dt.date
    df = (df.sort_values(["date","sim","datetime"], ascending=[True,False,False])
            .drop_duplicates("date", keep="first"))

    # 7) Сортируем по datetime DESC и сохраняем
    df = df.sort_values("datetime", ascending=False)
    df[["datetime","title","url","sim"]].to_csv(
        CSV_FILE, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL
    )

    print("Готово, всего строк:", len(df))

if __name__=="__main__":
    main()
