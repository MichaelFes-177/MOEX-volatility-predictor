# MOEX-volatility-predictor
# Прогнозирование волатильности акциq MOEX с учётом новостей

**Автор**: Михаил Фесенко  
**Дата**: июнь 2025  
**Проект**: школьный научно-исследовательский проект по машинному обучению и анализу данных от ЦУ

---

## Описание проекта

В рамках данного проекта мы построили сквозной пайплайн от сбора рыночных данных и новостей до интерактивного веб-приложения, способного в реальном времени прогнозировать, будет ли завтрашняя относительная волатильность акций MOEX (Московской биржи) выше медианы («High») или ниже («Low»). Модель сочетает классические технические признаки (из свечных графиков) и текстовые фичи из финансовых и геополитических новостей (SBERT → PCA). Для классификации используется CatBoostClassifier. Результат обернут в удобный Streamlit-интерфейс с визуализацией, SHAP-анализом и бэктестом простых торговых стратегий.

---

## Содержание документации

1. [Структура репозитория](#структура-репозитория)  
2. [Установка и запуск](#установка-и-запуск)  
3. [Описание данных](#описание-данных)  
   - [Свечные данные (candles.csv.gz)](#свечные-данные-candlescsvgz)  
   - [Новостной поток (news.csv)](#новостной-поток-newscsv)  
4. [Парсинг и обновление данных](#парсинг-и-обновление-данных)  
   - [Парсер финансовых новостей (Investing.com)](#парсер-финансовых-новостей-investingcom)  
   - [Парсер «горячих» гео-новостей (MK.ru)](#парсер-горячих-гео-новостей-mkru)  
   - [Скрипты обновления CSV](#скрипты-обновления-csv)  
5. [Предобработка и агрегация данных](#предобработка-и-агрегация-данных)  
   - [Технические признаки из свечей](#технические-признаки-из-свечей)  
   - [Текстовые признаки из новостей: SBERT + PCA](#текстовые-признаки-из-новостей-sbert--pca)  
   - [Shock-признаки и лаговые признаки](#shock-признаки-и-лаговые-признаки)  
6. [Построение финального датасета](#построение-финального-датасета)  
7. [Обучение модели CatBoostClassifier](#обучение-модели-catboostclassifier)  
   - [TimeSeriesSplit и подбор τ](#timeseriessplit-и-подбор-τ)  
   - [Валидация (CV & Hold-out)](#валидация-cv--hold-out)  
   - [Анализ ошибок и Confusion Matrix](#анализ-ошибок-и-confusion-matrix)  
8. [Streamlit-приложение](#streamlit-приложение)  
   - [Пример работы](#пример-работы)
9. [Торговые стратегии](#streamlit-приложение)  
   - [виды стратегий](#пример-работы)  
---

## Структура репозитория


- **main.py** — основной Streamlit-скрипт — веб-интерфейс для прогноза.  
- **cat_vol_model.pkl** — сериализованная CatBoostClassifier (без Optuna).  
- **tau_star.npy** — сохранённый порог τ=0.33 (float).  
- **pca_components.npz** — две матрицы PCA-компонентов (ключи `fin`, `geo`), формы `(64, 384)` каждая.  
- **candles.csv.gz** — gz-сжатый CSV с историческими свечными данными MOEX и тех-признаками.  
- **news.csv** — объединённый CSV новостей (`datetime,title,url,sim,src,date_only`).  
- **parser.py/** — скрипты для дозаписи новых данных в CSV.  
- **assets/** — графики для отчетности (PR-кривая, Confusion Matrix, residual plot, equity plots).  

---

## Установка и запуск

1. **Клонируйте репозиторий**  
   ```bash
   git clone https://github.com/MichaelFes-177/MOEX-volatility-predictor.git
   cd moex-volatility

2. **Создайте виртуальное окружение и установите зависимости**
```
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate     # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Поместите данные**

- candles.csv.gz и news.csv находятся в корне репозитория.

- Убедитесь, что файлы имеют корректный формат (см. следующий раздел).

4. **Запуск Streamlit-приложения**
```
streamlit run main.py
```
- Откроется веб-страница (по умолчанию http://localhost:8501), где можно взаимодействовать с прогнозом.


## Описание данных

*Свечные данные (candles.csv.gz)*
- Источник: MOEX ISS API, запрос candles.json по интервалу 24 часа.
- Формат CSV (после расчётов тех-признаков):


| date                | open  | high  | low   | close | volume     | ret1   | sma5 | sma20 | vol\_z | atr14 |
| ------------------- | ----- | ----- | ----- | ----- | ---------- | ------ | ---- | ----- | ------ | ----- |
| 2014-01-02 00:00:00 | 38.50 | 39.00 | 38.20 | 38.80 | 1200000000 | 0.0154 | ...  | ...   | ...    | ...   |
| 2014-01-03 00:00:00 | 38.80 | 39.20 | 38.40 | 39.00 | 1300000000 | 0.0052 | ...  | ...   | ...    | ...   |
| ...                 | ...   | ...   | ...   | ...   | ...        | ...    | ...  | ...   | ...    | ...   |



Описание колонок:
- date — начало торгового дня (normalized, 00:00:00).

- open, high, low, close — цены за день.

- volume — суммарный объём торгов за день.

- ret1 = close_t / close_{t-1} − 1 — дневная доходность.

- sma5 = rolling_mean(close, 5)

- sma20 = rolling_mean(close, 20)

- vol_z = (volume_t − rolling_mean(volume,20)) / rolling_std(volume,20)

- atr14 = rolling_mean(high−low, 14) / close

Количество строк: примерно 2822 (с 01.01.2014 по 21.05.2025).


*Новостной поток (news.csv)*

| datetime            | title                                          | url                                                                                                                      | sim    | src | date\_only |
| ------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------ | --- | ---------- |
| 2025-05-22 00:27:00 | Герой России генерал-полковник Андрей ...      | [https://www.mk.ru/politics/2025/05/22/geroy-rossii-](https://www.mk.ru/politics/2025/05/22/geroy-rossii-)...            | 0.5912 | geo | 2025-05-22 |
| 2025-05-22 11:36:00 | Мосбиржа добавит БПИФы ...                     | [https://ru.investing.com/news/general-news/article-2770929](https://ru.investing.com/news/general-news/article-2770929) | 1.0000 | fin | 2025-05-22 |
| 2025-05-22 14:05:00 | Мосбиржа обсудит с ЦБ расширение диапазона ... | [https://ru.investing.com/news/general-news/article-2771269](https://ru.investing.com/news/general-news/article-2771269) | 1.0000 | fin | 2025-05-22 |
| 2025-05-24 11:36:00 | Россиянка-поджигательница стала ...            | [https://www.mk.ru/incident/2025/05/24/rossiyankapodzhig](https://www.mk.ru/incident/2025/05/24/rossiyankapodzhig)...    | 0.6167 | geo | 2025-05-24 |
| 2025-05-25 09:20:00 | Недружественная страна подняла авиацию ...     | [https://www.mk.ru/politics/2025/05/25/nedruzhestvennaya-](https://www.mk.ru/politics/2025/05/25/nedruzhestvennaya-)...  | 0.7208 | geo | 2025-05-25 |
| …                   | …                                              | …                                                                                                                        | …      | …   | …          |

Описание колонок:
- datetime — точное время публикации новости.

- title — заголовок новости.

- url — ссылка (полная) на статью.

- sim — similarity (только для geo): косинус схожести с ключевыми фразами, ∈ [0,1]. Для fin всегда равен 1.0.

- src — источник: "fin" (финансовая новость) или "geo" (геополитическая).

- date_only = datetime.dt.date (дата без времени, чтобы агрегировать по дню).

- Количество строк: ~3525 (фин=~1300, гео=~2200), уникальных по url.


## Парсинг и обновление данных

В проекте используtтся единый скрипт для регулярного обновления новостей:

*parser.py*

Финансовые:
- Сканирует сайт Investing.com (раздел «МОEX»), постранично (≈42 страницы), собирает title, url, raw_date.

- Regex-добыча даты: ищет "(DD.MM.YYYY), HH:MM" в тексте страницы.

- Преобразует в pd.Timestamp(..., format="%d.%m.%Y, %H:%M").

- Останавливается, как только встречает дату ≤ последней из investing_jews.csv.

- Дозаписывает новые строки (append) в конец investing_jews.csv, затем полностью сортирует файл по дате (desc) для удобства.
```
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
```

Геополитика:
- Сканирует сайт MK.ru по дате (итерация от сегодняшней даты назад; while cur >= LAST.date()), вычитывая страницы /news/YYYY/M/D/ и /pageN.html.

- Для каждой li.news-listing__item проверяет класс news-listing__item_hot (горячая новость).

- Извлекает title, url и время публикации HH:MM из тега span.news-listing__item-time.

- Составляет datetime = combine(date_obj, time_obj).

- Останавливается, когда встречает дату ≤ последней из top_per_day_news.csv.

- Сохраняет в top_per_day_news.csv (или перезаписывает как топ-1 на каждый день с наибольшим sim).

- После завершения скрипта создаётся итоговый CSV, содержащий по 1 записи на день (самая высокая similarity +, tie-breaker: более поздний datetime).


```
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
```

Фильтруем по содержанию новости по 1 в день:

```
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
```


Скрипт обновления CSV

- Находится в  скрипте (update_investing_news.py и update_geo_hotnews.py) имеют схожую логику:

- Взять последнюю дату-время из существующего CSV (функция last_ts(path)).

- Перебирать новые страницы / новые даты, пока не встретили «старую» дату.

- Собрать новые записи, сохранить в DataFrame.

- Для geo: дополнительно SBERT-фильтровать → выбрать top-1 per day.

- Перезаписать/дозаписать CSV и отсортировать по datetime.


## Предобработка и агрегация данных
***Технические признаки из свечей***

1. Чтение исторических данных
```
import pandas as pd

candles = pd.read_csv(
    "candles.csv.gz",
    compression="gzip",
    parse_dates=["date"]
).set_index("date").sort_index()
```
2. Расчёт признаков

```
candles["ret1"] = candles["close"].pct_change()
candles["sma5"] = candles["close"].rolling(5).mean()
candles["sma20"] = candles["close"].rolling(20).mean()
candles["vol_z"] = (
    candles["volume"] - candles["volume"].rolling(20).mean()
) / candles["volume"].rolling(20).std(ddof=0)
candles["atr14"] = (
    candles["high"] - candles["low"]
).rolling(14).mean() / candles["close"]
candles = candles.dropna()
```
## Текстовые признаки из новостей: SBERT + PCA
1. Загрузка SBERT модели

```
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
```
- Модель предоставляет 384-мерный эмбеддинг для любого заголовка на русском языке.

2. Применение к новостным заголовкам

```
titles = news["title"].tolist()
emb_matrix = st_model.encode(titles, batch_size=128, show_progress_bar=False)
# emb_matrix.shape == (len(news), 384)
```
3. Группировка по дате

```
news["date"] = news["datetime"].dt.normalize()  # без времени, ровно на полночь
agg = (
    news.groupby("date")
        .agg(
          sent_mean=("title", lambda lst: np.tanh(
                        (sum(w in t.lower() for t in lst for w in POS) -
                         sum(w in t.lower() for t in lst for w in NEG)) / 2
                      )),
          n=("title", "size"),
          emb_stack=("emb", lambda arr: np.vstack(arr).mean(axis=0))
        )
)
```
- sent_mean = средний toy-sentiment (натянутое через tanh).

- emb_stack = усреднённый эмбеддинг всех заголовков в день.

4. PCA
```
from sklearn.decomposition import PCA

# Для финансовых новостей
fin_emb_matrix = np.vstack(fin_day["emb_stack"].values)
pca_fin = PCA(n_components=64, random_state=42).fit(fin_emb_matrix)
fin_pca_feats = pd.DataFrame(
    pca_fin.transform(fin_emb_matrix),
    index=fin_day.index,
    columns=[f"fin_p{i}" for i in range(64)]
)

# Для геополитических новостей
geo_emb_matrix = np.vstack(geo_day["emb_stack"].values)
pca_geo = PCA(n_components=64, random_state=42).fit(geo_emb_matrix)
geo_pca_feats = pd.DataFrame(
    pca_geo.transform(geo_emb_matrix),
    index=geo_day.index,
    columns=[f"geo_p{i}" for i in range(64)]
)

# Сохраняем компоненты в файл:
np.savez("pca_components.npz", fin=pca_fin.components_, geo=pca_geo.components_)
```
5. Сбор итоговых блоков

```
fin_day = pd.concat([fin_day.drop(columns=["emb_stack"]), fin_pca_feats], axis=1)
geo_day = pd.concat([geo_day.drop(columns=["emb_stack"]), geo_pca_feats], axis=1)
```
## Shock-признаки и лаговые признаки

1. Расчёт Z-score тональности

```
for df, prefix in [(fin_day, "fin_"), (geo_day, "geo_")]:
    roll30 = df["sent_mean"].rolling(30, min_periods=10)
    z = (df["sent_mean"] - roll30.mean()) / roll30.std(ddof=0)
    df[f"{prefix}shock"] = (z.abs() > 1.8).astype(int)
    df[f"{prefix}shock_mag"] = z.abs().fillna(0)
    for lag in [1,2,3]:
        df[f"{prefix}sent_lag{lag}"] = df["sent_mean"].shift(lag)
        df[f"{prefix}shock_lag{lag}"] = df[f"{prefix}shock"].shift(lag)
    df[f"{prefix}sent_roll7"] = df["sent_mean"].rolling(7).mean()
```
2. Результат
Для каждого дня сформированы колонки:
```
fin_sent_mean, fin_n, fin_p0…fin_p63, fin_shock, fin_shock_mag,
fin_sent_lag1, fin_sent_lag2, fin_sent_lag3, fin_shock_lag1, …
fin_sent_roll7
```
- и аналогично с geo_-префиксом.

##Обучение модели CatBoostClassifier

**CatBoost: основные настройки**
```
from catboost import CatBoostClassifier

cat_features = [
    X.columns.get_loc("weekday"),
    X.columns.get_loc("month"),
]

model = CatBoostClassifier(
    iterations=400,
    depth=7,
    learning_rate=0.05,
    colsample_bylevel=0.8,
    random_seed=42,
    loss_function="Logloss",
    auto_class_weights="Balanced",
    verbose=0,
)
```
**TimeSeriesSplit (5-фолдов) и получение CV-предсказаний**
```
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_curve, classification_report

tscv = TimeSeriesSplit(n_splits=5)
probas_all = []
y_all = []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(
        X_train, y_train,
        cat_features=cat_features
    )

    probas = model.predict_proba(X_test)[:, 1]  # P(high)
    probas_all.extend(probas)
    y_all.extend(y_test.values)
```

# Поиск порога τ (Precision-Recall)

```
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_all, probas_all)

# Выбираем τ так, чтобы Recall(high) ≥ 0.65
target_recall = 0.65
idx = np.where(recall >= target_recall)[0][-1]
tau = thresholds[idx]
print(f"Selected τ = {tau:.2f} при recall ≥ {recall[idx]:.3f}")
```
- Результат: τ ≈ 0.33.

# Оценка качества на CV (τ=0.33)
```
from sklearn.metrics import classification_report, confusion_matrix

y_pred = (np.array(probas_all) >= tau).astype(int)
print(classification_report(y_all, y_pred, digits=3, target_names=["Low","High"]))

cm_cv = confusion_matrix(y_all, y_pred)
print("Confusion Matrix (CV):")
print(cm_cv)
```
- CV-результаты:
```
                 precision  recall  f1-score  support
Low               0.642   0.513     0.570    1305
High              0.523   0.650     0.580    1070

accuracy                       0.575    2375
macro avg          0.582   0.582     0.575    2375
weighted avg       0.588   0.575     0.575    2375
```
- Confusion Matrix (CV):

![image](https://github.com/user-attachments/assets/d859955f-293f-4571-b7a4-8e49147f3f4e)


## Валидация (Hold-Out, 2025)
**Результаты Hold-Out:**
```
                 precision  recall  f1-score  support
Low               0.562   0.184     0.277       49
High              0.588   0.891     0.708       64

accuracy                       0.584      113
macro avg          0.575   0.537     0.492      113
weighted avg       0.577   0.584     0.521      113
```
Confusion Matrix (Hold-Out):

```
[[  9  40]
 [  7  57]]
```
Интерпретация:
- Модель настроена на высокий recall класса High (0.89), пожертвовав точностью для Low.
- Баланс между Precision и Recall выбран осознанно: легче допустить ложную тревогу (FP_Load → High), чем упустить реальный «всплеск» σ (FN_High).

## Streamlit-приложение

![image](https://github.com/user-attachments/assets/3f508842-1145-4d4a-8285-6ec486f25c1f)

![image](https://github.com/user-attachments/assets/a68eff3b-f5eb-4d48-9e0d-04c339881f3d)

## Торговые стратегии
**Использовались 3 стратегии:**
- Buy&Hold
![image](https://github.com/user-attachments/assets/bf91c1fb-932c-4079-b2ce-1f6fba6f322e)
- Hold-Lowσ
![image](https://github.com/user-attachments/assets/093b7ff2-fb67-401c-a8b4-40d012a56b01)

- Short-Vol
![image](https://github.com/user-attachments/assets/b1afc71a-1383-40b6-bcbe-6cf659d97f24)


Hold-Lowσ ожидаемо оказалась самой прибыльной:
![image](https://github.com/user-attachments/assets/1fbfb805-4edc-4a2b-8a3d-4e55b6dae222)
