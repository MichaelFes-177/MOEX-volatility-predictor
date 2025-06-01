import pathlib
import re
import csv
import pandas as pd

FIN_CSV = pathlib.Path("investing_news.csv")      # финансовый поток
GEO_CSV = pathlib.Path("top_per_day_news.csv")    # гео-топ-1
OUT_CSV = pathlib.Path("news.csv")                # итоговый файл

# ────────────────────────────────────────────────────────────────────────────────
def load_fin():
    """
    Читает investing_jews.csv построчно и возвращает DataFrame со столбцами:
      - datetime (pd.Timestamp)
      - title    (str)
      - url      (str)
      - sim      == 1.0
      - src      == "fin"
    """
    records = []
    with FIN_CSV.open("r", encoding="utf-8-sig") as f:
        header = f.readline()  # пропускаем строку-заголовок
        for lineno, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            # 1) КОГДА ДАТА: ищем "DD.MM.YYYY, HH:MM" (с опциональным "Опубликовано ")
            date_match = re.search(r"(\d{2}\.\d{2}\.\d{4}),\s*(\d{2}:\d{2})", line)
            if not date_match:
                continue
            date_part, time_part = date_match.groups()
            datetime_str = f"{date_part} {time_part}"
            try:
                dt = pd.to_datetime(datetime_str, format="%d.%m.%Y %H:%M", dayfirst=True)
            except Exception:
                dt = pd.to_datetime(datetime_str, dayfirst=True, errors="coerce")
            if pd.isna(dt):
                continue

            # 2) КОГДА URL: ищем http://… или https://… или что-то вида ru.investing.com/…
            url_match = re.search(r"https?://[^\s\",]+|[a-z]+\.investing\.com/[^\s\",]+", line)
            if not url_match:
                continue
            url = url_match.group(0)

            # 3) Заголовок: всё до начала URL, убираем обрамляющие кавычки/запятые
            title_raw = line[: url_match.start()]
            title = title_raw.rstrip(" ,").lstrip('"').rstrip('"')

            # 4) Добавляем запись с sim=1.0
            records.append({
                "datetime": dt,
                "title": title,
                "url": url,
                "sim": 1.0,
                "src": "fin"
            })

    if not records:
        # Если ни одной строки не распарсилось, возвращаем пустой DataFrame с нужными столбцами
        return pd.DataFrame(columns=["datetime", "title", "url", "sim", "src"])

    return pd.DataFrame.from_records(records, columns=["datetime", "title", "url", "sim", "src"])


# ────────────────────────────────────────────────────────────────────────────────
def load_geo(path: pathlib.Path, label: str):
    """
    Читает CSV-файл по указанному пути (если существует) через csv.reader, чтобы
    правильно учитывать кавычки и запятые. Возвращает DataFrame со столбцами:
      - datetime (pd.Timestamp)
      - title    (str)
      - url      (str)
      - sim      (float или pd.NA)
      - src      == label
    Если файл не найден, возвращает пустой DataFrame с такими же столбцами.
    """
    if not path.exists():
        return pd.DataFrame(columns=["datetime", "title", "url", "sim", "src"])

    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame(columns=["datetime", "title", "url", "sim", "src"])

        # Нормализуем имена колонок
        header_clean = [col.strip().lower() for col in header]

        # Ищем индексы нужных столбцов
        try:
            idx_datetime = header_clean.index("datetime")
        except ValueError:
            raise KeyError(f"Файл {path.name} не содержит столбца 'datetime' (есть: {header_clean})")
        try:
            idx_title = header_clean.index("title")
        except ValueError:
            raise KeyError(f"Файл {path.name} не содержит столбца 'title' (есть: {header_clean})")
        try:
            idx_url = header_clean.index("url")
        except ValueError:
            raise KeyError(f"Файл {path.name} не содержит столбца 'url'   (есть: {header_clean})")

        # Колонка 'sim' может быть необязательной
        idx_sim = header_clean.index("sim") if "sim" in header_clean else None

        for lineno, row in enumerate(reader, start=2):
            if len(row) < 3:
                continue

            # 1) Парсим datetime
            dt_str = row[idx_datetime].strip()
            dt = pd.to_datetime(dt_str, dayfirst=True, errors="coerce")
            if pd.isna(dt):
                continue

            # 2) Берём title и url
            title = row[idx_title].strip()
            url   = row[idx_url].strip()
            if not url:
                continue

            # 3) Симилиарити (если есть)
            if idx_sim is not None and idx_sim < len(row):
                sim_raw = row[idx_sim].strip()
                try:
                    sim_val = float(sim_raw)
                except Exception:
                    sim_val = pd.NA
            else:
                sim_val = pd.NA

            records.append({
                "datetime": dt,
                "title": title,
                "url": url,
                "sim": sim_val,
                "src": label
            })

    if not records:
        return pd.DataFrame(columns=["datetime", "title", "url", "sim", "src"])

    return pd.DataFrame.from_records(records, columns=["datetime", "title", "url", "sim", "src"])


# ────────────────────────────────────────────────────────────────────────────────
# Собираем все источники в один DataFrame
# ────────────────────────────────────────────────────────────────────────────────

# 1) Считываем финансовые новости
df_fin = load_fin()
print(f"> load_fin(): прочитано {len(df_fin)} финансовых строк (sim=1.0)")

# 2) Считываем гео-новости
df_geo = load_geo(GEO_CSV, "geo")
print(f"> load_geo(top_per_day_news.csv): прочитано {len(df_geo)} гео-строк")

# 3) Считываем «красный» поток

# 4) Объединяем
news = pd.concat([df_fin, df_geo], ignore_index=True)
print(f"> Всего строк до удаления дублей: {len(news)}")

# 5) Удаляем дубли по URL (оставляем самую свежую по datetime), затем сортируем от старых к новым
news = (
    news.sort_values("datetime", ascending=False)
        .drop_duplicates(subset=["url"], keep="first")
        .sort_values("datetime")
)
print(f"> После удаления дублей: {len(news)} строк")

# 6) Сохраняем результат в UTF-8 с BOM
news.to_csv(OUT_CSV, index=False, encoding="utf-8-sig", quoting=1)
print(f"✓ Файл «{OUT_CSV.name}» готов — всего {len(news)} строк")
