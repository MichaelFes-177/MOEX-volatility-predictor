"""
Volatility Dashboard – полная, устойчиво работающая версия (2025-06)
--------------------------------------------------------------------
* Shock-markers (оранжевые точки) на candlestick
* Слайдер τ + кнопка «Подобрать τ» (макс balanced-accuracy)
* SHAP топ-5 фичей (учитываются embedding-компоненты)
* Переход к произвольной дате (date_input)
* 3 демо-стратегии + Buy&Hold, equity-кривые
"""
import os
os.environ["USE_TF"] = "0"            # отключаем TensorFlow/Keras

import streamlit as st
import pandas as pd, numpy as np, joblib, csv, datetime as dt, re
import plotly.graph_objects as go
from pathlib import Path
from catboost import Pool
CANDLE_CSV =  "candles.csv.gz"
NEWS_CSV   =  "news.csv"
MODEL_FILE = "cat_vol_model.pkl"
TAU_FILE   = "tau_star.npy"
PCA_FILE   = "pca_components.npz"

TECH_COLS = ["ret1","sma5","sma20","vol_z","atr14"]
POS = ("рост","прибавил","увелич","рекорд")
NEG = ("падени","снизил","упал","минимум")

# ─── StaticPCA (простой класс без explained_variance_) ──────────
class StaticPCA:
    def __init__(self, comps: np.ndarray):
        self.components_ = comps  # (k, d)

    def transform(self, X: np.ndarray) -> np.ndarray:
        # X shape: (n, d), return: (n, k)
        return X @ self.components_.T

def load_pca_dict(path=PCA_FILE):
    arr = np.load(path, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):
        return {k: arr[k] for k in arr.files}
    return {"fin": arr}

PCA_DICT = load_pca_dict()
pca_fin   = StaticPCA(PCA_DICT["fin"])
pca_geo   = StaticPCA(PCA_DICT.get("geo", PCA_DICT["fin"]))
EMB_DIM   = pca_fin.components_.shape[1]

# ─── SBERT или Dummy (если SBERT падает) ─────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
except Exception:
    st.warning("⚠ SBERT не загрузился – эмбеддинги будут нулевыми")
    class DummyEmbedder:
        def encode(self, sents, **kwargs):
            return np.zeros((len(sents), EMB_DIM))
    st_model = DummyEmbedder()

# ─── Загрузка CatBoost-модели + τ ─────────────────────────────────
@st.cache_resource
def load_model():
    mdl = joblib.load(MODEL_FILE)
    tau = float(np.load(TAU_FILE))
    return mdl, tau, mdl.feature_names_

# ─── Загрузка CSV-таблиц ─────────────────────────────────────────
@st.cache_data(ttl=300)
def load_tables():
    candles = (
        pd.read_csv(CANDLE_CSV, compression="gzip", parse_dates=["date"])
          .set_index("date")
          .sort_index()
    )
    news = (
        pd.read_csv(NEWS_CSV, parse_dates=["datetime"],
                    dtype={"title":"string","url":"string","src":"category","sim":float})
          .rename(columns={"datetime": "date"})
    )
    news["date_only"] = news.date.dt.date
    return candles, news

# ─── Загружаем модель и данные ────────────────────────────────────
model, TAU_DEF, FEAT_NAMES = load_model()
candles, news = load_tables()

# ─── Вычисляем, какие индексы FEAT_NAMES отвечают за embedding-компоненты ─
FIN_SLOTS, GEO_SLOTS = [], []
pat_fin = re.compile(r"^(?:fin_)?p(\d+)$")
pat_geo = re.compile(r"^(?:geo_)?p(\d+)$")
for idx, name in enumerate(FEAT_NAMES):
    if (m := pat_fin.match(name)):
        FIN_SLOTS.append((idx, int(m.group(1))))
    elif (m := pat_geo.match(name)):
        GEO_SLOTS.append((idx, int(m.group(1))))

# ─── Группируем «shock days» (средняя sent_val > 0.8 по модулю) ────
news["sent_val"] = news.title.str.lower().apply(
    lambda t: np.tanh((sum(w in t for w in POS) - sum(w in t for w in NEG))/2)
)
shock_days = (news.groupby("date_only")["sent_val"].mean().abs() > 0.8)

# ─── Feature builder ─────────────────────────────────────────────
def build_vec(df: pd.DataFrame, pca: StaticPCA) -> np.ndarray:
    if df.empty:
        return np.zeros(pca.components_.shape[0])
    emb = st_model.encode(df.title.tolist(), show_progress_bar=False)
    return pca.transform(np.vstack(emb)).mean(axis=0)

def make_features(row: pd.Series, day_news: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает DataFrame (1×len(FEAT_NAMES)) с точно теми колонками,
    которые ждёт модель (model.feature_names_).
    """
    vec = np.zeros(len(FEAT_NAMES), dtype=float)
    idx_map = {n: i for i, n in enumerate(FEAT_NAMES)}

    # 1) технические фичи
    for c in TECH_COLS:
        vec[idx_map[c]] = float(row[c])

    # 2) календарные (как строки!)
    vec[idx_map["weekday"]] = row.name.weekday()
    vec[idx_map["month"]]   = row.name.month

    # 3) эмбеддинги
    df_fin = day_news.loc[day_news.src == "fin"]
    df_geo = day_news.loc[day_news.src != "fin"]
    v_fin = build_vec(df_fin, pca_fin)
    v_geo = build_vec(df_geo, pca_geo)

    for slot_idx, comp in FIN_SLOTS:
        if comp < len(v_fin):
            vec[slot_idx] = float(v_fin[comp])
    for slot_idx, comp in GEO_SLOTS:
        if comp < len(v_geo):
            vec[slot_idx] = float(v_geo[comp])

    # 4) sentiment (если есть колонка «sent»)
    if "sent" in idx_map and not day_news.empty:
        vec[idx_map["sent"]] = float(day_news["sent_val"].mean())

    # Собираем в DataFrame и конвертируем weekday/month в строки
    df = pd.DataFrame([vec], columns=FEAT_NAMES)
    for cat in ("weekday", "month"):
        if cat in df.columns:
            df[cat] = df[cat].astype("string")
    return df

# ─── Sidebar: параметры ─────────────────────────────────────────
st.sidebar.header("Настройки")
period = st.sidebar.selectbox("История (дней)", [7, 30, 90], index=1)
sel_date = st.sidebar.date_input("Центр графика", candles.index[-1].date())

tau_ui = st.sidebar.slider("Порог τ", min_value=0.1, max_value=0.9,
                           value=TAU_DEF, step=0.01)

if st.sidebar.button("Подобрать τ"):
    # берём срез от (sel_date - period) до sel_date
    rng = candles.loc[
        pd.Timestamp(sel_date) - pd.Timedelta(days=period):
        pd.Timestamp(sel_date)
    ]
    # истинные классы: выше/ниже медианы волатильности
    realized = ((rng.high - rng.low) / rng.close)
    thr_med = realized.median()
    y_true = (realized > thr_med).astype(int).values

    # прогнозы модели (на этом же срезе)
    y_proba = []
    for d, row in rng.iterrows():
        df_feats = make_features(row, news[news.date_only == d.date()])
        y_proba.append(model.predict_proba(df_feats)[0, 1])

    from sklearn.metrics import balanced_accuracy_score
    best_tau, best_ba = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 81):
        ba = balanced_accuracy_score(y_true, np.array(y_proba) >= t)
        if ba > best_ba:
            best_ba, best_tau = ba, t
    tau_ui = best_tau
    st.sidebar.success(f"Найден τ = {tau_ui:.2f}, BA = {best_ba:.3f}")

# ─── Основные прогнозы по срезу ─────────────────────────────────
window_start = pd.Timestamp(sel_date) - pd.Timedelta(days=period)
window_end   = pd.Timestamp(sel_date)
sub = candles.loc[window_start: window_end].copy()

probs, preds = [], []
for d, row in sub.iterrows():
    df_feats = make_features(row, news[news.date_only == d.date()])
    p = model.predict_proba(df_feats)[0, 1]
    probs.append(p)
    preds.append(p >= tau_ui)

sub["p_high"], sub["pred"] = probs, preds

# ─── График: Candlestick + Shock markers ───────────────────────
fig = go.Figure(
    go.Candlestick(
        x=sub.index,
        open=sub.open, high=sub.high,
        low=sub.low, close=sub.close
    )
)
for d in sub.index:
    if shock_days.get(d.date(), False):
        # рисуем оранжевую точку на уровне high
        fig.add_scatter(
            x=[d], y=[sub.loc[d, "high"]],
            mode="markers",
            marker=dict(color="orange", size=10),
            name="Shock"
        )
fig.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, use_container_width=True)

# ─── Цветная полоса прогнозов ───────────────────────────────────
stripe = go.Figure(
    go.Bar(
        x=sub.index,
        y=[1]*len(sub),
        marker_color=["red" if p else "green" for p in sub.pred],
        hovertext=sub.p_high.round(2)
    )
)
stripe.update_layout(
    height=60,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis_visible=False,
    xaxis_visible=False
)
st.plotly_chart(stripe, use_container_width=True)

# ─── Детали выбранного дня ─────────────────────────────────────
sel_idx = st.slider("День", 0, len(sub)-1, len(sub)-1)
row = sub.iloc[sel_idx]

st.subheader(row.name.strftime("%d %b %Y"))
st.metric("Класс", "High" if row.pred else "Low")
st.metric("P(high)", f"{row.p_high:.2f}")

# ─── SHAP: топ-5 фичей ─────────────────────────────────────────
with st.expander("Топ-5 SHAP фичей"):
    import shap
    explainer = shap.TreeExplainer(model)
    # shap_values возвращает либо array, либо list из двух массивов
    sv = explainer.shap_values(
        make_features(row, news[news.date_only == row.name.date()])
    )
    vals = sv[0] if isinstance(sv, list) else sv
    vals = vals.flatten()    # shape may be (1,N) → (N,)
    top5 = np.argsort(np.abs(vals))[::-1][:5]
    for i in top5:
        st.write(f"{FEAT_NAMES[i]} : {float(vals[i]):+0.3f}")

# ─── Драйверы новостей ─────────────────────────────────────────
st.write("### Драйверы новостей")
titles = news.loc[news.date_only == row.name.date(), "title"].head(5).tolist()
if titles:
    st.write("<br>".join(f"• {t}" for t in titles), unsafe_allow_html=True)
else:
    st.info("Новостей не было")

# ─── Симуляция 3 стратегий + Buy&Hold ────────────────────────
with st.expander("⚡ Симуляция стратегий"):
    # 1) Buy & Hold (на ежедневную доходность ret1)
    df_sim = pd.DataFrame(index=candles.index)
    df_sim["ret1"] = candles["ret1"]

    # 2) Hold-Lowσ: если предсказано Low (pred=False), берем ret1, иначе 0
    df_sim["prob"] = sub["p_high"].reindex(df_sim.index).fillna(method="ffill")
    df_sim["pred"] = df_sim["prob"] >= tau_ui
    df_sim["hold_low"] = (1 + df_sim["ret1"].where(~df_sim["pred"], 0)).cumprod()

    # 3) Short-Vol: если pred=True, “коротим волатильность” (−ret1 − cost), иначе 1
    cost = 0.001  # комиссия/сужение спреда
    df_sim["short_vol"] = (1 - df_sim["ret1"].where(df_sim["pred"], 0) - cost * df_sim["pred"]).cumprod()

    # 4) Buy & Hold (baseline)
    df_sim["bh"] = (1 + df_sim["ret1"]).cumprod()

    # Рисуем equity-кривые
    st.line_chart(df_sim[["bh", "hold_low", "short_vol"]])

    # Выводим финальное значение
    final = df_sim.iloc[-1][["bh", "hold_low", "short_vol"]]
    st.write(
        final.rename({
            "bh": "Buy&Hold",
            "hold_low": "Hold-Lowσ",
            "short_vol": "Short-Vol"
        }).to_frame("Equity").style.format("{:.2f}")
    )

st.caption("Учебный проект — не инвестиционная рекомендация.")
