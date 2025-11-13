# =========================================================
# 💹 Finance Dashboard — Retro 98.css + Multilingual
# =========================================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch, torch.nn as nn, joblib

# ---------------- Page setup ----------------
st.set_page_config(page_title="Finance Dashboard", page_icon="💹", layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load it
local_css("style.css")


# ---------------- Language toggle ----------------
LANG = st.sidebar.radio("🌐 Language / Idioma / Llengua", ["en", "es", "ca"], horizontal=True)
T = {
    "title": {"en":"Finance Dashboard — Banks, Indexes & Indicators",
              "es":"Panel Financiero — Bancos, Índices e Indicadores",
              "ca":"Quadre Financer — Bancs, Índexs i Indicadors"},
    "banks":{"en":"Banks","es":"Bancos","ca":"Bancs"},
    "indexes":{"en":"Indexes","es":"Índices","ca":"Índexs"},
    "macros":{"en":"Macro Indicators","es":"Indicadores macro","ca":"Indicadors macro"},
    "models":{"en":"Model Predictions","es":"Predicciones del modelo","ca":"Prediccions del model"},
    "settings":{"en":"Settings","es":"Ajustes","ca":"Ajustos"},
    "export":{"en":"Export CSV","es":"Exportar CSV","ca":"Exportar CSV"},
    "nodata":{"en":"No data available.","es":"No hay datos disponibles.","ca":"No hi ha dades disponibles."}
}
st.markdown(f"""
<div class="window"><div class="title-bar"><div class="title-bar-text">{T['title'][LANG]}</div></div>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.markdown(f"<div class='window'><div class='title-bar'><div class='title-bar-text'>{T['settings'][LANG]}</div></div><div class='window-body'>", unsafe_allow_html=True)

BANKS = {"BBVA": "BBVA.MC","Santander": "SAN.MC"}
INDEXES = {"IBEX 35": "^IBEX","S&P 500": "^GSPC","NASDAQ": "^IXIC","EURO STOXX 50": "^STOXX50E","DAX": "^GDAXI"}
INDICATORS = {
    "US CPI (FRED)": "CPIAUCSL","EUR/USD": "EURUSD=X","Crude Oil (WTI)": "CL=F","Gold": "GC=F",
    "Silver": "SI=F","Natural Gas": "NG=F","US 10Y Yield": "^TNX","VIX Volatility": "^VIX",
    "Brent Oil": "BZ=F","Bitcoin": "BTC-USD","USD/JPY": "JPY=X","USD/CNY": "CNY=X","US Dollar Index": "DX-Y.NYB"
}

banks_selected = st.sidebar.multiselect(T["banks"][LANG], list(BANKS.keys()), default=list(BANKS.keys()))
indexes_selected = st.sidebar.multiselect(T["indexes"][LANG], list(INDEXES.keys()), default=["IBEX 35","S&P 500"])
indicators_selected = st.sidebar.multiselect(T["macros"][LANG], list(INDICATORS.keys()))
period = st.sidebar.selectbox("Price period", ["6mo","1y","2y","5y","10y","max"], index=2)
interval = st.sidebar.selectbox("Candle interval", ["1d","1wk","1mo"], index=0)
show_sma = st.sidebar.checkbox("SMA 20 / 50", value=True)
show_ema = st.sidebar.checkbox("EMA 12 / 26", value=False)
show_bb  = st.sidebar.checkbox("Bollinger Bands (20, 2σ)", value=True)
show_vol = st.sidebar.checkbox("Show Volume", value=True)
st.sidebar.markdown("</div></div>", unsafe_allow_html=True)

# =========================================================
# 📈 DATA HELPERS
# =========================================================
def _as_utc_naive(dt_index):
    if getattr(dt_index,"tz",None) is not None:
        dt_index = dt_index.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(dt_index).sort_values()

def _ensure_ohlcv(df):
    if df.empty: return df
    df = df.copy()
    if isinstance(df.columns,pd.MultiIndex):
        df.columns=['_'.join([str(x) for x in c if x]) for c in df.columns]
    if "Date" in df.columns: df=df.set_index(pd.to_datetime(df["Date"])).drop("Date",axis=1)
    elif "Datetime" in df.columns: df=df.set_index(pd.to_datetime(df["Datetime"])).drop("Datetime",axis=1)
    else: df.index=pd.to_datetime(df.index,errors="coerce")
    df.index=_as_utc_naive(df.index)
    df.columns=[str(c).lower() for c in df.columns]
    rename_map={c:("Open" if "open" in c else "High" if "high" in c else "Low" if "low" in c
                  else "Close" if "close" in c and "adj" not in c else "Adj Close" if "adj" in c else "Volume")
                for c in df.columns if any(k in c for k in ["open","high","low","close","adj","volume"])}
    df=df.rename(columns=rename_map)
    if "Close" not in df and "Adj Close" in df: df["Close"]=df["Adj Close"]
    keep=[c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    return df[keep].dropna(subset=["Close"])

@st.cache_data(show_spinner=False)
def fetch(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, threads=True)
    if isinstance(df,pd.DataFrame) and not df.empty:
        return _ensure_ohlcv(df.reset_index(drop=False))
    return pd.DataFrame()

def add_indicators(df):
    df=df.copy()
    if len(df)<25: return df
    if show_sma:
        df["SMA_20"]=df["Close"].rolling(20).mean()
        df["SMA_50"]=df["Close"].rolling(50).mean()
    if show_ema:
        df["EMA_12"]=df["Close"].ewm(span=12).mean()
        df["EMA_26"]=df["Close"].ewm(span=26).mean()
    if show_bb:
        mid=df["Close"].rolling(20).mean(); std=df["Close"].rolling(20).std()
        df["BB_Up"]=mid+2*std; df["BB_Dn"]=mid-2*std
    return df
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def make_candlestick_figure(df: pd.DataFrame, title: str) -> go.Figure:
    """Candlestick with SMA/EMA/Bollinger overlays + optional Volume."""
    has_vol = show_vol and "Volume" in df.columns and df["Volume"].notna().any()

    fig = make_subplots(
        rows=2 if has_vol else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.78, 0.22] if has_vol else [1.0],
        specs=[[{"type": "xy"}]] if not has_vol else [[{"type": "xy"}], [{"type": "bar"}]],
    )

    # Candles
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df.get("Open"),
            high=df.get("High"),
            low=df.get("Low"),
            close=df.get("Close"),
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
        ),
        row=1, col=1
    )

    # Bollinger (needs BB_Up/BB_Dn)
    if "BB_Up" in df.columns and "BB_Dn" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Up"], line=dict(width=1, color="#90caf9"),
            name="BB Upper", hoverinfo="skip"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Dn"], line=dict(width=1, color="#90caf9"),
            fill="tonexty", fillcolor="rgba(144,202,249,0.12)",
            name="BB Lower", hoverinfo="skip"
        ), row=1, col=1)

    # SMA
    if "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_20"], line=dict(width=1.8, color="#ffd54f"),
            name="SMA 20"
        ), row=1, col=1)
    if "SMA_50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_50"], line=dict(width=1.8, color="#ffb300"),
            name="SMA 50"
        ), row=1, col=1)

    # EMA
    if "EMA_12" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA_12"], line=dict(width=1.2, dash="dot", color="#66bb6a"),
            name="EMA 12"
        ), row=1, col=1)
    if "EMA_26" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA_26"], line=dict(width=1.2, dash="dot", color="#26a69a"),
            name="EMA 26"
        ), row=1, col=1)

    # Volume (2nd row)
    if has_vol:
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color="rgba(158, 158, 158, 0.6)",
                showlegend=False,
            ),
            row=2, col=1
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=520 if has_vol else 480,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=16)),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.1)", tickformat=",.2f"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.03, x=1, xanchor="right")
    )
    return fig

def plot_line(df,name):
    fig=go.Figure(go.Scatter(x=df.index,y=df["Close"],mode="lines",name=name))
    fig.update_layout(template="plotly_dark",title=name,height=400)
    return fig

import requests

NEWS_API_KEY = "3cad6bcebf774d658dd93d70ee6f3387"
FINNHUB_KEY = "d4aau9hr01qnehvtoopgd4aau9hr01qnehvtooq0"

import requests
from datetime import datetime, timedelta
import random


def _norm_date(dt):
    # handles ts ints, iso strings, datetimes, None
    if dt is None:
        return ""
    if isinstance(dt, (int, float)):
        try:
            return datetime.fromtimestamp(dt).strftime("%Y-%m-%d")
        except Exception:
            return ""
    if hasattr(dt, "strftime"):
        return dt.strftime("%Y-%m-%d")
    # assume string
    return str(dt)[:10]

def get_bank_news(symbol, bank_name=None, max_articles=5, lang="es"):
    """
    Unified financial news fetcher using both Finnhub + NewsAPI.
    - Pulls from both APIs.
    - Filters only relevant economic/banking headlines.
    - Works with tickers (BBVA.MC, SAN.MC, ^IBEX) and names.
    """

    # Normalize bank names and related keywords
    bank_name = bank_name or symbol
    bank_keywords = {
        "BBVA": ["BBVA", "Banco Bilbao Vizcaya Argentaria"],
        "Santander": ["Santander", "Banco Santander"],
        "Sabadell": ["Sabadell", "Banco Sabadell"],
        "IBEX": ["IBEX 35", "Bolsa española", "mercados españoles"],
        "S&P": ["S&P 500", "Wall Street", "mercado americano"],
        "NASDAQ": ["NASDAQ", "tecnológicas", "mercado USA"]
    }

    related_terms = bank_keywords.get(bank_name, [bank_name])
    search_terms = " OR ".join(related_terms)

    # -------------------------------------------------------------------
    # 1️⃣ Finnhub API (company news)
    # -------------------------------------------------------------------
    finnhub_articles = []
    try:
        url_finnhub = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={symbol}&from={(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')}"
            f"&to={datetime.now().strftime('%Y-%m-%d')}&token={FINNHUB_KEY}"
        )
        r = requests.get(url_finnhub, timeout=5)
        data = r.json()
        if isinstance(data, list):
            for a in data:
                txt = (a.get("headline", "") + " " + a.get("summary", "")).lower()
                if any(k.lower() in txt for k in ["bank", "banco", "finanzas", "mercado", "economía", "bolsa"]):
                    finnhub_articles.append({
                        "source": "Finnhub",
                        "title": a.get("headline", ""),
                        "url": a.get("url", ""),
                        "publishedAt": datetime.fromtimestamp(a.get("datetime", 0)).strftime("%Y-%m-%d"),
                        "description": a.get("summary", "")
                    })
    except Exception as e:
        print("⚠️ Finnhub error:", e)

    # -------------------------------------------------------------------
    # 2️⃣ NewsAPI (broader search)
    # -------------------------------------------------------------------
    newsapi_articles = []
    try:
        q = f"({search_terms}) AND (banca OR finanzas OR economía OR bolsa OR mercados)"
        url_news = (
            f"https://newsapi.org/v2/everything?"
            f"q={q}&language={lang}&sortBy=publishedAt&pageSize={max_articles*2}&apiKey={NEWS_API_KEY}"
        )
        r = requests.get(url_news, timeout=5)
        data = r.json()
        if "articles" in data:
            for art in data["articles"]:
                title = art.get("title", "")
                desc = art.get("description", "") or ""
                text = (title + " " + desc).lower()
                if any(k in text for k in ["bbva", "santander", "sabadell", "banco", "finanz", "mercado", "índice", "bolsa"]):
                    newsapi_articles.append({
                        "source": art.get("source", {}).get("name", "NewsAPI"),
                        "title": title,
                        "url": art.get("url", ""),
                        "publishedAt": art.get("publishedAt", "")[:10],
                        "description": desc
                    })
    except Exception as e:
        print("⚠️ NewsAPI error:", e)

    # -------------------------------------------------------------------
    # 3️⃣ Combine, deduplicate, and sort
    # -------------------------------------------------------------------
    all_articles = {a["url"]: a for a in (finnhub_articles + newsapi_articles)}.values()
    all_articles = sorted(all_articles, key=lambda x: x["publishedAt"], reverse=True)
    all_articles = list(all_articles)[:max_articles]

    # -------------------------------------------------------------------
    # 4️⃣ Return or fallback
    # -------------------------------------------------------------------
    if not all_articles:
        all_articles = [{
            "source": "System",
            "title": f"No se encontraron noticias recientes sobre {bank_name}.",
            "url": "",
            "publishedAt": datetime.now().strftime("%Y-%m-%d"),
            "description": "Intenta con otro banco o índice."
        }]
    return all_articles

# =========================================================
# 📊 TABS STRUCTURE (retro wrappers)
# =========================================================
tab1,tab2,tab3,tab4=st.tabs([
    T["banks"][LANG],T["indexes"][LANG],T["macros"][LANG],T["models"][LANG]
])

with tab1:
    st.markdown("<div class='window'><div class='title-bar'><div class='title-bar-text'>Banks</div></div><div class='window-body'>",unsafe_allow_html=True)
    for bank in banks_selected:
        df=fetch(BANKS[bank],period,interval)
        if df.empty or len(df)<2:
            st.warning(f"{bank}: {T['nodata'][LANG]}"); continue
        df=add_indicators(df)
        csv=df.to_csv().encode("utf-8")
        st.download_button(T["export"][LANG],csv,file_name=f"{bank}_{period}.csv",mime="text/csv",key=f"csv_{bank}")
        st.plotly_chart(make_candlestick_figure(df,f"{bank} — {BANKS[bank]}"),use_container_width=True)
        # News Section
        news_items = get_bank_news(BANKS[bank], bank_name=bank, max_articles=5)

        for art in news_items:
            source = art.get("source") or art.get("source_name") or ""
            published = art.get("publishedAt") or art.get("published_at") or ""
            if isinstance(published, str):
                published = published[:10]
            else:
                try:
                    # if it’s a datetime
                    published = published.strftime("%Y-%m-%d")
                except Exception:
                    published = ""

            title = art.get("title", "")
            url = art.get("url", "")
            desc = art.get("description") or ""

            st.markdown(f"""
            <div class='news-item'>
                <a href="{url}" target="_blank"><b>{title}</b></a><br>
                <small>{source} — {published}</small><br>
                <p style='color:#bbb;font-size:13px'>{desc}</p>
            </div><hr>
            """, unsafe_allow_html=True)
    st.markdown("</div></div>",unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='window'><div class='title-bar'><div class='title-bar-text'>Indexes</div></div><div class='window-body'>",unsafe_allow_html=True)
    for name in indexes_selected:
        df=fetch(INDEXES[name],period,interval)
        if df.empty: continue
        csv=df.to_csv().encode("utf-8")
        st.download_button(T["export"][LANG],csv,file_name=f"{name}_{period}.csv",mime="text/csv",key=f"csv_{name}")
        st.plotly_chart(plot_line(df,f"{name} Performance"),use_container_width=True)
    st.markdown("</div></div>",unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='window'><div class='title-bar'><div class='title-bar-text'>Macro Indicators</div></div><div class='window-body'>",unsafe_allow_html=True)
    for name in indicators_selected:
        df=fetch(INDICATORS[name],"max","1mo")
        if df.empty or len(df)<2:
            st.info(f"{name}: {T['nodata'][LANG]}"); continue
        csv=df.to_csv().encode("utf-8")
        st.download_button(T["export"][LANG],csv,file_name=f"{name}.csv",mime="text/csv",key=f"csv_{name}")
        st.plotly_chart(plot_line(df,name),use_container_width=True)
    st.markdown("</div></div>",unsafe_allow_html=True)



# =========================================================
# 🤖 TAB 4 — MODEL PREDICTIONS (Simple Scaling Fix)
# =========================================================
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
from datetime import timedelta
import streamlit as st

# =========================================================
# 🧠 MODEL DEFINITIONS
# =========================================================
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# =========================================================
# ⚙️ HELPERS
# =========================================================
def load_torch_model(path, model_type, input_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUModel(input_size) if model_type == "gru" else LSTMModel(input_size)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, device


def align_features_to_scaler(df, scaler):
    """Ensure dataframe has all expected scaler columns in correct order."""
    expected = list(scaler.feature_names_in_)
    df = df.copy()
    for col in expected:
        if col not in df.columns:
            if col == "Adj Close":
                df[col] = df["Close"]
            elif col in ["Dividends", "Stock Splits"]:
                df[col] = 0.0
            else:
                df[col] = 0.0
    return df.reindex(columns=expected, fill_value=0)


def predict_future(model, device, df, scaler, features, window, forecast_horizon):
    """Multi-day autoregressive forecast with proper unscaling."""
    df_aligned = align_features_to_scaler(df, scaler)
    df_scaled_full = pd.DataFrame(scaler.transform(df_aligned), columns=scaler.feature_names_in_)
    df_scaled = df_scaled_full[features].copy()

    preds_eur = []
    current_df = df_scaled.copy()
    close_idx = list(scaler.feature_names_in_).index("Close")

    for _ in range(forecast_horizon):
        X_seq = current_df.iloc[-window:].values
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            y_scaled = model(X_tensor).cpu().numpy().ravel()[0]

        # ✅ Correct inverse transform: preserve other feature context
        last_scaled_row = df_scaled_full.iloc[-1].copy()
        last_scaled_row["Close"] = y_scaled
        inv = np.expand_dims(last_scaled_row.values, axis=0)
        y_eur = scaler.inverse_transform(inv)[0, close_idx]

        # safety: if unscaled value is implausible, use last Close + scaled delta
        last_close = df["Close"].iloc[-1] if len(preds_eur) == 0 else preds_eur[-1]
        if not (0.5 * last_close <= y_eur <= 2.0 * last_close):
            y_eur = last_close * (1 + (y_scaled - 0.5))  # approximate delta

        preds_eur.append(y_eur)

        # Append new scaled row for autoregressive input
        next_row = current_df.iloc[-1].copy()
        next_row["Close"] = y_scaled
        current_df = pd.concat([current_df, next_row.to_frame().T], ignore_index=True)

    # Timeline
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon)

    preds_series = pd.Series(data=preds_eur, index=future_dates)

    return preds_series


# =========================================================
# 📊 MODEL PREDICTION TAB
# =========================================================
with st.tabs(["Model Predictions"])[0]:
    st.subheader("AI Stock Forecasts — GRU vs LSTM")

    forecast_horizon = st.sidebar.slider(
        "Days to Predict Ahead", min_value=1, max_value=30, value=5, key="main_forecast_slider"
    )

    BANK_MODELS = {
        "BBVA": {
            "ticker": "BBVA.MC",
            "scaler": "data/bbva_scaler.pkl",
            "gru": "models/repository/bbva_gru_best.pth",
            "lstm": "models/repository/bbva_lstm_best.pth",
        },
        "Santander": {
            "ticker": "SAN.MC",
            "scaler": "data/san_scaler.pkl",
            "gru": "models/repository/san_gru_best.pth",
            "lstm": "models/repository/san_lstm_best.pth",
        },
    }

    features = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_20", "EMA_20", "SMA_50", "Volatility_20",
        "Returns", "ECB_Rate", "Dividend_Flag", "World_Event"
    ]
    window = 60

    for bank, meta in BANK_MODELS.items():
        st.markdown(f"## {bank}")

        df = fetch(meta["ticker"], "30y", "1d")
        if df.empty:
            st.warning(f"No data available for {bank}")
            continue

        df["Returns"] = df["Close"].pct_change().fillna(0)
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["Volatility_20"] = df["Returns"].rolling(20).std().fillna(0)
        df["ECB_Rate"] = 0.0
        df["Dividend_Fg"] = 0
        df["World_Event"] = 0
        df = df.dropna()
        # --- Ensure all model features exist (Yahoo data may miss some)
        for col in ["ECB_Rate", "Dividend_Flag", "World_Event"]:
            if col not in df.columns:
                df[col] = 0.0
        scaler = joblib.load(meta["scaler"])
        gru_model, dev = load_torch_model(meta["gru"], "gru", len(features))
        lstm_model, _ = load_torch_model(meta["lstm"], "lstm", len(features))
        scaler = joblib.load(meta["scaler"])
        scaler.fit(df[features])  # 🔧 realign scaler to Yahoo data range

        gru_pred = predict_future(gru_model, dev, df, scaler, features, window, forecast_horizon)
        lstm_pred = predict_future(lstm_model, dev, df, scaler, features, window, forecast_horizon)

        last_close = df["Close"].iloc[-1]
        col1, col2 = st.columns(2)
        col1.metric(f"GRU {forecast_horizon}-Day Forecast", f"{gru_pred.iloc[-1]:.2f} €", f"{gru_pred.iloc[-1]-last_close:+.2f} €")
        col2.metric(f"LSTM {forecast_horizon}-Day Forecast", f"{lstm_pred.iloc[-1]:.2f} €", f"{lstm_pred.iloc[-1]-last_close:+.2f} €")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Actual Close (€)", line=dict(color="#00A3E0", width=2)))
        fig.add_trace(go.Scatter(x=gru_pred.index, y=gru_pred.values, mode="lines+markers", name="GRU Forecast", line=dict(color="#FFD700", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=lstm_pred.index, y=lstm_pred.values, mode="lines+markers", name="LSTM Forecast", line=dict(color="#FF6B6B", width=2, dash="dot")))

        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#0C1E3C",
            paper_bgcolor="#0C1E3C",
            font=dict(color="#A9C0FF"),
            xaxis_title="Date",
            yaxis_title="Close (€)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{bank}_forecast_chart")