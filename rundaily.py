# ============================================================
# 🔁 RUNDaily — Update data, scalers and GRU/LSTM models daily
# Compatible with existing Streamlit app (tab4)
# ============================================================

import os
from datetime import date

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from pandas_datareader import data as web
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------
# ⚙️ Configuración general
# ------------------------------------------------------------
START_DATE = "2000-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")

BANKS = {
    "BBVA": "BBVA.MC",
    "Santander": "SAN.MC",
}

DATA_DIR = "data"
MODEL_DIR = "models/repository"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "EMA_20", "SMA_50", "Volatility_20",
    "Returns", "ECB_Rate", "Dividend_Flag", "World_Event",
]

WINDOW_SIZE = 60
BATCH_SIZE = 32
EPOCHS = 200
PATIENCE = 40
LR = 1e-3

# ------------------------------------------------------------
# 🧠 Device
# ------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("🧠 Using device:", device)


# ============================================================
# 🧩 Secuencias
# ============================================================
def create_sequences(data, target, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, :])
        y.append(target[i])
    return np.array(X), np.array(y)


# ============================================================
# 🧱 Modelos GRU y LSTM (mismas arquitecturas que la app)
# ============================================================
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# ============================================================
# 📈 Enriquecimiento de datos (igual filosofía que tu script)
# ============================================================

def fix_yahoo_df(df):
    """
    Normaliza cualquier DataFrame devuelto por yfinance a columnas:
    [Date, Open, High, Low, Close, Volume]
    manejando auto_adjust, acciones=True, MultiIndex y nombres raros.
    """

    # Caso 1: df vacío
    if df is None or len(df) == 0:
        raise ValueError("Yahoo Finance devolvió un DataFrame vacío.")

    df = df.copy()

    # Caso 2: MultiIndex tipo ('Open', ''), ('Close','')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Caso 3: resetear index si la fecha está ahí
    if df.index.name is not None:
        df = df.reset_index()

    # Normalizar nombres
    renames = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adjclose": "Adj Close",
        "adj close": "Adj Close",
        "volume": "Volume",
        "date": "Date",
        "datetime": "Date",
    }

    df.columns = [renames.get(c.lower(), c) for c in df.columns]

    # Muchos tickers vienen sin Close pero con "Adj Close"
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    # Si sigue sin close = error
    if "Close" not in df.columns:
        raise ValueError(f"No se encontró columna 'Close'. Columnas actuales: {list(df.columns)}")

    # Si falta volumen por acciones deshabilitadas → ponerlo a 0
    if "Volume" not in df.columns:
        df["Volume"] = 0

    # Asegurar orden
    keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]]

    # Fix fechas
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["Volatility_20"] = df["Close"].rolling(window=20).std()
    df["Returns"] = df["Close"].pct_change()
    return df


def add_ecb_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        ecb = web.DataReader(
            "FM.M.U2.EUR.4F.KR.MRR_FR.LEV", "fred", START_DATE, END_DATE
        )
        ecb = ecb.rename(
            columns={"FM.M.U2.EUR.4F.KR.MRR_FR.LEV": "ECB_Rate"}
        ).fillna(method="ffill")
        df = df.merge(ecb, how="left", left_index=True, right_index=True)
        df["ECB_Rate"] = df["ECB_Rate"].fillna(method="ffill")
    except Exception as e:
        print(f"[!] ECB rates not available: {e}")
        df["ECB_Rate"] = np.nan
    return df


def add_dividend_flag(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy().reset_index()
    try:
        dividends = yf.Ticker(ticker).dividends
        dividends = dividends.reset_index().rename(
            columns={"Date": "Date", "Dividends": "Dividend"}
        )
        dividends["Dividend_Flag"] = np.where(dividends["Dividend"] > 0, 1, 0)
        df = df.merge(dividends[["Date", "Dividend_Flag"]], on="Date", how="left")
        df["Dividend_Flag"] = df["Dividend_Flag"].fillna(0).astype(int)
    except Exception as e:
        print(f"[!] Dividends not available for {ticker}: {e}")
        df["Dividend_Flag"] = 0

    df.set_index("Date", inplace=True)
    return df


def add_world_events(df):
    df = df.copy()

    # --- 1) Asegurar que las columnas NO son MultiIndex ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns]

    # --- 2) Asegurar que Date es una columna simple ---
    if "Date" in df.index.names:
        df = df.reset_index()
    if "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    if "Date" not in df.columns:
        raise ValueError("ERROR: No existe columna 'Date' después de resetear")

    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    # --- 3) Tabla de eventos ---
    event_dates = [
        ("2001-09-11", "9/11 Attacks"),
        ("2008-09-15", "Lehman Collapse"),
        ("2010-05-01", "Eurozone Debt Crisis"),
        ("2020-03-11", "COVID Pandemic Declared"),
        ("2024-06-24", "Spain Blocks BBVA–Sabadell Merger"),
        ("2014-09-10", "Death of Emilio Botín"),
        ("2023-10-07", "Israel–Hamas Escalation")
    ]
    events = pd.DataFrame(event_dates, columns=["Date", "Event"])
    events["Date"] = pd.to_datetime(events["Date"]).dt.tz_localize(None)

    # --- 4) Merge seguro (ya no da errores) ---
    df = df.merge(events[["Date"]].assign(World_Event=1), on="Date", how="left")
    df["World_Event"] = df["World_Event"].fillna(0).astype(int)

    return df


def download_and_enrich(name, ticker):
    print(f"📥 Downloading & enriching data for {name} ({ticker})...")

    df = yf.Ticker(ticker).history(
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        actions=True
    )

    df = fix_yahoo_df(df)  # 🔥 REPARA SIEMPRE EL DF AQUÍ

    df = add_technical_indicators(df)
    df = add_ecb_rates(df)
    df = add_dividend_flag(df, ticker)
    df = add_world_events(df)

    df = df.fillna(method="ffill").fillna(0)

    df.to_csv(f"data/{name.lower()}_enriched.csv", index=False)
    print(f"✔ Saved enriched data: data/{name.lower()}_enriched.csv")

    return df

# ============================================================
# 📏 Escalado y preparación de datos para modelos
# ============================================================
def prepare_training_data(df: pd.DataFrame, name: str):
    """
    - Ajusta scaler MinMax a FEATURES en unidades reales.
    - Crea X_seq, y_seq ESCALADOS para LSTM/GRU.
    - Devuelve scaler y también un pequeño 'test tail' para métricas.
    """
    df = df.copy().sort_values("Date")

    # Nos quedamos sólo con las columnas necesarias
    train_df = df[["Date"] + FEATURES].copy()

    # Fit scaler en FEATURES (en euros, tal cual)
    scaler = MinMaxScaler()
    scaler.fit(train_df[FEATURES])

    # Guardar scaler con el nombre esperado por la app
    scaler_path = os.path.join(DATA_DIR, f"{name.lower()}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to {scaler_path}")

    # Transformar para el modelo
    scaled_features = scaler.transform(train_df[FEATURES])
    close_idx = FEATURES.index("Close")
    X_scaled = scaled_features
    y_scaled = scaled_features[:, close_idx]

    # Crear secuencias para toda la historia (train = full history)
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, WINDOW_SIZE)

    # Guardamos también fechas para la "cola" de métricas (últimos n puntos)
    d_all = train_df["Date"].values
    d_tail = d_all[-len(y_seq):]  # alineado con y_seq
    return X_seq, y_seq, scaler, d_tail


# ============================================================
# 🏋️‍♂️ Entrenamiento (GRU y LSTM) usando TODA la historia
# ============================================================
def train_model(model, X_seq, y_seq, name, model_type):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    dataset = TensorDataset(
        torch.tensor(X_seq, dtype=torch.float32),
        torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_loss = np.inf
    best_state = None
    no_improve = 0
    losses = []

    print(f"\n🚀 Training {model_type} for {name} on full history ({len(X_seq)} samples)...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        epoch_loss = float(np.mean(epoch_losses))
        losses.append(epoch_loss)

        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(f"{model_type} {name} | Epoch {epoch:03d} | Loss: {epoch_loss:.6f} | no_improve={no_improve}")

        if no_improve >= PATIENCE:
            print(f"⏹️ Early stopping {model_type} {name} at epoch {epoch} (best_loss={best_loss:.6f})")
            break

    model.load_state_dict(best_state)
    return model, losses


def evaluate_tail(model, scaler, X_seq, y_seq, d_tail, name, model_type, tail_size=200):
    """
    Evalúa el modelo en los últimos `tail_size` puntos de la serie.
    Devuelve métricas en euros (MAE, RMSE, R2) — solo para logging.
    """
    model.eval()
    tail_size = min(tail_size, len(X_seq))
    X_eval = X_seq[-tail_size:]
    y_eval = y_seq[-tail_size:]
    dates_eval = d_tail[-tail_size:]

    with torch.no_grad():
        preds_scaled = model(
            torch.tensor(X_eval, dtype=torch.float32).to(device)
        ).cpu().numpy().ravel()

    # Desescalar usando el scaler conjunto
    feature_names = scaler.feature_names_in_
    close_idx = list(feature_names).index("Close")

    y_pred_full = np.zeros((len(preds_scaled), len(feature_names)))
    y_true_full = np.zeros_like(y_pred_full)
    y_pred_full[:, close_idx] = preds_scaled
    y_true_full[:, close_idx] = y_eval

    y_pred_eur = scaler.inverse_transform(y_pred_full)[:, close_idx]
    y_true_eur = scaler.inverse_transform(y_true_full)[:, close_idx]

    mae = mean_absolute_error(y_true_eur, y_pred_eur)
    rmse = np.sqrt(mean_squared_error(y_true_eur, y_pred_eur))
    r2 = r2_score(y_true_eur, y_pred_eur)

    print(f"\n📊 {name} {model_type} tail metrics (last {tail_size} days):")
    print(f"   MAE  = €{mae:.4f}")
    print(f"   RMSE = €{rmse:.4f}")
    print(f"   R²   = {r2:.4f}")

    # Si quieres, aquí podrías guardar un CSV con las predicciones tail
    metrics_df = pd.DataFrame({
        "Date": dates_eval,
        "True_Close": y_true_eur,
        "Pred_Close": y_pred_eur,
    })
    metrics_path = os.path.join(
        MODEL_DIR, f"{name.lower()}_{model_type.lower()}_tail_eval.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)
    print(f"   ↳ Tail eval CSV saved to {metrics_path}")


# ============================================================
# 🔁 Pipeline completo por banco
# ============================================================
def run_daily_for_bank(name: str, ticker: str):
    # 1. Descargar + enriquecer
    df_enriched = download_and_enrich(name, ticker)

    # 2. Preparar datos + scaler
    X_seq, y_seq, scaler, d_tail = prepare_training_data(df_enriched, name)

    input_size = X_seq.shape[2]

    # 3. Entrenar GRU
    gru = GRUModel(input_size=input_size, hidden_size=64, num_layers=1, dropout=0.1).to(device)
    gru, gru_losses = train_model(gru, X_seq, y_seq, name, "GRU")

    gru_path = os.path.join(MODEL_DIR, f"{name.lower()}_gru_best.pth")
    torch.save(gru.state_dict(), gru_path)
    print(f"✅ GRU model saved to {gru_path}")

    # 4. Entrenar LSTM
    lstm = LSTMModel(input_size=input_size, hidden_size=64, num_layers=1, dropout=0.1).to(device)
    lstm, lstm_losses = train_model(lstm, X_seq, y_seq, name, "LSTM")

    lstm_path = os.path.join(MODEL_DIR, f"{name.lower()}_lstm_best.pth")
    torch.save(lstm.state_dict(), lstm_path)
    print(f"✅ LSTM model saved to {lstm_path}")

    # 5. Métricas rápidas en la cola (últimos días)
    evaluate_tail(gru, scaler, X_seq, y_seq, d_tail, name, "GRU")
    evaluate_tail(lstm, scaler, X_seq, y_seq, d_tail, name, "LSTM")

    print(f"\n🎯 Finished daily update for {name}\n")


# ============================================================
# 🚀 MAIN: run daily for all banks
# ============================================================
def run_daily():
    for name, ticker in BANKS.items():
        run_daily_for_bank(name, ticker)


if __name__ == "__main__":
    run_daily()