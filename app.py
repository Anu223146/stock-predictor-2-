"""
Ensemble Stock Direction Predictor (corrected version)

Fixes compared with the previous app:
  1. Latest trading day is no longer dropped, so the signal uses the true last row.
  2. All features are stationary (MACD / Volatility are now relative to price).
  3. 1-day gap between train and test, scaler fitted on training data only.
  4. Honest evaluation: baseline, 95% CI, binomial test vs baseline, AUC,
     confusion matrix and walk-forward (expanding window) out-of-sample accuracy.
  5. Simple backtest with transaction cost vs buy-and-hold.
  6. No silent fallback to AAPL; the user sees a clear error instead.
  7. Final signal model is retrained on ALL labelled data.
"""

import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

# ─── Page config & styles ─────────────────────────────────────────────────────
st.set_page_config(page_title="Ensemble Predictor", page_icon="▲", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');
header[data-testid="stHeader"], #MainMenu, .stDeployButton, footer { display: none !important; }
html, body, [data-testid="stAppViewContainer"] {
  background-color: #080c14 !important; color: #d4d8e0 !important; font-family: 'DM Sans', sans-serif !important;
}
.block-container { padding: 2.5rem 2.5rem 3rem !important; max-width: 1400px !important; }
[data-testid="stSidebar"] { background-color: #0c1120 !important; border-right: 1px solid #1c2236 !important; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
  color: #8892a4 !important; font-size: 0.78rem !important; letter-spacing: 0.06em !important; text-transform: uppercase !important;
}
.page-header { display:flex; justify-content:space-between; margin-bottom:2rem; padding-bottom:1.5rem; border-bottom:1px solid #1c2236; }
.header-eyebrow { font-family:'DM Mono',monospace; font-size:0.68rem; letter-spacing:0.18em; color:#e8c97d; text-transform:uppercase; margin-bottom:0.5rem; }
.header-title { font-family:'Syne',sans-serif; font-size:2.1rem; font-weight:800; color:#eef0f4; line-height:1.1; }
.header-sub { font-size:0.85rem; color:#5a6478; margin-top:0.5rem; font-weight:300; }
.grid4 { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:2rem; }
.card { background:#0e1525; border:1px solid #1c2236; border-radius:10px; padding:1.2rem 1.4rem; }
.card-label { font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:0.14em; text-transform:uppercase; color:#4a5568; margin-bottom:0.5rem; }
.card-value { font-family:'DM Mono',monospace; font-size:1.5rem; font-weight:500; color:#eef0f4; line-height:1; }
.card-value.gold { color:#e8c97d; }
.card-sub { font-family:'DM Mono',monospace; font-size:0.74rem; margin-top:0.4rem; color:#3d4d66; }
.pos { color:#34d399 !important; } .neg { color:#f87171 !important; }
.signal-wrap { display:flex; justify-content:center; padding:2rem 1rem; }
.signal-card { background:#0e1525; border:1px solid #1c2236; border-radius:16px; padding:2.5rem 3rem; text-align:center; width:100%; max-width:520px; }
.signal-card.bullish { border-color:rgba(52,211,153,0.3); } .signal-card.bearish { border-color:rgba(248,113,113,0.3); }
.signal-card.neutral { border-color:rgba(232,201,125,0.3); }
.signal-type { font-family:'DM Mono',monospace; font-size:0.7rem; letter-spacing:0.2em; text-transform:uppercase; color:#4a5568; margin-bottom:0.4rem; }
.signal-heading { font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; margin-bottom:0.5rem; }
.signal-heading.bullish { color:#34d399; } .signal-heading.bearish { color:#f87171; } .signal-heading.neutral { color:#e8c97d; }
.signal-desc { font-size:0.82rem; color:#6b7689; font-weight:300; }
.section-label { font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:0.2em; text-transform:uppercase; color:#3d4d66; margin-bottom:0.75rem; }
.divider { height:1px; background:linear-gradient(90deg,transparent,#1c2236 20%,#1c2236 80%,transparent); margin:2rem 0; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { border-bottom:1px solid #1c2236 !important; margin-bottom:2rem !important; }
[data-testid="stTabs"] [data-baseweb="tab"] { background:transparent !important; font-family:'DM Mono',monospace !important; font-size:0.72rem !important; letter-spacing:0.12em !important; text-transform:uppercase !important; color:#3d4d66 !important; }
[data-testid="stTabs"] [aria-selected="true"] { color:#e8c97d !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color:#e8c97d !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("### Ensemble Predictor")
ticker = st.sidebar.text_input("Ticker Symbol", "AAPL", help="Examples: AAPL, TSLA, RELIANCE.NS").strip()
days_history = st.sidebar.slider("Historical Range (Days)", 365, 3650, 1825)
test_frac = st.sidebar.slider("Test share (latest data)", 0.15, 0.40, 0.20, 0.05)
cost_bp = st.sidebar.slider("Backtest cost per trade (bp)", 0, 30, 10)
st.sidebar.markdown("---")
st.sidebar.caption("DSN2099 · Exhibition II")

INDIAN_SUFFIXES = (".NS", ".BO")
FEATURES = [
    "Daily_Return", "Return_Lag1", "Return_Lag2", "Return_Lag3",
    "Close_to_5MA", "Close_to_10MA", "Close_to_20MA",
    "Price_Range_Pct", "Volatility", "RSI",
    "MACD_Pct", "Signal_Pct", "MACD_Hist_Pct",
    "BB_PctB", "Stoch_K", "Stoch_D", "Williams_R",
    "ATR_Pct", "ROC", "Volume_Change",
]


# ─── Data sources ─────────────────────────────────────────────────────────────
def fetch_stooq(tk, days):
    if tk.upper().endswith(INDIAN_SUFFIXES):
        raise ValueError("Ticker not supported on Stooq")
    t = tk.lower()
    sym = t if "." in t else f"{t}.us"
    resp = requests.get(f"https://stooq.com/q/d/l/?s={sym}&i=d", timeout=10,
                        headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty or "Date" not in df.columns:
        raise ValueError("Stooq returned no data")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df = df[df.index >= datetime.now() - timedelta(days=days)]
    if df.empty:
        raise ValueError("Stooq returned no rows in range")
    return df


def fetch_yfinance(tk, days):
    df = yf.Ticker(tk).history(start=datetime.now() - timedelta(days=days), auto_adjust=True)
    if df is None or df.empty:
        raise ValueError("Yahoo Finance returned no data")
    df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def build_features(df):
    """All features use only information available at the close of day t."""
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all").copy()
    c = df["Close"]

    df["5MA"] = c.rolling(5).mean()
    df["10MA"] = c.rolling(10).mean()
    df["20MA"] = c.rolling(20).mean()

    df["Daily_Return"] = c.pct_change()
    for k in (1, 2, 3):
        df[f"Return_Lag{k}"] = df["Daily_Return"].shift(k)
    df["Volatility"] = df["Daily_Return"].rolling(5).std()          # return units (stationary)
    df["Price_Range_Pct"] = (df["High"] - df["Low"]) / c

    for n in (5, 10, 20):
        df[f"Close_to_{n}MA"] = c / df[f"{n}MA"] - 1

    # RSI (14)
    d = c.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = np.where(loss == 0, 100.0, 100 - 100 / (1 + rs))

    # MACD, expressed as % of price so it is comparable across price regimes
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    df["MACD"], df["Signal_Line"] = macd, sig
    df["MACD_Pct"] = macd / c
    df["Signal_Pct"] = sig / c
    df["MACD_Hist_Pct"] = (macd - sig) / c

    # Bollinger Bands
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    df["BB_Upper"], df["BB_Lower"] = ma20 + 2 * sd20, ma20 - 2 * sd20
    df["BB_PctB"] = (c - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)

    # Stochastic / Williams %R
    lo14, hi14 = df["Low"].rolling(14).min(), df["High"].rolling(14).max()
    rng = (hi14 - lo14).replace(0, np.nan)
    df["Stoch_K"] = 100 * (c - lo14) / rng
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
    df["Williams_R"] = -100 * (hi14 - c) / rng

    # ATR %
    pc = c.shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    df["ATR_Pct"] = tr.rolling(14).mean() / c

    df["ROC"] = c.pct_change(10) * 100
    df["Volume_Change"] = df["Volume"].pct_change().clip(-5, 5)

    # Target: direction of NEXT close vs today's close. Last row has NO label (NaN),
    # but it is KEPT so the live signal uses the most recent trading day.
    df["Next_Close_Return"] = c.shift(-1) / c - 1
    df["Target"] = np.where(df["Next_Close_Return"] > 0, 1.0, 0.0)
    df.loc[df["Next_Close_Return"].isna(), "Target"] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=FEATURES)


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tk, days):
    symbol = "₹" if tk.upper().endswith(INDIAN_SUFFIXES) else "$"
    df, source = None, None
    try:
        df, source = fetch_stooq(tk, days), "Stooq"
    except Exception:
        pass
    if df is None or len(df) < 30:
        try:
            df, source = fetch_yfinance(tk, days), "Yahoo Finance"
        except Exception:
            df = None
    if df is None or len(df) < 30:
        return None, symbol, None
    return build_features(df), symbol, source


# ─── Header & data load ───────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="page-header"><div>
<div class="header-eyebrow">Multi-Model ML · Market Analysis</div>
<div class="header-title">{ticker.upper()}&nbsp;&nbsp;Trend Predictor</div>
<div class="header-sub">Ensemble of Random Forest, Gradient Boosting, AdaBoost &amp; Logistic Regression</div>
</div></div>
""",
    unsafe_allow_html=True,
)

with st.spinner("Loading market data…"):
    data, symbol, data_source = load_data(ticker, days_history)

if data is None or data.empty:
    st.error(f"Couldn't load data for '{ticker}' from Stooq or Yahoo Finance. Check the symbol and try again.")
    st.stop()

labeled = data[data["Target"].notna()]
latest = data.iloc[[-1]]  # true most recent trading day (unlabelled)

if len(labeled) < 200:
    st.error(f"Only {len(labeled)} labelled rows. Increase the historical range or pick another ticker.")
    st.stop()

st.sidebar.caption(f"Data source: {data_source}")

# ─── Chronological split (with 1-row gap) ─────────────────────────────────────
n = len(labeled)
split = int(n * (1 - test_frac))
GAP = 1
X_all = labeled[FEATURES].values
y_all = labeled["Target"].values.astype(int)
X_train, y_train = X_all[: split - GAP], y_all[: split - GAP]
X_test, y_test = X_all[split:], y_all[split:]
test_df = labeled.iloc[split:]


# ─── Training ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(X_train, y_train, X_all, y_all, split, gap, cache_key):
    scaler = RobustScaler().fit(X_train)            # fitted on TRAIN only
    Xs = scaler.transform(X_train)
    tscv = TimeSeriesSplit(n_splits=5)

    def tune(est, params, n_iter):
        s = RandomizedSearchCV(est, params, n_iter=n_iter, cv=tscv, scoring="accuracy",
                               random_state=42, n_jobs=-1)
        s.fit(Xs, y_train)
        return s.best_estimator_, s.best_score_

    rf, s_rf = tune(RandomForestClassifier(random_state=42, class_weight="balanced"),
                    {"n_estimators": [200, 300], "max_depth": [3, 4, 6], "min_samples_leaf": [5, 10, 20],
                     "max_features": ["sqrt", "log2"]}, 8)
    gb, s_gb = tune(GradientBoostingClassifier(random_state=42),
                    {"n_estimators": [100, 200], "max_depth": [2, 3], "learning_rate": [0.01, 0.05, 0.1],
                     "subsample": [0.7, 0.85]}, 8)
    ada, s_ada = tune(AdaBoostClassifier(random_state=42),
                      {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.3, 0.5, 1.0]}, 6)
    lr, s_lr = tune(LogisticRegression(random_state=42, max_iter=2000, class_weight="balanced"),
                    {"C": [0.01, 0.05, 0.1, 0.5, 1.0]}, 5)

    # Equal weights: CV scores of a tuned search are optimistic, so they should not drive the vote.
    ensemble = VotingClassifier([("RF", rf), ("GB", gb), ("ADA", ada), ("LR", lr)], voting="soft")
    ensemble.fit(Xs, y_train)

    # Walk-forward (expanding window) out-of-sample probabilities over the test period.
    # Hyper-parameters come from the train period only, so the test period stays unseen.
    n_all = len(X_all)
    wf_proba = np.zeros(n_all - split)
    edges = np.linspace(split, n_all, 6, dtype=int)
    for a, b in zip(edges[:-1], edges[1:]):
        pipe = make_pipeline(RobustScaler(), clone(ensemble))
        pipe.fit(X_all[: a - gap], y_all[: a - gap])
        wf_proba[a - split: b - split] = pipe.predict_proba(X_all[a:b])[:, 1]

    # Live-signal model: trained on ALL labelled data.
    final_pipe = make_pipeline(RobustScaler(), clone(ensemble)).fit(X_all, y_all)

    cv = {"RF": s_rf, "GB": s_gb, "ADA": s_ada, "LR": s_lr}
    return scaler, rf, gb, ada, lr, ensemble, cv, wf_proba, final_pipe


cache_key = (ticker.upper(), days_history, test_frac, len(labeled), str(labeled.index[-1]))
with st.spinner("Training and validating models (first run takes a minute)…"):
    scaler, rf, gb, ada, lr, ensemble, cv_scores, wf_proba, final_pipe = train_models(
        X_train, y_train, X_all, y_all, split, GAP, cache_key
    )

Xte = scaler.transform(X_test)

# ─── Metrics ──────────────────────────────────────────────────────────────────
PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Mono, monospace", color="#4a5568", size=11),
            margin=dict(t=20, b=10, l=0, r=0))

y_pred = ensemble.predict(Xte)
y_prob = ensemble.predict_proba(Xte)[:, 1]
acc = accuracy_score(y_test, y_pred)
baseline = max(y_test.mean(), 1 - y_test.mean())         # best "dumb" constant guess
ci = 1.96 * np.sqrt(acc * (1 - acc) / len(y_test))
p_val = binomtest(int((y_pred == y_test).sum()), len(y_test), baseline, alternative="greater").pvalue
auc_ens = roc_auc_score(y_test, y_prob)

wf_pred = (wf_proba > 0.5).astype(int)
wf_acc = accuracy_score(y_test, wf_pred)
chunks = np.array_split(np.arange(len(y_test)), 5)
wf_chunk_acc = [accuracy_score(y_test[i], wf_pred[i]) for i in chunks]


def card(label, value, sub="", cls=""):
    return (f'<div class="card"><div class="card-label">{label}</div>'
            f'<div class="card-value {cls}">{value}</div><div class="card-sub">{sub}</div></div>')


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Market Dashboard", "Trading Signal", "Model Performance", "Backtest", "AI Insights"]
)

# ── Tab 1: dashboard ──────────────────────────────────────────────────────────
with tab1:
    last = data.iloc[-1]
    chg = last["Daily_Return"] * 100
    st.markdown(
        '<div class="grid4">'
        + card("Latest Close", f"{symbol}{last['Close']:,.2f}",
               f'<span class="{"pos" if chg >= 0 else "neg"}">{"▲" if chg >= 0 else "▼"} {abs(chg):.2f}% on {data.index[-1].date()}</span>')
        + card("RSI · 14-Day", f"{last['RSI']:.1f}",
               "Overbought" if last["RSI"] > 70 else "Oversold" if last["RSI"] < 30 else "Neutral zone")
        + card("MACD (% of price)", f"{last['MACD_Pct'] * 100:.3f}%", f"Signal {last['Signal_Pct'] * 100:.3f}%")
        + card("Volatility · 5-Day", f"{last['Volatility'] * 100:.2f}%", "Std dev of daily returns")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">Price History</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close", line=dict(color="#e8c97d", width=1.5)))
    fig.add_trace(go.Scatter(x=data.index, y=data["5MA"], name="5-MA", line=dict(color="#60a5fa", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=data.index, y=data["10MA"], name="10-MA", line=dict(color="#a78bfa", width=1, dash="dot")))
    fig.add_vline(x=test_df.index[0], line_dash="dash", line_color="#34d399",
                  annotation_text="test period starts", annotation_font_color="#34d399")
    fig.update_layout(**PLOT, height=320, hovermode="x unified",
                      legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-label">Bollinger Bands</div>', unsafe_allow_html=True)
    fb = go.Figure()
    fb.add_trace(go.Scatter(x=data.index, y=data["BB_Upper"], name="Upper", line=dict(color="rgba(96,165,250,0.4)", width=1)))
    fb.add_trace(go.Scatter(x=data.index, y=data["BB_Lower"], name="Lower", line=dict(color="rgba(96,165,250,0.4)", width=1),
                            fill="tonexty", fillcolor="rgba(96,165,250,0.05)"))
    fb.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close", line=dict(color="#e8c97d", width=1.2)))
    fb.update_layout(**PLOT, height=240, legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fb, use_container_width=True)

# ── Tab 2: signal ─────────────────────────────────────────────────────────────
with tab2:
    prob_up = final_pipe.predict_proba(latest[FEATURES].values)[0, 1]
    if abs(prob_up - 0.5) < 0.03:
        cls, icon, head = "neutral", "→", "NO CLEAR EDGE"
        desc = "Probability is within ±3% of 50%. The models see no meaningful direction."
    elif prob_up > 0.5:
        cls, icon, head = "bullish", "↑", "LEANS UP"
        desc = "Ensemble leans towards an upward close next session."
    else:
        cls, icon, head = "bearish", "↓", "LEANS DOWN"
        desc = "Ensemble leans towards a flat or downward close next session."

    st.caption(f"Signal is computed from the latest available day ({data.index[-1].date()}) "
               f"and refers to the NEXT trading session.")
    st.markdown(
        f"""
<div class="signal-wrap"><div class="signal-card {cls}">
<div style="font-size:2.5rem;margin-bottom:0.8rem;">{icon}</div>
<div class="signal-type">Next-Day Projection · P(up) = {prob_up:.1%}</div>
<div class="signal-heading {cls}">{head}</div>
<div class="signal-desc">{desc}</div>
</div></div>
""",
        unsafe_allow_html=True,
    )
    st.warning(
        f"Out-of-sample accuracy is {wf_acc:.1%} vs a {baseline:.1%} always-same-class baseline. "
        "Daily direction is very noisy, so treat this as an academic experiment, not financial advice."
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Individual Model Accuracy (test period)</div>', unsafe_allow_html=True)
    accs = {
        "Random Forest": accuracy_score(y_test, rf.predict(Xte)),
        "Gradient Boost": accuracy_score(y_test, gb.predict(Xte)),
        "AdaBoost": accuracy_score(y_test, ada.predict(Xte)),
        "Logistic Reg": accuracy_score(y_test, lr.predict(Xte)),
        "Ensemble": acc,
    }
    fa = go.Figure(go.Bar(x=list(accs), y=list(accs.values()),
                          marker_color=["#1e3a5f"] * 4 + ["#e8c97d"],
                          text=[f"{v:.1%}" for v in accs.values()], textposition="outside"))
    fa.add_hline(y=baseline, line_dash="dash", line_color="#f87171",
                 annotation_text=f"baseline {baseline:.1%}", annotation_font_color="#f87171")
    fa.update_layout(**PLOT, height=300, yaxis=dict(range=[0, 1.1], tickformat=".0%"), bargap=0.35)
    st.plotly_chart(fa, use_container_width=True)

# ── Tab 3: performance ────────────────────────────────────────────────────────
with tab3:
    verdict = ("statistically better than the baseline (p < 0.05)" if p_val < 0.05
               else "NOT statistically better than the baseline")
    st.markdown(
        '<div class="grid4">'
        + card("Test Accuracy", f"{acc:.1%}", f"95% CI ± {ci * 100:.1f}%", "gold")
        + card("Baseline (constant guess)", f"{baseline:.1%}", f"n = {len(y_test)} test days")
        + card("Walk-forward Accuracy", f"{wf_acc:.1%}", "expanding-window, out-of-sample", "gold")
        + card("ROC AUC", f"{auc_ens:.3f}", "0.5 = random guessing")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="grid4">'
        + card("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.1%}")
        + card("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.1%}")
        + card("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.1%}")
        + card("p-value vs baseline", f"{p_val:.3f}")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.info(f"Result is {verdict}.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-label">Confusion Matrix (test period)</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fc = go.Figure(go.Heatmap(z=cm, x=["Pred Down", "Pred Up"], y=["Actual Down", "Actual Up"],
                                  text=cm, texttemplate="%{text}", colorscale=[[0, "#0e1525"], [1, "#e8c97d"]],
                                  showscale=False))
        fc.update_layout(**PLOT, height=300)
        st.plotly_chart(fc, use_container_width=True)
    with c2:
        st.markdown('<div class="section-label">Walk-forward Accuracy by Period</div>', unsafe_allow_html=True)
        fw = go.Figure(go.Bar(x=[f"Fold {i + 1}" for i in range(5)], y=wf_chunk_acc,
                              marker_color="#60a5fa", text=[f"{v:.1%}" for v in wf_chunk_acc],
                              textposition="outside"))
        fw.add_hline(y=0.5, line_dash="dash", line_color="#f87171")
        fw.update_layout(**PLOT, height=300, yaxis=dict(range=[0, 1], tickformat=".0%"), bargap=0.4)
        st.plotly_chart(fw, use_container_width=True)

    st.markdown('<div class="section-label">ROC Curves</div>', unsafe_allow_html=True)
    fr = go.Figure()
    fr.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color="#3d4d66", dash="dot")))
    for name, m, col in [("Random Forest", rf, "#3b82f6"), ("Gradient Boost", gb, "#a78bfa"),
                         ("Logistic Reg", lr, "#60a5fa"), ("Ensemble", ensemble, "#e8c97d")]:
        p = m.predict_proba(Xte)[:, 1]
        f_, t_, _ = roc_curve(y_test, p)
        fr.add_trace(go.Scatter(x=f_, y=t_, name=f"{name} (AUC {roc_auc_score(y_test, p):.2f})",
                                line=dict(color=col, width=2.5 if name == "Ensemble" else 1.5)))
    fr.update_layout(**PLOT, height=380, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                     legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.98, bgcolor="rgba(14,21,37,0.8)"))
    st.plotly_chart(fr, use_container_width=True)

    st.caption("Cross-validated accuracy on the training period (tuning score, optimistic by design): "
               + " · ".join(f"{k} {v:.1%}" for k, v in cv_scores.items()))

# ── Tab 4: backtest ───────────────────────────────────────────────────────────
with tab4:
    pos = wf_pred.astype(float)                                  # 1 = long, 0 = cash
    ret = test_df["Next_Close_Return"].values
    trades = np.abs(np.diff(np.r_[0.0, pos]))
    strat = pos * ret - trades * (cost_bp / 10000)
    strat_eq = np.cumprod(1 + strat)
    bh_eq = np.cumprod(1 + ret)

    def sharpe(r):
        return 0.0 if np.std(r) == 0 else np.mean(r) / np.std(r) * np.sqrt(252)

    def max_dd(eq):
        return float((eq / np.maximum.accumulate(eq) - 1).min())

    st.markdown(
        '<div class="grid4">'
        + card("Strategy Return", f"{strat_eq[-1] - 1:+.1%}", f"Sharpe {sharpe(strat):.2f}",
               "pos" if strat_eq[-1] >= bh_eq[-1] else "neg")
        + card("Buy & Hold Return", f"{bh_eq[-1] - 1:+.1%}", f"Sharpe {sharpe(ret):.2f}")
        + card("Max Drawdown (strategy)", f"{max_dd(strat_eq):.1%}", f"B&H {max_dd(bh_eq):.1%}")
        + card("Trades", f"{int(trades.sum())}", f"cost {cost_bp} bp per trade")
        + "</div>",
        unsafe_allow_html=True,
    )
    fe = go.Figure()
    fe.add_trace(go.Scatter(x=test_df.index, y=strat_eq, name="Model strategy (long / cash)", line=dict(color="#e8c97d", width=2)))
    fe.add_trace(go.Scatter(x=test_df.index, y=bh_eq, name="Buy & hold", line=dict(color="#60a5fa", width=1.5)))
    fe.update_layout(**PLOT, height=380, yaxis_title="Growth of 1 unit",
                     legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fe, use_container_width=True)
    st.caption("Uses only walk-forward predictions (the model never saw the day it traded). "
               "Ignores slippage, taxes and liquidity.")

# ── Tab 5: insights ───────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-label">Feature Importance — Random Forest</div>', unsafe_allow_html=True)
    imp = rf.feature_importances_
    order = np.argsort(imp)
    ff = go.Figure(go.Bar(x=imp[order], y=np.array(FEATURES)[order], orientation="h",
                          marker_color=["#e8c97d" if imp[i] == imp.max() else "#1e3a5f" for i in order],
                          text=[f"{v:.3f}" for v in imp[order]], textposition="outside"))
    ff.update_layout(**PLOT, height=520, xaxis_title="Relative importance")
    st.plotly_chart(ff, use_container_width=True)
    top3 = [FEATURES[i] for i in np.argsort(imp)[-3:][::-1]]
    st.markdown(f"**Top 3 features:** {', '.join(top3)}")
    st.caption("Importances that are all small and similar indicate the features carry little predictive signal.")
