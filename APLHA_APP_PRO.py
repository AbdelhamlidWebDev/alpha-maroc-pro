# ALPH_APP.py  — Alpha Maroc Pro (Fondamental + Technique + Export + Reco)
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Alpha Maroc – Analyseur Pro", layout="wide")

# ===========================
# Helpers (indicateurs tech)
# ===========================
def sma(series, window):
    return series.rolling(window).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_down = pd.Series(down, index=series.index).rolling(period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi_v = 100 - (100 / (1 + rs))
    return rsi_v.bfill()

def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ===========================
# Titre
# ===========================
st.title("📊 Alpha Maroc – Analyseur Pro (Fondamental + Technique)")

tabs = st.tabs(["🏦 Analyse Fondamentale", "📈 Analyse Technique", "🧠 Recommandation & Export"])

# ==========================================================
# 🏦 Onglet FONDAMENTAL
# ==========================================================
with tabs[0]:
    st.markdown("Entrez les données financières d’une société cotée pour obtenir les **ratios clés** automatiquement.")
    with st.sidebar:
        st.header("Paramètres de l'entreprise")
        price = st.number_input("Prix actuel (DH)", value=126.50, step=0.01)
        shares_outstanding = st.number_input("Actions en circulation", value=17_695_000, step=1_000)
        revenue = st.number_input("Chiffre d'affaires (DH)", value=373_400_000, step=100_000)
        net_income = st.number_input("Résultat net (DH)", value=44_642_000, step=100_000)
        total_assets = st.number_input("Total actif (DH)", value=468_000_000, step=100_000)
        total_equity = st.number_input("Capitaux propres (DH)", value=300_000_000, step=100_000)
        ebitda = st.number_input("EBITDA (DH)", value=70_000_000, step=100_000)
        total_debt = st.number_input("Dette totale (DH)", value=50_000_000, step=100_000)
        cash = st.number_input("Trésorerie (DH)", value=20_000_000, step=100_000)

    # Calculs
    market_cap = price * shares_outstanding
    eps = net_income / shares_outstanding if shares_outstanding else np.nan
    per = price / eps if eps and eps > 0 else np.nan
    bvps = total_equity / shares_outstanding if shares_outstanding else np.nan
    pb = price / bvps if bvps and bvps > 0 else np.nan
    roe = (net_income / total_equity) * 100 if total_equity else np.nan
    roa = (net_income / total_assets) * 100 if total_assets else np.nan
    ev = market_cap + total_debt - cash
    ev_ebitda = ev / ebitda if ebitda else np.nan
    net_margin = (net_income / revenue) * 100 if revenue else np.nan

    df_fonda = pd.DataFrame({
        "Market Cap (DH)": [market_cap],
        "EPS (DH)": [eps],
        "PER": [per],
        "BVPS": [bvps],
        "P/B": [pb],
        "ROE %": [roe],
        "ROA %": [roa],
        "EV/EBITDA": [ev_ebitda],
        "Net Margin %": [net_margin]
    })

    st.subheader("📈 Résultats Financiers")
    st.dataframe(df_fonda.style.format("{:,.2f}"), use_container_width=True)

    st.subheader("💡 Interprétation rapide")
    interp = []
    interp.append(f"- **PER ({per:.1f}x)** → {'élevé' if per and per>25 else 'raisonnable' if per and per>10 else 'faible' if per else 'n/d'}")
    interp.append(f"- **P/B ({pb:.2f}x)** → {'valorisation élevée' if pb and pb>3 else 'proche de la valeur comptable' if pb else 'n/d'}")
    interp.append(f"- **ROE ({roe:.1f}%)** → {'excellent' if roe and roe>15 else 'correct' if roe and roe>8 else 'faible' if roe==roe else 'n/d'}")
    interp.append(f"- **Marge nette ({net_margin:.1f}%)** → {'bonne rentabilité' if net_margin and net_margin>10 else 'faible marge' if net_margin==net_margin else 'n/d'}")
    st.markdown("\n".join(interp))
    st.success("✅ Calcul fondamental terminé. Passe à l’onglet **Analyse Technique** pour charger l’historique de prix.")

# ==========================================================
# 📈 Onglet TECHNIQUE — robuste & structuré
# ==========================================================
with tabs[1]:
    st.markdown(
        "Charge un **CSV Investing.com** (Données historiques). "
        "Colonnes attendues : **Date**, **Close/Price/Dernier**, **Volume** (facultatif)."
    )
    file = st.file_uploader("Uploader le CSV des prix (Investing)", type=["csv"])

    # ---------- Paramètres techniques ----------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        rsi_period = st.number_input("Période RSI", value=14, min_value=2, step=1)
    with col2:
        sma_fast = st.number_input("SMA courte", value=20, min_value=2, step=1)
    with col3:
        sma_mid = st.number_input("SMA moyenne", value=50, min_value=2, step=1)
    with col4:
        sma_slow = st.number_input("SMA longue", value=200, min_value=2, step=1)

    # ---------- Helpers robustes ----------
    def normalize_numeric(series: pd.Series) -> pd.Series:
        """
        Convertit une série texte FR (espaces insécables, milliers, virgules décimales)
        en float propre. Renvoie NaN si conversion impossible.
        """
        s = (series.astype(str)
                  .str.replace("\u00a0", "", regex=False)  # NBSP
                  .str.replace("\u202f", "", regex=False)  # fine NBSP
                  .str.replace(" ", "", regex=False)       # espaces
                  .str.replace(".", "", regex=False)       # . comme séparateur de milliers
                  .str.replace(",", ".", regex=False))     # , -> .
        return pd.to_numeric(s, errors="coerce")

    def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        """Cherche une colonne par liste de candidats (insensible à la casse)."""
        lower_map = {c.lower(): c for c in df.columns}
        for cand in candidates:
            key = cand.lower()
            # exact
            if key in lower_map:
                return lower_map[key]
            # commence par
            for k, orig in lower_map.items():
                if k.startswith(key):
                    return orig
        return None

    def safe_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """RSI robuste pour séries avec trous/NaN."""
        close = close.astype(float)
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)

        roll_up = up.rolling(period, min_periods=period).mean()
        roll_down = down.rolling(period, min_periods=period).mean()

        rs = roll_up / roll_down.replace(0, np.nan)
        out = 100 - (100 / (1 + rs))
        return out.bfill()

    # ---------- Lecture & nettoyage du CSV ----------
    df_sig = None

if file:
    try:
        # Lecture du CSV
        df = pd.read_csv(file, sep=None, engine="python")

        # --- Nettoyage des noms de colonnes ---
        def nettoyer_nom_colonne(colonne):
            return (str(colonne)
                    .replace("\ufeff", "")   # Supprime les caractères cachés (BOM)
                    .strip()                 # Supprime les espaces
                    .strip('"')              # Supprime les guillemets doubles
                    .strip("'"))             # Supprime les guillemets simples

        df.columns = [nettoyer_nom_colonne(c) for c in df.columns]

        # --- Détection automatique des colonnes ---
        date_col = find_col(df, ["date"])
        price_col = find_col(df, ["close", "prix", "price", "dernier", "last", "cloture", "clôture"])
        volume_col = find_col(df, ["volume", "vol"])

        st.success("✅ Fichier CSV lu et nettoyé avec succès !")
        # --- Sélection/standardisation des colonnes utilisées
cols = [date_col, price_col] + ([volume_col] if volume_col else [])
df = df[cols].copy()
df.rename(columns={date_col: "date", price_col: "close"}, inplace=True)
if volume_col:
    df.rename(columns={volume_col: "volume"}, inplace=True)

# --- Types & tri
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["close"] = pd.to_numeric(df["close"], errors="coerce")
if "volume" in df.columns:
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

# --- Paramètres depuis l’UI
rsi_period = int(st.session_state.get("rsi_period", 14))        # ou la variable que tu lis depuis le widget
sma_fast   = int(st.session_state.get("sma_fast", 20))
sma_mid    = int(st.session_state.get("sma_mid", 50))
sma_slow   = int(st.session_state.get("sma_slow", 200))

# --- Indicateurs
df["RSI"]      = rsi(df["close"], rsi_period)
df["SMA_fast"] = sma(df["close"], sma_fast)
df["SMA_mid"]  = sma(df["close"], sma_mid)
df["SMA_slow"] = sma(df["close"], sma_slow)
macd_line, macd_signal, macd_hist = macd(df["close"])
df["MACD"]        = macd_line
df["MACD_signal"] = macd_signal
df["MACD_hist"]   = macd_hist

# =============== Graphique principal (cours + SMA) ===============
st.subheader("📊 Graphique des prix & moyennes mobiles")
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close"))
fig.add_trace(go.Scatter(x=df["date"], y=df["SMA_fast"], name=f"SMA{sma_fast}"))
fig.add_trace(go.Scatter(x=df["date"], y=df["SMA_mid"],  name=f"SMA{sma_mid}"))
fig.add_trace(go.Scatter(x=df["date"], y=df["SMA_slow"], name=f"SMA{sma_slow}"))
fig.update_layout(height=420, legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, use_container_width=True)

# =============== RSI & MACD ===============
c1, c2 = st.columns(2)
with c1:
    st.subheader("RSI")
    frsi = go.Figure()
    frsi.add_trace(go.Scatter(x=df["date"], y=df["RSI"], name="RSI"))
    frsi.add_hline(y=70, line=dict(color="red", dash="dot"))
    frsi.add_hline(y=30, line=dict(color="green", dash="dot"))
    frsi.update_layout(height=250)
    st.plotly_chart(frsi, use_container_width=True)
with c2:
    st.subheader("MACD")
    fmacd = go.Figure()
    fmacd.add_trace(go.Scatter(x=df["date"], y=df["MACD"], name="MACD"))
    fmacd.add_trace(go.Scatter(x=df["date"], y=df["MACD_signal"], name="Signal"))
    fmacd.add_trace(go.Bar(x=df["date"], y=df["MACD_hist"], name="Hist"))
    fmacd.update_layout(height=250, barmode="relative")
    st.plotly_chart(fmacd, use_container_width=True)

# =============== Signaux & interprétation ===============
last = df.iloc[-1]
price_last    = float(last["close"])
rsi_last      = float(last["RSI"])
sma_mid_last  = float(last["SMA_mid"])
sma_slow_last = float(last["SMA_slow"])
macd_pos      = 1 if last["MACD_hist"] > 0 else 0
rsi_signal    = 1 if rsi_last <= 30 else (0 if rsi_last >= 70 else 0.5)
sma_cross     = 1 if (price_last > sma_mid_last and sma_mid_last > sma_slow_last) else 0

# Pondérations (modifiable)
w_rsi, w_sma, w_macd = 30, 40, 30
tech_score = round((rsi_signal*w_rsi + sma_cross*w_sma + macd_pos*w_macd)/(w_rsi+w_sma+w_macd)*100, 0)

df_sig = pd.DataFrame([{
    "Last Price": price_last,
    "RSI": rsi_last,
    f"SMA{sma_mid}": sma_mid_last,
    f"SMA{sma_slow}": sma_slow_last,
    "MACD>0": macd_pos,
    "RSI_signal(1/0/0.5)": rsi_signal,
    "SMA_cross(1/0)": sma_cross,
    "Tech Score (0-100)": tech_score
}])

st.subheader("🧪 Signaux techniques (instantané)")
st.dataframe(df_sig.style.format("{:,.2f}"), use_container_width=True)

# Interprétations rapides
bullet = []
bullet.append(f"• **RSI {rsi_last:.1f}** → survente <30, surachat >70.")
bullet.append("• **MACD** > 0 → biais haussier court terme." if macd_pos else "• **MACD** < 0 → biais baissier court terme.")
bullet.append("• **SMA_mid au-dessus de SMA_slow** et prix au-dessus de SMA_mid → tendance haussière" if sma_cross else "• Tendance non confirmée par les MM.")
st.markdown("\n".join(bullet))

# Partage pour l’onglet Recommandation
st.session_state["df_sig"] = df_sig
st.session_state["hist"]  = df


    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")
        # Détection souple des colonnes clés
        date_col = find_col(df, ["date"])
        price_col = find_col(df, ["close", "prix", "price", "dernier", "last"])
        volume_col = find_col(df, ["volume", "vol"])

        if not date_col or not price_col:
            st.error(
                "Colonnes non reconnues. Assure-toi d’avoir au moins **Date** "
                "et **Close/Price/Dernier** dans le CSV."
            )
            st.dataframe(df.head(20))
            st.stop()

        # Normalisation colonnes
        df = df.rename(columns={date_col: "date", price_col: "close"})
        if volume_col:
            df = df.rename(columns={volume_col: "volume"})

        # Parse des dates robuste (format fr ok)
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True, infer_datetime_format=True)
        df = df.dropna(subset=["date"])

        # Nettoyage des nombres (prix / volume)
        df["close"] = normalize_numeric(df["close"])
        if "volume" in df.columns:
            df["volume"] = normalize_numeric(df["volume"])

        # Trier et garder lignes valides pour le prix
        df = df.sort_values("date").dropna(subset=["close"])

        if len(df) < max(50, rsi_period + 5):
            st.warning("Pas assez d’historique pour calculer tous les indicateurs. Ajoute plus de données.")
            st.dataframe(df.tail(10))
            st.stop()

        # ---------- Indicateurs ----------
        df["RSI"] = safe_rsi(df["close"], rsi_period)
        df["SMA_fast"] = df["close"].rolling(sma_fast, min_periods=1).mean()
        df["SMA_mid"]  = df["close"].rolling(sma_mid,  min_periods=1).mean()
        df["SMA_slow"] = df["close"].rolling(sma_slow, min_periods=1).mean()

        macd_line = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_signal

        df["MACD"] = macd_line
        df["MACD_signal"] = macd_signal
        df["MACD_hist"] = macd_hist

        # ---------- Graphique prix + SMAs ----------
        st.subheader("📊 Graphique")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close"))
        fig.add_trace(go.Scatter(x=df["date"], y=df["SMA_fast"], name=f"SMA{sma_fast}"))
        fig.add_trace(go.Scatter(x=df["date"], y=df["SMA_mid"],  name=f"SMA{sma_mid}"))
        fig.add_trace(go.Scatter(x=df["date"], y=df["SMA_slow"], name=f"SMA{sma_slow}"))
        fig.update_layout(height=420, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

        # ---------- RSI + MACD ----------
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("RSI")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df["date"], y=df["RSI"], name="RSI"))
            fig_rsi.add_hline(y=70, line=dict(color="red", dash="dot"))
            fig_rsi.add_hline(y=30, line=dict(color="green", dash="dot"))
            fig_rsi.update_layout(height=250)
            st.plotly_chart(fig_rsi, use_container_width=True)

        with c2:
            st.subheader("MACD")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df["date"], y=df["MACD"], name="MACD"))
            fig_macd.add_trace(go.Scatter(x=df["date"], y=df["MACD_signal"], name="Signal"))
            fig_macd.add_trace(go.Bar(x=df["date"], y=df["MACD_hist"], name="Hist"))
            fig_macd.update_layout(height=250, barmode="relative")
            st.plotly_chart(fig_macd, use_container_width=True)

        # ---------- Signaux & score technique ----------
        last = df.iloc[-1]
        price_last = float(last["close"])
        rsi_last = float(last["RSI"])
        sma_mid_last = float(last["SMA_mid"])
        sma_slow_last = float(last["SMA_slow"])
        macd_pos = 1 if last["MACD_hist"] > 0 else 0
        rsi_signal = 1 if rsi_last <= 30 else (0 if rsi_last >= 70 else 0.5)
        sma_cross = 1 if (price_last > sma_mid_last and sma_mid_last > sma_slow_last) else 0

        w_rsi, w_sma, w_macd = 30, 40, 30
        score_tech = round((rsi_signal*w_rsi + sma_cross*w_sma + macd_pos*w_macd)/(w_rsi+w_sma+w_macd)*100, 0)

        df_sig = pd.DataFrame([{
            "Last Price": price_last,
            "RSI": rsi_last,
            f"SMA{sma_mid}": sma_mid_last,
            f"SMA{sma_slow}": sma_slow_last,
            "MACD>0": macd_pos,
            "RSI_signal(1/0/0.5)": rsi_signal,
            "SMA_cross(1/0)": sma_cross,
            "Tech Score (0-100)": score_tech
        }])

        st.subheader("🧪 Signaux techniques (instantané)")
        st.dataframe(df_sig.style.format("{:,.2f}"), use_container_width=True)
        st.info("✅ Analyse technique prête. Passe à l’onglet **Recommandation & Export**.")






