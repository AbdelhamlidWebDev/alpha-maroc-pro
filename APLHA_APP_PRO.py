# ALPH_APP_PRO.py — Alpha Maroc Pro (Fondamental + Technique + Export + Reco)

import io
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Alpha Maroc – Analyseur Pro", layout="wide")

# ------------------------
# Indicateurs techniques
# ------------------------
def sma(series, window):
    return series.rolling(window, min_periods=window).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi_calc(series, period=14):
    delta = series.diff()
    up   = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up   = up.rolling(period, min_periods=period).mean()
    roll_down = down.rolling(period, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    r = 100 - (100 / (1 + rs))
    return r.bfill()

def macd_calc(series, fast=12, slow=26, signal=9):
    macd_line   = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ------------------------
# Titre & onglets
# ------------------------
st.title("📊 Alpha Maroc – Analyseur Pro (Fondamental + Technique)")
tabs = st.tabs(["🏦 Analyse Fondamentale", "📈 Analyse Technique", "🧠 Recommandation & Export"])
# ==========================================================
# 📈 Onglet TECHNIQUE
# ==========================================================
with tabs[1]:
    file = st.file_uploader("Uploader le CSV des prix (Investing)", type=["csv"])

    # Paramètres techniques
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        rsi_period = st.number_input("Période RSI", value=14, min_value=2, step=1)
    with col2:
        sma_fast = st.number_input("SMA courte", value=20, min_value=2, step=1)
    with col3:
        sma_mid = st.number_input("SMA moyenne", value=50, min_value=2, step=1)
    with col4:
        sma_slow = st.number_input("SMA longue", value=200, min_value=2, step=1)

    if file:
        try:
            # --- Lecture du CSV ---
            file.seek(0)
            df_raw = pd.read_csv(file, sep=None, engine="python", encoding="utf-8", skip_blank_lines=True)
            if df_raw.shape[1] == 1:
                file.seek(0)
                df_raw = pd.read_csv(file, sep=None, engine="python", encoding="latin1", skip_blank_lines=True)

            # --- Nettoyage des noms de colonnes ---
            def _clean_colname(c: str) -> str:
                s = str(c).strip().lower()
                for ch in ['"', "'", "\ufeff", "\u00a0", "\u202f"]:
                    s = s.replace(ch, "")
                return (s.replace("clôture", "cloture")
                         .replace("close/dernier", "close")
                         .replace("close/price", "close")
                         .replace("closeprice", "close")
                         .replace("dernier", "close")
                         .replace("prix", "close")
                         .replace("vol.", "volume")
                         .replace("vol", "volume")
                         .replace("date", "date"))

            df_raw.columns = [_clean_colname(c) for c in df_raw.columns]
            st.caption(f"Colonnes détectées : {list(df_raw.columns)}")

            if "date" not in df_raw.columns or "close" not in df_raw.columns:
                st.error("Colonnes non reconnues. Assure-toi d’avoir **Date** et **Close/Dernier**.")
                st.stop()

            # --- Sélection utile ---
            keep = ["date", "close"] + (["volume"] if "volume" in df_raw.columns else [])
            df = df_raw[keep].copy()
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

            # --- Nettoyage des valeurs ---
            def parse_price(x):
                if pd.isna(x): return np.nan
                s = str(x).replace("\u00a0", "").replace(" ", "").replace(",", ".")
                s = re.sub(r"[^0-9\.\-]", "", s)
                try: return float(s)
                except: return np.nan

            df["close"] = df["close"].map(parse_price)

            if "volume" in df.columns:
                def parse_vol(v):
                    if pd.isna(v): return np.nan
                    s = str(v).replace("\u00a0", "").replace(" ", "").lower()
                    mult = 1
                    if s.endswith("k"): mult, s = 1_000, s[:-1]
                    elif s.endswith("m"): mult, s = 1_000_000, s[:-1]
                    s = s.replace(",", ".")
                    s = re.sub(r"[^0-9\.\-]", "", s)
                    try: return float(s) * mult
                    except: return np.nan
                df["volume"] = df["volume"].map(parse_vol)

            # --- Nettoyage global ---
            rows_in = len(df)
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            df = df.dropna(subset=["close"]).reset_index(drop=True)
            rows_out = len(df)
            st.info(f"📄 Lignes CSV : {rows_in} → après nettoyage : {rows_out}")

            if rows_out < 35:
                st.warning(f"⚠️ Historique insuffisant ({rows_out} lignes). Ajoute plus de données.")
                st.stop()

            # --- Calcul indicateurs ---
            def _sma(s, w): return s.rolling(w, min_periods=w).mean()
            def _ema(s, a): return s.ewm(span=a, adjust=False, min_periods=a).mean()
            def _rsi(s, p=14):
                d = s.diff()
                up = d.clip(lower=0)
                dn = (-d).clip(lower=0)
                ru = up.rolling(p, min_periods=p).mean()
                rd = dn.rolling(p, min_periods=p).mean()
                rs = ru / rd.replace(0, np.nan)
                return 100 - (100 / (1 + rs))

            macd_line   = _ema(df["close"], 12) - _ema(df["close"], 26)
            macd_signal = _ema(macd_line, 9)
            macd_hist   = macd_line - macd_signal

            df["RSI"] = _rsi(df["close"], int(rsi_period))
            df["SMA_fast"] = _sma(df["close"], int(sma_fast))
            df["SMA_mid"] = _sma(df["close"], int(sma_mid))
            df["SMA_slow"] = _sma(df["close"], int(sma_slow))
            df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd_line, macd_signal, macd_hist

            # --- Graphiques ---
            st.subheader("📊 Graphique des prix et moyennes mobiles")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close"))
            fig.add_trace(go.Scatter(x=df["date"], y=df["SMA_fast"], name=f"SMA{sma_fast}"))
            fig.add_trace(go.Scatter(x=df["date"], y=df["SMA_mid"], name=f"SMA{sma_mid}"))
            fig.add_trace(go.Scatter(x=df["date"], y=df["SMA_slow"], name=f"SMA{sma_slow}"))
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("RSI")
                frsi = go.Figure()
                frsi.add_trace(go.Scatter(x=df["date"], y=df["RSI"], name="RSI"))
                frsi.add_hline(y=70, line=dict(color="red", dash="dot"))
                frsi.add_hline(y=30, line=dict(color="green", dash="dot"))
                st.plotly_chart(frsi, use_container_width=True)
            with c2:
                st.subheader("MACD")
                fmacd = go.Figure()
                fmacd.add_trace(go.Scatter(x=df["date"], y=df["MACD"], name="MACD"))
                fmacd.add_trace(go.Scatter(x=df["date"], y=df["MACD_signal"], name="Signal"))
                fmacd.add_trace(go.Bar(x=df["date"], y=df["MACD_hist"], name="Hist"))
                st.plotly_chart(fmacd, use_container_width=True)

            # --- Signaux + Score ---
            last = df.iloc[-1]
            price_last, rsi_last = float(last["close"]), float(last["RSI"])
            sma_mid_last, sma_slow_last = float(last["SMA_mid"]), float(last["SMA_slow"])
            macd_pos = 1 if float(last["MACD_hist"]) > 0 else 0
            rsi_signal = 1 if rsi_last <= 30 else (0 if rsi_last >= 70 else 0.5)
            sma_cross = 1 if (price_last > sma_mid_last and sma_mid_last > sma_slow_last) else 0

            w_rsi, w_sma, w_macd = 30, 40, 30
            score_tech = round((rsi_signal*w_rsi + sma_cross*w_sma + macd_pos*w_macd) /
                               (w_rsi+w_sma+w_macd) * 100, 0)

            st.session_state["tech_score"] = score_tech
            st.session_state["tech_prices_csv"] = file

            st.subheader("🧪 Signaux techniques")
            st.metric("Score technique (0–100)", f"{score_tech:.0f}")
            st.success("✅ Analyse technique terminée.")

        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")

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

    # Calculs fondamentaux
    market_cap = price * shares_outstanding
    eps = net_income / shares_outstanding if shares_outstanding else np.nan
    per = price / eps if (eps and eps > 0) else np.nan
    bvps = total_equity / shares_outstanding if shares_outstanding else np.nan
    pb = price / bvps if (bvps and bvps > 0) else np.nan
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
# 🧠 Onglet RECOMMANDATION & EXPORT
# ==========================================================
with tabs[2]:
    st.markdown("Synthèse des signaux **Fondamentaux + Techniques** et **export Excel**.")

    # Recrée df_fonda localement si l'utilisateur n'a pas visité le premier onglet
    df_fonda_safe = pd.DataFrame({
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

    st.subheader("📌 Résumé technique")
    if "tech_score" in st.session_state:
        tech_score = float(st.session_state["tech_score"])
        st.metric("Score technique (0–100)", f"{tech_score:.0f}")
    else:
        st.warning("Charge d’abord un CSV dans **Analyse Technique**.")
        tech_score = np.nan
    # Score fondamental simple (0..50)
    score_fonda = 0
    if per and per > 0: score_fonda += 20 if 10 <= per <= 25 else (10 if per < 10 else 0)
    if roe and roe > 0: score_fonda += 25 if roe > 15 else (15 if roe >= 8 else 0)
    if net_margin and net_margin > 10: score_fonda += 15
    if pb and pb > 3: score_fonda -= 10
    score_fonda = max(0, min(50, score_fonda))

    # Score global
    global_score = None
    if not np.isnan(tech_score):
        global_score = round(score_fonda + (tech_score / 2), 0)  # /100

    st.subheader("🧠 Recommandation automatique")
    if global_score is not None:
        if global_score >= 70:
            verdict = "✅ **Acheter / Renforcer**"
        elif global_score >= 50:
            verdict = "🟡 **Conserver / Surveiller**"
        else:
            verdict = "🔻 **Alléger / Éviter**"
        st.metric("Score global (0–100)", f"{global_score:.0f}", help="50% Fondamental + 50% Technique")
        st.success(f"Verdict : {verdict}")
    else:
        st.info("Charge les données techniques pour calculer le score global.")

    # -------- Export Excel (fondamental + technique + historique optionnel)
    st.subheader("📤 Export Excel")
    include_hist = False
    if 'tech_prices_csv' in st.session_state and st.session_state["tech_prices_csv"] is not None:
        include_hist = st.checkbox("Inclure l'historique de prix dans l'Excel", value=True)

    if st.button("📥 Télécharger le rapport Excel"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_fonda_safe.to_excel(writer, sheet_name="Fondamental", index=False)
            if 'df_sig' in locals() and df_sig is not None:
                df_sig.to_excel(writer, sheet_name="Technique_Signaux", index=False)
            if include_hist and 'tech_prices_csv' in st.session_state:
                f = st.session_state["tech_prices_csv"]
                try:
                    f.seek(0)
                    pd.read_csv(f, sep=None, engine="python").to_excel(writer, sheet_name="Prix_Historique", index=False)
                except Exception:
                    pass

        st.download_button(
            label="⬇️ Télécharger AlphaMaroc_Report.xlsx",
            data=output.getvalue(),
            file_name="AlphaMaroc_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.success("Rapport prêt ✅")







