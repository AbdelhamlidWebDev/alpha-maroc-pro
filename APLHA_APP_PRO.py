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
# 📈 Onglet TECHNIQUE
# ==========================================================
with tabs[1]:
    st.markdown(
        "Charge un **CSV Investing.com** (Données historiques). "
        "Colonnes attendues : **Date**, **Close/Price/Dernier**, **Volume** (facultatif)."
    )

    # Un seul uploader (clé unique)
    file = st.file_uploader(
        "Uploader le CSV des prix (Investing)",
        type=["csv"],
        key="tech_prices_csv"
    )

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

    df_sig = None

    # ------ Lecture & Nettoyage robustes ------
    def _norm_cols(cols):
        out = []
        for c in cols:
            s = str(c).lower()
            for ch in ["\u00a0", "\ufeff", "’", "'", " "]:
                s = s.replace(ch, "")
            s = (s.replace("clôture", "cloture")
                   .replace("dernier", "close")
                   .replace("prix", "close")
                   .replace("vol.", "volume"))
            out.append(s)
        return out

    def _pick(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    if file:
        try:
            # 1) Lecture tolérante (séparateur auto) + fallback latin1
            raw = pd.read_csv(file, sep=None, engine="python", encoding="utf-8", skip_blank_lines=True)
            if raw.shape[1] == 1:
                file.seek(0)
                raw = pd.read_csv(file, sep=None, engine="python", encoding="latin1", skip_blank_lines=True)

            rows_in = len(raw)
            raw.columns = _norm_cols(raw.columns)

            # 2) Colonnes (reconnaissance FR + accents)
            date_col  = _pick(raw, ["date"])
            price_col = _pick(raw, [
                "close","prix","price","dernier","cloture","clôture","last","close/price","closeprice"
            ])
            vol_col   = _pick(raw, ["volume","vol","vol.","volume(m)","volume(k)"])
            if not date_col or not price_col:
                st.error("Colonnes non reconnues. Assure-toi d’avoir **Date** et **Close/Dernier**.")
                st.stop()

            keep = [date_col, price_col] + ([vol_col] if vol_col else [])
            df = raw[keep].copy()
            rename_map = {date_col: "date", price_col: "close"}
            if vol_col:
                rename_map[vol_col] = "volume"
            df = df.rename(columns=rename_map)

            # 3) Dates (Investing fr : jour/mois/année)
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

            # 4) Prix : normaliser (virgule décimale, espaces, tirets)
            def parse_price(x):
                if pd.isna(x): return np.nan
                s = str(x)
                s = (s.replace("\u00a0","").replace(" ", "")
                       .replace("−","-").replace("–","-")
                       .replace(",", "."))   # virgule -> point
                s = re.sub(r"[^0-9\.\-]", "", s)
                try:
                    return float(s)
                except:
                    return np.nan

            df["close"] = df["close"].map(parse_price)

            # 5) Volume (K/M) facultatif
            if "volume" in df.columns:
                def parse_vol(v):
                    if pd.isna(v): return np.nan
                    s = str(v).replace("\u00a0","").replace(" ", "").lower()
                    mult = 1
                    if s.endswith("k"): mult, s = 1_000, s[:-1]
                    elif s.endswith("m"): mult, s = 1_000_000, s[:-1]
                    s = s.replace(",", ".")
                    s = re.sub(r"[^0-9\.\-]", "", s)
                    try:
                        return float(s) * mult
                    except:
                        return np.nan
                df["volume"] = df["volume"].map(parse_vol)

            # 6) Trier, retirer NaN de base
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            df = df.dropna(subset=["close"]).reset_index(drop=True)

            rows_out = len(df)
            st.info(f"📄 Lignes CSV : **{rows_in}** → après nettoyage : **{rows_out}**")

            # 7) Garde-fou d’historique
            need = max(int(sma_slow), int(rsi_period) + 5, 35)
            if len(df) < need:
                st.warning(
                    f"⚠️ Historique insuffisant pour SMA {sma_slow}/RSI {rsi_period}. "
                    f"Il reste **{len(df)}** lignes après nettoyage, besoin ≈ **{need}**. "
                    "→ Télécharge plus de données ou diminue SMA longue."
                )
                st.stop()

            # ---- Indicateurs
            df["RSI"]      = rsi_calc(df["close"], period=int(rsi_period))
            df["SMA_fast"] = sma(df["close"], int(sma_fast))
            df["SMA_mid"]  = sma(df["close"], int(sma_mid))
            df["SMA_slow"] = sma(df["close"], int(sma_slow))
            macd_line, macd_sig, macd_hist = macd_calc(df["close"])
            df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd_line, macd_sig, macd_hist

            # ---- Graphiques
            st.subheader("📊 Graphique")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close"))
            fig.add_trace(go.Scatter(x=df["date"], y=df["SMA_fast"], name=f"SMA{sma_fast}"))
            fig.add_trace(go.Scatter(x=df["date"], y=df["SMA_mid"],  name=f"SMA{sma_mid}"))
            fig.add_trace(go.Scatter(x=df["date"], y=df["SMA_slow"], name=f"SMA{sma_slow}"))
            fig.update_layout(height=420, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)

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

            # ---- Signaux & score technique
            last = df.iloc[-1]
            price_last    = float(last["close"])
            rsi_last      = float(last["RSI"])
            sma_mid_last  = float(last["SMA_mid"])
            sma_slow_last = float(last["SMA_slow"])
            macd_pos      = 1 if float(last["MACD_hist"]) > 0 else 0
            rsi_signal    = 1 if rsi_last <= 30 else (0 if rsi_last >= 70 else 0.5)
            sma_cross     = 1 if (price_last > sma_mid_last and sma_mid_last > sma_slow_last) else 0

            w_rsi, w_sma, w_macd = 30, 40, 30
            score_tech = round((rsi_signal*w_rsi + sma_cross*w_sma + macd_pos*w_macd) / (w_rsi+w_sma+w_macd) * 100, 0)

            df_sig = pd.DataFrame([{
                "Last Price": price_last,
                "RSI": rsi_last,
                f"SMA{int(sma_mid)}": sma_mid_last,
                f"SMA{int(sma_slow)}": sma_slow_last,
                "MACD>0": macd_pos,
                "RSI_signal(1/0/0.5)": rsi_signal,
                "SMA_cross(1/0)": sma_cross,
                "Tech Score (0-100)": score_tech
            }])

            st.subheader("🧪 Signaux techniques (instantané)")
            st.dataframe(df_sig.style.format("{:,.2f}"), use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")
            st.stop()

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

    st.subheader("📌 Résumé fondamental")
    st.dataframe(df_fonda_safe.style.format("{:,.2f}"), use_container_width=True)

    st.subheader("📌 Résumé technique")
    if 'df_sig' in locals() and df_sig is not None:
        st.dataframe(df_sig.style.format("{:,.2f}"), use_container_width=True)
        tech_score = float(df_sig["Tech Score (0-100)"].iloc[0])
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

