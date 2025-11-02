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






