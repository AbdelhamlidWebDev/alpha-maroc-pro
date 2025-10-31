
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analyseur Fondamental – Alpha Maroc", layout="wide")

def interpret_fundamentals(per, pb, roe, net_margin, ev_ebitda):
    msgs = []
    def tag(txt, lvl="info"):
        palette = {"good":"🟢","ok":"🟡","warn":"🟠","bad":"🔴","info":"ℹ️"}
        return f"{palette.get(lvl,'ℹ️')} {txt}"
    if per==per:
        if per < 10:  msgs.append(tag(f"PER {per:.1f}× : décote potentielle", "good"))
        elif per <= 25: msgs.append(tag(f"PER {per:.1f}× : zone raisonnable", "ok"))
        else: msgs.append(tag(f"PER {per:.1f}× : valorisation élevée", "warn"))
    if pb==pb:
        if pb <= 1:   msgs.append(tag(f"P/B {pb:.2f}× : proche/inf. valeur comptable", "good"))
        elif pb <= 3: msgs.append(tag(f"P/B {pb:.2f}× : acceptable", "ok"))
        else:         msgs.append(tag(f"P/B {pb:.2f}× : cher (exige rentabilité)", "warn"))
    if roe==roe:
        lvl = "good" if roe>15 else ("ok" if roe>=8 else "bad")
        lab = "excellent" if roe>15 else "correct" if roe>=8 else "faible"
        msgs.append(tag(f"ROE {roe:.1f}% : {lab}", lvl))
    if net_margin==net_margin:
        lvl = "good" if net_margin>15 else ("ok" if net_margin>8 else "warn")
        lab = "confortable" if net_margin>15 else "correcte" if net_margin>8 else "tendue"
        msgs.append(tag(f"Marge nette {net_margin:.1f}% : {lab}", lvl))
    if ev_ebitda==ev_ebitda:
        if ev_ebitda < 6:    msgs.append(tag(f"EV/EBITDA {ev_ebitda:.1f}× : attractif", "good"))
        elif ev_ebitda<=12: msgs.append(tag(f"EV/EBITDA {ev_ebitda:.1f}× : normal", "ok"))
        else:               msgs.append(tag(f"EV/EBITDA {ev_ebitda:.1f}× : exige croissance/qualité", "warn"))
    return msgs

def simulate_order(price, qty, direction, courtage=0.006, reg_liv=0.002, bourse=0.001, min_fees=10.0, tva=0.1, tax_pv=0.1, sell_price=None):
    montant_brut = price * qty
    fees_ht = max(min_fees, montant_brut * (courtage + reg_liv + bourse))
    fees_ttc = fees_ht * (1 + tva)
    if direction == "achat":
        net = montant_brut + fees_ttc
        return {"montant_brut": montant_brut, "frais_ht": fees_ht, "tva": fees_ht*tva, "net": net}
    else:
        assert sell_price is not None
        brut_sell = sell_price * qty
        fees_sell_ht = max(min_fees, brut_sell * (courtage + reg_liv + bourse))
        fees_sell_ttc = fees_sell_ht * (1 + tva)
        pv = max(0.0, brut_sell - montant_brut)
        impots = pv * tax_pv
        net = brut_sell - fees_sell_ttc - impots
        return {"montant_brut": brut_sell, "frais_ht": fees_sell_ht, "tva": fees_sell_ht*tva, "impot_pv": impots, "net": net}

st.title("📊 Analyseur Fondamental – Alpha Maroc")

with st.sidebar:
    st.header("Paramètres de l'entreprise")
    price = st.number_input("Prix actuel (DH)", value=126.50, min_value=0.0, step=0.01)
    shares = st.number_input("Actions en circulation", value=17695000, min_value=0, step=1000)
    revenue = st.number_input("Chiffre d'affaires (DH)", value=373400000.0, min_value=0.0, step=1000.0)
    net_income = st.number_input("Résultat net (DH)", value=44642000.0, min_value=0.0, step=1000.0)
    total_assets = st.number_input("Total actif (DH)", value=468000000.0, min_value=0.0, step=1000.0)
    equity = st.number_input("Capitaux propres (DH)", value=300000000.0, min_value=0.0, step=1000.0)
    ebitda = st.number_input("EBITDA (DH)", value=70000000.0, min_value=0.0, step=1000.0)

market_cap = price * shares
eps = (net_income / shares) if shares else np.nan
per = (price / eps) if eps and eps!=0 else np.nan
bvps = (equity / shares) if shares else np.nan
pb = (price / bvps) if bvps and bvps!=0 else np.nan
roe = (net_income / equity * 100) if equity else np.nan
roa = (net_income / total_assets * 100) if total_assets else np.nan
ev = market_cap
ev_ebitda = (ev / ebitda) if ebitda else np.nan
net_margin = (net_income / revenue * 100) if revenue else np.nan

df = pd.DataFrame([{
    "Market Cap (DH)": market_cap,
    "EPS (DH)": eps,
    "PER": per,
    "BVPS": bvps,
    "P/B": pb,
    "ROE %": roe,
    "ROA %": roa,
    "EV/EBITDA": ev_ebitda,
    "Net Margin %": net_margin
}])
st.subheader("📈 Résultats financiers")
st.dataframe(df.style.format({"Market Cap (DH)":"{:,.2f}","EPS (DH)":"{:,.2f}","PER":"{:,.2f}","BVPS":"{:,.2f}","P/B":"{:,.2f}","ROE %":"{:,.2f}","ROA %":"{:,.2f}","EV/EBITDA":"{:,.2f}","Net Margin %":"{:,.2f}"}), use_container_width=True)

st.subheader("🧠 Interprétations")
for m in interpret_fundamentals(per, pb, roe, net_margin, ev_ebitda):
    st.markdown(m)

st.subheader("💸 Simulateur d'ordre (Bourse de Casablanca)")
col1, col2, col3 = st.columns(3)
with col1:
    qty = st.number_input("Quantité", value=3, min_value=1, step=1)
with col2:
    var_pct = st.number_input("Variation prix pour vente (%)", value=5.0, step=0.5)
with col3:
    tax_pv = st.number_input("Impôt sur plus-value (%)", value=10.0, step=0.5) / 100.0

achat = simulate_order(price, qty, "achat")
sell_price = price * (1 + var_pct/100.0)
vente = simulate_order(price, qty, "vente", tax_pv=tax_pv, sell_price=sell_price)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Achat**")
    st.write({k: round(v,2) for k,v in achat.items()})
with c2:
    st.markdown("**Vente (après variation)**")
    st.write({k: round(v,2) for k,v in vente.items()})
gain_net = vente["net"] - achat["net"]
st.success(f"Gain net estimé : {gain_net:,.2f} DH".replace(",", " "))
st.caption("Frais par défaut : 0,60% courtage + 0,20% R/L + 0,10% Bourse (min 10 DH) + TVA 10%. Ajustez dans le code selon votre courtier.")



#nouvelle




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
# 📈 Onglet TECHNIQUE
# ==========================================================
with tabs[1]:
    st.markdown("Charge un **CSV Investing.com** (Données historiques). Colonnes attendues : **Date**, **Close/Price**, **Volume**.")
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

    df_sig = None
    if file:
        df = pd.read_csv(file)
        # Essayer de reconnaître les colonnes
        date_col = next((c for c in df.columns if c.lower().startswith("date")), None)
        price_col = next((c for c in df.columns if c.lower() in ["close","prix","price","dernier","last"]), None)
        volume_col = next((c for c in df.columns if "vol" in c.lower()), None)

        if not date_col or not price_col:
            st.error("Colonnes non reconnues. Assure-toi d’avoir au moins **Date** et **Close/Price**.")
        else:
            df.rename(columns={date_col:"date", price_col:"close"}, inplace=True)
            if volume_col: df.rename(columns={volume_col:"volume"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            df["RSI"] = rsi(df["close"], rsi_period)
            df["SMA_fast"] = sma(df["close"], sma_fast)
            df["SMA_mid"]  = sma(df["close"], sma_mid)
            df["SMA_slow"] = sma(df["close"], sma_slow)
            macd_line, macd_signal, macd_hist = macd(df["close"])
            df["MACD"] = macd_line
            df["MACD_signal"] = macd_signal
            df["MACD_hist"] = macd_hist

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

            # Signaux & score technique
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

# ==========================================================
# 🧠 Onglet RECOMMANDATION & EXPORT
# ==========================================================
with tabs[2]:
    st.markdown("Synthèse des signaux **Fondamentaux + Techniques** et **export Excel**.")
    # Recrée df_fonda localement si l'utilisateur n'a pas visité le premier onglet (sécurité)
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

    # Score fondamental simple (paramétrable)
    score_fonda = 0
    # +20 si PER 10–25, +10 si <10
    if per and per>0:
        score_fonda += 20 if 10<=per<=25 else (10 if per<10 else 0)
    # +25 si ROE > 15, +15 si 8–15
    if roe and roe>0:
        score_fonda += 25 if roe>15 else (15 if roe>=8 else 0)
    # +15 si marge nette >10
    if net_margin and net_margin>10: score_fonda += 15
    # -10 si P/B > 3 (cher)
    if pb and pb>3: score_fonda -= 10
    score_fonda = max(0, min(50, score_fonda))  # borne 0..50

    # Score global : 50% fonda + 50% tech
    global_score = None
    if not np.isnan(tech_score):
        global_score = round(score_fonda + (tech_score/2), 0)  # fonda sur 50 + tech/2 (50) = /100

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
    if 'file' in locals() and file:
        include_hist = st.checkbox("Inclure l'historique de prix dans l'Excel", value=True)

    if st.button("📥 Télécharger le rapport Excel"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_fonda_safe.to_excel(writer, sheet_name="Fondamental", index=False)
            if 'df_sig' in locals() and df_sig is not None:
                df_sig.to_excel(writer, sheet_name="Technique_Signaux", index=False)
            if include_hist:
                # Relire le CSV (si chargé) pour l’export
                file.seek(0)
                pd.read_csv(file).to_excel(writer, sheet_name="Prix_Historique", index=False)
        st.download_button(
            label="⬇️ Télécharger AlphaMaroc_Report.xlsx",
            data=output.getvalue(),
            file_name="AlphaMaroc_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.success("Rapport prêt ✅")
