
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
    price = st.number_input("Prix actuel (DH)", value=00.00, min_value=0.0, step=0.01)
    shares = st.number_input("Actions en circulation", value=00, min_value=0, step=1000)
    revenue = st.number_input("Chiffre d'affaires (DH)", value=00.0, min_value=0.0, step=1000.0)
    net_income = st.number_input("Résultat net (DH)", value=00.0, min_value=0.0, step=1000.0)
    total_assets = st.number_input("Total actif (DH)", value=00.0, min_value=0.0, step=1000.0)
    equity = st.number_input("Capitaux propres (DH)", value=00.0, min_value=0.0, step=1000.0)
    ebitda = st.number_input("EBITDA (DH)", value=00.0, min_value=0.0, step=1000.0)

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

def compute_metrics(row):
    m = {}

    # --- bases sûres ---
    CA = max(1.0, float(row.get("chiffre_affaires", 0)))
    EBIT = float(row.get("resultat_exploitation", 0))
    RN = float(row.get("resultat_net", 0))
    CP = float(row.get("capitaux_propres", 0))
    AT = max(1.0, float(row.get("actif_total", 0)))
    DF = float(row.get("dettes_financement", 0))
    CF = float(row.get("charges_financieres", 0))
    PF = float(row.get("produits_financiers", 0))

    stocks = float(row.get("stocks_net", 0))
    clients = float(row.get("clients", 0))
    autres_creances = float(row.get("autres_creances", 0))
    treso_actif = float(row.get("tresorerie_actif", 0))
    treso_passif = float(row.get("tresorerie_passif", 0))
    fournisseurs = float(row.get("fournisseurs", 0))
    autres_dettes_ct = float(row.get("autres_dettes_ct", 0))
    immobs = float(row.get("immobilisations_net", 0))

    # --- profitabilité ---
    m["marge_nette"] = RN / CA
    m["marge_ebit"] = EBIT / CA
    m["roe"] = (RN / CP) if CP > 0 else np.nan
    m["roa"] = RN / AT

    # --- structure ---
    dette_nette = DF + treso_passif - treso_actif
    m["dette_nette"] = dette_nette
    m["gearing"] = (dette_nette / CP) if CP > 0 else np.nan
    m["couverture_interets"] = EBIT / max(1.0, CF)

    actif_circ = stocks + clients + autres_creances + treso_actif
    passif_circ = fournisseurs + autres_dettes_ct + treso_passif
    m["current_ratio"] = actif_circ / max(1.0, passif_circ)
    m["quick_ratio"] = (actif_circ - stocks) / max(1.0, passif_circ)
    m["wcr"] = (stocks + clients + autres_creances) - (fournisseurs + autres_dettes_ct)
    m["frng"] = (CP + DF) - immobs

    # --- efficacité (si achats fournis) ---
    achats = float(row.get("achats_revendus", 0)) or float(row.get("achats_revendus_marchandises", 0))
    cout_ventes = max(1.0, achats)
    m["dso"] = clients / CA * 365
    m["dio"] = stocks / cout_ventes * 365
    m["dpo"] = fournisseurs / cout_ventes * 365
    m["ccc"] = m["dso"] + m["dio"] - m["dpo"]

    # --- score simple ---
    score = 0; total = 0

    def add(points, cond):
        nonlocal score, total
        total += points
        if cond: score += points

    add(10, m["marge_nette"] > 0.05)
    add(10, m["marge_ebit"] > 0.07)
    add(10, (not np.isnan(m["roe"])) and m["roe"] > 0.12)
    add(10, m["current_ratio"] >= 1.2)
    add(10, m["quick_ratio"] >= 1.0)
    add(10, (not np.isnan(m["gearing"])) and m["gearing"] <= 1.0)
    add(10, m["couverture_interets"] >= 2.0)
    add(10, m["ccc"] <= 120)  # délai de conversion du cash raisonnable

    m["score"] = round(100 * score / max(1, total), 1)
    m["red_flags"] = []
    if CP <= 0: m["red_flags"].append("Capitaux propres négatifs")
    if m["couverture_interets"] < 1: m["red_flags"].append("Couverture intérêts < 1")
    if m["current_ratio"] < 1: m["red_flags"].append("Liquidité court terme < 1")

    return m



