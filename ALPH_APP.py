
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

    # Nouvelle bloc 
    # =======================
    # FONDAMENTAL - SAISIE & RATIOS
    # =======================
    import math
    import numpy as np
    import pandas as pd
    import streamlit as st
    
    st.subheader("🧮 Saisie des états financiers (K MAD sauf mention)")
    
    colA, colB, colC = st.columns(3)
    
    with colA:
        ca = st.number_input("Chiffre d'affaires (CA)", min_value=0.0, value=0.0, step=100.0)
        cogs = st.number_input("Coût des ventes / Achats revendus (COGS)", min_value=0.0, value=0.0, step=100.0)
        ebitda_in = st.number_input("EBITDA (si connu, sinon 0)", min_value=0.0, value=0.0, step=100.0)
        ebit = st.number_input("Résultat d’exploitation (EBIT)", min_value=0.0, value=0.0, step=100.0)
        rn = st.number_input("Résultat net (RN)", min_value=-1e9, value=0.0, step=100.0)
    
    with colB:
        actifs_courants = st.number_input("Actif courant", min_value=0.0, value=0.0, step=100.0)
        passifs_courants = st.number_input("Passif courant", min_value=0.0, value=0.0, step=100.0)
        actifs_totaux = st.number_input("Total Actif", min_value=0.0, value=0.0, step=100.0)
        passifs_totaux = st.number_input("Total Passif (CP inclus)", min_value=0.0, value=0.0, step=100.0)
        cp = st.number_input("Capitaux propres (CP)", min_value=-1e9, value=0.0, step=100.0)
    
    with colC:
        dettes_fin = st.number_input("Dettes financières (LT + CB)", min_value=0.0, value=0.0, step=100.0)
        dettes_tot = st.number_input("Dettes totales (hors CP)", min_value=0.0, value=0.0, step=100.0)
        treso_actif = st.number_input("Trésorerie actif (cash)", min_value=0.0, value=0.0, step=100.0)
        treso_passif = st.number_input("Trésorerie passif (découverts)", min_value=0.0, value=0.0, step=100.0)
        interets = st.number_input("Charges d'intérêts (K MAD)", min_value=0.0, value=0.0, step=10.0)
    
    st.markdown("---")
    st.subheader("📈 Données de marché")
    colM1, colM2, colM3 = st.columns(3)
    with colM1:
        prix = st.number_input("Prix par action (MAD)", min_value=0.0, value=0.0, step=0.1)
    with colM2:
        nb_actions = st.number_input("Nombre d’actions (unités)", min_value=0.0, value=0.0, step=1000.0)
    with colM3:
        taux_impot = st.number_input("Taux d’impôt (%)", min_value=0.0, max_value=60.0, value=30.0, step=1.0)
    
    mcap = prix * nb_actions  # MAD si prix est en MAD et actions en unités
    
    # ---------- Agrégats & garde-fous ----------
    CA = max(1.0, ca)
    COGS = max(0.0, cogs)
    
    # Marge brute : si COGS=0 et ebitda/ebit renseignés, on l’approxime à défaut
    marge_brute_amount = (CA - COGS) if COGS > 0 else max(0.0, ebit + (ebitda_in - ebit) if ebitda_in > 0 and ebit > 0 else CA*0.0)
    
    EBIT = ebit
    EBITDA = ebitda_in if ebitda_in > 0 else (EBIT + 0.0)  # si pas d'info, on ne fabrique pas d’EBITDA
    
    # Dette nette
    dette_nette = dettes_fin + treso_passif - treso_actif
    
    # ---------- Ratios demandés ----------
    def safe_div(a, b):
        b = float(b)
        return (float(a) / b) if abs(b) > 1e-9 else np.nan
    
    ratios = {}
    
    # Structure/solvabilité
    ratios["Dettes totales / Capitaux propres"] = safe_div(dettes_tot, cp)
    ratios["Autonomie financière (CP / Passif total)"] = safe_div(cp, passifs_totaux)
    ratios["Ratio dettes financières (Dettes fin. / Passif total)"] = safe_div(dettes_fin, passifs_totaux)
    ratios["Solvabilité (CP / Actif total)"] = safe_div(cp, actifs_totaux)
    
    # Liquidité
    ratios["Liquidité générale (Current ratio)"] = safe_div(actifs_courants, passifs_courants)
    ratios["Quick ratio ((Actif courant - Stocks)/Passif courant)"] = np.nan  # on n’a pas 'stocks' ici
    
    # Profitabilité / marges
    ratios["Marge brute"] = safe_div(marge_brute_amount, CA)
    ratios["Marge d’exploitation (EBIT)"] = safe_div(EBIT, CA)
    ratios["Marge opérationnelle"] = safe_div(EBIT, CA)  # assimilée ici à EBIT/CA
    ratios["Marge nette"] = safe_div(rn, CA)
    
    # Couverture intérêts
    ratios["Couverture des intérêts (EBIT / Intérêts)"] = safe_div(EBIT, interets)
    
    # ROE / ROA
    ratios["ROE (RN/CP)"] = safe_div(rn, cp)
    ratios["ROA (RN/Actif total)"] = safe_div(rn, actifs_totaux)
    
    # Levier / Gearing
    ratios["Gearing (Dette nette / CP)"] = safe_div(dette_nette, cp)
    
    # Actif courant / Passif total hors CP (interprétation : passif total - CP)
    passif_hors_cp = passifs_totaux - cp
    ratios["Actif courant / Passif total (hors CP)"] = safe_div(actifs_courants, passif_hors_cp)
    
    # Valorisation boursière
    # PER = Price / EPS ; EPS = RN (K MAD) / actions  => convertir en MAD/action
    eps = (rn * 1000.0) / nb_actions if nb_actions > 0 else np.nan  # rn K MAD -> MAD ; EPS en MAD
    ratios["PER (Prix/EPS)"] = safe_div(prix, eps)
    
    # Price/Sales = Market Cap / CA (converti en MAD)
    sales_mad = ca * 1000.0
    ratios["Price/Sales (P/S)"] = safe_div(mcap, sales_mad)
    
    # P/B = Market Cap / Capitaux propres (converti en MAD)
    book_mad = cp * 1000.0
    ratios["P/B (Price/Book)"] = safe_div(mcap, book_mad)
    
    # EBITDA & EV
    ratios["EBITDA (K MAD)"] = EBITDA
    EV = mcap + (dettes_fin*1000.0 + treso_passif*1000.0) - (treso_actif*1000.0)
    ratios["Enterprise Value (MAD)"] = EV
    
    # ---------- Valeur intrinsèque (2 méthodes simples) ----------
    
    st.markdown("---")
    st.subheader("📐 Valeur intrinsèque (approches rapides)")
    
    c1, c2 = st.columns(2)
    
    # A) Gordon sur Free Cash-Flow approx. : FCF ≈ EBITDA*(1 - T) – Intérêts*(1-T) – Capex – ΔBFR
    with c1:
        st.markdown("**Méthode DCF (Gordon simplifié)**")
        capex = st.number_input("Capex annuel (K MAD)", min_value=0.0, value=0.0, step=100.0, key="capex")
        delta_bfr = st.number_input("ΔBFR annuel (K MAD)", min_value=-1e9, value=0.0, step=100.0, key="dbfr")
        g = st.number_input("Croissance à long terme g (%)", min_value=0.0, max_value=8.0, value=2.0, step=0.5, key="g")
        wacc = st.number_input("WACC (%)", min_value=1.0, max_value=25.0, value=12.0, step=0.5, key="wacc")
        t = taux_impot/100.0
        g_rt = g/100.0
        wacc_rt = wacc/100.0
    
        if EBITDA > 0:
            fcf_k = EBITDA*(1 - t) - interets*(1 - t) - capex - delta_bfr  # K MAD
        else:
            # fallback grossier si pas d'EBITDA
            fcf_k = rn  # K MAD
    
        vi_dcf_mad = np.nan
        if wacc_rt > g_rt and fcf_k > 0:
            vi_dcf_mad = (fcf_k*1000.0) * (1 + g_rt) / (wacc_rt - g_rt)  # MAD
    
        valeur_par_action_dcf = vi_dcf_mad/nb_actions if (nb_actions>0 and not math.isnan(vi_dcf_mad)) else np.nan
        st.metric("Valeur DCF par action (MAD)", f"{valeur_par_action_dcf:,.2f}" if valeur_par_action_dcf==valeur_par_action_dcf else "n/d")
    
    # B) Méthode des multiples (cible sectorielle)
    with c2:
        st.markdown("**Méthode des multiples**")
        pe_cible = st.number_input("P/E cible secteur", min_value=0.0, value=12.0, step=0.5, key="pe")
        ps_cible = st.number_input("P/S cible secteur", min_value=0.0, value=1.0, step=0.1, key="ps")
        pb_cible = st.number_input("P/B cible secteur", min_value=0.0, value=1.2, step=0.1, key="pb")
    
        vi_pe = (pe_cible * eps) if eps==eps else np.nan
        vi_ps = (ps_cible * (sales_mad/nb_actions)) if nb_actions>0 else np.nan
        vi_pb = (pb_cible * (book_mad/nb_actions)) if nb_actions>0 else np.nan
    
        vi_mult_mad = np.nanmean([vi for vi in [vi_pe, vi_ps, vi_pb] if vi==vi])  # moyenne des valeurs par action MAD
        st.metric("Valeur par action (moy. multiples)", f"{vi_mult_mad:,.2f}" if vi_mult_mad==vi_mult_mad else "n/d")
    
    # ---------- Affichage table ----------
    st.markdown("---")
    st.subheader("📊 Ratios calculés")
    
    df = pd.DataFrame([
        {"Ratio": k, "Valeur": v if not isinstance(v, float) else (v if "MAD" in k or "K MAD" in k else (v*100 if "Marge" in k else v))}
        for k, v in ratios.items()
    ])
    
    def _fmt(k, v):
        if isinstance(v, float) and not math.isnan(v):
            if "Marge" in k or "ROE" in k or "ROA" in k or "Autonomie" in k or "Solvabilité" in k:
                return f"{v*100:.2f} %"
            if "PER" in k or "P/S" in k or "P/B" in k or "Gearing" in k \
               or "Couverture" in k or "Current" in k or "Actif courant" in k \
               or "Dettes totales" in k or "Ratio dettes financières" in k:
                return f"{v:.2f}"
            if "EBITDA" in k:
                return f"{v:,.0f} K"
            if "Enterprise Value" in k:
                return f"{v:,.0f} MAD"
            return f"{v:,.2f}"
        return "n/d"
    
    df["Formaté"] = [ _fmt(r["Ratio"], r["Valeur"]) for _, r in df.iterrows() ]
    st.dataframe(df[["Ratio","Formaté"]], use_container_width=True)
    
    # ---------- Verdict simple (facultatif) ----------
    st.markdown("---")
    st.subheader("🧠 Verdict rapide (règles simples)")
    
    flags = []
    current_ok = ratios["Liquidité générale (Current ratio)"]
    couverture = ratios["Couverture des intérêts (EBIT / Intérêts)"]
    autonomie = ratios["Autonomie financière (CP / Passif total)"]
    gearing = ratios["Gearing (Dette nette / CP)"]
    
    if (cp <= 0):
        flags.append("Capitaux propres négatifs")
    if (current_ok is not np.nan) and current_ok < 1.0:
        flags.append("Liquidité insuffisante (Current < 1)")
    if (couverture is not np.nan) and couverture < 1.0:
        flags.append("Couverture des intérêts < 1")
    if not math.isnan(gearing) and gearing > 1.5:
        flags.append("Gearing élevé (>1.5)")
    
    if flags:
        st.error("⚠️ Drapeaux rouges : " + " | ".join(flags))
    else:
        st.success("✅ Pas de drapeau rouge majeur sur ces indicateurs (à confirmer avec tendances pluriannuelles).")


