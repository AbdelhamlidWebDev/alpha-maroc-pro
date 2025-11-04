

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


