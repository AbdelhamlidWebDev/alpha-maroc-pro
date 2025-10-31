# Alpha Maroc Pro – Analyseur Fondamental & Simulateur (Streamlit)

Application Streamlit pour analyser rapidement une société cotée (ratios fondamentaux) et
simuler un ordre d'achat/vente à la Bourse de Casablanca (frais, TVA et impôt sur plus-value).

## Fonctionnalités
- Saisie des données : prix, actions en circulation, CA, résultat net, actifs, capitaux propres, EBITDA.
- Calculs automatiques : Market Cap, EPS, PER, BVPS, P/B, ROE, ROA, EV/EBITDA, marge nette.
- Interprétations lisibles (couleurs et messages).
- Simulateur d’ordre **achat/vente** avec frais par défaut (0,60% + 0,20% + 0,10%, min 10 DH, TVA 10%) et impôt PV.
- Interface responsive (sidebar + tableau de résultats).

## Déploiement (Streamlit Cloud)
1. Pousser `ALPH_APP.py` et `requirements.txt` dans un dépôt GitHub public.
2. Aller sur https://share.streamlit.io → Create app → sélectionner le repo.
3. Main file path : `ALPH_APP.py` → Deploy

## Local
```bash
pip install -r requirements.txt
streamlit run ALPH_APP.py
```
