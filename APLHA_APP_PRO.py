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
            # sep=None + engine='python' -> auto-détection ; utile pour CSV fr
            df = pd.read_csv(file, sep=None, engine="python")
        except Exception as e:
            st.error(f"Impossible de lire ce fichier CSV : {e}")
            st.stop()

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
