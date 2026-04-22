import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats as scipy_stats
from datetime import date

API_URL = "http://127.0.0.1:8000/global_predict"

st.set_page_config(layout="wide")
st.title("Bitcoin — Signal ML")

# ── SIDE PANEL ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Paramètres")

    date_pivot = st.date_input("Date cutoff training", value=date(2024, 1, 1), min_value=date(2023, 11, 4), max_value=date.today())
    date_pivot = date_pivot.strftime("%Y-%m-%d")

    st.divider()
    st.subheader("Stratégies")

    show_signal = st.checkbox("Signal ML (threshold 0.5)", value=True)
    show_flex = st.checkbox("Signal personnalisé (threshold ajustable)", value=True)
    show_long_only = st.checkbox("Long only", value=True)
    show_random = st.checkbox("Random", value=False)

    st.divider()
    st.subheader("Seuil de proba pour signal d'achat")
    threshold = st.slider("Seuil", min_value=0.5, max_value=0.75, value=0.5, step=0.005)

    horizon = 5

    st.divider()
    st.subheader("Modèle")
    model_name = st.selectbox(
    "Modèle",
    options=["xgb", "rnn", "linear"],
    format_func=lambda x: {"xgb": "XGBoost", "rnn": "RNN", "linear": "Régression linéaire"}[x]
    )

    load = st.button("Charger les données", type="primary")

# ── DATA LOADING ─────────────────────────────────────────────────
if load:
    response = requests.get(API_URL, params={"date_pivot": date_pivot, "model_name": model_name})
    data = response.json()
    df = pd.DataFrame(data["df_for_streamlit"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    st.session_state["df"] = df

# ── CHARTS ───────────────────────────────────────────────────────
if "df" in st.session_state:
    df = st.session_state["df"]

    # Calcul des stratégies
    def simulate_portfolio(df, mode, horizon, threshold=0.5):
        n = len(df)

        # Décision par batch (tous les horizon jours)
        decisions = []
        for i in range(0, n, horizon):
            row = df.iloc[i]
            end_idx = min(i + horizon, n - 1)
            price_end = df.iloc[end_idx]["Close"]

            if mode == "long_only":
                invest = True
            elif mode == "signal":
                invest = float(row["probability"]) >= 0.5
            elif mode == "flex signal":
                invest = float(row["probability"]) >= threshold
            elif mode == "random":
                import random
                random.seed(i)
                invest = random.random() > 0.5

            for j in range(i, min(i + horizon, n)):
                decisions.append(invest)

        decisions = decisions[:n]

        # Portfolio évolue jour par jour selon la décision
        portfolio = 100.0
        values = [100.0]
        for i in range(1, n):
            daily_ret = (df.iloc[i]["Close"] - df.iloc[i-1]["Close"]) / df.iloc[i-1]["Close"]
            if decisions[i]:
                portfolio *= (1 + daily_ret)
            values.append(portfolio)

        return values

    fig = go.Figure()

    # Prix BTC en base 100
    btc_base100 = df["Close"] / df["Close"].iloc[0] * 100
    fig.add_trace(go.Scatter(
        x=df["Date"], y=btc_base100,
        name="Prix BTC (base 100)",
        line=dict(color="#378ADD", width=1.5, dash="dot"),
        opacity=0.5
    ))

    strat_colors = {
        "signal": "#EF9F27",
        "long_only": "#5B8DEF",
        "flex signal": "#1D9E75",
        "random": "#888780"
    }
    strat_labels = {
        "signal": "Signal ML",
        "long_only": "Long only",
        "flex signal": "Signal personnalisé",
        "random": "Random"
    }
    active = {
        "signal": show_signal,
        "flex signal": show_flex,
        "long_only": show_long_only,
        "random": show_random
    }

    for mode, show in active.items():
        if show:
            vals = simulate_portfolio(df, mode, horizon, threshold)
            fig.add_trace(go.Scatter(
                x=df["Date"], y=vals,
                name=strat_labels[mode],
                line=dict(color=strat_colors[mode], width=2)
            ))

    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.4)

    fig.update_layout(
        template="plotly_dark",
        title="Performance des stratégies (base 100)",
        height=500,
        xaxis_title="Date",
        yaxis_title="Valeur (base 100)",
        legend=dict(orientation="h", y=-0.25)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Nuage de points : probabilité vs mouvement réel
    st.subheader("Probabilité du signal vs mouvement réel")

    df_scatter = df.copy()
    df_scatter["real_return"] = df_scatter["Close"].shift(-horizon) / df_scatter["Close"] - 1
    df_scatter = df_scatter.dropna(subset=["real_return"])

    x_all = y_all_orig = df_scatter["probability"].values  # proba en x
    y_all = df_scatter["real_return"].values * 100          # rendement en y

    mask_above = x_all >= threshold
    mask_below = x_all < threshold

    slope_all, intercept_all, _, _, _ = scipy_stats.linregress(x_all, y_all)
    slope_above, intercept_above, _, _, _ = scipy_stats.linregress(x_all[mask_above], y_all[mask_above])
    slope_below, intercept_below, _, _, _ = scipy_stats.linregress(x_all[mask_below], y_all[mask_below])

    x_line = np.linspace(x_all.min(), x_all.max(), 200)

    fig2 = go.Figure()

    # Points
    fig2.add_trace(go.Scatter(
        x=x_all[mask_above], y=y_all[mask_above],
        mode="markers",
        marker=dict(color="#1D9E75", size=3, opacity=0.6),
        text=df_scatter[mask_above]["Date"].dt.strftime("%Y-%m-%d"),
        hovertemplate="Date: %{text}<br>Proba: %{x:.2f}<br>Rendement: %{y:.1f}%<extra></extra>",
        name="Investi"
    ))

    fig2.add_trace(go.Scatter(
        x=x_all[mask_below], y=y_all[mask_below],
        mode="markers",
        marker=dict(color="#E24B4A", size=3, opacity=0.6),
        text=df_scatter[mask_below]["Date"].dt.strftime("%Y-%m-%d"),
        hovertemplate="Date: %{text}<br>Proba: %{x:.2f}<br>Rendement: %{y:.1f}%<extra></extra>",
        name="Cash"
    ))

    # Régression générale
    fig2.add_trace(go.Scatter(
        x=x_line, y=slope_all * x_line + intercept_all,
        mode="lines", line=dict(color="#378ADD", width=1.5, dash="dot"),
        name="Régression générale"
    ))

    # Régression linéaire investi
    fig2.add_trace(go.Scatter(
        x=x_line, y=slope_above * x_line + intercept_above,
        mode="lines", line=dict(color="#1D9E75", width=2),
        name=f"Régression investi (≥ {threshold})"
    ))

    # Régression linéaire cash
    fig2.add_trace(go.Scatter(
        x=x_line, y=slope_below * x_line + intercept_below,
        mode="lines", line=dict(color="#E24B4A", width=2),
        name=f"Régression cash (< {threshold})"
    ))

    # # Boxplot cash — rendements réels sur yaxis2
    # fig2.add_trace(go.Box(
    #     y=x_all[mask_below],
    #     x=["Cash"] * mask_below.sum(),
    #     name="Cash — rendements réels",
    #     marker_color="#E24B4A",
    #     boxmean=True,
    #     width=0.3,
    #     xaxis="x2",
    #     yaxis="y2"
    # ))

    # # Boxplot investi — rendements réels sur yaxis2
    # fig2.add_trace(go.Box(
    #     y=x_all[mask_above],
    #     x=["Investi"] * mask_above.sum(),
    #     name="Investi — rendements réels",
    #     marker_color="#1D9E75",
    #     boxmean=True,
    #     width=0.3,
    #     xaxis="x2",
    #     yaxis="y2"
    # ))

    fig2.add_vline(x=threshold, line_dash="dash", line_color="#1D9E75", opacity=0.7,
                   annotation_text=f"Threshold ({threshold})", annotation_position="top right")
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)

    fig2.update_layout(
        template="plotly_dark",
        height=500,
        xaxis=dict(title="Probabilité du signal", domain=[0, 0.78]),
        xaxis2=dict(domain=[0.82, 1]),
        yaxis=dict(title="Mouvement réel à t+horizon (%)"),
        yaxis2=dict(title="Rendement réel (%)", anchor="x2"),
        legend=dict(orientation="h", y=-0.2),
        boxmode="group"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Résumé
    st.subheader("Résumé")
    n_buy = int((df["prediction"] == 1).sum())
    n_total = len(df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Signaux achat", f"{n_buy}/{n_total}")
    col2.metric("Probabilité moyenne", f"{df['probability'].mean():.2%}")
    col3.metric("Dernière prédiction", "ACHETER" if df['prediction'].iloc[-1] == 1 else "CASH")

    st.divider()
    st.dataframe(
        df[["Date", "Close", "prediction", "probability"]].tail(20),
        use_container_width=True
    )


# ── STATS DE PERFORMANCE ─────────────────────────────────────
    st.subheader("Statistiques de performance")

    import numpy as np
    from scipy import stats as scipy_stats

    TRADING_DAYS_PER_YEAR = 365  # crypto = 365j

    def compute_stats(values, label):
        s = pd.Series(values)
        daily_ret = s.pct_change().dropna()
        n = len(daily_ret)
        if n == 0:
            return {}

        # Rendement annualisé géométrique
        wealth = (1 + daily_ret).prod()
        ann_ret = wealth ** (TRADING_DAYS_PER_YEAR / n) - 1

        # Volatilité annualisée
        ann_vol = daily_ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Sharpe (rf=0)
        sharpe = (daily_ret.mean() / daily_ret.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR) if daily_ret.std() > 0 else np.nan

        # Max drawdown
        w = s / s.iloc[0]
        dd = w / w.cummax() - 1
        mdd = dd.min()

        # Skewness / Kurtosis
        skew = scipy_stats.skew(daily_ret, bias=False)
        kurt = scipy_stats.kurtosis(daily_ret, fisher=True, bias=False)

        # Return total
        total_ret = (s.iloc[-1] / s.iloc[0] - 1)

        return {
            "Stratégie": label,
            "Perf. totale": f"{total_ret:.1%}",
            "Perf. annualisée": f"{ann_ret:.1%}",
            "Volatilité ann.": f"{ann_vol:.1%}",
            "Sharpe": f"{sharpe:.2f}",
            "Max drawdown": f"{mdd:.1%}",
            "Skewness": f"{skew:.2f}",
            "Kurtosis": f"{kurt:.2f}",
        }

    stats_rows = []
    strat_vals = {
        "Prix BTC": btc_base100.tolist(),
    }
    if show_signal:
        strat_vals["Signal ML"] = simulate_portfolio(df, "signal", horizon, threshold)
        strat_vals["Signal personnalisé"] = simulate_portfolio(df, "flex signal", horizon, threshold)
    if show_long_only:
        strat_vals["Long only"] = simulate_portfolio(df, "long_only", horizon, threshold)
    if show_random:
        strat_vals["Random"] = simulate_portfolio(df, "random", horizon, threshold)

    for label, vals in strat_vals.items():
        stats_rows.append(compute_stats(vals, label))

    stats_df = pd.DataFrame(stats_rows).set_index("Stratégie")
    st.dataframe(stats_df, use_container_width=True)

    # ── PERFORMANCES MENSUELLES ──────────────────────────────────
    st.subheader("Performances mensuelles — Signal ML")

    signal_vals = simulate_portfolio(df, "flex signal", horizon, threshold)
    perf_df = pd.DataFrame({
        "Date": df["Date"].values,
        "value": signal_vals
    }).set_index("Date")
    perf_df.index = pd.DatetimeIndex(perf_df.index)

    month_end = perf_df.resample("ME").last()
    monthly_ret = month_end["value"].pct_change().dropna()
    monthly_ret.index = monthly_ret.index.to_period("M")

    pivot = pd.DataFrame({
        "year": monthly_ret.index.year,
        "month": monthly_ret.index.month,
        "ret": monthly_ret.values
    }).pivot(index="year", columns="month", values="ret")

    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
    annual = monthly_ret.groupby(monthly_ret.index.year).apply(lambda x: (1+x).prod()-1)
    pivot["Year"] = annual.values
    pivot = (pivot * 100).round(2)

    st.dataframe(
        pivot.style.background_gradient(cmap="RdYlGn", axis=None, vmin=-15, vmax=15),
        use_container_width=True
    )
