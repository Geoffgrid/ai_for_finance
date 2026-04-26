"""
Backtest Module
Applies a binary signal (long=1 / flat=0) to a price series
and computes comprehensive financial performance metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class Backtest:
    """
    Simple long/flat backtest engine.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices of the underlying (e.g. SPX)
    signals : pd.Series
        Binary signal aligned to prices index: 1=long, 0=flat
    transaction_cost_bps : float
        One-way transaction cost in basis points (default 5 bps)
    """

    def __init__(
        self,
        prices: pd.Series,
        signals: pd.Series,
        transaction_cost_bps: float = 5.0,
    ):
        self.prices = prices.copy()
        self.signals = signals.copy()
        self.tc = transaction_cost_bps / 10_000
        self._results = None

    def run(self) -> "Backtest":
        """Compute strategy and benchmark return series."""
        idx = self.prices.index.intersection(self.signals.index)
        prices = self.prices.reindex(idx)
        signals = self.signals.reindex(idx).fillna(0)

        daily_ret = prices.pct_change().fillna(0)

        # Lag signal by 1 day: signal today → trade at tomorrow's open ≈ today's close
        position = signals.shift(1).fillna(0)

        # Transaction costs on signal changes
        turnover = position.diff().abs().fillna(0)
        tc_drag = turnover * self.tc

        strategy_ret = position * daily_ret - tc_drag
        benchmark_ret = daily_ret

        self._results = pd.DataFrame(
            {
                "price": prices,
                "daily_ret": daily_ret,
                "position": position,
                "strategy_ret": strategy_ret,
                "benchmark_ret": benchmark_ret,
                "strategy_cum": (1 + strategy_ret).cumprod(),
                "benchmark_cum": (1 + benchmark_ret).cumprod(),
            },
            index=idx,
        )
        return self

    @property
    def results(self) -> pd.DataFrame:
        if self._results is None:
            raise RuntimeError("Call .run() first")
        return self._results

    # ---------------------------------------------------------------------------
    # Financial metrics
    # ---------------------------------------------------------------------------

    @staticmethod
    def _metrics(ret: pd.Series, label: str = "") -> dict:
        """Compute a standard set of financial metrics from a daily return series."""
        ret = ret.dropna()
        n = len(ret)
        ann = 252

        cum_ret = (1 + ret).prod() - 1
        n_years = n / ann
        cagr = (1 + cum_ret) ** (1 / max(n_years, 1e-9)) - 1
        vol = ret.std() * np.sqrt(ann)
        sharpe = (ret.mean() * ann) / (ret.std() * np.sqrt(ann) + 1e-9)

        # Drawdown
        cum_curve = (1 + ret).cumprod()
        rolling_max = cum_curve.cummax()
        dd = (cum_curve - rolling_max) / rolling_max
        max_dd = dd.min()
        calmar = cagr / (abs(max_dd) + 1e-9)

        # Downside deviation (Sortino)
        downside = ret[ret < 0]
        sortino = (ret.mean() * ann) / (
            downside.std() * np.sqrt(ann) + 1e-9
        )

        # Distribution
        skew = stats.skew(ret)
        kurt = stats.kurtosis(ret)

        # Win stats
        win_rate = (ret > 0).mean()
        avg_win = ret[ret > 0].mean() if (ret > 0).any() else 0
        avg_loss = ret[ret < 0].mean() if (ret < 0).any() else 0
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan

        # VaR / CVaR (95%)
        var_95 = np.percentile(ret, 5)
        cvar_95 = ret[ret <= var_95].mean()

        prefix = f"{label}_" if label else ""
        return {
            f"{prefix}Total Return": f"{cum_ret:.2%}",
            f"{prefix}CAGR": f"{cagr:.2%}",
            f"{prefix}Volatility (ann.)": f"{vol:.2%}",
            f"{prefix}Sharpe Ratio": f"{sharpe:.3f}",
            f"{prefix}Sortino Ratio": f"{sortino:.3f}",
            f"{prefix}Max Drawdown": f"{max_dd:.2%}",
            f"{prefix}Calmar Ratio": f"{calmar:.3f}",
            f"{prefix}Skewness": f"{skew:.3f}",
            f"{prefix}Kurtosis": f"{kurt:.3f}",
            f"{prefix}Win Rate": f"{win_rate:.2%}",
            f"{prefix}Avg Win": f"{avg_win:.4%}",
            f"{prefix}Avg Loss": f"{avg_loss:.4%}",
            f"{prefix}Payoff Ratio": f"{payoff_ratio:.3f}",
            f"{prefix}VaR 95%": f"{var_95:.4%}",
            f"{prefix}CVaR 95%": f"{cvar_95:.4%}",
        }

    def get_metrics(self, compare_benchmark: bool = True) -> pd.DataFrame:
        """Return a comparison table of strategy vs benchmark metrics."""
        res = self.results
        strategy_metrics = self._metrics(res["strategy_ret"], "Strategy")
        if compare_benchmark:
            bench_metrics = self._metrics(res["benchmark_ret"], "Benchmark")
            combined = {**strategy_metrics, **bench_metrics}
            # Build tidy table
            keys = list(self._metrics(res["strategy_ret"]).keys())
            rows = []
            for k in keys:
                strat_val = strategy_metrics.get(f"Strategy_{k}", "—")
                bench_val = bench_metrics.get(f"Benchmark_{k}", "—")
                rows.append({"Metric": k, "Strategy": strat_val, "Benchmark": bench_val})
            return pd.DataFrame(rows).set_index("Metric")
        else:
            rows = [
                {"Metric": k.replace("Strategy_", ""), "Value": v}
                for k, v in strategy_metrics.items()
            ]
            return pd.DataFrame(rows).set_index("Metric")

    # ---------------------------------------------------------------------------
    # Monthly performance
    # ---------------------------------------------------------------------------

    def monthly_returns(self) -> pd.DataFrame:
        """
        Monthly returns table (rows = years, columns = months).
        Last column is the annual return.
        """
        ret = self.results["strategy_ret"].copy()
        ret.index = pd.to_datetime(ret.index)

        monthly = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        monthly_df = monthly.to_frame("ret")
        monthly_df["Year"] = monthly_df.index.year
        monthly_df["Month"] = monthly_df.index.strftime("%b")

        pivot = monthly_df.pivot(index="Year", columns="Month", values="ret")

        # Reorder months
        month_order = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])

        # Annual return
        annual = ret.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        annual.index = annual.index.year
        pivot["Annual"] = annual

        return pivot

    def exposure(self) -> pd.Series:
        """Fraction of days in long position per year."""
        pos = self.results["position"]
        pos.index = pd.to_datetime(pos.index)
        return pos.resample("YE").mean().rename("Exposure (long %)")

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------

    def plot(self, figsize=(16, 14)) -> None:
        """Full backtest dashboard."""
        res = self.results

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

        # 1) Cumulative returns
        ax1 = fig.add_subplot(gs[0, :])
        res["strategy_cum"].plot(ax=ax1, label="Strategy", color="steelblue", lw=1.8)
        res["benchmark_cum"].plot(ax=ax1, label="Benchmark", color="orange", lw=1.2, alpha=0.8)
        ax1.set_title("Cumulative Returns")
        ax1.set_ylabel("Equity")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2) Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        cum_curve = res["strategy_cum"]
        dd = (cum_curve - cum_curve.cummax()) / cum_curve.cummax()
        dd.plot(ax=ax2, color="crimson", lw=1)
        ax2.fill_between(dd.index, dd, 0, alpha=0.3, color="crimson")
        ax2.set_title("Strategy Drawdown")
        ax2.set_ylabel("Drawdown")
        ax2.grid(alpha=0.3)

        # 3) Rolling Sharpe (1Y)
        ax3 = fig.add_subplot(gs[2, 0])
        roll_sharpe = (
            res["strategy_ret"].rolling(252).mean()
            / (res["strategy_ret"].rolling(252).std() + 1e-9)
            * np.sqrt(252)
        )
        roll_sharpe.plot(ax=ax3, color="steelblue")
        ax3.axhline(0, color="black", lw=0.8, ls="--")
        ax3.set_title("Rolling 1Y Sharpe Ratio")
        ax3.grid(alpha=0.3)

        # 4) Monthly heatmap
        ax4 = fig.add_subplot(gs[2, 1])
        monthly = self.monthly_returns().drop(columns="Annual", errors="ignore")
        import seaborn as sns
        sns.heatmap(
            monthly.astype(float),
            annot=True,
            fmt=".1%",
            cmap="RdYlGn",
            center=0,
            ax=ax4,
            cbar=False,
            annot_kws={"size": 7},
        )
        ax4.set_title("Monthly Returns Heatmap")
        ax4.set_ylabel("Year")

        # 5) Return distribution
        ax5 = fig.add_subplot(gs[3, 0])
        strat_ret = res["strategy_ret"].dropna()
        bench_ret = res["benchmark_ret"].dropna()
        ax5.hist(bench_ret, bins=80, alpha=0.5, color="orange", label="Benchmark", density=True)
        ax5.hist(strat_ret, bins=80, alpha=0.6, color="steelblue", label="Strategy", density=True)
        ax5.set_title("Return Distribution")
        ax5.legend()
        ax5.grid(alpha=0.3)

        # 6) Position / exposure over time
        ax6 = fig.add_subplot(gs[3, 1])
        exp = self.exposure()
        exp.index = exp.index.year
        ax6.bar(exp.index, exp.values, color="steelblue", alpha=0.8)
        ax6.set_title("Annual Long Exposure")
        ax6.set_ylabel("Fraction")
        ax6.set_ylim(0, 1)
        ax6.grid(alpha=0.3, axis="y")

        plt.suptitle("Backtest Performance Dashboard", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.show()

    def plot_monthly_heatmap(self, figsize=(12, 6)) -> None:
        """Standalone monthly returns heatmap."""
        import seaborn as sns
        monthly = self.monthly_returns()
        annual = monthly.pop("Annual")

        fig, axes = plt.subplots(1, 2, figsize=figsize,
                                 gridspec_kw={"width_ratios": [12, 1]})

        sns.heatmap(
            monthly.astype(float),
            annot=True,
            fmt=".1%",
            cmap="RdYlGn",
            center=0,
            ax=axes[0],
            cbar=False,
            annot_kws={"size": 8},
        )
        axes[0].set_title("Monthly Returns (%)")

        sns.heatmap(
            annual.to_frame("Annual").astype(float),
            annot=True,
            fmt=".1%",
            cmap="RdYlGn",
            center=0,
            ax=axes[1],
            cbar=False,
            annot_kws={"size": 8},
        )
        axes[1].set_title("Annual")

        plt.tight_layout()
        plt.show()
