"""
Advanced Risk Management System

Comprehensive risk management for trading strategies including:
- Portfolio-level risk metrics (VaR, CVaR, Maximum Drawdown)
- Position-level risk controls (stop-loss, take-profit, position sizing)
- Market risk factors (volatility, correlation, beta)
- Stress testing and scenario analysis
- Risk-adjusted performance metrics
- Compliance and regulatory reporting

Author: Enhanced Monte Carlo Trading System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for portfolio analysis."""
    # Basic risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    maximum_drawdown: float
    calmar_ratio: float

    # Value at Risk metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    # Performance metrics
    total_return: float
    annualized_return: float
    win_rate: float
    profit_factor: float

    # Risk-adjusted metrics
    information_ratio: float
    omega_ratio: float
    kappa_ratio: float

    # Drawdown metrics
    average_drawdown: float
    longest_drawdown_period: int
    recovery_time: int

@dataclass
class PositionRisk:
    """Risk metrics for individual positions."""
    symbol: str
    current_price: float
    position_size: float
    market_value: float
    unrealized_pnl: float

    # Risk controls
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    max_position_size: float
    max_loss_limit: float

    # Risk metrics
    volatility: float
    beta: float
    correlation: float
    var_contribution: float

@dataclass
class RiskLimits:
    """Risk limits and thresholds for portfolio management."""
    max_portfolio_volatility: float = 0.20  # 20% max volatility
    max_single_position: float = 0.10      # 10% max single position
    max_drawdown_limit: float = 0.15       # 15% max drawdown
    var_limit_95: float = 0.10             # 10% VaR limit at 95%
    var_limit_99: float = 0.15             # 15% VaR limit at 99%
    max_correlation: float = 0.80          # Max correlation between positions
    max_leverage: float = 2.0              # Max leverage ratio

class RiskManager:
    """Advanced risk management system for trading portfolios."""

    def __init__(self, risk_limits: RiskLimits = None):
        self.risk_limits = risk_limits or RiskLimits()
        self.portfolio_positions: Dict[str, PositionRisk] = {}
        self.portfolio_history: pd.DataFrame = pd.DataFrame()
        self.benchmark_returns: pd.Series = pd.Series()
        self.risk_free_rate = 0.02  # 2% risk-free rate

    def add_position(self, position: PositionRisk):
        """Add a position to the portfolio."""
        self.portfolio_positions[position.symbol] = position

    def remove_position(self, symbol: str):
        """Remove a position from the portfolio."""
        if symbol in self.portfolio_positions:
            del self.portfolio_positions[symbol]

    def update_portfolio_history(self, returns: pd.Series, prices: pd.DataFrame = None):
        """Update portfolio historical returns and prices."""
        self.portfolio_history = pd.DataFrame({'returns': returns})
        if prices is not None:
            self.portfolio_history = pd.concat([self.portfolio_history, prices], axis=1)

    def calculate_portfolio_risk_metrics(self,
                                       confidence_level: float = 0.95,
                                       annualize: bool = True) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics.

        Args:
            confidence_level: Confidence level for VaR calculations (default 95%)
            annualize: Whether to annualize metrics (default True)

        Returns:
            RiskMetrics: Complete set of risk and performance metrics
        """
        if self.portfolio_history.empty:
            raise ValueError("Portfolio history is required for risk calculations")

        returns = self.portfolio_history['returns'].dropna()

        if len(returns) < 30:
            raise ValueError("Insufficient data for reliable risk calculations")

        # Basic return and volatility metrics
        total_return = (1 + returns).prod() - 1
        volatility = returns.std()

        if annualize:
            # Assume daily returns for annualization
            trading_days = 252
            annualized_return = (1 + total_return) ** (trading_days / len(returns)) - 1
            volatility = volatility * np.sqrt(trading_days)
        else:
            annualized_return = total_return

        # Sharpe and Sortino ratios
        excess_returns = returns - self.risk_free_rate / 252  # Daily risk-free rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if annualize else excess_returns.mean() / excess_returns.std()

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        sortino_ratio = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252) if len(downside_returns) > 0 and annualize else (excess_returns.mean() / downside_returns.std()) if len(downside_returns) > 0 else 0

        # Maximum drawdown calculation
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Calmar ratio
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0

        # Value at Risk (VaR) using historical simulation
        var_95 = np.percentile(returns, (1 - 0.95) * 100)
        var_99 = np.percentile(returns, (1 - 0.99) * 100)

        # Conditional VaR (CVaR/Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99

        # Win rate and profit factor
        winning_trades = len(returns[returns > 0])
        total_trades = len(returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')

        # Information ratio (if benchmark available)
        if not self.benchmark_returns.empty:
            tracking_error = (returns - self.benchmark_returns).std()
            information_ratio = (returns.mean() - self.benchmark_returns.mean()) / tracking_error * np.sqrt(252) if tracking_error != 0 else 0
        else:
            information_ratio = 0

        # Omega ratio (probability weighted ratio of gains to losses)
        threshold = 0  # Minimum acceptable return
        omega_ratio = self._calculate_omega_ratio(returns, threshold)

        # Kappa ratio (generalized Sharpe ratio)
        kappa = 3  # Risk aversion parameter
        kappa_ratio = self._calculate_kappa_ratio(returns, kappa)

        # Drawdown analysis
        drawdown_periods = self._analyze_drawdowns(cumulative)
        average_drawdown = drawdowns[drawdowns < 0].mean()
        longest_drawdown_period = max([len(period) for period in drawdown_periods]) if drawdown_periods else 0
        recovery_time = self._calculate_recovery_time(cumulative)

        return RiskMetrics(
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            maximum_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            total_return=total_return,
            annualized_return=annualized_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            information_ratio=information_ratio,
            omega_ratio=omega_ratio,
            kappa_ratio=kappa_ratio,
            average_drawdown=average_drawdown,
            longest_drawdown_period=longest_drawdown_period,
            recovery_time=recovery_time
        )

    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float) -> float:
        """Calculate Omega ratio."""
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())
        return gains / losses if losses != 0 else float('inf')

    def _calculate_kappa_ratio(self, returns: pd.Series, kappa: float) -> float:
        """Calculate Kappa ratio (generalized Sharpe ratio)."""
        mu = returns.mean()
        sigma = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()

        if sigma == 0:
            return 0

        # Kappa ratio formula
        kappa_ratio = mu / (sigma ** kappa)
        return kappa_ratio

    def _analyze_drawdowns(self, cumulative: pd.Series) -> List[List[int]]:
        """Analyze drawdown periods."""
        drawdowns = []
        in_drawdown = False
        current_drawdown = []

        for i, val in enumerate(cumulative):
            if val < cumulative[:i+1].max():
                if not in_drawdown:
                    in_drawdown = True
                    current_drawdown = [i]
                else:
                    current_drawdown.append(i)
            else:
                if in_drawdown:
                    drawdowns.append(current_drawdown)
                    in_drawdown = False
                    current_drawdown = []

        if in_drawdown:
            drawdowns.append(current_drawdown)

        return drawdowns

    def _calculate_recovery_time(self, cumulative: pd.Series) -> int:
        """Calculate average recovery time from drawdowns."""
        recovery_times = []

        for i in range(1, len(cumulative)):
            if cumulative.iloc[i] >= cumulative.iloc[:i].max():
                # Find when we recovered to previous peak
                peak_idx = cumulative.iloc[:i].idxmax()
                recovery_times.append(i - peak_idx)

        return int(np.mean(recovery_times)) if recovery_times else 0

    def calculate_position_risks(self, prices: pd.DataFrame,
                               market_returns: pd.Series) -> Dict[str, PositionRisk]:
        """Calculate risk metrics for individual positions."""
        position_risks = {}

        for symbol, position in self.portfolio_positions.items():
            if symbol not in prices.columns:
                continue

            price_series = prices[symbol].dropna()
            returns = price_series.pct_change().dropna()

            # Basic volatility
            volatility = returns.std() * np.sqrt(252)  # Annualized

            # Beta calculation
            if not market_returns.empty:
                covariance = np.cov(returns, market_returns)[0, 1]
                market_variance = market_returns.var()
                beta = covariance / market_variance if market_variance != 0 else 1.0
            else:
                beta = 1.0

            # Correlation with market
            correlation = returns.corr(market_returns) if not market_returns.empty else 0.0

            # VaR contribution (simplified)
            position_weight = position.market_value / sum(p.market_value for p in self.portfolio_positions.values())
            var_contribution = position_weight * volatility

            # Update position with calculated metrics
            position.volatility = volatility
            position.beta = beta
            position.correlation = correlation
            position.var_contribution = var_contribution

            position_risks[symbol] = position

        return position_risks

    def check_risk_limits(self, metrics: RiskMetrics,
                         position_risks: Dict[str, PositionRisk]) -> List[str]:
        """Check if portfolio exceeds risk limits."""
        violations = []

        # Portfolio-level checks
        if metrics.volatility > self.risk_limits.max_portfolio_volatility:
            violations.append(f"Portfolio volatility ({metrics.volatility:.2%}) exceeds limit ({self.risk_limits.max_portfolio_volatility:.2%})")

        if abs(metrics.maximum_drawdown) > self.risk_limits.max_drawdown_limit:
            violations.append(f"Maximum drawdown ({abs(metrics.maximum_drawdown):.2%}) exceeds limit ({self.risk_limits.max_drawdown_limit:.2%})")

        if abs(metrics.var_95) > self.risk_limits.var_limit_95:
            violations.append(f"VaR 95% ({abs(metrics.var_95):.2%}) exceeds limit ({self.risk_limits.var_limit_95:.2%})")

        # Position-level checks
        total_value = sum(p.market_value for p in self.portfolio_positions.values())

        for symbol, position in position_risks.items():
            position_weight = position.market_value / total_value

            if position_weight > self.risk_limits.max_single_position:
                violations.append(f"Position {symbol} weight ({position_weight:.2%}) exceeds limit ({self.risk_limits.max_single_position:.2%})")

        # Correlation checks
        high_corr_positions = []
        for symbol, position in position_risks.items():
            if abs(position.correlation) > self.risk_limits.max_correlation:
                high_corr_positions.append(symbol)

        if high_corr_positions:
            violations.append(f"High correlation positions: {', '.join(high_corr_positions)}")

        return violations

    def optimize_portfolio_risk(self, returns: pd.DataFrame,
                              target_volatility: float = None) -> Dict[str, float]:
        """Optimize portfolio weights for risk management."""
        if target_volatility is None:
            target_volatility = self.risk_limits.max_portfolio_volatility

        # Mean-variance optimization with risk constraints
        n_assets = len(returns.columns)
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        # Objective: minimize portfolio variance subject to return constraint
        def portfolio_variance(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # weights sum to 1
        ]

        # Bounds: 0 to max position limit
        bounds = [(0, self.risk_limits.max_single_position) for _ in range(n_assets)]

        # Initial guess: equal weights
        initial_weights = np.array([1/n_assets] * n_assets)

        # Optimize
        result = minimize(portfolio_variance, initial_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            return dict(zip(returns.columns, result.x))
        else:
            # Fallback to equal weights
            equal_weight = 1 / n_assets
            return dict(zip(returns.columns, [equal_weight] * n_assets))

    def stress_test_portfolio(self, returns: pd.Series,
                            scenarios: Dict[str, pd.Series]) -> Dict[str, Dict[str, float]]:
        """Perform stress testing on portfolio."""
        results = {}

        for scenario_name, stress_returns in scenarios.items():
            # Combine portfolio returns with stress scenario
            stressed_returns = returns + stress_returns

            # Calculate stressed metrics
            stressed_cumulative = (1 + stressed_returns).cumprod()
            stressed_peak = stressed_cumulative.expanding().max()
            stressed_drawdown = (stressed_cumulative - stressed_peak) / stressed_peak

            results[scenario_name] = {
                'max_drawdown': stressed_drawdown.min(),
                'final_return': stressed_cumulative.iloc[-1] - 1,
                'volatility': stressed_returns.std() * np.sqrt(252),
                'var_95': np.percentile(stressed_returns, 5),
                'worst_month': stressed_returns.groupby(pd.Grouper(freq='M')).sum().min()
            }

        return results

    def generate_risk_report(self, metrics: RiskMetrics,
                           position_risks: Dict[str, PositionRisk],
                           violations: List[str]) -> str:
        """Generate a comprehensive risk report."""
        report = []
        report.append("=" * 60)
        report.append("PORTFOLIO RISK MANAGEMENT REPORT")
        report.append("=" * 60)
        report.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Risk Metrics Summary
        report.append("📊 RISK METRICS SUMMARY")
        report.append("-" * 30)
        report.append(".2%")
        report.append(".2%")
        report.append(".2%")
        report.append(".2%")
        report.append(".2%")
        report.append(".2%")
        report.append(".2%")
        report.append(".2%")
        report.append("")

        # Performance Metrics
        report.append("💰 PERFORMANCE METRICS")
        report.append("-" * 25)
        report.append(".2%")
        report.append(".2%")
        report.append(".1%")
        report.append(".2f")
        report.append("")

        # Risk Limits Status
        report.append("⚠️  RISK LIMITS STATUS")
        report.append("-" * 22)
        if violations:
            report.append("❌ VIOLATIONS DETECTED:")
            for violation in violations:
                report.append(f"   • {violation}")
        else:
            report.append("✅ All risk limits within acceptable ranges")

        report.append("")

        # Position Risk Analysis
        report.append("📈 POSITION RISK ANALYSIS")
        report.append("-" * 27)
        report.append("<12")
        report.append("-" * 70)

        total_value = sum(p.market_value for p in self.portfolio_positions.values())
        for symbol, position in position_risks.items():
            weight = position.market_value / total_value
            report.append("<12")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

# Example usage
def example_risk_analysis():
    """Example of comprehensive risk analysis."""

    # Create sample portfolio returns (simulate a trading strategy)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    n_days = len(dates)

    # Generate sample returns with some volatility and drawdowns
    base_returns = np.random.normal(0.0005, 0.02, n_days)  # Mean 0.05%, vol 2%
    # Add some drawdown periods
    drawdown_periods = np.random.choice(n_days, size=5, replace=False)
    for period in drawdown_periods:
        if period + 20 < n_days:
            base_returns[period:period+20] -= np.linspace(0, 0.05, 20)  # 5% drawdown

    portfolio_returns = pd.Series(base_returns, index=dates)

    # Initialize risk manager
    risk_manager = RiskManager()

    # Add sample positions
    positions = [
        PositionRisk("AAPL", 150.0, 100, 15000, 500, None, None, 20000, 2000, 0, 0, 0, 0),
        PositionRisk("MSFT", 280.0, 50, 14000, 1000, None, None, 15000, 1500, 0, 0, 0, 0),
        PositionRisk("GOOGL", 2500.0, 10, 25000, -500, None, None, 30000, 3000, 0, 0, 0, 0),
    ]

    for position in positions:
        risk_manager.add_position(position)

    # Update portfolio history
    risk_manager.update_portfolio_history(portfolio_returns)

    # Calculate comprehensive risk metrics
    try:
        metrics = risk_manager.calculate_portfolio_risk_metrics()

        # Generate sample price data for position risk calculation
        sample_prices = pd.DataFrame({
            'AAPL': 150 + np.cumsum(np.random.normal(0, 2, n_days)),
            'MSFT': 280 + np.cumsum(np.random.normal(0, 3, n_days)),
            'GOOGL': 2500 + np.cumsum(np.random.normal(0, 20, n_days))
        }, index=dates)

        market_returns = pd.Series(np.random.normal(0.0003, 0.015, n_days), index=dates)
        position_risks = risk_manager.calculate_position_risks(sample_prices, market_returns)

        # Check risk limits
        violations = risk_manager.check_risk_limits(metrics, position_risks)

        # Generate risk report
        report = risk_manager.generate_risk_report(metrics, position_risks, violations)

        print(report)

        # Additional analysis
        print("\n🔍 DETAILED ANALYSIS")
        print("-" * 20)

        # Stress testing scenarios
        stress_scenarios = {
            'Market Crash': pd.Series([-0.05] * 5 + [0] * (n_days-5), index=dates),  # 5% drop
            'High Volatility': pd.Series(np.random.normal(0, 0.04, n_days), index=dates),  # 4% vol
            'Liquidity Crisis': pd.Series([-0.02] * 10 + [0] * (n_days-10), index=dates),  # 2% drop for 10 days
        }

        stress_results = risk_manager.stress_test_portfolio(portfolio_returns, stress_scenarios)
        print("\n📊 STRESS TEST RESULTS")
        print("-" * 22)
        for scenario, results in stress_results.items():
            print(f"{scenario}:")
            print(".2%")
            print(".2%")
            print()

    except Exception as e:
        print(f"Error in risk analysis: {e}")

if __name__ == "__main__":
    example_risk_analysis()
