from dataclasses import dataclass, field
import numpy as np
from robustness.monte_carlo import MonteCarloSim, MCSimulationResult
from robustness.cscv import CSCVAnalyzer, CSCVResult
from robustness.sensitivity import SensitivityAnalyzer, SensitivityResult
from robustness.statistics import deflated_sharpe_ratio


@dataclass
class RobustnessReport:
    monte_carlo: MCSimulationResult = field(default_factory=MCSimulationResult)
    cscv: CSCVResult = field(default_factory=CSCVResult)
    deflated_sharpe: float = 0.0
    summary: str = ""


class RobustnessAnalyzer:
    """稳健性检验门面 — 一站式调用"""

    def __init__(self, performance_analyzer):
        self.analyzer = performance_analyzer
        self._rf = getattr(performance_analyzer, 'risk_free_rate', 0.025)
        self._ppy = getattr(performance_analyzer, 'periods_per_year', 252)

    def run_all(self, n_mc: int = 2000, n_cscv_comb: int = 100) -> RobustnessReport:
        """运行全部稳健性检验（不含敏感性，因为需要 engine_factory）"""
        daily_returns = getattr(self.analyzer, 'daily_returns', None)
        report = RobustnessReport()

        if daily_returns is not None and len(daily_returns) > 5:
            mc = MonteCarloSim(rf_annual=self._rf, periods_per_year=self._ppy)
            report.monte_carlo = mc.run(daily_returns, n_sim=n_mc)

            cscv = CSCVAnalyzer(rf_annual=self._rf, periods_per_year=self._ppy)
            report.cscv = cscv.run(daily_returns, n_comb=n_cscv_comb)

            report.deflated_sharpe = deflated_sharpe_ratio(
                daily_returns, rf_annual=self._rf, periods_per_year=self._ppy
            )

        report.summary = self._build_summary(report)
        return report

    def run_sensitivity(self, params: dict, engine_factory) -> SensitivityResult:
        """单独运行敏感性分析（需要 engine_factory 重跑回测）"""
        sa = SensitivityAnalyzer()
        return sa.run(params, engine_factory)

    def _build_summary(self, report: RobustnessReport) -> str:
        lines = []
        mc = report.monte_carlo
        if mc.sharpe_95ci[1] > 0:
            lines.append(
                f"蒙特卡洛: 均值Sharpe={mc.mean_sharpe:.2f}, "
                f"95%CI=[{mc.sharpe_95ci[0]:.2f}, {mc.sharpe_95ci[1]:.2f}]"
            )

        cscv = report.cscv
        if cscv.overfit_probability > 0:
            status = '通过' if cscv.is_robust else '警告: 过拟合风险'
            lines.append(f"CSCV: PBO={cscv.overfit_probability:.2%}, {status}")

        if report.deflated_sharpe != 0:
            lines.append(f"Deflated Sharpe: {report.deflated_sharpe:.2f}")

        return "\n".join(lines)
