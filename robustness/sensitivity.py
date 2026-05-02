from dataclasses import dataclass, field
from typing import Dict, Callable, Any
from copy import deepcopy


@dataclass
class SensitivityResult:
    per_param: Dict[str, dict] = field(default_factory=dict)
    overall_stability_score: float = 1.0  # 0~1，越高越稳定


class SensitivityAnalyzer:
    """参数敏感性分析：对每个参数 ±20% 扰动，对比 Sharpe 变化"""

    def __init__(self, perturbation_pct: float = 0.2):
        self.perturbation_pct = perturbation_pct

    def run(self, params: Dict[str, float],
            engine_factory: Callable[[Dict], Any]) -> SensitivityResult:
        """params: 策略参数字典
           engine_factory: 接收参数字典，返回包含 'sharpe_ratio' 的 dict 的对象"""
        return self._evaluate(params, engine_factory)

    def _evaluate(self, params: Dict[str, float],
                  run_fn: Callable[[Dict], Any]) -> SensitivityResult:
        base_result = run_fn(deepcopy(params))
        base_sharpe = self._extract_sharpe(base_result)

        per_param = {}
        sharpe_changes = []

        for key, value in params.items():
            if not isinstance(value, (int, float)):
                continue

            delta = abs(value) * self.perturbation_pct if value != 0 else self.perturbation_pct
            perturbed_sharpes = []

            for perturbed_value in [value + delta, value - delta]:
                cfg = deepcopy(params)
                cfg[key] = perturbed_value
                result = run_fn(cfg)
                perturbed_sharpes.append(self._extract_sharpe(result))

            avg_perturbed = sum(perturbed_sharpes) / 2
            avg_change = abs((avg_perturbed - base_sharpe) / base_sharpe) if base_sharpe != 0 else 0.0
            sharpe_changes.append(avg_change)

            per_param[key] = {
                'base_value': value,
                'delta': delta,
                'sharpe_change_pct': round(avg_change * 100, 2),
                'stable': avg_change < 0.10,
            }

        if sharpe_changes:
            avg_sensitivity = sum(sharpe_changes) / len(sharpe_changes)
            stability = max(0.0, min(1.0, 1.0 - avg_sensitivity * 5))
        else:
            stability = 1.0

        return SensitivityResult(per_param=per_param, overall_stability_score=round(stability, 4))

    def _extract_sharpe(self, result: Any) -> float:
        if isinstance(result, dict):
            return float(result.get('sharpe_ratio', 0))
        if hasattr(result, 'sharpe_ratio'):
            return float(result.sharpe_ratio)
        return 0.0
