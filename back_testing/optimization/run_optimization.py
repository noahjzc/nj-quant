"""
遗传算法因子权重优化 - 运行脚本

本脚本执行完整的Walk-Forward遗传算法优化流程:

1. 配置参数
   - 数据路径、时间范围
   - GA参数(种群大小、代数等)
   - Walk-Forward窗口参数

2. 对每个Walk-Forward窗口:
   - 用训练期数据运行GA优化
   - 在验证集上评估
   - 在测试集上评估

3. 聚合多窗口结果:
   - 汇总各窗口最优权重
   - 输出最终权重配置
   - 敏感性分析

4. 保存结果:
   - 权重配置保存到JSON
   - 因子重要性表格

使用示例:
    python back_testing/optimization/run_optimization.py
"""
import pandas as pd
import json
from back_testing.optimization.genetic_optimizer.optimizer import GeneticOptimizer
from back_testing.optimization.genetic_optimizer.fitness import FitnessEvaluator
from back_testing.optimization.genetic_optimizer.walk_forward import WalkForwardAnalyzer
from back_testing.optimization.genetic_optimizer.sensitivity import SensitivityAnalyzer


def main():
    """运行因子权重优化"""

    # ========================
    # 1. 配置参数
    # ========================
    DATA_PATH = 'data/daily_ycz'
    START_DATE = pd.Timestamp('2020-01-01')
    END_DATE = pd.Timestamp('2024-01-01')

    print("=" * 60)
    print("遗传算法因子权重优化")
    print("=" * 60)
    print(f"数据路径: {DATA_PATH}")
    print(f"时间范围: {START_DATE.date()} ~ {END_DATE.date()}")
    print("=" * 60)

    # ========================
    # 2. 初始化组件
    # ========================

    # 适应度评估器
    evaluator = FitnessEvaluator(
        max_drawdown_constraint=0.20,  # 最大回撤约束20%
        n_stocks=5                       # 持仓5只股票
    )

    # Walk-Forward分析器
    wf_analyzer = WalkForwardAnalyzer(
        train_window_years=3,   # 训练3年
        val_window_years=1,     # 验证1年
        test_window_years=1,    # 测试1年
        step_months=3           # 每3个月滚动一次
    )

    # ========================
    # 3. 生成Walk-Forward窗口
    # ========================
    windows = wf_analyzer.get_windows(START_DATE, END_DATE)
    print(f"\nWalk-Forward窗口数: {len(windows)}")

    all_optimal_weights = []  # 收集各窗口最优权重

    # ========================
    # 4. 遍历每个窗口执行优化
    # ========================
    for i, window in enumerate(windows):
        print(f"\n=== 窗口 {i + 1}/{len(windows)} ===")
        print(f"训练: {window['train'][0].date()} ~ {window['train'][1].date()}")
        print(f"验证: {window['val'][0].date()} ~ {window['val'][1].date()}")
        print(f"测试: {window['test'][0].date()} ~ {window['test'][1].date()}")

        # 适应度函数包装
        def fitness_func(weights, data):
            return evaluator.evaluate(weights, data[0], data[1])

        # 创建GA优化器
        optimizer = GeneticOptimizer(
            population_size=50,     # 种群50
            max_generations=100,    # 最大100代
            elite_ratio=0.1,       # 10%精英
            crossover_rate=0.7,     # 70%交叉率
            mutation_rate=0.05,     # 5%变异率
            patience=20,            # 20代早停
            seed=42                 # 固定种子
        )

        # 执行优化
        optimal = optimizer.optimize(
            fitness_func=lambda w, _: fitness_func(w, (window['train'][0], window['train'][1])),
            train_data=(window['train'][0], window['train'][1]),
            val_data=(window['val'][0], window['val'][1]),
            verbose=True
        )

        # 在验证集和测试集上评估
        val_fitness = evaluator.evaluate(optimal, window['val'][0], window['val'][1])
        test_fitness = evaluator.evaluate(optimal, window['test'][0], window['test'][1])

        print(f"验证集Calmar: {val_fitness:.4f}")
        print(f"测试集Calmar: {test_fitness:.4f}")

        all_optimal_weights.append(optimal)

    # ========================
    # 5. 聚合多窗口结果
    # ========================
    final_weights = wf_analyzer.aggregate_weights(all_optimal_weights)

    print("\n" + "=" * 60)
    print("最终聚合权重")
    print("=" * 60)
    for k, v in sorted(final_weights.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.4f}")

    # ========================
    # 6. 敏感性分析
    # ========================
    print("\n=== 敏感性分析 ===")
    sensitivity = SensitivityAnalyzer(evaluator)
    importance = sensitivity.analyze_factor_importance(
        final_weights,
        (START_DATE, END_DATE)
    )
    print(importance.to_string(index=False))

    # ========================
    # 7. 保存结果
    # ========================
    results = {
        'final_weights': final_weights,
        'window_weights': all_optimal_weights,
        'factor_importance': importance.to_dict()
    }

    with open('optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n结果已保存到 optimization_results.json")


if __name__ == '__main__':
    main()
