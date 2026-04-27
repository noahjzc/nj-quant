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
from back_testing.data.data_provider import DataProvider


def main():
    """运行因子权重优化"""

    # ========================
    # 1. 配置参数
    # ========================
    START_DATE = pd.Timestamp('2022-01-01')
    END_DATE = pd.Timestamp('2024-12-31')

    print("=" * 60)
    print("遗传算法因子权重优化")
    print("=" * 60)
    print(f"时间范围: {START_DATE.date()} ~ {END_DATE.date()}")
    print("=" * 60)

    # ========================
    # 2. 初始化组件
    # ========================

    # 适应度评估器
    all_codes = DataProvider().get_all_stock_codes()
    stock_codes = all_codes[1200:1500]  # 取中间300只作为测试股票池
    print(f"股票池: 共 {len(stock_codes)} 只 (从第1201只到第1500只)")

    evaluator = FitnessEvaluator(
        max_drawdown_constraint=0.30,  # 最大回撤约束30%
        n_stocks=5,                     # 持仓5只股票
        stock_codes=stock_codes,
    )

    # Walk-Forward分析器
    wf_analyzer = WalkForwardAnalyzer(
        train_window_months=12,
        val_window_months=6,
        test_window_months=6,
        step_months=3
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
        # 每个窗口开始前清缓存
        evaluator.clear_cache()

        print(f"\n=== 窗口 {i + 1}/{len(windows)} ===")
        print(f"训练: {window['train'][0].date()} ~ {window['train'][1].date()}")
        print(f"验证: {window['val'][0].date()} ~ {window['val'][1].date()}")
        print(f"测试: {window['test'][0].date()} ~ {window['test'][1].date()}")

        # 重要: 因子数据缓存跨代共享，首个个体加载后后续个体命中缓存

        # 创建GA优化器MultiFactorSelector
        optimizer = GeneticOptimizer(
            population_size=50,     # 种群规模（因子数×5）
            max_generations=100,     # 最大迭代代数
            elite_ratio=0.1,       # 10%精英
            crossover_rate=0.7,     # 70%交叉率
            mutation_rate=0.05,     # 5%变异率
            patience=40,            # 20代无提升则早停
            seed=42
        )

        # 训练适应度函数
        train_fitness_func = lambda w, _: evaluator.evaluate(w, window['train'][0], window['train'][1])
        # 验证适应度函数（用于早停，真正样本外评估）
        val_fitness_func = lambda w, _: evaluator.evaluate(w, window['val'][0], window['val'][1])

        # 执行优化
        optimal = optimizer.optimize(
            fitness_func=train_fitness_func,
            train_data=(window['train'][0], window['train'][1]),
            val_data=(window['val'][0], window['val'][1]),
            val_fitness_func=val_fitness_func,
            verbose=True
        )

        # 在验证集和测试集上评估
        val_fitness = evaluator.evaluate(optimal, window['val'][0], window['val'][1])
        test_fitness = evaluator.evaluate(optimal, window['test'][0], window['test'][1])

        # 打印缓存统计
        cache_stats = evaluator.get_cache_stats()
        print(f"验证集信息比率: {val_fitness:.4f}")
        print(f"测试集信息比率: {test_fitness:.4f}")
        print(f"  ↳ 因子缓存命中: {cache_stats['hits']}次, 未命中: {cache_stats['misses']}次, 命中率: {cache_stats['hit_rate']:.1%}")

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
