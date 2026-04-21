"""Factor Weight Optimization Entry Script"""
import pandas as pd
import json
from back_testing.optimization.genetic_optimizer.optimizer import GeneticOptimizer
from back_testing.optimization.genetic_optimizer.fitness import FitnessEvaluator
from back_testing.optimization.genetic_optimizer.walk_forward import WalkForwardAnalyzer
from back_testing.optimization.genetic_optimizer.sensitivity import SensitivityAnalyzer


def main():
    """Run factor weight optimization"""
    # Configuration
    DATA_PATH = 'data/daily_ycz'
    START_DATE = pd.Timestamp('2019-01-01')
    END_DATE = pd.Timestamp('2024-01-01')

    print("=" * 60)
    print("Genetic Algorithm Factor Weight Optimization")
    print("=" * 60)
    print(f"Data path: {DATA_PATH}")
    print(f"Period: {START_DATE.date()} ~ {END_DATE.date()}")
    print("=" * 60)

    # Create evaluator
    evaluator = FitnessEvaluator(
        data_path=DATA_PATH,
        max_drawdown_constraint=0.20,
        n_stocks=5
    )

    # Create walk-forward analyzer
    wf_analyzer = WalkForwardAnalyzer(
        train_window_years=3,
        val_window_years=1,
        test_window_years=1,
        step_months=3
    )

    windows = wf_analyzer.get_windows(START_DATE, END_DATE)
    print(f"\nWalk-Forward Windows: {len(windows)}")

    all_optimal_weights = []

    for i, window in enumerate(windows):
        print(f"\n=== Window {i + 1}/{len(windows)} ===")
        print(f"Train: {window['train'][0].date()} ~ {window['train'][1].date()}")
        print(f"Val:   {window['val'][0].date()} ~ {window['val'][1].date()}")
        print(f"Test:  {window['test'][0].date()} ~ {window['test'][1].date()}")

        def fitness_func(weights, data):
            return evaluator.evaluate(weights, data[0], data[1])

        optimizer = GeneticOptimizer(
            population_size=50,
            max_generations=100,
            elite_ratio=0.1,
            crossover_rate=0.7,
            mutation_rate=0.05,
            patience=20,
            seed=42
        )

        optimal = optimizer.optimize(
            fitness_func=lambda w: fitness_func(w, (window['train'][0], window['train'][1])),
            train_data=(window['train'][0], window['train'][1]),
            val_data=(window['val'][0], window['val'][1]),
            verbose=True
        )

        val_fitness = evaluator.evaluate(optimal, window['val'][0], window['val'][1])
        test_fitness = evaluator.evaluate(optimal, window['test'][0], window['test'][1])

        print(f"Val Calmar: {val_fitness:.4f}")
        print(f"Test Calmar: {test_fitness:.4f}")

        all_optimal_weights.append(optimal)

    final_weights = wf_analyzer.aggregate_weights(all_optimal_weights)

    print("\n" + "=" * 60)
    print("Final Aggregated Weights")
    print("=" * 60)
    for k, v in sorted(final_weights.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.4f}")

    print("\n=== Sensitivity Analysis ===")
    sensitivity = SensitivityAnalyzer(evaluator)
    importance = sensitivity.analyze_factor_importance(
        final_weights,
        (START_DATE, END_DATE)
    )
    print(importance.to_string(index=False))

    results = {
        'final_weights': final_weights,
        'window_weights': all_optimal_weights,
        'factor_importance': importance.to_dict()
    }

    with open('optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to optimization_results.json")


if __name__ == '__main__':
    main()
