"""Portfolio optimization and evaluation logic

NOTE: This module is not currently used in the portfolio testing workflow.
It is planned for future implementation of portfolio weight optimization.
The current portfolio testing uses builder.py for all metric calculations.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


def optimize_weights(portfolio, scenario, metric: str = "sharpe", n_iterations: int = 100,
                    **kwargs) -> Dict[str, Any]:
    """
    Optimize portfolio weights using Bayesian optimization
    
    Args:
        portfolio: Portfolio instance
        scenario: Scenario instance with forecast data
        metric: Optimization metric ("sharpe", "return", "risk_adjusted")
        n_iterations: Number of optimization iterations
        **kwargs: Additional optimization parameters
        
    Returns:
        Dictionary with optimized weights and metrics
        
    Note:
        Portfolios support long-short positions (negative weights allowed).
        Optimized weights are normalized so sum of absolute values equals 1.0.
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real
        from skopt.utils import use_named_args
    except ImportError:
        logger.error("scikit-optimize not installed. Please install with: pip install scikit-optimize")
        raise ImportError("scikit-optimize is required for portfolio optimization")
    
    # Extract scenario data
    scenario_data = extract_scenario_data(scenario, portfolio.assets)
    
    # Define search space for weights (always long-short: -1 to 1)
    n_assets = len(portfolio.assets)
    dimensions = []
    for i in range(n_assets):
        dimensions.append(Real(-1.0, 1.0, name=f'weight_{i}'))
    
    # Objective function
    @use_named_args(dimensions=dimensions)
    def objective(**params):
        weights = [params[f'weight_{i}'] for i in range(n_assets)]
        weights_dict = dict(zip(portfolio.assets, weights))
        
        # Normalize weights (sum of absolute values = 1.0)
        weights_dict = _apply_constraints(weights_dict)
        
        # Calculate portfolio returns
        returns_data = calculate_portfolio_returns(scenario_data, weights_dict)
        
        # Return negative metric for minimization
        if metric == "sharpe":
            return -returns_data['sharpe']
        elif metric == "return":
            return -returns_data['mean_return']
        elif metric == "risk_adjusted":
            # Custom risk-adjusted metric
            return -(returns_data['mean_return'] / returns_data['std_return'])
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    # Run optimization
    logger.info(f"Starting optimization with {n_iterations} iterations...")
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_iterations,
        random_state=42
    )
    
    # Extract optimized weights and normalize
    optimized_weights = dict(zip(portfolio.assets, result.x))
    optimized_weights = _apply_constraints(optimized_weights)
    
    # Calculate final metrics
    final_returns = calculate_portfolio_returns(scenario_data, optimized_weights)
    
    # Save optimization history
    _save_optimization_history(portfolio, scenario, metric, n_iterations, final_returns)
    
    return {
        'weights': optimized_weights,
        'sharpe': final_returns['sharpe'],
        'mean_return': final_returns['mean_return'],
        'std_return': final_returns['std_return'],
        'optimization_history': result.func_vals,
        'n_iterations': len(result.func_vals)
    }


def evaluate_portfolio(portfolio, scenario) -> Dict[str, Any]:
    """
    Evaluate portfolio performance metrics for a scenario
    
    Args:
        portfolio: Portfolio instance
        scenario: Scenario instance with forecast data
        
    Returns:
        Dictionary with performance metrics
    """
    # Extract scenario data
    scenario_data = extract_scenario_data(scenario, portfolio.assets)
    
    # Calculate portfolio returns
    returns_data = calculate_portfolio_returns(scenario_data, portfolio.weights)
    
    # Save evaluation history
    _save_evaluation_history(portfolio, scenario, returns_data)
    
    return returns_data


def extract_scenario_data(scenario, assets: List[str]) -> Dict[str, Any]:
    """
    Extract forecast data for specified assets from scenario
    
    Args:
        scenario: Scenario instance
        assets: List of asset names
        
    Returns:
        Dictionary with extracted data organized by sample_idx
    """
    if not scenario.output:
        raise ValueError("Scenario must be simulated before extracting data")
    
    reconstructed_windows = scenario.output.get('conditioning_info', {}).get('reconstructed', [])
    
    if not reconstructed_windows:
        raise ValueError("No reconstructed data found in scenario output")
    
    # Find forecast windows (future, not historical pattern)
    forecast_windows = [w for w in reconstructed_windows if 
                       w.get('temporal_tag') == 'future' and 
                       w.get('_is_historical_pattern') == False]
    
    if not forecast_windows:
        raise ValueError("No forecast windows found in scenario output")
    
    # Organize data by sample_idx and asset
    sample_data = {}
    
    for window in forecast_windows:
        sample_idx = window.get('_sample_idx')
        feature = window.get('feature')
        values = window.get('reconstructed_values', [])
        
        if sample_idx is None or feature not in assets or not values:
            continue
        
        if sample_idx not in sample_data:
            sample_data[sample_idx] = {}
        
        sample_data[sample_idx][feature] = values
    
    # Verify we have data for all assets
    missing_assets = set(assets) - set(sample_data.get(list(sample_data.keys())[0], {}).keys())
    if missing_assets:
        raise ValueError(f"Missing forecast data for assets: {missing_assets}")
    
    logger.info(f"Extracted data for {len(sample_data)} samples and {len(assets)} assets")
    return sample_data


def calculate_portfolio_returns(scenario_data: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate portfolio returns from scenario data
    
    Args:
        scenario_data: Data organized by sample_idx and asset
        weights: Portfolio weights for each asset
        
    Returns:
        Dictionary with portfolio performance metrics
    """
    portfolio_returns = []
    cumulative_paths = []
    
    for sample_idx, sample_assets in scenario_data.items():
        # Calculate returns for each asset in this sample
        asset_returns = {}
        
        for asset, values in sample_assets.items():
            if len(values) < 2:
                continue
            
            # Calculate period returns: (value_t - value_t-1) / value_t-1
            returns = []
            for i in range(1, len(values)):
                if values[i-1] != 0:  # Avoid division by zero
                    ret = (values[i] - values[i-1]) / values[i-1]
                    returns.append(ret)
            
            asset_returns[asset] = returns
        
        # Calculate portfolio returns for this sample
        if asset_returns:
            # Find minimum length to align all assets
            min_length = min(len(returns) for returns in asset_returns.values())
            
            if min_length > 0:
                sample_portfolio_returns = []
                for t in range(min_length):
                    portfolio_return_t = sum(
                        weights.get(asset, 0) * asset_returns[asset][t]
                        for asset in asset_returns.keys()
                    )
                    sample_portfolio_returns.append(portfolio_return_t)
                
                # Calculate cumulative returns
                cumulative_return = 1.0
                cumulative_path = [cumulative_return]
                for ret in sample_portfolio_returns:
                    cumulative_return *= (1 + ret)
                    cumulative_path.append(cumulative_return)
                
                portfolio_returns.extend(sample_portfolio_returns)
                cumulative_paths.append(cumulative_path)
    
    if not portfolio_returns:
        raise ValueError("No portfolio returns calculated")
    
    # Calculate metrics
    returns_array = np.array(portfolio_returns)
    
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array)
    sharpe = mean_return / std_return if std_return > 0 else 0
    
    # Calculate VaR
    var_95 = np.percentile(returns_array, 5)
    var_99 = np.percentile(returns_array, 1)
    
    # Calculate max drawdown from cumulative paths
    max_drawdown = 0
    if cumulative_paths:
        for path in cumulative_paths:
            peak = path[0]
            for value in path:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
    
    return {
        'returns': portfolio_returns,
        'cumulative_paths': cumulative_paths,
        'mean_return': mean_return,
        'std_return': std_return,
        'sharpe': sharpe,
        'var_95': var_95,
        'var_99': var_99,
        'max_drawdown': max_drawdown,
        'n_samples': len(cumulative_paths),
        'n_periods': len(portfolio_returns) // len(cumulative_paths) if cumulative_paths else 0
    }


def _apply_constraints(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize portfolio weights so sum of absolute values equals 1.0
    
    Args:
        weights: Dictionary of asset weights
        
    Returns:
        Normalized weights dictionary
    """
    # Normalize absolute weights to sum to 1.0
    abs_total = sum(abs(w) for w in weights.values())
    if abs_total < 1e-6:  # Avoid division by zero
        # Equal weights if all weights are zero
        n_assets = len(weights)
        return {asset: 1.0 / n_assets for asset in weights.keys()}
    else:
        return {asset: weight / abs_total for asset, weight in weights.items()}


def _save_optimization_history(portfolio, scenario, metric: str, n_iterations: int, 
                              final_returns: Dict[str, Any]) -> None:
    """Save optimization history to database"""
    import sqlite3
    import uuid
    from datetime import datetime
    
    db_path = os.path.expanduser("~/.sablier/portfolios.db")
    
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            INSERT INTO portfolio_optimizations 
            (id, portfolio_id, scenario_id, scenario_name, metric, n_iterations,
             final_sharpe, final_return, final_risk, optimization_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            portfolio.id,
            scenario.id,
            scenario.name,
            metric,
            n_iterations,
            final_returns.get('sharpe'),
            final_returns.get('mean_return'),
            final_returns.get('std_return'),
            datetime.utcnow().isoformat() + 'Z'
        ))
        conn.commit()


def _save_evaluation_history(portfolio, scenario, returns_data: Dict[str, Any]) -> None:
    """Save evaluation history to database"""
    import sqlite3
    import uuid
    from datetime import datetime
    
    db_path = os.path.expanduser("~/.sablier/portfolios.db")
    
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            INSERT INTO portfolio_evaluations 
            (id, portfolio_id, scenario_id, scenario_name, sharpe, mean_return, 
             std_return, var_95, var_99, max_drawdown, evaluation_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            portfolio.id,
            scenario.id,
            scenario.name,
            returns_data.get('sharpe'),
            returns_data.get('mean_return'),
            returns_data.get('std_return'),
            returns_data.get('var_95'),
            returns_data.get('var_99'),
            returns_data.get('max_drawdown'),
            datetime.utcnow().isoformat() + 'Z'
        ))
        conn.commit()
