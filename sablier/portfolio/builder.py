"""Portfolio class for asset allocation optimization and analysis"""

import json
import os
import uuid
import sqlite3
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .test import Test
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Represents a portfolio of assets for optimization and analysis
    
    A portfolio defines:
    - Asset allocation weights
    - Optimization constraints
    - Performance evaluation methods
    
    Workflow:
    1. Create portfolio from target set
    2. Set or optimize weights
    3. Evaluate performance across scenarios
    4. Compare scenarios or portfolios
    """
    
    def __init__(self, http_client, portfolio_data: dict):
        """
        Initialize Portfolio instance
        
        Args:
            http_client: HTTP client for API calls
            portfolio_data: Portfolio metadata dictionary
        """
        self.http = http_client
        self._data = portfolio_data
        
        # Core attributes
        self.id = portfolio_data.get('id')
        self.name = portfolio_data.get('name')
        self.description = portfolio_data.get('description', '')
        self.target_set_id = portfolio_data.get('target_set_id')
        self.target_set_name = portfolio_data.get('target_set_name')
        self.assets = portfolio_data.get('assets', [])
        self.weights = portfolio_data.get('weights', {})
        self.capital = portfolio_data.get('capital', 100000.0)  # Default $100k
        self.asset_configs = portfolio_data.get('asset_configs', {})
        self.created_at = portfolio_data.get('created_at')
        
        # Validate weights if provided
        if self.weights:
            self._validate_weights()
        self.updated_at = portfolio_data.get('updated_at')
    
    def _validate_weights(self) -> None:
        """Validate portfolio weights"""
        if not self.weights:
            raise ValueError("Portfolio weights cannot be empty")
        
        # Check that all assets have weights
        missing_assets = set(self.assets) - set(self.weights.keys())
        if missing_assets:
            raise ValueError(f"Missing weights for assets: {missing_assets}")
        
        # Check for extra weights
        extra_assets = set(self.weights.keys()) - set(self.assets)
        if extra_assets:
            raise ValueError(f"Extra weights for assets not in portfolio: {extra_assets}")
        
        # Check that absolute weights sum to 1.0 (long-short portfolios)
        abs_weight_sum = sum(abs(w) for w in self.weights.values())
        if abs(abs_weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Absolute weights must sum to 1.0, got {abs_weight_sum:.6f}")
    
    @property
    def is_optimized(self) -> bool:
        """Check if portfolio has optimized weights"""
        return bool(self.weights)
    
    def save(self) -> None:
        """Save portfolio to local database"""
        # Update timestamp
        self.updated_at = datetime.utcnow().isoformat() + 'Z'
        self._data['updated_at'] = self.updated_at
        
        # Save directly to database
        self._save_to_database()
        
        logger.info(f"Portfolio '{self.name}' saved to database")
    
    def delete(self) -> None:
        """Remove portfolio from local storage"""
        success = self._delete_from_database()
        if success:
            logger.info(f"Portfolio '{self.name}' deleted")
        else:
            logger.warning(f"Failed to delete portfolio '{self.name}'")
    
    def rename(self, new_name: str) -> None:
        """Update portfolio name and save"""
        success = self._rename_in_database(new_name)
        if success:
            self.name = new_name
            self._data['name'] = new_name
            self._data['updated_at'] = datetime.utcnow().isoformat() + 'Z'
            logger.info(f"Portfolio renamed to '{new_name}'")
        else:
            raise ValueError(f"Failed to rename portfolio to '{new_name}'")
    
    def set_weights(self, weights_dict: Dict[str, float]) -> None:
        """
        Manually set portfolio weights
        
        Args:
            weights_dict: Dictionary mapping asset names to weights
        """
        # Validate weights
        if not weights_dict:
            raise ValueError("Weights dictionary cannot be empty")
        
        # Check all assets are in portfolio
        missing_assets = set(weights_dict.keys()) - set(self.assets)
        if missing_assets:
            raise ValueError(f"Assets not in portfolio: {missing_assets}")
        
        # Apply constraints
        weights_dict = self._apply_constraints(weights_dict)
        
        self.weights = weights_dict
        self._data['weights'] = weights_dict
        self.save()
    
    def get_weights(self) -> Dict[str, float]:
        """Return current portfolio weights"""
        return self.weights.copy()
    
    def list_assets(self) -> List[str]:
        """List all assets in this portfolio"""
        return self.assets.copy()
    
    def get_weight_summary(self) -> Dict[str, Any]:
        """Get summary of portfolio weights"""
        if not self.weights:
            return {
                'total_weight': 0,
                'long_weight': 0,
                'short_weight': 0,
                'num_assets': len(self.assets),
                'num_positioned': 0
            }
        
        total_weight = sum(self.weights.values())
        long_weight = sum(w for w in self.weights.values() if w > 0)
        short_weight = sum(w for w in self.weights.values() if w < 0)
        num_positioned = sum(1 for w in self.weights.values() if abs(w) > 1e-6)
        
        return {
            'total_weight': total_weight,
            'long_weight': long_weight,
            'short_weight': short_weight,
            'num_assets': len(self.assets),
            'num_positioned': num_positioned,
            'weights': self.weights
        }
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history for this portfolio"""
        # DEPRECATED: Optimization history is no longer tracked in database
        return []
    
    def _is_compatible_with_scenario(self, scenario) -> bool:
        """
        Check if portfolio is compatible with a scenario (internal method)
        
        Args:
            scenario: Scenario instance
            
        Returns:
            True if portfolio assets match scenario's target set assets
        """
        # Get scenario's target set assets
        scenario_target_set = scenario.model.get_target_set()
        # Extract asset names from feature dictionaries
        scenario_features = scenario_target_set.features
        scenario_assets = set([feature.get('name', feature.get('id', str(feature))) for feature in scenario_features])
        
        # Get portfolio assets
        portfolio_assets = set(self.assets)
        
        # Check if they match exactly
        return scenario_assets == portfolio_assets
    
    def _validate_scenario_compatibility(self, scenario) -> None:
        """
        Validate that portfolio is compatible with scenario (internal method)
        
        Args:
            scenario: Scenario instance
            
        Raises:
            ValueError: If portfolio is not compatible with scenario
        """
        if not self._is_compatible_with_scenario(scenario):
            scenario_target_set = scenario.model.get_target_set()
            # Extract asset names from feature dictionaries
            scenario_features = scenario_target_set.features
            scenario_assets = set([feature.get('name', feature.get('id', str(feature))) for feature in scenario_features])
            portfolio_assets = set(self.assets)
            
            missing_in_portfolio = scenario_assets - portfolio_assets
            missing_in_scenario = portfolio_assets - scenario_assets
            
            error_msg = f"Portfolio '{self.name}' is not compatible with scenario '{scenario.name}':\n"
            
            if missing_in_portfolio:
                error_msg += f"  Portfolio missing assets: {sorted(missing_in_portfolio)}\n"
            
            if missing_in_scenario:
                error_msg += f"  Scenario missing assets: {sorted(missing_in_scenario)}\n"
            
            error_msg += f"  Portfolio assets: {sorted(self.assets)}\n"
            error_msg += f"  Scenario assets: {sorted(scenario_target_set.features)}"
            
            raise ValueError(error_msg)
    
    def optimize(self, scenario, metric: str = "sharpe", n_iterations: int = 100, **kwargs) -> Dict[str, Any]:
        """
        Optimize portfolio weights for a given scenario
        
        Args:
            scenario: Scenario instance with forecast data
            metric: Optimization metric ("sharpe", "return", "risk_adjusted")
            n_iterations: Number of optimization iterations
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary with optimized weights and metrics
        """
        # Validate compatibility
        self._validate_scenario_compatibility(scenario)
        
        from .optimizer import optimize_weights
        
        logger.info(f"Optimizing portfolio '{self.name}' for scenario '{scenario.name}'")
        logger.info(f"  Metric: {metric}, Iterations: {n_iterations}")
        
        result = optimize_weights(
            portfolio=self,
            scenario=scenario,
            metric=metric,
            n_iterations=n_iterations,
            **kwargs
        )
        
        # Update portfolio with optimized weights
        self.set_weights(result['weights'])
        
        logger.info(f"Optimization complete. Sharpe ratio: {result['sharpe']:.3f}")
        return result
    
    def evaluate(self, scenario) -> Dict[str, Any]:
        """
        Calculate portfolio performance metrics for a scenario
        
        Args:
            scenario: Scenario instance with forecast data
            
        Returns:
            Dictionary with performance metrics
        """
        # Validate compatibility
        self._validate_scenario_compatibility(scenario)
        
        from .optimizer import evaluate_portfolio
        
        logger.info(f"Evaluating portfolio '{self.name}' on scenario '{scenario.name}'")
        
        metrics = evaluate_portfolio(
            portfolio=self,
            scenario=scenario
        )
        
        logger.info(f"Evaluation complete. Sharpe: {metrics['sharpe']:.3f}, Return: {metrics['mean_return']:.3f}")
        return metrics
    
    def compare_scenarios(self, scenarios: List, labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare portfolio performance across multiple scenarios
        
        Args:
            scenarios: List of scenario instances
            labels: Optional labels for scenarios
            
        Returns:
            Dictionary with comparison metrics
        """
        if not scenarios:
            raise ValueError("At least one scenario required")
        
        if labels and len(labels) != len(scenarios):
            raise ValueError("Number of labels must match number of scenarios")
        
        if not labels:
            labels = [f"Scenario {i+1}" for i in range(len(scenarios))]
        
        logger.info(f"Comparing portfolio '{self.name}' across {len(scenarios)} scenarios")
        
        comparison = {}
        for scenario, label in zip(scenarios, labels):
            metrics = self.evaluate(scenario)
            comparison[label] = metrics
        
        return comparison
    
    def plot_performance(self, scenario, save: bool = False, save_dir: str = "./portfolio_plots/") -> List[str]:
        """
        Plot portfolio performance for a scenario
        
        Args:
            scenario: Scenario instance
            save: Whether to save plots to disk
            save_dir: Directory to save plots
            
        Returns:
            List of saved plot file paths
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Suppress matplotlib INFO messages
        logging.getLogger('matplotlib.category').setLevel(logging.WARNING)
        
        # Create directory if saving
        if save:
            os.makedirs(save_dir, exist_ok=True)
        
        # Get portfolio returns data
        returns_data = self._extract_portfolio_returns(scenario)
        
        saved_files = []
        
        # Plot 1: Returns distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(returns_data['returns'], bins=50, alpha=0.7, density=True)
        ax.axvline(returns_data['mean_return'], color='red', linestyle='--', 
                  label=f"Mean: {returns_data['mean_return']:.3f}")
        ax.set_xlabel('Portfolio Return')
        ax.set_ylabel('Density')
        ax.set_title(f'Portfolio Returns Distribution\n{self.name} - {scenario.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            file_path = os.path.join(save_dir, f"{self.name}_returns_distribution.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            saved_files.append(file_path)
        
        plt.show()
        
        # Plot 2: Cumulative returns paths
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot sample paths
        for i in range(min(50, len(returns_data['cumulative_paths']))):
            ax.plot(returns_data['cumulative_paths'][i], alpha=0.1, color='blue')
        
        # Plot mean path
        mean_path = np.mean(returns_data['cumulative_paths'], axis=0)
        ax.plot(mean_path, color='red', linewidth=2, label='Mean')
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Cumulative Return')
        ax.set_title(f'Portfolio Cumulative Returns\n{self.name} - {scenario.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            file_path = os.path.join(save_dir, f"{self.name}_cumulative_returns.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            saved_files.append(file_path)
        
        plt.show()
        
        if save and saved_files:
            print(f"\n✅ Saved {len(saved_files)} portfolio plots to {save_dir}")
        
        return saved_files
    
    def plot_scenario_comparison(self, scenarios: List, labels: List[str], save: bool = False) -> List[str]:
        """
        Plot side-by-side scenario comparison
        
        Args:
            scenarios: List of scenario instances
            labels: Labels for scenarios
            save: Whether to save plots
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if len(scenarios) != len(labels):
            raise ValueError("Number of scenarios must match number of labels")
        
        # Suppress matplotlib INFO messages
        logging.getLogger('matplotlib.category').setLevel(logging.WARNING)
        
        saved_files = []
        
        # Get data for all scenarios
        scenario_data = {}
        for scenario, label in zip(scenarios, labels):
            returns_data = self._extract_portfolio_returns(scenario)
            scenario_data[label] = returns_data
        
        # Plot 1: Side-by-side returns distributions
        fig, axes = plt.subplots(1, len(scenarios), figsize=(5*len(scenarios), 6))
        if len(scenarios) == 1:
            axes = [axes]
        
        for i, (label, data) in enumerate(scenario_data.items()):
            axes[i].hist(data['returns'], bins=30, alpha=0.7, density=True)
            axes[i].axvline(data['mean_return'], color='red', linestyle='--',
                           label=f"Mean: {data['mean_return']:.3f}")
            axes[i].set_xlabel('Portfolio Return')
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'{label}\nSharpe: {data["sharpe"]:.3f}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(f'Portfolio Returns Comparison\n{self.name}')
        plt.tight_layout()
        
        if save:
            file_path = f"./portfolio_plots/{self.name}_scenario_comparison.png"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            saved_files.append(file_path)
        
        plt.show()
        
        # Plot 2: Overlaid cumulative returns
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(scenarios)))
        for i, (label, data) in enumerate(scenario_data.items()):
            mean_path = np.mean(data['cumulative_paths'], axis=0)
            ax.plot(mean_path, color=colors[i], linewidth=2, label=f'{label} (Sharpe: {data["sharpe"]:.3f})')
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Cumulative Return')
        ax.set_title(f'Portfolio Cumulative Returns Comparison\n{self.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            file_path = f"./portfolio_plots/{self.name}_cumulative_comparison.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            saved_files.append(file_path)
        
        plt.show()
        
        if save and saved_files:
            print(f"\n✅ Saved {len(saved_files)} comparison plots")
        
        return saved_files
    
    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize portfolio weights so sum of absolute values equals 1.0"""
        # Normalize absolute weights to sum to 1.0
        abs_total = sum(abs(w) for w in weights.values())
        if abs_total < 1e-6:  # Avoid division by zero
            # Equal weights if all weights are zero
            n_assets = len(weights)
            return {asset: 1.0 / n_assets for asset in weights.keys()}
        else:
            return {asset: weight / abs_total for asset, weight in weights.items()}
    
    def _extract_portfolio_returns(self, scenario) -> Dict[str, Any]:
        """Extract portfolio returns data from scenario"""
        from .optimizer import extract_scenario_data, calculate_portfolio_returns
        
        # Extract scenario data
        scenario_data = extract_scenario_data(scenario, self.assets)
        
        # Calculate portfolio returns
        returns_data = calculate_portfolio_returns(scenario_data, self.weights)
        
        return returns_data
    
    def _save_to_database(self) -> None:
        """Save portfolio data to database"""
        import sqlite3
        import json
        
        db_path = os.path.expanduser("~/.sablier/portfolios.db")
        
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO portfolios 
                (id, name, description, target_set_id, target_set_name, assets, 
                 weights, capital, asset_configs, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.id,
                self.name,
                self.description,
                self.target_set_id,
                self.target_set_name,
                json.dumps(self.assets),
                json.dumps(self.weights),
                self.capital,
                json.dumps(self.asset_configs),
                self.created_at,
                self.updated_at
            ))
            conn.commit()
    
    def _delete_from_database(self) -> bool:
        """Delete portfolio from database"""
        import sqlite3
        
        db_path = os.path.expanduser("~/.sablier/portfolios.db")
        
        with sqlite3.connect(db_path) as conn:
            # Delete portfolio (tests are now CASCADE deleted automatically)
            # Delete portfolio
            cursor = conn.execute("DELETE FROM portfolios WHERE id = ?", (self.id,))
            deleted = cursor.rowcount > 0
            
            conn.commit()
        
        return deleted
    
    def _rename_in_database(self, new_name: str) -> bool:
        """Rename portfolio in database"""
        import sqlite3
        
        db_path = os.path.expanduser("~/.sablier/portfolios.db")
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("""
                UPDATE portfolios 
                SET name = ?, updated_at = ?
                WHERE id = ?
            """, (new_name, datetime.utcnow().isoformat() + 'Z', self.id))
            
            updated = cursor.rowcount > 0
            conn.commit()
        
        return updated
    
    
    
    def _aggregate_sample_metrics(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across all samples with time-series analysis"""
        import numpy as np
        
        if not sample_results:
            return {}
        
        # Extract arrays for analysis
        total_returns = np.array([r['total_return'] for r in sample_results])
        pnls = np.array([r['pnl'] for r in sample_results])
        max_drawdowns = np.array([r['max_drawdown'] for r in sample_results])
        sharpe_ratios = np.array([r['sharpe_ratio'] for r in sample_results])
        sortino_ratios = np.array([r['sortino_ratio'] for r in sample_results])
        
        # Extract additional risk metrics
        average_drawdowns = np.array([r['average_drawdown'] for r in sample_results])
        downside_deviations = np.array([r['downside_deviation'] for r in sample_results])
        
        # Calculate annualized volatility for each sample
        annualized_volatilities = []
        for r in sample_results:
            daily_returns = np.array(r.get('daily_returns', []))
            if len(daily_returns) > 0:
                # Annualized volatility = std(daily returns) * sqrt(252 trading days per year)
                daily_std = np.std(daily_returns)
                ann_vol = daily_std * np.sqrt(252)
                annualized_volatilities.append(ann_vol)
            else:
                annualized_volatilities.append(0.0)
        annualized_volatilities = np.array(annualized_volatilities)
        
        # Compute VaR
        var_95 = np.percentile(total_returns, 5)  # 95% VaR (5th percentile)
        var_99 = np.percentile(total_returns, 1)  # 99% VaR (1st percentile)
        
        # Compute CVaR/Expected Shortfall
        cvar_95 = np.mean(total_returns[total_returns <= var_95])
        cvar_99 = np.mean(total_returns[total_returns <= var_99])
        
        # Compute profitability metrics
        profitable_samples = sum(1 for r in sample_results if r['is_profitable'])
        profitability_rate = profitable_samples / len(sample_results)  # Percentage of profitable paths
        
        # NEW: Compute Tail Ratio (upside vs downside)
        tail_ratio = np.percentile(total_returns, 95) / np.percentile(total_returns, 5) if np.percentile(total_returns, 5) != 0 else 0
        
        # NEW: Time-series aggregation
        # Get the number of days from the first sample
        n_days = len(sample_results[0]['daily_metrics']) if sample_results else 0
        time_series_metrics = {}
        
        if n_days > 0:
            # Aggregate metrics for each day across all samples
            for day_idx in range(n_days):
                day_metrics = []
                for sample in sample_results:
                    if day_idx < len(sample['daily_metrics']):
                        day_metrics.append(sample['daily_metrics'][day_idx])
                
                if day_metrics:
                    # Extract arrays for this day (only meaningful daily metrics)
                    pnls_day = np.array([dm['pnl'] for dm in day_metrics])
                    returns_day = np.array([dm['cumulative_return'] for dm in day_metrics])
                    
                    # Compute daily VaR
                    var_95_day = np.percentile(returns_day, 5)
                    var_99_day = np.percentile(returns_day, 1)
                    
                    # Compute daily CVaR
                    cvar_95_day = np.mean(returns_day[returns_day <= var_95_day])
                    cvar_99_day = np.mean(returns_day[returns_day <= var_99_day])
                    
                    time_series_metrics[f'day_{day_idx + 1}'] = {
                        'day': day_idx + 1,
                        'pnl': {
                            'mean': float(np.mean(pnls_day)),
                            'std': float(np.std(pnls_day)),
                            'min': float(np.min(pnls_day)),
                            'max': float(np.max(pnls_day)),
                            'var_95': float(np.percentile(pnls_day, 5)),
                            'var_99': float(np.percentile(pnls_day, 1))
                        },
                        'returns': {
                            'mean': float(np.mean(returns_day)),
                            'std': float(np.std(returns_day)),
                            'min': float(np.min(returns_day)),
                            'max': float(np.max(returns_day)),
                            'var_95': float(var_95_day),
                            'var_99': float(var_99_day),
                            'cvar_95': float(cvar_95_day),
                            'cvar_99': float(cvar_99_day)
                        }
                    }
        
        return {
            'var_95': float(var_95),
            'var_99': float(var_99),
            'cvar_95': float(cvar_95),
            'cvar_99': float(cvar_99),
            'profitability_rate': float(profitability_rate),
            'profitable_samples': profitable_samples,
            'total_samples': len(sample_results),
            'tail_ratio': float(tail_ratio),
            'return_distribution': {
                'mean': float(np.mean(total_returns)),
                'std': float(np.std(total_returns)),
                'skewness': float(self._compute_skewness(total_returns)),
                'kurtosis': float(self._compute_kurtosis(total_returns)),
                'min': float(np.min(total_returns)),
                'max': float(np.max(total_returns))
            },
            'sharpe_distribution': {
                'mean': float(np.mean(sharpe_ratios)),
                'std': float(np.std(sharpe_ratios)),
                'min': float(np.min(sharpe_ratios)),
                'max': float(np.max(sharpe_ratios))
            },
            'drawdown_distribution': {
                'mean': float(np.mean(max_drawdowns)),
                'std': float(np.std(max_drawdowns)),
                'max': float(np.max(max_drawdowns))
            },
            # NEW: Additional risk metric distributions
            'average_drawdown_distribution': {
                'mean': float(np.mean(average_drawdowns)),
                'std': float(np.std(average_drawdowns)),
                'min': float(np.min(average_drawdowns)),
                'max': float(np.max(average_drawdowns))
            },
            'downside_deviation_distribution': {
                'mean': float(np.mean(downside_deviations)),
                'std': float(np.std(downside_deviations)),
                'min': float(np.min(downside_deviations)),
                'max': float(np.max(downside_deviations))
            },
            'annualized_volatility_distribution': {
                'mean': float(np.mean(annualized_volatilities)),
                'median': float(np.median(annualized_volatilities)),
                'std': float(np.std(annualized_volatilities)),
                'min': float(np.min(annualized_volatilities)),
                'max': float(np.max(annualized_volatilities))
            },
            # Time-series aggregated metrics
            'time_series': time_series_metrics,
            'n_days': n_days
        }
    
    def _compute_skewness(self, data):
        """Compute skewness"""
        import numpy as np
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data):
        """Compute kurtosis"""
        import numpy as np
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    
   
    
    

    
    # ============================================
    # TEST MANAGEMENT METHODS
    # ============================================
    
    def test(self, scenario) -> 'Test':
        """
        Run portfolio test against a scenario
        
        Args:
            scenario: Scenario object to test against
            
        Returns:
            Test: Test instance with results
        """
        from .test import Test
        
        # Validate that portfolio has weights
        if not self.weights:
            raise ValueError("Portfolio must have weights before testing. Use set_weights() first.")
        
        # Validate scenario compatibility
        self._validate_scenario_compatibility(scenario)
        
        # Ensure scenario is simulated
        if not scenario.is_simulated:
            print(f"🔄 Simulating scenario '{scenario.name}'...")
            scenario.simulate(n_samples=50)
            print("✅ Scenario simulation complete")
        
        # Run the test and get results
        print(f"🔄 Running portfolio test analysis for scenario '{scenario.name}'...")
        results = self._run_test_analysis(scenario)
        print(f"✅ Portfolio test analysis complete!")
        
        # Save test results to SQLite
        test_id = self._save_test_results(scenario, results)
        
        # Load and return Test instance
        test_data = self._load_test_from_db(test_id)
        return Test(self.id, test_data)
    
    def _find_existing_test(self, scenario) -> Optional['Test']:
        """
        DEPRECATED: This method no longer returns cached tests.
        We always create a new test to allow running tests with different parameters
        (e.g., different number of samples).
        """
        # Always return None to force creation of new test
        return None
    
    def list_tests(self) -> List['Test']:
        """List all tests for this portfolio"""
        from .test import Test
        
        db_path = os.path.expanduser("~/.sablier/portfolios.db")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM portfolio_tests 
                WHERE portfolio_id = ? 
                ORDER BY test_date DESC
            """, (self.id,))
            
            tests = []
            for row in cursor.fetchall():
                test_data = dict(zip([col[0] for col in cursor.description], row))
                tests.append(Test(self.id, test_data))
            
            return tests
    
    def delete_all_tests(self) -> int:
        """Delete all tests for this portfolio"""
        import sqlite3
        
        db_path = os.path.expanduser("~/.sablier/portfolios.db")
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("DELETE FROM portfolio_tests WHERE portfolio_id = ?", (self.id,))
                deleted_count = cursor.rowcount
                conn.commit()
                print(f"✅ Deleted {deleted_count} tests for portfolio '{self.name}'")
                return deleted_count
        except Exception as e:
            print(f"❌ Failed to delete tests: {e}")
            return 0
    
    def get_test(self, identifier) -> Optional['Test']:
        """
        Get test by ID (str) or index (int)
        
        Args:
            identifier: Test ID (str) or index (int)
            
        Returns:
            Test instance or None if not found
        """
        from .test import Test
        
        if isinstance(identifier, int):
            # Get by index
            tests = self.list_tests()
            if 0 <= identifier < len(tests):
                return tests[identifier]
            return None
        elif isinstance(identifier, str):
            # Get by ID
            db_path = os.path.expanduser("~/.sablier/portfolios.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM portfolio_tests 
                    WHERE portfolio_id = ? AND id = ?
                """, (self.id, identifier))
                
                row = cursor.fetchone()
                if row:
                    test_data = dict(zip([col[0] for col in cursor.description], row))
                    return Test(self.id, test_data)
                return None
        else:
            raise ValueError("Identifier must be string (ID) or int (index)")
    
    def _run_test_analysis(self, scenario) -> Dict[str, Any]:
        """Run the actual portfolio test analysis"""
        # Extract scenario data
        price_matrix, future_dates = self._extract_scenario_data(scenario)
        
        # Compute sample metrics
        sample_results = self._compute_sample_metrics(price_matrix, future_dates)
        
        # Aggregate sample metrics
        aggregated_results = self._aggregate_sample_metrics(sample_results)
        
        # Compute summary stats
        summary_stats = self._compute_summary_stats(sample_results)
        
        return {
            'sample_results': sample_results,
            'aggregated_results': aggregated_results,
            'summary_stats': summary_stats
        }
    
    def _extract_scenario_data(self, scenario) -> tuple[np.ndarray, List[str]]:
        """Extract price data and future dates from scenario for portfolio testing"""
        import numpy as np
        
        # Get scenario output
        output = scenario.output
        if not output:
            raise ValueError("Scenario must be simulated before testing")
        
        # Extract reconstructed data
        reconstructed = output.get('conditioning_info', {}).get('reconstructed', [])
        if not reconstructed:
            raise ValueError("No reconstructed data found in scenario output")
        
        # Filter for future forecast windows
        forecast_windows = [
            w for w in reconstructed 
            if w.get('temporal_tag') == 'future' and w.get('_is_historical_pattern') == False
        ]
        
        if not forecast_windows:
            raise ValueError("No forecast windows found in scenario output")
        
        # Extract future dates from scenario output
        # The future_dates are in the simulation_result returned by simulate()
        # We need to get them from the scenario's output structure
        future_dates = output.get('future_dates', [])
        
        if not future_dates:
            raise ValueError("No future_dates found in scenario output")
        
        # Debug temporal tags
        temporal_tags = set()
        historical_patterns = set()
        for window in reconstructed:
            temporal_tags.add(window.get('temporal_tag'))
            historical_patterns.add(window.get('_is_historical_pattern'))
        
        
        # Group by feature (same logic as plotting function)
        feature_data = {}
        for window in forecast_windows:
            # Use 'feature' field like the plotting function does
            feature_name = window.get('feature')
            if feature_name and feature_name in self.assets:
                if feature_name not in feature_data:
                    feature_data[feature_name] = []
                # Use 'reconstructed_values' field like the plotting function does
                feature_data[feature_name].append(window.get('reconstructed_values', []))
        
        
        # Check we have data for all assets
        missing_assets = set(self.assets) - set(feature_data.keys())
        if missing_assets:
            raise ValueError(f"Missing price data for assets: {missing_assets}")
        
        # Debug the data structure
        first_asset = self.assets[0]
        first_window = feature_data[first_asset][0]
        # Convert to numpy array [n_samples, n_days, n_assets]
        # Each feature_data[asset] is a list of samples, each sample is a list of values
        n_samples = len(feature_data[self.assets[0]])  # Number of samples
        n_days = len(feature_data[self.assets[0]][0])  # Number of days per sample
        n_assets = len(self.assets)
        
        price_matrix = np.zeros((n_samples, n_days, n_assets))
        
        for i, asset in enumerate(self.assets):
            # Each feature_data[asset] is a list of samples
            # Each sample is a list of daily values
            asset_samples = feature_data[asset]  # List of samples
            for sample_idx, sample_data in enumerate(asset_samples):
                price_matrix[sample_idx, :, i] = np.array(sample_data)
        
        return price_matrix, future_dates
    
    def _calculate_asset_returns(self, asset_name: str, price_path: np.ndarray, future_dates: List[str]) -> np.ndarray:
        """
        Calculate returns for a specific asset using its configured return calculation method
        
        Args:
            asset_name: Name of the asset
            price_path: Array of prices/yields over time [n_days]
            future_dates: List of date strings ['2025-10-29', ...]
            
        Returns:
            Array of returns over time [n_days-1]
        """
        # Check if asset has custom config
        if asset_name in self.asset_configs:
            config = self.asset_configs[asset_name]
            asset_type = config["type"]
            params = config["params"]
            
            if asset_type == "treasury_bond":
                return self._calculate_treasury_bond_returns(price_path, future_dates, params)
            elif asset_type == "default":
                return self._calculate_default_returns(price_path)
            else:
                raise ValueError(f"Unknown asset type: {asset_type}")
        else:
            # No config = use default calculation (backward compatibility)
            return self._calculate_default_returns(price_path)
    
    def _calculate_default_returns(self, price_path: np.ndarray) -> np.ndarray:
        """Calculate simple price returns"""
        return np.diff(price_path) / price_path[:-1]
    
    def _calculate_treasury_bond_returns(self, yield_path: np.ndarray, future_dates: List[str], params: Dict[str, Any]) -> np.ndarray:
        """
        Calculate Treasury bond returns using YTM method
        
        Args:
            yield_path: Array of yields over time [n_days]
            future_dates: List of date strings ['2025-10-29', ...]
            params: Dict with coupon_rate, face_value, issue_date, payment_frequency
            
        Returns:
            Array of bond returns for each time step [n_days-1]
        """
        # Validate required parameters
        required_params = ['coupon_rate', 'face_value', 'issue_date', 'payment_frequency']
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter '{param}' for Treasury bond")
        
        coupon_rate = params['coupon_rate']
        face_value = params['face_value']
        issue_date_str = params['issue_date']
        payment_frequency = params['payment_frequency']
        
        # Parse dates
        issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d').date()
        forecast_start = datetime.strptime(future_dates[0], '%Y-%m-%d').date()
        
        # Validate issue_date is before forecast start
        if issue_date >= forecast_start:
            raise ValueError(f"Issue date {issue_date_str} must be before forecast start date {future_dates[0]}")
        
        # Calculate coupon payment dates within forecast window
        coupon_dates = self._get_coupon_payment_dates(issue_date, future_dates, payment_frequency)
        
        # Calculate bond prices and returns
        n_days = len(yield_path)
        bond_prices = np.zeros(n_days)
        returns = np.zeros(n_days - 1)
        
        # Calculate bond price for each day
        for t in range(n_days):
            bond_prices[t] = self._calculate_bond_price(
                yield_path[t], coupon_rate, face_value, issue_date, 
                datetime.strptime(future_dates[t], '%Y-%m-%d').date(), payment_frequency
            )
        
        # Calculate returns (price change + coupon payments)
        for t in range(1, n_days):
            price_return = (bond_prices[t] - bond_prices[t-1]) / bond_prices[t-1]
            
            # Add coupon payment if due on this date
            coupon_return = 0.0
            current_date = datetime.strptime(future_dates[t], '%Y-%m-%d').date()
            if current_date in coupon_dates:
                coupon_return = (coupon_rate / payment_frequency) / bond_prices[t-1]
            
            returns[t-1] = price_return + coupon_return
        
        return returns
    
    def _get_coupon_payment_dates(self, issue_date: date, future_dates: List[str], payment_frequency: int) -> List[date]:
        """Find which dates in future_dates have coupon payments"""
        from dateutil.relativedelta import relativedelta
        
        coupon_dates = []
        forecast_start = datetime.strptime(future_dates[0], '%Y-%m-%d').date()
        forecast_end = datetime.strptime(future_dates[-1], '%Y-%m-%d').date()
        
        # Calculate coupon payment interval
        months_per_payment = 12 // payment_frequency
        
        # Find first coupon date after issue date
        current_coupon_date = issue_date
        while current_coupon_date <= forecast_start:
            current_coupon_date = current_coupon_date + relativedelta(months=months_per_payment)
        
        # Find all coupon dates within forecast window
        while current_coupon_date <= forecast_end:
            # Convert to business day if weekend
            if current_coupon_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                # Move to next Monday
                days_to_add = 7 - current_coupon_date.weekday()
                current_coupon_date = current_coupon_date + relativedelta(days=days_to_add)
            
            coupon_dates.append(current_coupon_date)
            current_coupon_date = current_coupon_date + relativedelta(months=months_per_payment)
        
        return coupon_dates
    
    def _calculate_bond_price(self, yield_rate: float, coupon_rate: float, face_value: float, 
                            issue_date: date, current_date: date, payment_frequency: int) -> float:
        """
        Calculate bond price using yield-to-maturity formula
        
        Args:
            yield_rate: Current market yield (annual)
            coupon_rate: Annual coupon rate
            face_value: Bond face value
            issue_date: When bond was issued
            current_date: Current date
            payment_frequency: Payments per year
            
        Returns:
            Bond price
        """
        from dateutil.relativedelta import relativedelta
        
        # Calculate years to maturity from current date
        years_to_maturity = (issue_date + relativedelta(years=10) - current_date).days / 365.25
        
        if years_to_maturity <= 0:
            return face_value  # Bond at maturity
        
        # Calculate coupon payment
        coupon_payment = (coupon_rate / payment_frequency) * face_value
        
        # Calculate number of remaining payments
        payments_per_year = payment_frequency
        total_payments = int(years_to_maturity * payments_per_year)
        
        # Calculate present value of coupon payments
        coupon_pv = 0.0
        for i in range(1, total_payments + 1):
            coupon_pv += coupon_payment / ((1 + yield_rate / payments_per_year) ** i)
        
        # Calculate present value of face value
        face_pv = face_value / ((1 + yield_rate / payments_per_year) ** total_payments)
        
        return coupon_pv + face_pv
    
    def _compute_sample_metrics(self, price_matrix: np.ndarray, future_dates: List[str]) -> List[Dict[str, Any]]:
        """Compute metrics for each sample path using asset-specific return calculations"""
        import numpy as np
        
        n_samples, n_days, n_assets = price_matrix.shape
        sample_results = []
        
        for sample_idx in range(n_samples):
            # Get price path for this sample
            price_path = price_matrix[sample_idx]  # [n_days, n_assets]
            
            # Calculate returns for each asset using their specific method
            asset_returns = {}
            for i, asset in enumerate(self.assets):
                asset_price_path = price_path[:, i]  # [n_days]
                asset_returns[asset] = self._calculate_asset_returns(asset, asset_price_path, future_dates)
            
            # Compute portfolio values using returns
            portfolio_values = np.zeros(n_days)
            portfolio_values[0] = self.capital  # Start with initial capital
            
            for t in range(1, n_days):
                portfolio_value = portfolio_values[t-1]
                for asset in self.assets:
                    weight = self.weights[asset]
                    asset_return = asset_returns[asset][t-1]  # Returns are n_days-1 length
                    portfolio_value += weight * portfolio_values[t-1] * asset_return
                portfolio_values[t] = portfolio_value
            
            # Compute returns
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            cumulative_returns = (portfolio_values / portfolio_values[0]) - 1
            
            # Compute PnL and total return (these should match!)
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            pnl = final_value - initial_value
            total_return = pnl / initial_value  # This should equal cumulative_returns[-1]
            
            # Compute risk metrics
            if len(daily_returns) > 0:
                # Sharpe ratio (excess return over risk-free rate / volatility)
                risk_free_rate_daily = 0.02 / 252  # 2% annual risk-free rate
                excess_returns = daily_returns - risk_free_rate_daily
                sharpe_ratio = np.mean(excess_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
                # Sortino ratio (excess return over risk-free rate / downside deviation)
                negative_returns = daily_returns[daily_returns < 0]
                downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0
                sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
                
                # Max drawdown - CORRECTED VERSION
                # Drawdown = (Current - Peak) / Peak (negative when below peak)
                running_max = np.maximum.accumulate(portfolio_values)
                drawdowns = (portfolio_values - running_max) / running_max
                max_drawdown = np.min(drawdowns)  # Most negative drawdown (worst case)
                
                
                
                # Average drawdown - FIXED!
                # Average of all drawdowns (including zeros)
                average_drawdown = np.mean(drawdowns)
                
                # Downside deviation
                downside_returns = daily_returns[daily_returns < 0]
                downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
                
            else:
                sharpe_ratio = sortino_ratio = 0
                max_drawdown = average_drawdown = downside_deviation = 0.0
            
            # Daily metrics for time-series analysis
            daily_metrics = []
            for t in range(n_days):
                daily_pnl = portfolio_values[t] - initial_value
                daily_cumulative_return = (portfolio_values[t] / initial_value) - 1
                daily_return = daily_returns[t-1] if t > 0 else 0
                
                daily_metric = {
                    'day': t,
                    'portfolio_value': float(portfolio_values[t]),
                    'pnl': float(daily_pnl),
                    'cumulative_return': float(daily_cumulative_return),
                    'daily_return': float(daily_return),
                    'drawdown': float(drawdowns[t])
                }
                daily_metrics.append(daily_metric)
            
            sample_result = {
                'sample_idx': sample_idx,
                'total_return': float(total_return),
                'pnl': float(pnl),
                'max_drawdown': float(max_drawdown),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'average_drawdown': float(average_drawdown),
                'downside_deviation': float(downside_deviation),
                'daily_returns': daily_returns.tolist(),
                'cumulative_returns': cumulative_returns.tolist(),
                'is_profitable': bool(pnl > 0),
                'daily_metrics': daily_metrics,
                'initial_value': float(initial_value),
                'final_value': float(final_value)
            }
            sample_results.append(sample_result)
        
        return sample_results
    
    def _aggregate_sample_metrics(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across all samples"""
        import numpy as np
        
        # Extract arrays for aggregation
        total_returns = np.array([s['total_return'] for s in sample_results])
        sharpe_ratios = np.array([s['sharpe_ratio'] for s in sample_results])
        max_drawdowns = np.array([s['max_drawdown'] for s in sample_results])
        average_drawdowns = np.array([s['average_drawdown'] for s in sample_results])
        downside_deviations = np.array([s['downside_deviation'] for s in sample_results])
        
        # Count samples
        profitable_samples = sum(1 for s in sample_results if s['is_profitable'])
        total_samples = len(sample_results)
        
        # Compute VaR and CVaR
        var_95 = np.percentile(total_returns, 5)
        var_99 = np.percentile(total_returns, 1)
        cvar_95 = np.mean(total_returns[total_returns <= var_95])
        cvar_99 = np.mean(total_returns[total_returns <= var_99])
        
        # Tail ratio
        tail_ratio = np.percentile(total_returns, 95) / abs(np.percentile(total_returns, 5)) if np.percentile(total_returns, 5) != 0 else 0
        
        # Time-series aggregation
        time_series_metrics = {}
        if sample_results and 'daily_metrics' in sample_results[0]:
            n_days = len(sample_results[0]['daily_metrics'])
            for day in range(n_days):
                daily_pnls = [s['daily_metrics'][day]['pnl'] for s in sample_results]
                daily_returns = [s['daily_metrics'][day]['cumulative_return'] for s in sample_results]
                daily_portfolio_values = [s['daily_metrics'][day]['portfolio_value'] for s in sample_results]
                daily_drawdowns = [s['daily_metrics'][day]['drawdown'] for s in sample_results]

                # Convert to numpy arrays for proper indexing
                daily_pnls_array = np.array(daily_pnls)
                daily_returns_array = np.array(daily_returns)
                daily_portfolio_values_array = np.array(daily_portfolio_values)
                daily_drawdowns_array = np.array(daily_drawdowns)
                
                # Compute CVaR (Conditional Value at Risk)
                var_95_pnl = np.percentile(daily_pnls_array, 5)
                var_99_pnl = np.percentile(daily_pnls_array, 1)
                var_95_returns = np.percentile(daily_returns_array, 5)
                var_99_returns = np.percentile(daily_returns_array, 1)
                var_95_portfolio = np.percentile(daily_portfolio_values_array, 5)
                var_99_portfolio = np.percentile(daily_portfolio_values_array, 1)
                
                cvar_95_pnl = np.mean(daily_pnls_array[daily_pnls_array <= var_95_pnl]) if np.any(daily_pnls_array <= var_95_pnl) else var_95_pnl
                cvar_99_pnl = np.mean(daily_pnls_array[daily_pnls_array <= var_99_pnl]) if np.any(daily_pnls_array <= var_99_pnl) else var_99_pnl
                cvar_95_returns = np.mean(daily_returns_array[daily_returns_array <= var_95_returns]) if np.any(daily_returns_array <= var_95_returns) else var_95_returns
                cvar_99_returns = np.mean(daily_returns_array[daily_returns_array <= var_99_returns]) if np.any(daily_returns_array <= var_99_returns) else var_99_returns
                cvar_95_portfolio = np.mean(daily_portfolio_values_array[daily_portfolio_values_array <= var_95_portfolio]) if np.any(daily_portfolio_values_array <= var_95_portfolio) else var_95_portfolio
                cvar_99_portfolio = np.mean(daily_portfolio_values_array[daily_portfolio_values_array <= var_99_portfolio]) if np.any(daily_portfolio_values_array <= var_99_portfolio) else var_99_portfolio

                time_series_metrics[f'day_{day}'] = {
                    'day': day,
                    'pnl': {
                        'mean': float(np.mean(daily_pnls_array)),
                        'std': float(np.std(daily_pnls_array)),
                        'var_95': float(var_95_pnl),
                        'var_99': float(var_99_pnl),
                        'cvar_95': float(cvar_95_pnl),
                        'cvar_99': float(cvar_99_pnl)
                    },
                    'returns': {
                        'mean': float(np.mean(daily_returns_array)),
                        'std': float(np.std(daily_returns_array)),
                        'var_95': float(var_95_returns),
                        'var_99': float(var_99_returns),
                        'cvar_95': float(cvar_95_returns),
                        'cvar_99': float(cvar_99_returns)
                    },
                    'portfolio_value': {
                        'mean': float(np.mean(daily_portfolio_values_array)),
                        'std': float(np.std(daily_portfolio_values_array)),
                        'var_95': float(var_95_portfolio),
                        'var_99': float(var_99_portfolio),
                        'cvar_95': float(cvar_95_portfolio),
                        'cvar_99': float(cvar_99_portfolio)
                    },
                    'drawdown': {
                        'mean': float(np.mean(daily_drawdowns_array)),
                        'std': float(np.std(daily_drawdowns_array)),
                        'min': float(np.min(daily_drawdowns_array)),
                        'max': float(np.max(daily_drawdowns_array))
                    }
                }
        
        # Calculate annualized volatility for each sample
        annualized_volatilities = []
        for s in sample_results:
            daily_returns = np.array(s.get('daily_returns', []))
            if len(daily_returns) > 0:
                daily_std = np.std(daily_returns)
                ann_vol = daily_std * np.sqrt(252)
                annualized_volatilities.append(ann_vol)
            else:
                annualized_volatilities.append(0.0)
        annualized_volatilities = np.array(annualized_volatilities)
        
        return {
            'profitability_rate': profitable_samples / total_samples,
            'tail_ratio': float(tail_ratio),
            'var_95': float(var_95),
            'var_99': float(var_99),
            'cvar_95': float(cvar_95),
            'cvar_99': float(cvar_99),
            'profitable_samples': profitable_samples,
            'total_samples': total_samples,
            'return_distribution': {
                'mean': float(np.mean(total_returns)),
                'std': float(np.std(total_returns)),
                'min': float(np.min(total_returns)),
                'max': float(np.max(total_returns))
            },
            'sharpe_distribution': {
                'mean': float(np.mean(sharpe_ratios)),
                'std': float(np.std(sharpe_ratios)),
                'min': float(np.min(sharpe_ratios)),
                'max': float(np.max(sharpe_ratios))
            },
            'drawdown_distribution': {
                'mean': float(np.mean(max_drawdowns)),
                'std': float(np.std(max_drawdowns)),
                'min': float(np.min(max_drawdowns)),
                'max': float(np.max(max_drawdowns))
            },
            'average_drawdown_distribution': {
                'mean': float(np.mean(average_drawdowns)),
                'std': float(np.std(average_drawdowns)),
                'min': float(np.min(average_drawdowns)),
                'max': float(np.max(average_drawdowns))
            },
            'downside_deviation_distribution': {
                'mean': float(np.mean(downside_deviations)),
                'std': float(np.std(downside_deviations)),
                'min': float(np.min(downside_deviations)),
                'max': float(np.max(downside_deviations))
            },
            'annualized_volatility_distribution': {
                'mean': float(np.mean(annualized_volatilities)),
                'median': float(np.median(annualized_volatilities)),
                'std': float(np.std(annualized_volatilities)),
                'min': float(np.min(annualized_volatilities)),
                'max': float(np.max(annualized_volatilities))
            },
            'time_series': time_series_metrics,
            'n_days': n_days if sample_results else 0
        }
    
    def _compute_summary_stats(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics"""
        import numpy as np
        
        total_returns = np.array([s['total_return'] for s in sample_results])
        sharpe_ratios = np.array([s['sharpe_ratio'] for s in sample_results])
        max_drawdowns = np.array([s['max_drawdown'] for s in sample_results])
        average_drawdowns = np.array([s['average_drawdown'] for s in sample_results])
        downside_deviations = np.array([s['downside_deviation'] for s in sample_results])
        
        return {
            'total_return': {
                'mean': float(np.mean(total_returns)),
                'median': float(np.median(total_returns)),
                'std': float(np.std(total_returns)),
                'p25': float(np.percentile(total_returns, 25)),
                'p75': float(np.percentile(total_returns, 75))
            },
            'sharpe_ratio': {
                'mean': float(np.mean(sharpe_ratios)),
                'median': float(np.median(sharpe_ratios)),
                'std': float(np.std(sharpe_ratios)),
                'p25': float(np.percentile(sharpe_ratios, 25)),
                'p75': float(np.percentile(sharpe_ratios, 75))
            },
            'max_drawdown': {
                'mean': float(np.mean(max_drawdowns)),
                'median': float(np.median(max_drawdowns)),
                'std': float(np.std(max_drawdowns)),
                'p25': float(np.percentile(max_drawdowns, 25)),
                'p75': float(np.percentile(max_drawdowns, 75))
            },
            'average_drawdown': {
                'mean': float(np.mean(average_drawdowns)),
                'median': float(np.median(average_drawdowns)),
                'std': float(np.std(average_drawdowns))
            },
            'downside_deviation': {
                'mean': float(np.mean(downside_deviations)),
                'median': float(np.median(downside_deviations)),
                'std': float(np.std(downside_deviations))
            }
        }
    
    def _save_test_results(self, scenario, results: Dict[str, Any]) -> str:
        """Save test results to SQLite database"""
        import sqlite3
        import uuid
        from datetime import datetime
        
        test_id = str(uuid.uuid4())
        test_date = datetime.utcnow().isoformat() + 'Z'
        
        db_path = os.path.expanduser("~/.sablier/portfolios.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                INSERT INTO portfolio_tests 
                (id, portfolio_id, scenario_id, scenario_name, test_date,
                 sample_results, aggregated_results, summary_stats, time_series_metrics, n_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_id,
                self.id,
                scenario.id,
                scenario.name,
                test_date,
                json.dumps(results['sample_results']),
                json.dumps(results['aggregated_results']),
                json.dumps(results['summary_stats']),
                json.dumps(results['aggregated_results'].get('time_series', {})),
                results['aggregated_results'].get('n_days', 0)
            ))
            conn.commit()
        
        return test_id
    
    def _load_test_from_db(self, test_id: str) -> Dict[str, Any]:
        """Load test data from SQLite database"""
        import sqlite3
        
        db_path = os.path.expanduser("~/.sablier/portfolios.db")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM portfolio_tests WHERE id = ?
            """, (test_id,))
            
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Test {test_id} not found in database")
            
            return dict(zip([col[0] for col in cursor.description], row))
    
    def info(self) -> None:
        """Display comprehensive portfolio information"""
        print(f"📊 PORTFOLIO INFORMATION")
        print("=" * 50)
        print(f"Name: {self.name}")
        print(f"ID: {self.id}")
        print(f"Description: {self.description}")
        print(f"Capital: ${self.capital:,.2f}")
        print(f"Target Set: {self.target_set_name} (ID: {self.target_set_id})")
        print(f"Created: {self.created_at}")
        print(f"Updated: {self.updated_at}")
        
        print(f"\n📈 ASSET ALLOCATION")
        print("-" * 30)
        if self.weights:
            # For long-short: show absolute weights sum
            total_weight = sum(abs(w) for w in self.weights.values())
            total_allocation = self.capital
            
            for asset, weight in self.weights.items():
                percentage = weight * 100
                allocation = weight * self.capital
                position_type = "LONG" if weight >= 0 else "SHORT"
                print(f"{asset}: {percentage:6.1f}% (${allocation:8,.2f}) [{position_type}]")
            
            print(f"{'Total:':<20} {total_weight*100:6.1f}% (${total_allocation:8,.2f})")
            
            # Debug info
            print(f"\n🔍 DEBUG INFO")
            print(f"Absolute weights sum: {total_weight:.6f}")
            print(f"Raw weights sum: {sum(self.weights.values()):.6f}")
            print(f"Capital: ${self.capital:,.2f}")
        else:
            print("No weights assigned")
        
        print(f"\n📋 ASSETS ({len(self.assets)})")
        print("-" * 20)
        for i, asset in enumerate(self.assets, 1):
            print(f"{i:2d}. {asset}")
        
        print(f"\n🧪 TESTS")
        print("-" * 15)
        tests = self.list_tests()
        if tests:
            print(f"Total tests: {len(tests)}")
            for i, test in enumerate(tests, 1):
                print(f"{i:2d}. {test.scenario_name} ({test.test_date})")
        else:
            print("No tests run yet")
    
    def __repr__(self) -> str:
        return f"Portfolio(id='{self.id}', name='{self.name}', assets={len(self.assets)})"
