"""Portfolio class for asset allocation optimization and analysis"""

import json
import os
import uuid
import sqlite3
from datetime import datetime
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
        self.constraint_type = portfolio_data.get('constraint_type', 'long_short')
        self.custom_constraints = portfolio_data.get('custom_constraints')
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
        
        # Check for negative weights if not long_short
        if self.constraint_type == 'long_only':
            negative_weights = [asset for asset, weight in self.weights.items() if weight < 0]
            if negative_weights:
                raise ValueError(f"Long-only constraint violated for assets: {negative_weights}")
        
        # Check weight sum based on constraint type
        if self.constraint_type == 'long_only':
            # For long-only: raw weights must sum to 1.0
            weight_sum = sum(self.weights.values())
            if abs(weight_sum - 1.0) > 1e-6:
                raise ValueError(f"Weights must sum to 1.0, got {weight_sum:.6f}")
        else:
            # For long-short: absolute weights must sum to 1.0
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
        return self._get_optimization_history()
    
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
            constraint_type=self.constraint_type,
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
        """Apply portfolio constraints to weights"""
        if self.constraint_type == "long_only":
            # Ensure all weights are non-negative
            weights = {asset: max(0, weight) for asset, weight in weights.items()}
        elif self.constraint_type == "long_short":
            # Allow negative weights (no change needed)
            pass
        elif self.constraint_type == "custom":
            # Apply custom constraints if defined
            if self.custom_constraints:
                # Implementation depends on custom constraint format
                pass
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if abs(total_weight) > 1e-6:  # Avoid division by zero
            weights = {asset: weight / total_weight for asset, weight in weights.items()}
        
        return weights
    
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
                 constraint_type, custom_constraints, weights, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.id,
                self.name,
                self.description,
                self.target_set_id,
                self.target_set_name,
                json.dumps(self.assets),
                self.constraint_type,
                json.dumps(self.custom_constraints),
                json.dumps(self.weights),
                self.created_at,
                self.updated_at
            ))
            conn.commit()
    
    def _delete_from_database(self) -> bool:
        """Delete portfolio from database"""
        import sqlite3
        
        db_path = os.path.expanduser("~/.sablier/portfolios.db")
        
        with sqlite3.connect(db_path) as conn:
            # Delete related records first
            conn.execute("DELETE FROM portfolio_optimizations WHERE portfolio_id = ?", (self.id,))
            conn.execute("DELETE FROM portfolio_evaluations WHERE portfolio_id = ?", (self.id,))
            
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
    
    def _get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history from database"""
        import sqlite3
        
        db_path = os.path.expanduser("~/.sablier/portfolios.db")
        
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM portfolio_optimizations 
                WHERE portfolio_id = ? 
                ORDER BY optimization_date DESC
            """, (self.id,))
            
            rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def _get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get evaluation history from database"""
        import sqlite3
        
        db_path = os.path.expanduser("~/.sablier/portfolios.db")
        
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM portfolio_evaluations 
                WHERE portfolio_id = ? 
                ORDER BY evaluation_date DESC
            """, (self.id,))
            
            rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    # ============================================
    # PORTFOLIO TESTING METHODS
    # ============================================
    
    def test(self, scenario, risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Test portfolio performance against a scenario simulation
        
        Args:
            scenario: Scenario instance with simulated data
            risk_free_rate: Risk-free rate for Sharpe/Sortino calculations (annual)
            
        Returns:
            Dict containing comprehensive test results
        """
        # Validate scenario compatibility
        self._validate_scenario_compatibility(scenario)
        
        # Ensure scenario is simulated
        if not scenario.is_simulated:
            raise ValueError("Scenario must be simulated before testing. Run scenario.simulate() first.")
        
        # Extract scenario data
        scenario_data = self._extract_scenario_data(scenario)
        
        # Compute per-sample metrics
        sample_results = self._compute_sample_metrics(scenario_data, risk_free_rate)
        
        # Aggregate cross-sample metrics
        aggregated_results = self._aggregate_sample_metrics(sample_results)
        
        # Compute summary statistics
        summary_stats = self._compute_summary_stats(sample_results)
        
        # Store test results
        test_results = {
            'scenario_id': scenario.id,
            'scenario_name': scenario.name,
            'test_date': datetime.utcnow().isoformat() + 'Z',
            'sample_results': sample_results,
            'aggregated_results': aggregated_results,
            'summary_stats': summary_stats
        }
        
        # Save to database
        self._save_test_results(test_results)
        
        logger.info(f"Portfolio '{self.name}' tested against scenario '{scenario.name}'")
        return test_results
    
    def _extract_scenario_data(self, scenario) -> Dict[str, Any]:
        """Extract price paths for portfolio assets from scenario output"""
        import numpy as np
        
        # Get scenario output
        output = scenario.output
        if not output:
            raise ValueError("Scenario output is empty")
        
        # Get reconstructed windows from output (same as plotting method)
        reconstructed_windows = output.get('conditioning_info', {}).get('reconstructed', [])
        
        if not reconstructed_windows:
            raise ValueError("No reconstructed data found in scenario output")
        
        # Find forecast windows - these are future windows that are NOT historical patterns
        forecast_windows = [w for w in reconstructed_windows if 
                          w.get('temporal_tag') == 'future' and 
                          w.get('_is_historical_pattern') == False]
        
        if not forecast_windows:
            raise ValueError("No forecast windows found in scenario output")
        
        # Group by feature (same logic as plotting method)
        feature_forecasts = {}
        for window in forecast_windows:
            feat = window.get('feature')
            if feat:
                if feat not in feature_forecasts:
                    feature_forecasts[feat] = []
                feature_forecasts[feat].append(window.get('reconstructed_values', []))
        
        # Extract price paths for each asset
        asset_paths = {}
        dates = output.get('future_dates', [])
        
        for asset in self.assets:
            if asset in feature_forecasts:
                # Convert to numpy array: [n_samples, n_timesteps]
                asset_paths[asset] = np.array(feature_forecasts[asset])
            else:
                raise ValueError(f"Asset '{asset}' not found in scenario forecast data")
        
        # Get dimensions
        first_asset = self.assets[0]
        n_samples, n_days = asset_paths[first_asset].shape
        
        # Create price matrix: [n_samples, n_days, n_assets]
        price_matrix = np.zeros((n_samples, n_days, len(self.assets)))
        
        for i, asset in enumerate(self.assets):
            price_matrix[:, :, i] = asset_paths[asset]
        
        return {
            'price_matrix': price_matrix,
            'dates': dates,
            'assets': self.assets,
            'n_samples': n_samples,
            'n_days': n_days
        }
    
    def _compute_sample_metrics(self, scenario_data: Dict[str, Any], risk_free_rate: float) -> List[Dict[str, Any]]:
        """Compute metrics for each sample path with time-series analysis"""
        import numpy as np
        
        price_matrix = scenario_data['price_matrix']
        n_samples, n_days, n_assets = price_matrix.shape
        sample_results = []
        
        for sample_idx in range(n_samples):
            # Get price path for this sample
            prices = price_matrix[sample_idx, :, :]  # [n_days, n_assets]
            
            # Compute daily returns
            daily_returns = np.zeros(n_days - 1)
            for t in range(1, n_days):
                for i, asset in enumerate(self.assets):
                    if prices[t-1, i] > 0:  # Avoid division by zero
                        asset_return = (prices[t, i] / prices[t-1, i]) - 1
                        daily_returns[t-1] += self.weights.get(asset, 0) * asset_return
            
            # Compute cumulative returns
            cumulative_returns = np.cumprod(1 + daily_returns) - 1
            
            # Compute initial portfolio value
            initial_value = np.sum([self.weights.get(asset, 0) * prices[0, i] for i, asset in enumerate(self.assets)])
            
            # Compute time-series metrics for each day (only PnL and cumulative returns)
            daily_metrics = []
            for t in range(1, n_days):
                # Portfolio value at day t
                portfolio_value_t = np.sum([self.weights.get(asset, 0) * prices[t, i] for i, asset in enumerate(self.assets)])
                
                # PnL at day t
                pnl_t = portfolio_value_t - initial_value
                
                # Cumulative return at day t
                cumulative_return_t = (portfolio_value_t / initial_value) - 1
                
                daily_metric = {
                    'day': t,
                    'portfolio_value': float(portfolio_value_t),
                    'pnl': float(pnl_t),
                    'cumulative_return': float(cumulative_return_t),
                    'daily_return': float(daily_returns[t-1]) if t > 0 else 0
                }
                daily_metrics.append(daily_metric)
            
            # Final metrics (end of path)
            total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
            final_value = np.sum([self.weights.get(asset, 0) * prices[-1, i] for i, asset in enumerate(self.assets)])
            pnl = final_value - initial_value
            
            # Final maximum drawdown
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns) / (1 + peak)
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            # Final Sharpe ratio (annualized)
            if len(daily_returns) > 1:
                excess_returns = daily_returns - (risk_free_rate / 252)
                sharpe_ratio = np.mean(excess_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Final Sortino ratio (annualized)
            if len(daily_returns) > 1:
                negative_returns = daily_returns[daily_returns < 0]
                downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0
                sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
            else:
                sortino_ratio = 0
            
            # Final Calmar ratio
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
            
            # NEW: Additional risk metrics
            # Information Ratio (using risk-free rate as benchmark)
            excess_returns = daily_returns - (risk_free_rate / 252)
            
            # Average Drawdown (mean of all drawdowns)
            average_drawdown = np.mean(drawdown) if len(drawdown) > 0 else 0
            
            # Downside Deviation (std of negative returns only)
            negative_returns = daily_returns[daily_returns < 0]
            downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0
            
            sample_result = {
                'sample_idx': sample_idx,
                'total_return': float(total_return),
                'pnl': float(pnl),
                'max_drawdown': float(max_drawdown),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'calmar_ratio': float(calmar_ratio),
                'average_drawdown': float(average_drawdown),
                'downside_deviation': float(downside_deviation),
                'daily_returns': daily_returns.tolist(),
                'cumulative_returns': cumulative_returns.tolist(),
                'is_profitable': bool(pnl > 0),
                'survives': bool(total_return > -0.5),
                # NEW: Time-series metrics
                'daily_metrics': daily_metrics,
                'initial_value': float(initial_value),
                'final_value': float(final_value)
            }
            
            sample_results.append(sample_result)
        
        return sample_results
    
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
        calmar_ratios = np.array([r['calmar_ratio'] for r in sample_results])
        
        # NEW: Extract additional risk metrics
        average_drawdowns = np.array([r['average_drawdown'] for r in sample_results])
        downside_deviations = np.array([r['downside_deviation'] for r in sample_results])
        
        # Compute VaR
        var_95 = np.percentile(total_returns, 5)  # 95% VaR (5th percentile)
        var_99 = np.percentile(total_returns, 1)  # 99% VaR (1st percentile)
        
        # Compute CVaR/Expected Shortfall
        cvar_95 = np.mean(total_returns[total_returns <= var_95])
        cvar_99 = np.mean(total_returns[total_returns <= var_99])
        
        # Compute survival metrics
        profitable_samples = sum(1 for r in sample_results if r['is_profitable'])
        surviving_samples = sum(1 for r in sample_results if r['survives'])
        survival_rate = surviving_samples / len(sample_results)
        profit_probability = profitable_samples / len(sample_results)
        
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
            'survival_rate': float(survival_rate),
            'profit_probability': float(profit_probability),
            'profitable_samples': profitable_samples,
            'surviving_samples': surviving_samples,
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
            # NEW: Time-series aggregated metrics
            'time_series': time_series_metrics,
            'n_days': n_days
        }
    
    def _compute_summary_stats(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics"""
        import numpy as np
        
        if not sample_results:
            return {}
        
        total_returns = np.array([r['total_return'] for r in sample_results])
        sharpe_ratios = np.array([r['sharpe_ratio'] for r in sample_results])
        max_drawdowns = np.array([r['max_drawdown'] for r in sample_results])
        
        # NEW: Extract additional risk metrics
        average_drawdowns = np.array([r['average_drawdown'] for r in sample_results])
        downside_deviations = np.array([r['downside_deviation'] for r in sample_results])
        
        return {
            'total_return': {
                'mean': float(np.mean(total_returns)),
                'median': float(np.median(total_returns)),
                'std': float(np.std(total_returns)),
                'percentile_25': float(np.percentile(total_returns, 25)),
                'percentile_75': float(np.percentile(total_returns, 75))
            },
            'sharpe_ratio': {
                'mean': float(np.mean(sharpe_ratios)),
                'median': float(np.median(sharpe_ratios)),
                'std': float(np.std(sharpe_ratios))
            },
            'max_drawdown': {
                'mean': float(np.mean(max_drawdowns)),
                'median': float(np.median(max_drawdowns)),
                'max': float(np.max(max_drawdowns))
            },
            # NEW: Additional risk metric summaries
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
    
    def _save_test_results(self, test_results: Dict[str, Any]):
        """Save test results to database with enhanced time-series support"""
        import sqlite3
        import json
        
        test_id = str(uuid.uuid4())
        db_path = os.path.expanduser("~/.sablier/portfolios.db")
        
        # Apply migration if needed
        self._apply_migration_002(db_path)
        
        with sqlite3.connect(db_path) as conn:
            # Extract time-series data
            aggregated_results = test_results['aggregated_results']
            time_series_metrics = aggregated_results.get('time_series', {})
            n_days = aggregated_results.get('n_days', 0)
            
            conn.execute("""
                INSERT INTO portfolio_tests 
                (id, portfolio_id, scenario_id, scenario_name, test_date,
                 sample_results, aggregated_results, summary_stats, 
                 time_series_metrics, n_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_id,
                self.id,
                test_results['scenario_id'],
                test_results['scenario_name'],
                test_results['test_date'],
                json.dumps(test_results['sample_results']),
                json.dumps(test_results['aggregated_results']),
                json.dumps(test_results['summary_stats']),
                json.dumps(time_series_metrics),
                n_days
            ))
            conn.commit()
    
    def _apply_migration_002(self, db_path: str):
        """Apply migration 002 to enhance portfolio tests with time-series metrics"""
        import sqlite3
        
        with sqlite3.connect(db_path) as conn:
            # Check if columns already exist
            cursor = conn.execute("PRAGMA table_info(portfolio_tests)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'time_series_metrics' not in columns:
                conn.execute("ALTER TABLE portfolio_tests ADD COLUMN time_series_metrics TEXT")
                print("✅ Added time_series_metrics column to portfolio_tests")
            
            if 'n_days' not in columns:
                conn.execute("ALTER TABLE portfolio_tests ADD COLUMN n_days INTEGER")
                print("✅ Added n_days column to portfolio_tests")
            
            # Create index if it doesn't exist
            try:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_tests_n_days ON portfolio_tests(n_days)")
            except sqlite3.OperationalError:
                pass  # Index might already exist
            
            conn.commit()
    
    def get_test_history(self) -> List[Dict[str, Any]]:
        """Get historical test results for this portfolio"""
        import sqlite3
        
        db_path = os.path.expanduser("~/.sablier/portfolios.db")
        
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM portfolio_tests 
                WHERE portfolio_id = ?
                ORDER BY test_date DESC
            """, (self.id,))
            
            rows = cursor.fetchall()
        
        results = []
        for row in rows:
            result = {
                'id': row['id'],
                'scenario_id': row['scenario_id'],
                'scenario_name': row['scenario_name'],
                'test_date': row['test_date'],
                'sample_results': json.loads(row['sample_results']),
                'aggregated_results': json.loads(row['aggregated_results']),
                'summary_stats': json.loads(row['summary_stats'])
            }
            results.append(result)
        
        return results
    
    def get_test_results(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Get test results for a specific scenario"""
        history = self.get_test_history()
        for result in history:
            if result['scenario_id'] == scenario_id:
                return result
        return None
    
    # ============================================
    # PORTFOLIO TEST PLOTTING METHODS
    # ============================================
    
    def plot_test_results(self, scenario, save: bool = False, save_dir: str = None, 
                         display: bool = True) -> List[str]:
        """
        Plot comprehensive portfolio test results
        
        Args:
            scenario: Scenario instance or scenario_id string
            save: Whether to save plots to disk
            save_dir: Directory to save plots (default: ./portfolio_plots/)
            display: Whether to display plots inline
            
        Returns:
            List of saved plot file paths
        """
        # Get test results
        if isinstance(scenario, str):
            scenario_id = scenario
            results = self.get_test_results(scenario_id)
            if not results:
                raise ValueError(f"No test results found for scenario {scenario_id}")
        else:
            scenario_id = scenario.id
            results = self.get_test_results(scenario_id)
            if not results:
                raise ValueError(f"No test results found for scenario {scenario.name}. Run portfolio.test(scenario) first.")
        
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        saved_files = []
        
        # Plot 1: Return Distribution
        saved_files.extend(self._plot_return_distribution(results, save, save_dir, display))
        
        # Plot 2: Sharpe Ratio Distribution
        saved_files.extend(self._plot_sharpe_distribution(results, save, save_dir, display))
        
        # Plot 3: Drawdown Distribution
        saved_files.extend(self._plot_drawdown_distribution(results, save, save_dir, display))
        
        # Plot 4: Sample Paths (first 10 samples)
        saved_files.extend(self._plot_sample_paths(results, n_samples=10, save=save, save_dir=save_dir, display=display))
        
        # Plot 5: Risk-Return Scatter
        saved_files.extend(self._plot_risk_return_scatter(results, save, save_dir, display))
        
        return saved_files
    
    def plot_sample_path(self, scenario, sample_idx: int, save: bool = False, 
                        save_dir: str = None, display: bool = True) -> str:
        """
        Plot a single sample path showing portfolio performance over time
        
        Args:
            scenario: Scenario instance or scenario_id string
            sample_idx: Index of the sample to plot
            save: Whether to save plot to disk
            save_dir: Directory to save plot
            display: Whether to display plot inline
            
        Returns:
            Path to saved plot file (if saved)
        """
        # Get test results
        if isinstance(scenario, str):
            scenario_id = scenario
            results = self.get_test_results(scenario_id)
        else:
            scenario_id = scenario.id
            results = self.get_test_results(scenario_id)
        
        if not results:
            raise ValueError(f"No test results found for scenario {scenario_id}")
        
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        sample_results = results['sample_results']
        if sample_idx >= len(sample_results):
            raise ValueError(f"Sample index {sample_idx} out of range. Available: 0-{len(sample_results)-1}")
        
        sample = sample_results[sample_idx]
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Setup save directory
        if save_dir is None:
            save_dir = './portfolio_plots/'
        os.makedirs(save_dir, exist_ok=True)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Cumulative Returns
        cumulative_returns = np.array(sample['cumulative_returns'])
        days = np.arange(len(cumulative_returns))
        
        ax1.plot(days, cumulative_returns * 100, 'b-', linewidth=2, label=f'Sample {sample_idx}')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_title(f'Portfolio Cumulative Returns - Sample {sample_idx}')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add metrics text
        metrics_text = f"Total Return: {sample['total_return']:.1%}\n"
        metrics_text += f"Sharpe Ratio: {sample['sharpe_ratio']:.2f}\n"
        metrics_text += f"Max Drawdown: {sample['max_drawdown']:.1%}\n"
        metrics_text += f"PnL: {sample['pnl']:.2f}\n"
        metrics_text += f"Profitable: {sample['is_profitable']}\n"
        metrics_text += f"Survives: {sample['survives']}"
        
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Daily Returns
        daily_returns = np.array(sample['daily_returns'])
        daily_days = np.arange(1, len(daily_returns) + 1)  # Match daily_returns length
        ax2.plot(daily_days, daily_returns * 100, 'r-', linewidth=1, alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_title(f'Portfolio Daily Returns - Sample {sample_idx}')
        ax2.set_xlabel('Days')
        ax2.set_ylabel('Daily Return (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save or display
        saved_files = []
        if save:
            filename = f"portfolio_sample_{sample_idx}_{self.name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files[0] if saved_files else ""
    
    def _plot_return_distribution(self, results: Dict[str, Any], save: bool, save_dir: str, display: bool) -> List[str]:
        """Plot return distribution across samples"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        sample_results = results['sample_results']
        returns = [s['total_return'] for s in sample_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1.hist(returns, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(np.mean(returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.1%}')
        ax1.axvline(np.percentile(returns, 5), color='orange', linestyle='--', linewidth=2, label=f'VaR 95%: {np.percentile(returns, 5):.1%}')
        ax1.set_title('Portfolio Return Distribution')
        ax1.set_xlabel('Total Return')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(returns, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        ax2.set_title('Portfolio Return Distribution (Box Plot)')
        ax2.set_ylabel('Total Return')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        saved_files = []
        if save:
            filename = f"return_distribution_{self.name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def _plot_sharpe_distribution(self, results: Dict[str, Any], save: bool, save_dir: str, display: bool) -> List[str]:
        """Plot Sharpe ratio distribution across samples"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        sample_results = results['sample_results']
        sharpe_ratios = [s['sharpe_ratio'] for s in sample_results]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.hist(sharpe_ratios, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(np.mean(sharpe_ratios), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sharpe_ratios):.2f}')
        ax.axvline(np.median(sharpe_ratios), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(sharpe_ratios):.2f}')
        ax.set_title('Portfolio Sharpe Ratio Distribution')
        ax.set_xlabel('Sharpe Ratio')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        saved_files = []
        if save:
            filename = f"sharpe_distribution_{self.name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def _plot_drawdown_distribution(self, results: Dict[str, Any], save: bool, save_dir: str, display: bool) -> List[str]:
        """Plot maximum drawdown distribution across samples"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        sample_results = results['sample_results']
        drawdowns = [s['max_drawdown'] for s in sample_results]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.hist(drawdowns, bins=30, alpha=0.7, color='red', edgecolor='black')
        ax.axvline(np.mean(drawdowns), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(drawdowns):.1%}')
        ax.axvline(np.max(drawdowns), color='darkred', linestyle='--', linewidth=2, label=f'Max: {np.max(drawdowns):.1%}')
        ax.set_title('Portfolio Maximum Drawdown Distribution')
        ax.set_xlabel('Maximum Drawdown')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        saved_files = []
        if save:
            filename = f"drawdown_distribution_{self.name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def _plot_sample_paths(self, results: Dict[str, Any], n_samples: int = 10, 
                          save: bool = False, save_dir: str = None, display: bool = True) -> List[str]:
        """Plot multiple sample paths showing portfolio performance"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        sample_results = results['sample_results']
        n_to_plot = min(n_samples, len(sample_results))
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_to_plot))
        
        for i in range(n_to_plot):
            sample = sample_results[i]
            cumulative_returns = np.array(sample['cumulative_returns'])
            days = np.arange(len(cumulative_returns))
            
            ax.plot(days, cumulative_returns * 100, '-', color=colors[i], 
                   alpha=0.7, linewidth=1, label=f'Sample {i}' if i < 5 else "")
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_title(f'Portfolio Cumulative Returns - {n_to_plot} Sample Paths')
        ax.set_xlabel('Days')
        ax.set_ylabel('Cumulative Return (%)')
        ax.grid(True, alpha=0.3)
        
        if n_to_plot <= 5:
            ax.legend()
        
        plt.tight_layout()
        
        saved_files = []
        if save:
            filename = f"sample_paths_{self.name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def _plot_risk_return_scatter(self, results: Dict[str, Any], save: bool, save_dir: str, display: bool) -> List[str]:
        """Plot risk-return scatter plot for all samples"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        sample_results = results['sample_results']
        returns = [s['total_return'] for s in sample_results]
        sharpe_ratios = [s['sharpe_ratio'] for s in sample_results]
        drawdowns = [s['max_drawdown'] for s in sample_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Return vs Sharpe
        scatter1 = ax1.scatter(returns, sharpe_ratios, alpha=0.6, c=drawdowns, 
                              cmap='RdYlBu_r', s=50)
        ax1.set_xlabel('Total Return')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_title('Return vs Sharpe Ratio')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Max Drawdown')
        
        # Return vs Drawdown
        scatter2 = ax2.scatter(returns, drawdowns, alpha=0.6, c=sharpe_ratios, 
                              cmap='viridis', s=50)
        ax2.set_xlabel('Total Return')
        ax2.set_ylabel('Max Drawdown')
        ax2.set_title('Return vs Max Drawdown')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Sharpe Ratio')
        
        plt.tight_layout()
        
        saved_files = []
        if save:
            filename = f"risk_return_scatter_{self.name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def plot_time_series_evolution(self, scenario, save: bool = False, save_dir: str = None, 
                                  display: bool = True) -> List[str]:
        """
        Plot time-series evolution of portfolio metrics over the forecast horizon
        
        Generates 3 plots showing meaningful daily metrics:
        - PnL Evolution: Portfolio profit/loss over time with VaR bands
        - Return Evolution: Cumulative returns over time with confidence bands  
        - Risk Evolution: VaR and CVaR evolution over time
        
        Args:
            scenario: Scenario instance or scenario_id string
            save: Whether to save plots to disk
            save_dir: Directory to save plots (default: ./portfolio_plots/)
            display: Whether to display plots inline
            
        Returns:
            List of saved plot file paths
        """
        # Get test results
        if isinstance(scenario, str):
            scenario_id = scenario
            results = self.get_test_results(scenario_id)
        else:
            scenario_id = scenario.id
            results = self.get_test_results(scenario_id)
        
        if not results:
            raise ValueError(f"No test results found for scenario {scenario_id}")
        
        # Set default save directory and create it if needed
        if save:
            if save_dir is None:
                save_dir = './portfolio_plots/'
            os.makedirs(save_dir, exist_ok=True)
        
        time_series = results['aggregated_results'].get('time_series', {})
        if not time_series:
            raise ValueError("No time-series data found. Run portfolio.test(scenario) first.")
        
        saved_files = []
        
        # Plot 1: PnL Evolution
        saved_files.extend(self._plot_pnl_evolution(time_series, save, save_dir, display))
        
        # Plot 2: Return Evolution
        saved_files.extend(self._plot_return_evolution(time_series, save, save_dir, display))
        
        # Plot 3: Risk Metrics Evolution (VaR/CVaR only)
        saved_files.extend(self._plot_risk_evolution(time_series, save, save_dir, display))
        
        return saved_files
    
    def _plot_pnl_evolution(self, time_series: Dict[str, Any], save: bool, save_dir: str, display: bool) -> List[str]:
        """Plot PnL evolution over time"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        days = []
        mean_pnl = []
        var_95_pnl = []
        var_99_pnl = []
        std_pnl = []
        
        for day_key, day_data in time_series.items():
            days.append(day_data['day'])
            mean_pnl.append(day_data['pnl']['mean'])
            var_95_pnl.append(day_data['pnl']['var_95'])
            var_99_pnl.append(day_data['pnl']['var_99'])
            std_pnl.append(day_data['pnl']['std'])
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot mean PnL
        ax.plot(days, mean_pnl, 'b-', linewidth=2, label='Mean PnL')
        
        # Plot confidence bands
        ax.fill_between(days, var_95_pnl, var_99_pnl, alpha=0.2, color='red', 
                       label='95%-99% VaR Band')
        
        # Plot standard deviation bands
        upper_std = np.array(mean_pnl) + np.array(std_pnl)
        lower_std = np.array(mean_pnl) - np.array(std_pnl)
        ax.fill_between(days, lower_std, upper_std, alpha=0.3, color='blue', 
                       label='±1 Std Dev')
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Days')
        ax.set_ylabel('PnL')
        ax.set_title(f'Portfolio PnL Evolution Over Time\n{self.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"pnl_evolution_{self.name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def _plot_return_evolution(self, time_series: Dict[str, Any], save: bool, save_dir: str, display: bool) -> List[str]:
        """Plot cumulative return evolution over time"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        days = []
        mean_return = []
        var_95_return = []
        var_99_return = []
        std_return = []
        
        for day_key, day_data in time_series.items():
            days.append(day_data['day'])
            mean_return.append(day_data['returns']['mean'])
            var_95_return.append(day_data['returns']['var_95'])
            var_99_return.append(day_data['returns']['var_99'])
            std_return.append(day_data['returns']['std'])
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot mean return
        ax.plot(days, np.array(mean_return) * 100, 'g-', linewidth=2, label='Mean Return (%)')
        
        # Plot confidence bands
        ax.fill_between(days, np.array(var_95_return) * 100, np.array(var_99_return) * 100, 
                       alpha=0.2, color='red', label='95%-99% VaR Band')
        
        # Plot standard deviation bands
        upper_std = (np.array(mean_return) + np.array(std_return)) * 100
        lower_std = (np.array(mean_return) - np.array(std_return)) * 100
        ax.fill_between(days, lower_std, upper_std, alpha=0.3, color='green', 
                       label='±1 Std Dev')
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Days')
        ax.set_ylabel('Cumulative Return (%)')
        ax.set_title(f'Portfolio Cumulative Return Evolution\n{self.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"return_evolution_{self.name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
    def _plot_risk_evolution(self, time_series: Dict[str, Any], save: bool, save_dir: str, display: bool) -> List[str]:
        """Plot risk metrics evolution over time (VaR and CVaR only)"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        days = []
        var_95_returns = []
        var_99_returns = []
        cvar_95_returns = []
        cvar_99_returns = []
        
        for day_key, day_data in time_series.items():
            days.append(day_data['day'])
            var_95_returns.append(day_data['returns']['var_95'])
            var_99_returns.append(day_data['returns']['var_99'])
            cvar_95_returns.append(day_data['returns']['cvar_95'])
            cvar_99_returns.append(day_data['returns']['cvar_99'])
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot VaR evolution
        ax.plot(days, np.array(var_95_returns) * 100, 'b-', linewidth=2, label='95% VaR (%)')
        ax.plot(days, np.array(var_99_returns) * 100, 'r-', linewidth=2, label='99% VaR (%)')
        
        # Plot CVaR evolution
        ax.plot(days, np.array(cvar_95_returns) * 100, 'b--', linewidth=2, label='95% CVaR (%)')
        ax.plot(days, np.array(cvar_99_returns) * 100, 'r--', linewidth=2, label='99% CVaR (%)')
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Days')
        ax.set_ylabel('Risk Level (%)')
        ax.set_title(f'Portfolio Risk Evolution (VaR/CVaR)\n{self.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        saved_files = []
        if save:
            filename = f"risk_evolution_{self.name.replace(' ', '_')}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"✅ Saved: {filepath}")
        
        if display:
            plt.show()
        else:
            plt.close()
        
        return saved_files
    
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
        
        # Check if test already exists for this portfolio-scenario combination
        existing_test = self._find_existing_test(scenario)
        if existing_test:
            print(f"📊 Found existing test for scenario '{scenario.name}' - returning cached results")
            return existing_test
        
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
        """Find existing test for the same portfolio-scenario combination"""
        import sqlite3
        from .test import Test
        
        db_path = os.path.expanduser("~/.sablier/portfolios.db")
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM portfolio_tests 
                    WHERE portfolio_id = ? AND scenario_id = ?
                    ORDER BY test_date DESC
                    LIMIT 1
                """, (self.id, scenario.id))
                
                row = cursor.fetchone()
                if row:
                    # Convert row to dict
                    columns = [description[0] for description in cursor.description]
                    test_data = dict(zip(columns, row))
                    return Test(self.id, test_data)
        except Exception as e:
            print(f"⚠️ Could not check for existing tests: {e}")
        
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
        print(f"🔍 DEBUG: Starting _run_test_analysis for scenario '{scenario.name}'")
        
        # Extract scenario data
        price_matrix = self._extract_scenario_data(scenario)
        print(f"🔍 DEBUG: Extracted price matrix shape: {price_matrix.shape}")
        
        # Compute sample metrics
        print(f"🔍 DEBUG: Computing sample metrics...")
        sample_results = self._compute_sample_metrics(price_matrix)
        print(f"🔍 DEBUG: Computed metrics for {len(sample_results)} samples")
        
        # Aggregate sample metrics
        aggregated_results = self._aggregate_sample_metrics(sample_results)
        
        # Compute summary stats
        summary_stats = self._compute_summary_stats(sample_results)
        
        return {
            'sample_results': sample_results,
            'aggregated_results': aggregated_results,
            'summary_stats': summary_stats
        }
    
    def _extract_scenario_data(self, scenario) -> np.ndarray:
        """Extract price data from scenario for portfolio testing"""
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
        
        return price_matrix
    
    def _compute_sample_metrics(self, price_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Compute metrics for each sample path"""
        import numpy as np
        
        n_samples, n_days, n_assets = price_matrix.shape
        sample_results = []
        
        for sample_idx in range(n_samples):
            # Get price path for this sample
            price_path = price_matrix[sample_idx]  # [n_days, n_assets]
            
            # Compute portfolio values (correct for long-short)
            portfolio_values = np.zeros(n_days)
            for t in range(n_days):
                portfolio_value = 0
                for i, asset in enumerate(self.assets):
                    weight = self.weights[asset]
                    # For long-short: positive weights = long, negative weights = short
                    portfolio_value += weight * price_path[t, i] * self.capital
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
                
                # Max drawdown - DEBUG VERSION
                # Drawdown = (Peak - Current) / Peak
                running_max = np.maximum.accumulate(portfolio_values)
                drawdowns = (running_max - portfolio_values) / running_max
                max_drawdown = np.min(drawdowns)  # Most negative drawdown (worst case)
                
                
                
                # Average drawdown - FIXED!
                # Average of all drawdowns (including zeros)
                average_drawdown = np.mean(drawdowns)
                
                # Downside deviation
                downside_returns = daily_returns[daily_returns < 0]
                downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
                
                # Calmar ratio - FIXED!
                # Calmar = Annual Return / Max Drawdown
                calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
                
            else:
                sharpe_ratio = sortino_ratio = calmar_ratio = 0
                max_drawdown = average_drawdown = downside_deviation = 0
            
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
                'calmar_ratio': float(calmar_ratio),
                'average_drawdown': float(average_drawdown),
                'downside_deviation': float(downside_deviation),
                'daily_returns': daily_returns.tolist(),
                'cumulative_returns': cumulative_returns.tolist(),
                'is_profitable': bool(pnl > 0),
                'survives': bool(total_return > -0.5),
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
        surviving_samples = sum(1 for s in sample_results if s['survives'])
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
        
        return {
            'survival_rate': surviving_samples / total_samples,
            'profit_probability': profitable_samples / total_samples,
            'tail_ratio': float(tail_ratio),
            'var_95': float(var_95),
            'var_99': float(var_99),
            'cvar_95': float(cvar_95),
            'cvar_99': float(cvar_99),
            'profitable_samples': profitable_samples,
            'surviving_samples': surviving_samples,
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
        print(f"Constraint Type: {self.constraint_type}")
        print(f"Target Set: {self.target_set_name} (ID: {self.target_set_id})")
        print(f"Created: {self.created_at}")
        print(f"Updated: {self.updated_at}")
        
        print(f"\n📈 ASSET ALLOCATION")
        print("-" * 30)
        if self.weights:
            if self.constraint_type == 'long_only':
                total_weight = sum(self.weights.values())
                total_allocation = self.capital
            else:
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
            if self.constraint_type == 'long_only':
                print(f"Weights sum: {total_weight:.6f}")
            else:
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
