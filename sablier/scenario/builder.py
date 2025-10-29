"""Scenario class representing a market scenario for conditional generation"""

import logging
from typing import Optional, Any, List, Dict
from datetime import datetime, timedelta
from ..http_client import HTTPClient

logger = logging.getLogger(__name__)


class Scenario:
    """
    Represents a market scenario for conditional synthetic data generation
    
    A scenario defines the conditioning context for generating synthetic market paths:
    - Past: Recent historical data (fetched from live sources)
    - Future: User-defined or sample-based conditioning
    
    Workflow:
    1. Create scenario (linked to trained model)
    2. Fetch recent past data
    3. Configure future conditioning
    4. Generate synthetic paths
    5. Analyze and validate results
    """
    
    def __init__(self, http_client: HTTPClient, scenario_data: dict, model, interactive: bool = True):
        """
        Initialize Scenario instance
        
        Args:
            http_client: HTTP client for API requests
            scenario_data: Scenario data from API
            model: Associated Model instance
            interactive: Whether to prompt for confirmations (default: True)
        """
        self.http = http_client
        self._data = scenario_data
        self.model = model
        self.interactive = interactive
        
        # Core attributes
        self.id = scenario_data.get('id')
        self.name = scenario_data.get('name')
        self.description = scenario_data.get('description', '')
        self.model_id = scenario_data.get('model_id')
        self.simulation_date = scenario_data.get('simulation_date')
        self.feature_simulation_dates = scenario_data.get('feature_simulation_dates', {})
        self.status = scenario_data.get('status', 'created')
        self.output = scenario_data.get('output')
        self.last_simulated_date = scenario_data.get('last_simulated_date')
    
    @property
    def is_simulated(self) -> bool:
        """Check if scenario has been simulated"""
        return self.status == 'simulation_done' and self.output is not None
    
    def __repr__(self):
        return f"Scenario(id='{self.id}', name='{self.name}', status='{self.status}')"
    
    # ============================================
    # SIMULATION
    # ============================================
    
    def simulate(self, n_samples: int = 1000, force: bool = False) -> Dict[str, Any]:
        """
        Run the scenario simulation by calling the forecast endpoint.
        
        If the scenario was already simulated, this will re-run it with fresh data,
        overwriting the previous results.
        
        Args:
            n_samples: Number of forecast samples to generate
            force: Skip confirmation prompt for re-simulation (default: False)
            
        Returns:
            Forecast response data
            
        Example:
            >>> scenario.simulate(n_samples=1000)
            >>> # Re-run with fresh data
            >>> scenario.simulate(n_samples=1000)  # Will prompt for confirmation
            >>> scenario.simulate(n_samples=1000, force=True)  # Skip confirmation
        """
        if not self.simulation_date:
            raise ValueError("Scenario must have a simulation_date configured")
        
        # Check if already simulated
        is_resimulation = bool(self._data.get('output'))
        
        if is_resimulation and not force and self.interactive:
            print(f"\n⚠️  Scenario '{self.name}' has already been simulated.")
            print(f"   Re-running will fetch fresh market data and overwrite the previous results.")
            response = input("   Continue? (y/N): ").strip().lower()
            if response != 'y':
                print("❌ Re-simulation cancelled.")
                return self._data.get('output', {})
            print()
        
        if is_resimulation:
            print(f"[Scenario {self.name}] Re-running simulation with fresh data...")
        else:
            print(f"[Scenario {self.name}] Running simulation...")
        
        print(f"  Simulation date: {self.simulation_date}")
        print(f"  Number of samples: {n_samples}")
        
        # Call forecast endpoint with scenario-based conditioning
        # Use the authenticated user's ID (scenario owner), not the model's user_id
        # This allows using template models (owned by template user) while accessing
        # the current user's scenarios
        user_id = self._data.get("user_id")
        if not user_id:
            # Fallback: get from HTTP client's auth (if available)
            user_id = getattr(self.http.auth_handler, 'user_id', None)
        
        response = self.http.post('/api/v1/ml/forecast', {
            'user_id': user_id,
            'model_id': self.model_id,
            'conditioning_source': 'scenario',
            'scenario_id': self.id,
            'n_samples': n_samples
        })
        
        print(f"✅ Simulation complete!")
        print(f"  Status: {response.get('status')}")
        print(f"  Generated {response.get('n_samples', 0)} forecast samples")
        
        # Refetch scenario from database to get the full output structure
        refreshed_data = self.http.get(f'/api/v1/scenarios/{self.id}')
        
        # Update local data with refreshed scenario (includes full output)
        self._data.update(refreshed_data)
        self.output = refreshed_data.get('output')
        self.status = refreshed_data.get('status', 'simulation_done')
        
        return response
    
    # ============================================
    # PLOTTING AND ANALYSIS
    # ============================================
    
    def plot_forecasts(
        self, 
        feature: Optional[str] = None,
        features: Optional[List[str]] = None, 
        save: bool = False,
        save_dir: Optional[str] = None,
        display: bool = True
    ) -> List[str]:
        """
        Plot forecast paths with conditioning and ground truth
        
        Shows:
        - Past trajectories (historical data)
        - Future ground truth (if available from historical simulation)
        - Confidence intervals (68% and 95%)
        - Limited number of individual forecast paths
        - Median forecast
        
        Args:
            feature: Single feature to plot and display inline (optional)
            features: List of features to plot when saving (default: all)
            save: Whether to save plots to disk (default: False)
            save_dir: Directory to save plots (default: ./forecasts/)
            display: Whether to display plot inline (default: True, only when feature specified)
            
        Returns:
            List of saved plot file paths (empty if save=False)
            
        Examples:
            >>> # Display a single feature inline
            >>> scenario.plot_forecasts(feature="10-Year Treasury Rate")
            
            >>> # Save all features to disk without displaying
            >>> scenario.plot_forecasts(save=True, display=False)
            
            >>> # Display one feature AND save all features
            >>> scenario.plot_forecasts(feature="S&P 500", save=True)
        """
        if not self.is_simulated:
            # Try to refresh from database
            refreshed_data = self.http.get(f'/api/v1/scenarios/{self.id}')
            self._data.update(refreshed_data)
            self.output = refreshed_data.get('output')
            self.status = refreshed_data.get('status', 'created')
            
            if not self.is_simulated:
                print(f"Debug: status={self.status}, output type={type(self.output)}, output is not None={self.output is not None}")
                raise ValueError("Scenario must be simulated before plotting. Run scenario.simulate() first.")
        
        import matplotlib.pyplot as plt
        import numpy as np
        import logging
        import os
        
        # Suppress matplotlib INFO messages about categorical units
        logging.getLogger('matplotlib.category').setLevel(logging.WARNING)
        
        # Default save directory
        if save_dir is None:
            save_dir = './forecasts/'
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Get reconstructed windows from output
        output = self.output
        reconstructed_windows = output.get('conditioning_info', {}).get('reconstructed', [])
        
        if not reconstructed_windows:
            raise ValueError("No reconstructed data found in scenario output")
        
        # Find forecast windows - these are future windows that are NOT historical patterns
        forecast_windows = [w for w in reconstructed_windows if 
                          w.get('temporal_tag') == 'future' and 
                          w.get('_is_historical_pattern') == False]
        
        if not forecast_windows:
            print("Warning: No forecast windows found")
            return []
        
        # Group by feature
        feature_forecasts = {}
        for window in forecast_windows:
            feat = window.get('feature')
            if feat:
                if feat not in feature_forecasts:
                    feature_forecasts[feat] = []
                feature_forecasts[feat].append(window.get('reconstructed_values', []))
        
        if not feature_forecasts:
            print("Warning: No features available to plot")
            return []
        
        # Determine which features to process
        features_to_plot = []
        
        # If a single feature is specified for display, use that
        if feature is not None:
            if feature in feature_forecasts:
                features_to_plot = [feature]
            else:
                available = ', '.join(list(feature_forecasts.keys())[:5])
                raise ValueError(f"Feature '{feature}' not found. Available: {available}...")
        
        # If saving, determine which features to save
        if save:
            if features is None:
                # Save all features
                save_features = list(feature_forecasts.keys())
            else:
                # Save specified features
                save_features = [f for f in features if f in feature_forecasts]
        else:
            save_features = []
        
        # Combine both lists (for display and/or save)
        all_features_to_process = list(set(features_to_plot + save_features))
        
        if not all_features_to_process and not feature:
            raise ValueError("No features specified. Use feature='name' to display or save=True to save all")
        
        saved_files = []
        
        # Plot each feature
        for feature_name in all_features_to_process:
            should_display_this = (feature_name == feature and display)
            should_save_this = (save and feature_name in save_features)
            fig, ax = plt.subplots(1, 1, figsize=(14, 6))
            
            # Get forecast data
            forecast_paths = feature_forecasts[feature_name]
            forecasts_array = np.array(forecast_paths)
            n_samples, n_timesteps = forecasts_array.shape
            
            # Get ground truth data (past and future)
            past_values = self._get_past_values(feature_name)
            future_gt_values = self._get_ground_truth_values(feature_name)
            
            # Setup time axis with dates (if available)
            past_dates = self.output.get('past_dates', [])
            future_dates = self.output.get('future_dates', [])
            
            if past_dates and future_dates:
                # Use actual dates from forecast response
                past_t = past_dates
                future_t = future_dates
                use_dates = True
            elif past_values:
                # Fallback to numeric indices if dates not available
                past_t = np.arange(len(past_values))
                future_t = np.arange(len(past_values), len(past_values) + n_timesteps)
                use_dates = False
            else:
                # No past data, just use future indices
                past_t = []
                future_t = np.arange(n_timesteps)
                use_dates = False
            
            # Plot ground truth past (black line with markers)
            if past_values and len(past_t) > 0:
                ax.plot(past_t, past_values, 'o-', color='black', linewidth=2, 
                       markersize=4, alpha=0.8, label='Recent Past', zorder=5)
                
                # Plot ground truth future (green line with markers)
                if future_gt_values and len(future_gt_values) > 0:
                    # Handle case where ground truth might be longer than forecast
                    if use_dates:
                        # Use actual dates for ground truth
                        future_t_gt = future_dates[:len(future_gt_values)]
                    else:
                        # Use numeric indices for ground truth
                        future_t_gt = np.arange(len(past_values), len(past_values) + len(future_gt_values))
                    
                    # Get simulation_date from scenario for label
                    simulation_date = self._data.get('simulation_date', '')
                    hist_label = f'Historical Path ({simulation_date})' if simulation_date else 'Historical Path'
                    
                    ax.plot(future_t_gt, future_gt_values, 'o-', color='green', linewidth=2.5, 
                           markersize=5, alpha=0.9, label=hist_label, zorder=6)
                
                # Vertical line at forecast start (red dotted)
                if not use_dates:
                    ax.axvline(x=len(past_values), color='red', linestyle=':', 
                              linewidth=2, alpha=0.5, label='Forecast Start', zorder=4)
            
            # Plot individual forecast paths (light blue, semi-transparent)
            n_to_plot = min(50, n_samples)  # Show up to 50 paths
            for i in range(n_to_plot):
                ax.plot(future_t, forecasts_array[i], '-', alpha=0.2,
                       linewidth=0.8, color='steelblue', zorder=2)
            
            # Add legend entry for forecast paths
            ax.plot([], [], '-', alpha=0.5, linewidth=1.5, color='steelblue',
                   label=f'Forecast Paths (n={n_samples})', zorder=2)
            
            # Add confidence intervals (68% and 95%)
            ci_levels = [0.68, 0.95]
            ci_colors = ['darkblue', 'steelblue']
            ci_alphas = [0.2, 0.15]
            
            for ci_idx, ci_level in enumerate(ci_levels):
                lower_q = (1 - ci_level) / 2
                upper_q = 1 - lower_q
                
                lower = np.percentile(forecasts_array, lower_q * 100, axis=0)
                upper = np.percentile(forecasts_array, upper_q * 100, axis=0)
                
                ax.fill_between(future_t, lower, upper, alpha=ci_alphas[ci_idx], 
                               color=ci_colors[ci_idx],
                               label=f'{int(ci_level*100)}% CI', zorder=3)
            
            # Plot median forecast (dark red line)
            median_forecast = np.median(forecasts_array, axis=0)
            ax.plot(future_t, median_forecast, '-', color='darkred', 
                   linewidth=2.5, alpha=0.9, label='Median Forecast', zorder=7)
            
            # Formatting
            ax.set_title(f'{feature_name} - Conditional Forecast', fontsize=14, fontweight='bold')
            
            if use_dates:
                # Format x-axis for dates
                ax.set_xlabel('Date', fontsize=12)
                # Rotate labels and show every Nth date to avoid crowding
                n_dates = len(past_t) + len(future_t)
                tick_interval = max(1, n_dates // 10)  # Show ~10 ticks
                all_dates = list(past_t) + list(future_t)
                tick_indices = range(0, n_dates, tick_interval)
                ax.set_xticks([all_dates[i] for i in tick_indices if i < len(all_dates)])
                ax.tick_params(axis='x', rotation=45)
            else:
                # Numeric time steps
                ax.set_xlabel('Time Step', fontsize=12)
            
            ax.set_ylabel(f'{feature_name} Value', fontsize=12)
            ax.legend(loc='best', fontsize=10, framealpha=0.95)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add statistics text box
            stats_text = f'Min: {np.min(forecasts_array):.3f}\n'
            stats_text += f'Max: {np.max(forecasts_array):.3f}\n'
            stats_text += f'Median: {np.median(median_forecast):.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            # Save if requested
            if should_save_this:
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                safe_feature_name = feature_name.replace('/', '_').replace('\\', '_')
                save_path = os.path.join(save_dir, f'{safe_feature_name}_forecast.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                saved_files.append(save_path)
                if display:
                    print(f"  ✅ Saved: {save_path}")
            
            # Display if requested
            if should_display_this:
                plt.show()
            else:
                plt.close()
        
        if save and saved_files:
            print(f"\n✅ Saved {len(saved_files)} forecast plots to {save_dir}")
        
        return saved_files
    

    
    def plot_conditioning(
        self,
        feature: Optional[str] = None,
        features: Optional[List[str]] = None,
        save: bool = False,
        save_dir: Optional[str] = None,
        display: bool = True
    ) -> List[str]:
        """
        Plot conditioning data (past and future conditioning windows)
        
        Shows:
        - Past conditioning (fetched recent data)
        - Future conditioning (from selected historical sample)
        - Boundary line separating past from future
        
        Args:
            feature: Single feature to plot and display inline (optional)
            features: List of features to plot when saving (default: all)
            save: Whether to save plots to disk (default: False)
            save_dir: Directory to save plots (default: ./conditioning/)
            display: Whether to display plot inline (default: True, only when feature specified)
            
        Returns:
            List of saved plot file paths (empty if save=False)
            
        Examples:
            >>> # Display a single feature inline
            >>> scenario.plot_conditioning(feature="10-Year Treasury Rate")
            
            >>> # Save all features to disk without displaying
            >>> scenario.plot_conditioning(save=True, display=False)
            
            >>> # Display one feature AND save all features
            >>> scenario.plot_conditioning(feature="S&P 500", save=True)
        """
        if not self.is_simulated:
            # Try to refresh from database
            refreshed_data = self.http.get(f'/api/v1/scenarios/{self.id}')
            self._data.update(refreshed_data)
            self.output = refreshed_data.get('output')
            self.status = refreshed_data.get('status', 'created')
            
            if not self.is_simulated:
                print(f"Debug: status={self.status}, output type={type(self.output)}, output is not None={self.output is not None}")
                raise ValueError("Scenario must be simulated before plotting. Run scenario.simulate() first.")
        
        import matplotlib.pyplot as plt
        import numpy as np
        import logging
        import os
        
        # Suppress matplotlib INFO messages about categorical units
        logging.getLogger('matplotlib.category').setLevel(logging.WARNING)
        
        # Default save directory
        if save_dir is None:
            save_dir = './conditioning/'
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Get reconstructed windows from output
        reconstructed_windows = self.output.get('conditioning_info', {}).get('reconstructed', [])
        
        if not reconstructed_windows:
            raise ValueError("No reconstructed data found in scenario output")
        
        # Find past and future conditioning windows
        past_windows = [w for w in reconstructed_windows if w.get('temporal_tag') == 'past']
        future_cond_windows = [w for w in reconstructed_windows if 
                              w.get('temporal_tag') == 'future' and 
                              w.get('_is_historical_pattern') == True]
        
        # Get available features from past and future conditioning
        available_features = set()
        for window in past_windows + future_cond_windows:
            feat = window.get('feature')
            if feat:
                available_features.add(feat)
        
        # Determine which features to process
        features_to_plot = []
        
        # If a single feature is specified for display, use that
        if feature is not None:
            if feature in available_features:
                features_to_plot = [feature]
            else:
                available = ', '.join(list(available_features)[:5])
                raise ValueError(f"Feature '{feature}' not found. Available: {available}...")
        
        # If saving, determine which features to save
        if save:
            if features is None:
                # Save all features
                save_features = list(available_features)
            else:
                # Save specified features
                save_features = [f for f in features if f in available_features]
        else:
            save_features = []
        
        # Combine both lists (for display and/or save)
        all_features_to_process = list(set(features_to_plot + save_features))
        
        if not all_features_to_process and not feature:
            raise ValueError("No features specified. Use feature='name' to display or save=True to save all")
        
        saved_files = []
        
        # Plot each feature
        for feature_name in all_features_to_process:
            should_display_this = (feature_name == feature and display)
            should_save_this = (save and feature_name in save_features)
            fig, ax = plt.subplots(1, 1, figsize=(14, 6))
            
            # Get conditioning data
            past_values = self._get_past_conditioning_values(feature_name)
            future_values = self._get_future_conditioning_values(feature_name)
            
            if not past_values and not future_values:
                continue
            
            # Create time axis with dates (if available)
            past_dates = self.output.get('past_dates', [])
            future_dates = self.output.get('future_dates', [])
            
            if past_dates and future_dates:
                past_t = past_dates if past_values else []
                future_t = future_dates if future_values else []
                use_dates = True
            else:
                past_t = np.arange(len(past_values)) if past_values else []
                future_t = np.arange(len(past_values), len(past_values) + len(future_values)) if future_values else []
                use_dates = False
            
            # Plot past conditioning (blue line)
            if past_values:
                ax.plot(past_t, past_values, '-', color='blue', linewidth=2, 
                       alpha=0.8, label='Past (Fetched)', zorder=3)
            
            # Plot future conditioning (orange line)
            if future_values:
                ax.plot(future_t, future_values, '-', color='orange', linewidth=2, 
                       alpha=0.8, label='Future (Conditioning)', zorder=3)
            
            # Add boundary line (dashed vertical line at reference date)
            if past_values and future_values:
                if use_dates and self.output.get('reference_date'):
                    # Use actual reference date for boundary
                    boundary_x = self.output.get('reference_date')
                elif past_values:
                    # Use numeric index
                    boundary_x = len(past_values)
                else:
                    boundary_x = None
                
                if boundary_x is not None:
                    ax.axvline(x=boundary_x, color='red', linestyle='--', 
                              linewidth=2, alpha=0.7, label='Boundary', zorder=4)
            
            # Formatting
            ax.set_title(f'{feature_name} - Conditioning Scenario', fontsize=14, fontweight='bold')
            
            if use_dates:
                # Format x-axis for dates
                ax.set_xlabel('Date', fontsize=12)
                # Rotate labels and show every Nth date
                n_dates = len(past_t) + len(future_t)
                tick_interval = max(1, n_dates // 10)  # Show ~10 ticks
                all_dates = list(past_t) + list(future_t)
                tick_indices = range(0, n_dates, tick_interval)
                ax.set_xticks([all_dates[i] for i in tick_indices if i < len(all_dates)])
                ax.tick_params(axis='x', rotation=45)
            else:
                # Numeric time steps
                ax.set_xlabel('Time Step', fontsize=12)
            
            ax.set_ylabel('Value', fontsize=12)
            ax.legend(loc='best', fontsize=10, framealpha=0.95)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            
            # Save if requested
            if should_save_this:
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                safe_feature_name = feature_name.replace('/', '_').replace('\\', '_')
                save_path = os.path.join(save_dir, f'{safe_feature_name}_conditioning.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                saved_files.append(save_path)
                if display:
                    print(f"  ✅ Saved: {save_path}")
            
            # Display if requested
            if should_display_this:
                plt.show()
            else:
                plt.close()
        
        if save and saved_files:
            print(f"\n✅ Saved {len(saved_files)} conditioning plots to {save_dir}")
        
        return saved_files
    
    def _get_past_values(self, feature_name: str) -> Optional[List[float]]:
        """Get past values for a feature from the forecast output"""
        reconstructed_windows = self.output.get('conditioning_info', {}).get('reconstructed', [])
        for window in reconstructed_windows:
            if (window.get('feature') == feature_name and 
                window.get('temporal_tag') == 'past'):
                return window.get('reconstructed_values', [])
        return None
    
    def _get_ground_truth_values(self, feature_name: str) -> Optional[List[float]]:
        """Get ground truth values for a feature from the forecast output"""
        reconstructed_windows = self.output.get('conditioning_info', {}).get('reconstructed', [])
        for window in reconstructed_windows:
            if (window.get('feature') == feature_name and 
                window.get('temporal_tag') == 'future' and
                window.get('_is_historical_pattern') == True):
                return window.get('reconstructed_values', [])
        return None
    
    def _get_past_conditioning_values(self, feature_name: str) -> Optional[List[float]]:
        """Get past conditioning values for a feature"""
        return self._get_past_values(feature_name)
    
    def _get_future_conditioning_values(self, feature_name: str) -> Optional[List[float]]:
        """Get future conditioning values for a feature"""
        return self._get_ground_truth_values(feature_name)
    
    def _update_step(self, step: str):
        """Update scenario step both locally and in database"""
        try:
            self.http.patch(f'/api/v1/scenarios/{self.id}', {'current_step': step})
            self._data['current_step'] = step
            self.current_step = step
        except Exception as e:
            logger.warning(f"Could not update scenario step: {e}")
    
    
    
    def refresh(self):
        """Refresh scenario data from database"""
        response = self.http.get(f'/api/v1/scenarios/{self.id}')
        self._data = response
        self.current_step = response.get('current_step', 'model-selection')
        # n_scenarios removed - number of paths is specified per forecast request
    
    def delete(self, confirm: bool = False) -> Dict[str, Any]:
        """
        Delete this scenario
        
        Args:
            confirm: Skip confirmation prompt if True
        
        Returns:
            Deletion result
        
        Example:
            >>> scenario.delete(confirm=True)
        """
        if self.interactive and not confirm:
            response = input(f"Delete scenario '{self.name}'? [y/N]: ")
            if response.lower() != 'y':
                print("❌ Deletion cancelled")
                return {"status": "cancelled"}
        
        print(f"🗑️  Deleting scenario: {self.name}...")
        
        try:
            # Delete via API (backend handles cascade)
            self.http.delete(f'/api/v1/scenarios/{self.id}')
            print("✅ Scenario deleted")
            
            return {"status": "success"}
        except Exception as e:
            print(f"❌ Deletion failed: {e}")
            raise
