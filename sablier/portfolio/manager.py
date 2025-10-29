"""Enhanced Portfolio Manager with SQLite + JSON hybrid storage"""

import json
import os
import uuid
import sqlite3
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
import logging

from .builder import Portfolio

logger = logging.getLogger(__name__)

# Schema version - increment this when making database schema changes
SCHEMA_VERSION = 1


class PortfolioManager:
    """Enhanced portfolio manager with SQLite metadata and JSON data storage"""
    
    def __init__(self, http_client):
        """
        Initialize PortfolioManager
        
        Args:
            http_client: HTTP client for API calls (for scenario data access)
        """
        self.http = http_client
        
        # Initialize local database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for portfolio metadata"""
        # Create directory
        sablier_dir = os.path.expanduser("~/.sablier")
        os.makedirs(sablier_dir, exist_ok=True)
        
        # Database path
        self.db_path = os.path.join(sablier_dir, "portfolios.db")
        
        with sqlite3.connect(self.db_path) as conn:
            # Create schema_version table to track migrations
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
            """)
            
            # Check if schema_version table exists and has any records
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
            schema_version_exists = cursor.fetchone() is not None
            
            if schema_version_exists:
                # Check current schema version
                cursor = conn.execute("SELECT MAX(version) FROM schema_version")
                result = cursor.fetchone()
                current_version = (result[0] if result and result[0] is not None else 0)
            else:
                # Old database without schema_version - set to version 0 to force all migrations
                current_version = 0
                logger.info("⚠️  Old database detected (no schema_version). Running all migrations...")
            
            # Run migrations to bring database to latest version
            self._run_migrations(conn, current_version, SCHEMA_VERSION)
            
            # Create tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolios (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    target_set_id TEXT NOT NULL,
                    target_set_name TEXT NOT NULL,
                    assets TEXT NOT NULL,  -- JSON array of asset names
                    constraint_type TEXT NOT NULL DEFAULT 'long_short',
                    custom_constraints TEXT,  -- JSON object
                    weights TEXT,  -- JSON object
                    capital REAL NOT NULL DEFAULT 100000.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(name)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_optimizations (
                    id TEXT PRIMARY KEY,
                    portfolio_id TEXT NOT NULL,
                    scenario_id TEXT NOT NULL,
                    scenario_name TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    n_iterations INTEGER NOT NULL,
                    final_sharpe REAL,
                    final_return REAL,
                    final_risk REAL,
                    optimization_date TEXT NOT NULL,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_evaluations (
                    id TEXT PRIMARY KEY,
                    portfolio_id TEXT NOT NULL,
                    scenario_id TEXT NOT NULL,
                    scenario_name TEXT NOT NULL,
                    sharpe REAL,
                    mean_return REAL,
                    std_return REAL,
                    var_95 REAL,
                    var_99 REAL,
                    max_drawdown REAL,
                    evaluation_date TEXT NOT NULL,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios (id)
                )
            """)
            
            # Add portfolio_tests table for comprehensive scenario testing
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_tests (
                    id TEXT PRIMARY KEY,
                    portfolio_id TEXT NOT NULL,
                    scenario_id TEXT NOT NULL,
                    scenario_name TEXT NOT NULL,
                    test_date TEXT NOT NULL,
                    sample_results TEXT NOT NULL,  -- JSON array of per-sample metrics
                    aggregated_results TEXT NOT NULL,  -- JSON object
                    summary_stats TEXT NOT NULL,  -- JSON object
                    time_series_metrics TEXT,  -- JSON object with time-series aggregated data
                    n_days INTEGER,  -- Number of days in the test period
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
                )
            """)
            
            # Create indexes for portfolio_tests
            conn.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_tests_portfolio_id ON portfolio_tests(portfolio_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_tests_scenario_id ON portfolio_tests(scenario_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_tests_date ON portfolio_tests(test_date)")
            
            conn.commit()
    
    def _run_migrations(self, conn, current_version: int, target_version: int):
        """Run migrations to bring database from current_version to target_version"""
        
        for version in range(current_version + 1, target_version + 1):
            if version == 1:
                # Migration 1: Add asset_configs column to portfolios table
                # AND Add time_series_metrics and n_days to portfolio_tests table
                try:
                    # Add asset_configs to portfolios
                    cursor = conn.execute("PRAGMA table_info(portfolios)")
                    columns = {row[1] for row in cursor.fetchall()}
                    
                    if 'asset_configs' not in columns:
                        conn.execute("ALTER TABLE portfolios ADD COLUMN asset_configs TEXT")
                        logger.info("✅ Applied portfolio migration 1: Added asset_configs column")
                    else:
                        logger.info("ℹ️  Portfolio migration 1: asset_configs column already exists, skipping")
                    
                    # Add time_series_metrics and n_days to portfolio_tests
                    # First, ensure portfolio_tests table exists
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS portfolio_tests (
                            id TEXT PRIMARY KEY,
                            portfolio_id TEXT NOT NULL,
                            scenario_id TEXT NOT NULL,
                            scenario_name TEXT NOT NULL,
                            test_date TEXT NOT NULL,
                            sample_results TEXT NOT NULL,
                            aggregated_results TEXT NOT NULL,
                            summary_stats TEXT NOT NULL,
                            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
                        )
                    """)
                    
                    # Now check if columns need to be added
                    cursor = conn.execute("PRAGMA table_info(portfolio_tests)")
                    columns = {row[1] for row in cursor.fetchall()}
                    
                    if 'time_series_metrics' not in columns:
                        conn.execute("ALTER TABLE portfolio_tests ADD COLUMN time_series_metrics TEXT")
                        logger.info("✅ Applied portfolio migration 1: Added time_series_metrics column")
                    
                    if 'n_days' not in columns:
                        conn.execute("ALTER TABLE portfolio_tests ADD COLUMN n_days INTEGER")
                        logger.info("✅ Applied portfolio migration 1: Added n_days column")
                        
                except Exception as e:
                    logger.error(f"Portfolio migration 1 failed: {e}")
                    # Continue anyway - columns might already exist
                
                # Record migration was applied
                try:
                    conn.execute("""
                        INSERT INTO schema_version (version, applied_at) 
                        VALUES (?, ?)
                    """, (1, datetime.utcnow().isoformat() + 'Z'))
                except sqlite3.IntegrityError:
                    # Version already recorded
                    pass
            
            # Add future migrations here:
            # if version == 2:
            #     ...
    
    def create(self, name: str, target_set, weights: Optional[Union[Dict[str, float], List[float]]] = None, 
               capital: float = 100000.0, description: Optional[str] = None, 
               constraint_type: str = "long_short", asset_configs: Optional[Dict[str, Dict[str, Any]]] = None) -> Portfolio:
        """
        Create a new portfolio from a target set
        
        Args:
            name: Portfolio name
            target_set: Target feature set instance
            weights: Either:
                - Dict[str, float]: Dictionary of asset weights (must sum to 1.0)
                - List[float]: List of weights assigned to assets in order (must sum to 1.0)
                - None: Random weights will be generated (sum to 1.0)
            capital: Total capital allocation (default $100k)
            description: Optional description
            constraint_type: "long_only", "long_short", or "custom"
            asset_configs: Optional dict mapping asset names to their return calculation config
            
        Returns:
            New Portfolio instance
        """
        # Check if name already exists
        if self._portfolio_exists(name):
            raise ValueError(f"Portfolio '{name}' already exists")
        
        # Generate unique ID
        portfolio_id = str(uuid.uuid4())
        
        # Get assets from target set (extract feature names)
        assets = [feature.get('name', feature.get('id', str(feature))) for feature in target_set.features]
        
        # Process weights based on type
        processed_weights = self._process_weights(weights, assets, constraint_type)
        
        # Create portfolio data
        portfolio_data = {
            "id": portfolio_id,
            "name": name,
            "description": description or "",
            "target_set_id": target_set.id,
            "target_set_name": target_set.name,
            "assets": assets,
            "weights": processed_weights,
            "capital": capital,
            "constraint_type": constraint_type,
            "custom_constraints": None,
            "asset_configs": asset_configs or {},
            "created_at": datetime.utcnow().isoformat() + 'Z',
            "updated_at": datetime.utcnow().isoformat() + 'Z'
        }
        
        # Save to database
        self._save_portfolio_to_db(portfolio_data)
        
        # Create portfolio instance
        portfolio = Portfolio(self.http, portfolio_data)
        
        logger.info(f"Created portfolio '{name}' with {len(assets)} assets from target set '{target_set.name}'")
        return portfolio
    
    def _process_weights(self, weights: Optional[Union[Dict[str, float], List[float]]], 
                        assets: List[str], constraint_type: str) -> Dict[str, float]:
        """
        Process weights based on input type and generate random weights if None
        
        Args:
            weights: Input weights (Dict, List, or None)
            assets: List of asset names
            constraint_type: Constraint type for validation
            
        Returns:
            Dict[str, float]: Processed weights dictionary
        """
        import random
        
        if weights is None:
            # Generate random weights
            return self._generate_random_weights(assets, constraint_type)
        elif isinstance(weights, list):
            # Convert list to dict by assigning to assets in order
            return self._convert_list_to_dict(weights, assets, constraint_type)
        elif isinstance(weights, dict):
            # Validate dict weights
            self._validate_dict_weights(weights, assets, constraint_type)
            return weights
        else:
            raise ValueError("Weights must be Dict[str, float], List[float], or None")
    
    def _generate_random_weights(self, assets: List[str], constraint_type: str) -> Dict[str, float]:
        """Generate random weights that sum to 1.0"""
        import random
        
        n_assets = len(assets)
        
        if constraint_type == 'long_only':
            # Generate positive random numbers
            random_values = [random.uniform(0.01, 1.0) for _ in range(n_assets)]
            # Normalize to sum to 1.0
            total = sum(random_values)
            if abs(total) < 1e-10:  # Avoid division by zero
                normalized_weights = [1.0 / n_assets] * n_assets
            else:
                normalized_weights = [w / total for w in random_values]
        else:
            # Generate random numbers (can be negative for long_short)
            random_values = [random.uniform(-0.5, 1.0) for _ in range(n_assets)]
            # For long-short: normalize absolute values to sum to 1.0
            abs_total = sum(abs(w) for w in random_values)
            if abs_total < 1e-10:  # Avoid division by zero
                normalized_weights = [1.0 / n_assets] * n_assets
            else:
                normalized_weights = [w / abs_total for w in random_values]
        
        # Create dictionary
        return {asset: weight for asset, weight in zip(assets, normalized_weights)}
    
    def _convert_list_to_dict(self, weights: List[float], assets: List[str], 
                             constraint_type: str) -> Dict[str, float]:
        """Convert list of weights to dictionary"""
        if len(weights) != len(assets):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of assets ({len(assets)})")
        
        # Check for negative weights if not long_short
        if constraint_type == 'long_only':
            negative_indices = [i for i, w in enumerate(weights) if w < 0]
            if negative_indices:
                negative_assets = [assets[i] for i in negative_indices]
                raise ValueError(f"Long-only constraint violated for assets: {negative_assets}")
        
        # Check weight sum based on constraint type
        if constraint_type == 'long_only':
            # For long-only: raw weights must sum to 1.0
            weight_sum = sum(weights)
            if abs(weight_sum - 1.0) > 1e-6:
                raise ValueError(f"Weights must sum to 1.0, got {weight_sum:.6f}")
        else:
            # For long-short: absolute weights must sum to 1.0
            abs_weight_sum = sum(abs(w) for w in weights)
            if abs(abs_weight_sum - 1.0) > 1e-6:
                raise ValueError(f"Absolute weights must sum to 1.0, got {abs_weight_sum:.6f}")
        
        # Create dictionary
        return {asset: weight for asset, weight in zip(assets, weights)}
    
    def _validate_dict_weights(self, weights: Dict[str, float], assets: List[str], 
                              constraint_type: str) -> None:
        """Validate dictionary weights"""
        # Check that all assets have weights
        missing_assets = set(assets) - set(weights.keys())
        if missing_assets:
            raise ValueError(f"Missing weights for assets: {missing_assets}")
        
        # Check for extra weights
        extra_assets = set(weights.keys()) - set(assets)
        if extra_assets:
            raise ValueError(f"Extra weights for assets not in portfolio: {extra_assets}")
        
        # Check for negative weights if not long_short
        if constraint_type == 'long_only':
            negative_weights = [asset for asset, weight in weights.items() if weight < 0]
            if negative_weights:
                raise ValueError(f"Long-only constraint violated for assets: {negative_weights}")
        
        # Check weight sum based on constraint type
        if constraint_type == 'long_only':
            # For long-only: raw weights must sum to 1.0
            weight_sum = sum(weights.values())
            if abs(weight_sum - 1.0) > 1e-6:
                raise ValueError(f"Weights must sum to 1.0, got {weight_sum:.6f}")
        else:
            # For long-short: absolute weights must sum to 1.0
            abs_weight_sum = sum(abs(w) for w in weights.values())
            if abs(abs_weight_sum - 1.0) > 1e-6:
                raise ValueError(f"Absolute weights must sum to 1.0, got {abs_weight_sum:.6f}")
    
    def get(self, portfolio_id: str) -> Optional[Portfolio]:
        """
        Load portfolio from local storage
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            Portfolio instance or None if not found
        """
        portfolio_data = self._load_portfolio_from_db(portfolio_id)
        if not portfolio_data:
            return None
        
        portfolio = Portfolio(self.http, portfolio_data)
        logger.info(f"Loaded portfolio '{portfolio.name}'")
        return portfolio
    
    def get_by_name(self, name: str) -> Optional[Portfolio]:
        """Get portfolio by name"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM portfolios 
                WHERE name = ?
            """, (name,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            portfolio_data = self._row_to_portfolio_data(row)
            return Portfolio(self.http, portfolio_data)
    
    def list(self, limit: Optional[int] = None, offset: int = 0) -> List[Portfolio]:
        """
        List all portfolios
        
        Args:
            limit: Maximum number of portfolios to return
            offset: Number of portfolios to skip
            
        Returns:
            List of Portfolio instances
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = """
                SELECT * FROM portfolios 
                ORDER BY created_at DESC
            """
            params = []
            
            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        portfolios = []
        for row in rows:
            portfolio_data = self._row_to_portfolio_data(row)
            portfolio = Portfolio(self.http, portfolio_data)
            portfolios.append(portfolio)
        
        logger.info(f"Found {len(portfolios)} portfolios")
        return portfolios
    
    def search(self, query: str) -> List[Portfolio]:
        """
        Search portfolios by name or description
        
        Args:
            query: Search query
            
        Returns:
            List of matching Portfolio instances
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM portfolios 
                WHERE (name LIKE ? OR description LIKE ?)
                ORDER BY created_at DESC
            """, (f"%{query}%", f"%{query}%"))
            
            rows = cursor.fetchall()
        
        portfolios = []
        for row in rows:
            portfolio_data = self._row_to_portfolio_data(row)
            portfolio = Portfolio(self.http, portfolio_data)
            portfolios.append(portfolio)
        
        return portfolios
    
    def list_by_assets(self, assets: List[str]) -> List[Portfolio]:
        """
        List portfolios containing specific assets
        
        Args:
            assets: List of asset names to search for
            
        Returns:
            List of Portfolio instances containing these assets
        """
        portfolios = []
        all_portfolios = self.list()
        
        for portfolio in all_portfolios:
            if all(asset in portfolio.assets for asset in assets):
                portfolios.append(portfolio)
        
        return portfolios
    
    def get_optimization_history(self, portfolio_id: str) -> List[Dict[str, Any]]:
        """Get optimization history for a portfolio"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM portfolio_optimizations 
                WHERE portfolio_id = ? 
                ORDER BY optimization_date DESC
            """, (portfolio_id,))
            
            rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def get_evaluation_history(self, portfolio_id: str) -> List[Dict[str, Any]]:
        """Get evaluation history for a portfolio"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM portfolio_evaluations 
                WHERE portfolio_id = ? 
                ORDER BY evaluation_date DESC
            """, (portfolio_id,))
            
            rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def delete(self, portfolio_id: str) -> bool:
        """
        Delete a portfolio and all its history
        
        Args:
            portfolio_id: Portfolio ID to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            # Delete related records first
            conn.execute("DELETE FROM portfolio_optimizations WHERE portfolio_id = ?", (portfolio_id,))
            conn.execute("DELETE FROM portfolio_evaluations WHERE portfolio_id = ?", (portfolio_id,))
            conn.execute("DELETE FROM portfolio_tests WHERE portfolio_id = ?", (portfolio_id,))
            
            # Delete portfolio
            cursor = conn.execute("DELETE FROM portfolios WHERE id = ?", (portfolio_id,))
            deleted = cursor.rowcount > 0
            
            conn.commit()
        
        if deleted:
            logger.info(f"Deleted portfolio {portfolio_id}")
        else:
            logger.warning(f"Portfolio {portfolio_id} not found")
        
        return deleted
    
    def rename(self, portfolio_id: str, new_name: str) -> bool:
        """Rename a portfolio"""
        if self._portfolio_exists(new_name, exclude_id=portfolio_id):
            raise ValueError(f"Portfolio '{new_name}' already exists")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE portfolios 
                SET name = ?, updated_at = ?
                WHERE id = ?
            """, (new_name, datetime.utcnow().isoformat() + 'Z', portfolio_id))
            
            updated = cursor.rowcount > 0
            conn.commit()
        
        return updated
    
    def _portfolio_exists(self, name: str, exclude_id: Optional[str] = None) -> bool:
        """Check if portfolio name exists"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT 1 FROM portfolios WHERE name = ?"
            params = [name]
            
            if exclude_id:
                query += " AND id != ?"
                params.append(exclude_id)
            
            cursor = conn.execute(query, params)
            return cursor.fetchone() is not None
    
    def _save_portfolio_to_db(self, portfolio_data: Dict[str, Any]):
        """Save portfolio data to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO portfolios 
                (id, name, description, target_set_id, target_set_name, assets, 
                 constraint_type, custom_constraints, weights, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                portfolio_data['id'],
                portfolio_data['name'],
                portfolio_data['description'],
                portfolio_data['target_set_id'],
                portfolio_data['target_set_name'],
                json.dumps(portfolio_data['assets']),
                portfolio_data['constraint_type'],
                json.dumps(portfolio_data.get('custom_constraints')),
                json.dumps(portfolio_data['weights']),
                portfolio_data['created_at'],
                portfolio_data['updated_at']
            ))
            conn.commit()
    
    def _load_portfolio_from_db(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Load portfolio data from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM portfolios WHERE id = ?", (portfolio_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_portfolio_data(row)
    
    def _row_to_portfolio_data(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert database row to portfolio data dictionary"""
        return {
            'id': row['id'],
            'name': row['name'],
            'description': row['description'],
            'target_set_id': row['target_set_id'],
            'target_set_name': row['target_set_name'],
            'assets': json.loads(row['assets']),  # Should be list of strings
            'weights': json.loads(row['weights']) if row['weights'] else {},
            'constraint_type': row['constraint_type'],
            'custom_constraints': json.loads(row['custom_constraints']) if row['custom_constraints'] else None,
            'created_at': row['created_at'],
            'updated_at': row['updated_at']
        }
    
    def stats(self) -> Dict[str, Any]:
        """Get portfolio statistics for this project"""
        with sqlite3.connect(self.db_path) as conn:
            # Count portfolios
            cursor = conn.execute("SELECT COUNT(*) FROM portfolios")
            total_portfolios = cursor.fetchone()[0]
            
            # Count optimized portfolios
            cursor = conn.execute("""
                SELECT COUNT(*) FROM portfolios 
                WHERE weights != '{}'
            """)
            optimized_portfolios = cursor.fetchone()[0]
            
            # Count optimizations
            cursor = conn.execute("""
                SELECT COUNT(*) FROM portfolio_optimizations
            """)
            total_optimizations = cursor.fetchone()[0]
            
            # Count evaluations
            cursor = conn.execute("""
                SELECT COUNT(*) FROM portfolio_evaluations
            """)
            total_evaluations = cursor.fetchone()[0]
        
        return {
            'total_portfolios': total_portfolios,
            'optimized_portfolios': optimized_portfolios,
            'total_optimizations': total_optimizations,
            'total_evaluations': total_evaluations
        }