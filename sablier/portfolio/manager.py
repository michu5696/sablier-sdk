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

# Desired table schemas - modify these when you need to change the schema
# The schema diffing system will automatically detect and apply differences
DESIRED_SCHEMAS = {
    'portfolios': {
        'columns': [
            {'name': 'id', 'type': 'TEXT', 'constraints': 'PRIMARY KEY'},
            {'name': 'name', 'type': 'TEXT', 'constraints': 'NOT NULL'},
            {'name': 'description', 'type': 'TEXT', 'constraints': None},
            {'name': 'target_set_id', 'type': 'TEXT', 'constraints': 'NOT NULL'},
            {'name': 'target_set_name', 'type': 'TEXT', 'constraints': 'NOT NULL'},
            {'name': 'assets', 'type': 'TEXT', 'constraints': 'NOT NULL'},
            {'name': 'weights', 'type': 'TEXT', 'constraints': None},
            {'name': 'capital', 'type': 'REAL', 'constraints': 'NOT NULL DEFAULT 100000.0'},
            {'name': 'asset_configs', 'type': 'TEXT', 'constraints': None},
            {'name': 'created_at', 'type': 'TEXT', 'constraints': 'NOT NULL'},
            {'name': 'updated_at', 'type': 'TEXT', 'constraints': 'NOT NULL'},
        ],
        'table_constraints': ['UNIQUE(name)']
    },
    'portfolio_tests': {
        'columns': [
            {'name': 'id', 'type': 'TEXT', 'constraints': 'PRIMARY KEY'},
            {'name': 'portfolio_id', 'type': 'TEXT', 'constraints': 'NOT NULL'},
            {'name': 'scenario_id', 'type': 'TEXT', 'constraints': 'NOT NULL'},
            {'name': 'scenario_name', 'type': 'TEXT', 'constraints': 'NOT NULL'},
            {'name': 'test_date', 'type': 'TEXT', 'constraints': 'NOT NULL'},
            {'name': 'sample_results', 'type': 'TEXT', 'constraints': 'NOT NULL'},
            {'name': 'aggregated_results', 'type': 'TEXT', 'constraints': 'NOT NULL'},
            {'name': 'summary_stats', 'type': 'TEXT', 'constraints': 'NOT NULL'},
            {'name': 'time_series_metrics', 'type': 'TEXT', 'constraints': None},
            {'name': 'n_days', 'type': 'INTEGER', 'constraints': None},
        ],
        'foreign_keys': [{
            'column': 'portfolio_id',
            'references': 'portfolios(id)',
            'on_delete': 'CASCADE'
        }]
    }
}


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
        """Initialize SQLite database for portfolio metadata using schema diffing"""
        # Create directory
        sablier_dir = os.path.expanduser("~/.sablier")
        os.makedirs(sablier_dir, exist_ok=True)
        
        # Database path
        self.db_path = os.path.join(sablier_dir, "portfolios.db")
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if this is a new database (no tables exist yet)
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('portfolios', 'portfolio_tests')")
            existing_tables = {row[0] for row in cursor.fetchall()}
            is_new_database = 'portfolios' not in existing_tables
            
            if is_new_database:
                # New database - create all tables with latest schema directly
                logger.info("📝 Creating new database with latest schema")
                self._create_tables(conn, DESIRED_SCHEMAS)
                self._create_indexes(conn)
            else:
                # Existing database - apply schema diffing to update to latest schema
                logger.info("🔧 Existing database detected - applying schema updates")
                self._apply_schema_diff(conn, DESIRED_SCHEMAS)
            
            conn.commit()
    
    def _create_tables(self, conn, schemas: Dict[str, Dict[str, Any]]):
        """Create tables with the desired schema"""
        for table_name, schema in schemas.items():
            columns_sql = []
            for col in schema['columns']:
                col_def = f"{col['name']} {col['type']}"
                if col['constraints']:
                    col_def += f" {col['constraints']}"
                columns_sql.append(col_def)
            
            # Add table-level constraints
            if 'table_constraints' in schema:
                columns_sql.extend(schema['table_constraints'])
            
            # Add foreign keys
            if 'foreign_keys' in schema:
                for fk in schema['foreign_keys']:
                    fk_sql = f"FOREIGN KEY ({fk['column']}) REFERENCES {fk['references']}"
                    if 'on_delete' in fk:
                        fk_sql += f" ON DELETE {fk['on_delete']}"
                    columns_sql.append(fk_sql)
            
            create_sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {', '.join(columns_sql)}
                )
            """
            conn.execute(create_sql)
            logger.info(f"✅ Created/verified table: {table_name}")
    
    def _create_indexes(self, conn):
        """Create indexes for portfolio_tests table"""
        conn.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_tests_portfolio_id ON portfolio_tests(portfolio_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_tests_scenario_id ON portfolio_tests(scenario_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_tests_date ON portfolio_tests(test_date)")
    
    def _apply_schema_diff(self, conn, desired_schemas: Dict[str, Dict[str, Any]]):
        """
        Apply schema differences between existing database and desired schema.
        This automatically detects missing columns and adds them.
        
        For new users: If database doesn't exist, tables are created directly.
        For existing users: Missing columns are automatically added.
        """
        for table_name, desired_schema in desired_schemas.items():
            # Check if table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (table_name,))
            if not cursor.fetchone():
                # Table doesn't exist - create it
                logger.info(f"📋 Table {table_name} doesn't exist - creating it")
                self._create_tables(conn, {table_name: desired_schema})
                continue
            
            # Get existing columns
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            existing_columns = {row[1]: {'type': row[2], 'notnull': row[3], 'dflt_value': row[4]} 
                               for row in cursor.fetchall()}
            
            # Compare with desired columns
            desired_column_names = {col['name'] for col in desired_schema['columns']}
            existing_column_names = set(existing_columns.keys())
            
            # Find missing columns
            missing_columns = desired_column_names - existing_column_names
            
            if missing_columns:
                logger.info(f"🔧 Table {table_name}: Adding {len(missing_columns)} missing column(s)")
                for col in desired_schema['columns']:
                    if col['name'] in missing_columns:
                        # Build column definition
                        col_def = f"{col['name']} {col['type']}"
                        if col['constraints']:
                            col_def += f" {col['constraints']}"
                        
                        # Add column using ALTER TABLE
                        try:
                            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_def}")
                            logger.info(f"  ✅ Added column: {col['name']} ({col['type']})")
                        except sqlite3.OperationalError as e:
                            # Column might already exist (race condition) or error adding
                            if "duplicate column" in str(e).lower():
                                logger.info(f"  ℹ️  Column {col['name']} already exists, skipping")
                            else:
                                logger.warning(f"  ⚠️  Failed to add column {col['name']}: {e}")
            
            # Verify indexes
            self._create_indexes(conn)
            
            # Note: We don't handle:
            # - Dropping columns (SQLite limitation - would need table recreation)
            # - Changing column types (SQLite limitation - would need table recreation)
            # - Modifying constraints (complex, usually not needed)
            # These are rare operations that would require manual migration if needed.
    
    def create(self, name: str, target_set, weights: Optional[Union[Dict[str, float], List[float]]] = None, 
               capital: float = 100000.0, description: Optional[str] = None, 
               asset_configs: Optional[Dict[str, Dict[str, Any]]] = None) -> Portfolio:
        """
        Create a new portfolio from a target set
        
        Args:
            name: Portfolio name
            target_set: Target feature set instance
            weights: Either:
                - Dict[str, float]: Dictionary of asset weights (sum of absolute values must equal 1.0)
                - List[float]: List of weights assigned to assets in order (sum of absolute values must equal 1.0)
                - None: Random weights will be generated (sum of absolute values = 1.0)
            capital: Total capital allocation (default $100k)
            description: Optional description
            asset_configs: Optional dict mapping asset names to their return calculation config
            
        Returns:
            New Portfolio instance
            
        Note:
            Portfolios support long-short positions (negative weights allowed).
            The sum of absolute values of weights must equal 1.0.
        """
        # Check if name already exists
        if self._portfolio_exists(name):
            raise ValueError(f"Portfolio '{name}' already exists")
        
        # Generate unique ID
        portfolio_id = str(uuid.uuid4())
        
        # Get assets from target set (extract feature names)
        assets = [feature.get('name', feature.get('id', str(feature))) for feature in target_set.features]
        
        # Process weights based on type (always long-short)
        processed_weights = self._process_weights(weights, assets)
        
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
                        assets: List[str]) -> Dict[str, float]:
        """
        Process weights based on input type and generate random weights if None
        
        Args:
            weights: Input weights (Dict, List, or None)
            assets: List of asset names
            
        Returns:
            Dict[str, float]: Processed weights dictionary (sum of absolute values = 1.0)
        """
        import random
        
        if weights is None:
            # Generate random weights
            return self._generate_random_weights(assets)
        elif isinstance(weights, list):
            # Convert list to dict by assigning to assets in order
            return self._convert_list_to_dict(weights, assets)
        elif isinstance(weights, dict):
            # Validate dict weights
            self._validate_dict_weights(weights, assets)
            return weights
        else:
            raise ValueError("Weights must be Dict[str, float], List[float], or None")
    
    def _generate_random_weights(self, assets: List[str]) -> Dict[str, float]:
        """Generate random weights where sum of absolute values equals 1.0"""
        import random
        
        n_assets = len(assets)
        # Generate random numbers (can be negative for long-short)
        random_values = [random.uniform(-0.5, 1.0) for _ in range(n_assets)]
        # Normalize absolute values to sum to 1.0
        abs_total = sum(abs(w) for w in random_values)
        if abs_total < 1e-10:  # Avoid division by zero
            normalized_weights = [1.0 / n_assets] * n_assets
        else:
            normalized_weights = [w / abs_total for w in random_values]
        
        # Create dictionary
        return {asset: weight for asset, weight in zip(assets, normalized_weights)}
    
    def _convert_list_to_dict(self, weights: List[float], assets: List[str]) -> Dict[str, float]:
        """Convert list of weights to dictionary"""
        if len(weights) != len(assets):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of assets ({len(assets)})")
        
        # Check that absolute weights sum to 1.0
        abs_weight_sum = sum(abs(w) for w in weights)
        if abs(abs_weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Absolute weights must sum to 1.0, got {abs_weight_sum:.6f}")
        
        # Create dictionary
        return {asset: weight for asset, weight in zip(assets, weights)}
    
    def _validate_dict_weights(self, weights: Dict[str, float], assets: List[str]) -> None:
        """Validate dictionary weights"""
        # Check that all assets have weights
        missing_assets = set(assets) - set(weights.keys())
        if missing_assets:
            raise ValueError(f"Missing weights for assets: {missing_assets}")
        
        # Check for extra weights
        extra_assets = set(weights.keys()) - set(assets)
        if extra_assets:
            raise ValueError(f"Extra weights for assets not in portfolio: {extra_assets}")
        
        # Check that absolute weights sum to 1.0
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
    
    
    def delete(self, portfolio_id: str) -> bool:
        """
        Delete a portfolio and all its history
        
        Args:
            portfolio_id: Portfolio ID to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            # Delete related records first (portfolio_tests is now CASCADE, but explicit delete ensures cleanup)
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
                 weights, capital, asset_configs, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                portfolio_data['id'],
                portfolio_data['name'],
                portfolio_data['description'],
                portfolio_data['target_set_id'],
                portfolio_data['target_set_name'],
                json.dumps(portfolio_data['assets']),
                json.dumps(portfolio_data['weights']),
                portfolio_data.get('capital', 100000.0),
                json.dumps(portfolio_data.get('asset_configs', {})),
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
        # sqlite3.Row doesn't have .get(), need to use try/except or check keys
        row_dict = dict(row)
        portfolio_data = {
            'id': row_dict['id'],
            'name': row_dict['name'],
            'description': row_dict['description'],
            'target_set_id': row_dict['target_set_id'],
            'target_set_name': row_dict['target_set_name'],
            'assets': json.loads(row_dict['assets']),  # Should be list of strings
            'weights': json.loads(row_dict['weights']) if row_dict['weights'] else {},
            'capital': row_dict.get('capital', 100000.0),
            'asset_configs': json.loads(row_dict['asset_configs']) if row_dict.get('asset_configs') else {},
            'created_at': row_dict['created_at'],
            'updated_at': row_dict['updated_at']
        }
        # Backward compatibility: if constraint_type exists in DB, ignore it (legacy)
        return portfolio_data
    
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
            
            # Count tests (replacement for optimizations/evaluations)
            cursor = conn.execute("""
                SELECT COUNT(*) FROM portfolio_tests
            """)
            total_tests = cursor.fetchone()[0]
        
        return {
            'total_portfolios': total_portfolios,
            'optimized_portfolios': optimized_portfolios,
            'total_tests': total_tests
        }