"""
User Settings Manager for Sablier SDK

Handles local storage of user settings including API keys, API URLs, and other preferences.
Uses SQLite database in ~/.sablier/user_settings.db
"""

import os
import sqlite3
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Schema version - increment this when making database schema changes
SCHEMA_VERSION = 1


class UserSettingsManager:
    """Manages user settings including API keys and URLs"""
    
    def __init__(self):
        """Initialize the user settings manager"""
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for user settings"""
        # Create directory
        sablier_dir = os.path.expanduser("~/.sablier")
        os.makedirs(sablier_dir, exist_ok=True)
        
        # Database path
        self.db_path = os.path.join(sablier_dir, "user_settings.db")
        
        with sqlite3.connect(self.db_path) as conn:
            # Create schema_version table to track migrations
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
            """)
            
            # Check current schema version
            cursor = conn.execute("SELECT MAX(version) FROM schema_version")
            result = cursor.fetchone()
            current_version = (result[0] if result and result[0] is not None else 0)
            
            # Check if this is a new database (no tables exist yet)
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('user_settings', 'api_keys')")
            existing_tables = {row[0] for row in cursor.fetchall()}
            is_new_database = 'user_settings' not in existing_tables
            
            if is_new_database:
                # New database - skip migrations and mark as up-to-date
                logger.info("📝 Creating new user settings database with schema version 1")
                try:
                    conn.execute("""
                        INSERT INTO schema_version (version, applied_at) 
                        VALUES (?, ?)
                    """, (SCHEMA_VERSION, datetime.utcnow().isoformat() + 'Z'))
                except sqlite3.IntegrityError:
                    pass
            elif current_version < SCHEMA_VERSION:
                # Existing database - run migrations
                logger.info(f"🔧 User settings database version {current_version}, migrating to {SCHEMA_VERSION}")
                self._run_migrations(conn, current_version, SCHEMA_VERSION)
            
            # Create tables with latest schema (includes description column)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    setting_key TEXT NOT NULL UNIQUE,
                    setting_value TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_key TEXT NOT NULL UNIQUE,
                    api_url TEXT NOT NULL,
                    user_email TEXT,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    description TEXT  -- Added in schema version 1
                )
            """)
            
            conn.commit()
    
    def _run_migrations(self, conn, current_version: int, target_version: int):
        """Run migrations to bring database from current_version to target_version"""
        
        for version in range(current_version + 1, target_version + 1):
            if version == 1:
                # Migration 1: Add description column to api_keys
                try:
                    cursor = conn.execute("PRAGMA table_info(api_keys)")
                    columns = {row[1] for row in cursor.fetchall()}
                    
                    if 'description' not in columns:
                        conn.execute("ALTER TABLE api_keys ADD COLUMN description TEXT")
                        logger.info("✅ Applied migration 1: Added description column to api_keys")
                    else:
                        logger.info("ℹ️  Migration 1: description column already exists, skipping")
                except Exception as e:
                    logger.error(f"Migration 1 failed: {e}")
                    # Continue anyway - column might already exist
                
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
    
    def save_api_key(self, api_key: str, api_url: str, 
                     description: Optional[str] = None, is_default: bool = False) -> bool:
        """
        Save an API key to the database
        
        Args:
            api_key: The API key to save
            api_url: The API URL associated with this key
            description: Optional name/description for the key (e.g., "default", "template", "production")
            is_default: Whether this should be the default key
            
        Returns:
            bool: True if saved successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # If this is the default key, unset any other default keys
                if is_default:
                    conn.execute("""
                        UPDATE api_keys 
                        SET description = NULL 
                        WHERE description = 'default'
                    """)
                
                # Deactivate any existing active keys for this URL
                conn.execute("""
                    UPDATE api_keys 
                    SET is_active = 0 
                    WHERE api_url = ? AND is_active = 1
                """, (api_url,))
                
                # Use the provided description (should already be set by client.py)
                final_description = description
                
                # Insert new API key
                conn.execute("""
                    INSERT OR REPLACE INTO api_keys 
                    (api_key, api_url, user_email, is_active, created_at, last_used_at, description)
                    VALUES (?, ?, ?, 1, ?, ?, ?)
                """, (
                    api_key,
                    api_url,
                    None,  # user_email not used anymore
                    datetime.utcnow().isoformat() + 'Z',
                    datetime.utcnow().isoformat() + 'Z',
                    final_description
                ))
                
                conn.commit()
                logger.info(f"API key saved for URL: {api_url}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save API key: {e}")
            return False
    
    def update_api_key_description(self, api_key: str, description: str) -> bool:
        """
        Update the description of an existing API key
        
        Args:
            api_key: The API key to update
            description: New description/name
            
        Returns:
            bool: True if updated successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    UPDATE api_keys 
                    SET description = ?
                    WHERE api_key = ?
                """, (description, api_key))
                
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update API key description: {e}")
            return False
    
    def get_active_api_key(self, api_url: str) -> Optional[str]:
        """
        Get the active API key for a given URL
        
        Args:
            api_url: The API URL to get the key for
            
        Returns:
            str: The active API key, or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT api_key FROM api_keys 
                    WHERE api_url = ? AND is_active = 1
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (api_url,))
                
                result = cursor.fetchone()
                if result:
                    # Update last used timestamp
                    conn.execute("""
                        UPDATE api_keys 
                        SET last_used_at = ? 
                        WHERE api_key = ?
                    """, (datetime.utcnow().isoformat() + 'Z', result[0]))
                    conn.commit()
                    
                    return result[0]
                return None
                
        except Exception as e:
            logger.error(f"Failed to get API key: {e}")
            return None
    
    def get_api_key_by_name(self, name: str) -> Optional[str]:
        """
        Get an API key by its name/description
        
        Args:
            name: The name/description of the API key (e.g., "default", "template", "production")
            
        Returns:
            str: The API key, or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT api_key FROM api_keys 
                    WHERE description = ?
                    LIMIT 1
                """, (name,))
                
                result = cursor.fetchone()
                if result:
                    # Update last used timestamp
                    conn.execute("""
                        UPDATE api_keys 
                        SET last_used_at = ? 
                        WHERE api_key = ?
                    """, (datetime.utcnow().isoformat() + 'Z', result[0]))
                    conn.commit()
                    
                    return result[0]
                return None
                
        except Exception as e:
            logger.error(f"Failed to get API key by name: {e}")
            return None
    
    def get_default_api_key(self) -> Optional[str]:
        """
        Get the default API key (the one with description='default')
        
        Returns:
            str: The default API key, or None if not found
        """
        return self.get_api_key_by_name('default')
    
    def list_api_keys(self) -> List[Dict[str, Any]]:
        """
        List all API keys
        
        Returns:
            List of API key dictionaries with keys: api_key, api_url, 
            is_active, created_at, last_used_at, description
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT api_key, api_url, is_active, 
                           created_at, last_used_at, description
                    FROM api_keys 
                    ORDER BY created_at DESC
                """)
                
                keys = []
                for row in cursor.fetchall():
                    keys.append({
                        'api_key': row[0],
                        'api_url': row[1],
                        'is_active': bool(row[2]),
                        'created_at': row[3],
                        'last_used_at': row[4],
                        'description': row[5]
                    })
                
                return keys
                
        except Exception as e:
            logger.error(f"Failed to list API keys: {e}")
            return []
    
    def delete_api_key(self, api_key: str) -> bool:
        """
        Delete an API key
        
        Args:
            api_key: The API key to delete
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM api_keys WHERE api_key = ?", (api_key,))
                deleted = cursor.rowcount > 0
                conn.commit()
                
                if deleted:
                    logger.info(f"API key deleted: {api_key[:10]}...")
                
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to delete API key: {e}")
            return False
    
    def set_default_api_url(self, api_url: str) -> bool:
        """
        Set the default API URL
        
        Args:
            api_url: The default API URL
            
        Returns:
            bool: True if set successfully
        """
        return self._save_setting('default_api_url', api_url)
    
    def get_default_api_url(self) -> Optional[str]:
        """
        Get the default API URL
        
        Returns:
            str: The default API URL, or None if not set
        """
        return self._get_setting('default_api_url')
    
    def _save_setting(self, key: str, value: str) -> bool:
        """Save a setting to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO user_settings 
                    (setting_key, setting_value, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    key,
                    value,
                    datetime.utcnow().isoformat() + 'Z',
                    datetime.utcnow().isoformat() + 'Z'
                ))
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to save setting {key}: {e}")
            return False
    
    def _get_setting(self, key: str) -> Optional[str]:
        """Get a setting from the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT setting_value FROM user_settings 
                    WHERE setting_key = ?
                """, (key,))
                
                result = cursor.fetchone()
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"Failed to get setting {key}: {e}")
            return None
    
    def get_all_settings(self) -> Dict[str, str]:
        """Get all settings"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT setting_key, setting_value FROM user_settings")
                
                settings = {}
                for row in cursor.fetchall():
                    settings[row[0]] = row[1]
                
                return settings
                
        except Exception as e:
            logger.error(f"Failed to get all settings: {e}")
            return {}
    
    def clear_all_data(self) -> bool:
        """Clear all user data (API keys and settings)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM api_keys")
                conn.execute("DELETE FROM user_settings")
                conn.commit()
                logger.info("All user data cleared")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear all data: {e}")
            return False
