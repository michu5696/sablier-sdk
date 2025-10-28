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
        
        # Create tables
        with sqlite3.connect(self.db_path) as conn:
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
                    description TEXT
                )
            """)
            
            conn.commit()
    
    def save_api_key(self, api_key: str, api_url: str, user_email: Optional[str] = None, 
                     description: Optional[str] = None) -> bool:
        """
        Save an API key to the database
        
        Args:
            api_key: The API key to save
            api_url: The API URL associated with this key
            user_email: Optional user email
            description: Optional description for the key
            
        Returns:
            bool: True if saved successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Deactivate any existing active keys for this URL
                conn.execute("""
                    UPDATE api_keys 
                    SET is_active = 0 
                    WHERE api_url = ? AND is_active = 1
                """, (api_url,))
                
                # Insert new API key
                conn.execute("""
                    INSERT OR REPLACE INTO api_keys 
                    (api_key, api_url, user_email, is_active, created_at, last_used_at, description)
                    VALUES (?, ?, ?, 1, ?, ?, ?)
                """, (
                    api_key,
                    api_url,
                    user_email,
                    datetime.utcnow().isoformat() + 'Z',
                    datetime.utcnow().isoformat() + 'Z',
                    description
                ))
                
                conn.commit()
                logger.info(f"API key saved for URL: {api_url}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save API key: {e}")
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
    
    def list_api_keys(self) -> List[Dict[str, Any]]:
        """
        List all API keys
        
        Returns:
            List of API key dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT api_key, api_url, user_email, is_active, 
                           created_at, last_used_at, description
                    FROM api_keys 
                    ORDER BY created_at DESC
                """)
                
                keys = []
                for row in cursor.fetchall():
                    keys.append({
                        'api_key': row[0],
                        'api_url': row[1],
                        'user_email': row[2],
                        'is_active': bool(row[3]),
                        'created_at': row[4],
                        'last_used_at': row[5],
                        'description': row[6]
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
