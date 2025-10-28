-- Migration 002: Enhance portfolio tests with time-series metrics
-- This migration adds support for storing detailed time-series analysis

-- Add new columns to portfolio_tests table for enhanced metrics
ALTER TABLE portfolio_tests ADD COLUMN time_series_metrics TEXT;  -- JSON for daily aggregated metrics
ALTER TABLE portfolio_tests ADD COLUMN n_days INTEGER;  -- Number of days in forecast horizon

-- Create index for faster time-series queries
CREATE INDEX IF NOT EXISTS idx_portfolio_tests_n_days ON portfolio_tests(n_days);

-- Update the existing sample_results, aggregated_results, and summary_stats columns
-- to handle the enhanced data structure (no schema change needed, just JSON content)
