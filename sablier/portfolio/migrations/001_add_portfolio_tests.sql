-- Migration 001: Add portfolio_tests table for scenario-based portfolio testing
-- This migration adds support for storing portfolio test results against scenarios

CREATE TABLE IF NOT EXISTS portfolio_tests (
    id TEXT PRIMARY KEY,
    portfolio_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    scenario_name TEXT NOT NULL,
    test_date TEXT NOT NULL,
    
    -- Per-sample results (JSON array of metrics for each sample path)
    sample_results TEXT NOT NULL,  -- Array of per-sample metrics
    
    -- Cross-sample aggregated results (JSON)
    aggregated_results TEXT NOT NULL,  -- VaR, survival rate, distribution stats
    
    -- Summary statistics (JSON)
    summary_stats TEXT NOT NULL,  -- Mean, std, percentiles across samples
    
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
);

-- Index for efficient querying
CREATE INDEX IF NOT EXISTS idx_portfolio_tests_portfolio_id ON portfolio_tests(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_tests_scenario_id ON portfolio_tests(scenario_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_tests_date ON portfolio_tests(test_date);
