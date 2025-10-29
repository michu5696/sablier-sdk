# Sablier SDK

Python SDK for the Sablier Market Scenario Generator - Create scenario-conditioned synthetic financial data for portfolio testing and risk analysis.

## Installation

### From PyPI
```bash
pip install sablier-sdk
```

### From Source
```bash
# Clone the repository
git clone https://github.com/michu5696/sablier-sdk.git
cd sablier-sdk/sdk

# Install in editable mode
pip install -e ".[all]"
```

### Requirements
- Python 3.8 or higher
- Core dependencies: requests, pandas, numpy, pydantic, python-dateutil, matplotlib

## Quick Start

### Using Template Projects 

Template projects provide pre-configured models that you can use immediately. This is the fastest way to get started:

```python
from sablier import SablierClient

# Initialize client with your API URL
client = SablierClient(api_url="https://your-backend.run.app")

# List available projects (includes template projects)
projects = client.list_projects()

# Find the template project
template_project = [p for p in projects if p.is_template][0]
print(f"Using template: {template_project.name}")

# Get the first model from the template
models = template_project.list_models()
model = models[0]
print(f"Model: {model.name}")

# Create a scenario on the template model
scenario = model.create_scenario(
    simulation_date="2020-03-15",  # COVID crash date
    name="COVID Scenario"
)

# Simulate the scenario to generate synthetic market paths
result = scenario.simulate(n_samples=50)

# Access the forecast data
forecast_data = result.get('forecast_windows', [])
print(f"Generated {len(forecast_data)} forecast samples")

# Visualize the forecasts
scenario.plot_forecasts(save=True, save_dir="./forecasts")

# Test a portfolio against the scenario
portfolio = client.create_portfolio(
    name="Test Portfolio",
    target_set=model.get_target_set(),
    asset_configs={
        "10-Year Treasury Constant Maturity Rate": {
            "type": "treasury_bond",
            "params": {
                "coupon_rate": 0.04,  # 4% coupon rate
                "face_value": 1000,
                "issue_date": "2018-08-15",
                "payment_frequency": 2  # Semi-annual
            }
        },
        "20-Year Treasury Constant Maturity Rate": {
            "type": "treasury_bond",
            "params": {
                "coupon_rate": 0.042,  # 4.2% coupon rate
                "face_value": 1000,
                "issue_date": "2018-08-15",
                "payment_frequency": 2
            }
        },
        "30-Year Treasury Constant Maturity Rate": {
            "type": "treasury_bond",
            "params": {
                "coupon_rate": 0.041,  # 4.1% coupon rate
                "face_value": 1000,
                "issue_date": "2018-08-15",
                "payment_frequency": 2
            }
        }
    }
)

# Set portfolio weights
portfolio.set_weights({
    "10-Year Treasury Constant Maturity Rate": 0.4,
    "20-Year Treasury Constant Maturity Rate": -0.3,
    "30-Year Treasury Constant Maturity Rate": 0.3
})

# Run portfolio test
test = portfolio.test(scenario)

# View portfolio metrics
print(f"Sharpe Ratio: {test.summary_stats['sharpe']:.3f}")
print(f"Total Return: {test.summary_stats['total_return']:.2%}")
print(f"Max Drawdown: {test.summary_stats['max_drawdown']:.2%}")

# Plot portfolio performance evolution
test.plot_evolution('pnl')
test.plot_evolution('drawdown')
```


## Features

- **Template Projects**: Access pre-trained models immediately
- **Scenario Generation**: Define custom market scenarios with historical or synthetic conditions
- **Synthetic Data**: Generate thousands of realistic market paths
- **Portfolio Testing**: Test portfolios against synthetic scenarios with comprehensive metrics
- **Visualization**: Built-in plotting for scenarios, forecasts, and portfolio performance

## Portfolio Testing

The SDK includes comprehensive portfolio testing capabilities:

```python
# Create portfolio with asset configurations (for bonds, equities, etc.)
portfolio = client.create_portfolio(
    name="Treasury Portfolio",
    target_set=target_set,
    asset_configs={
        "DGS10": {
            "type": "treasury_bond",
            "params": {
                "coupon_rate": 0.04,
                "face_value": 1000,
                "issue_date": "2018-08-15",
                "payment_frequency": 2
            }
        }
    }
)

# Set portfolio weights
portfolio.set_weights({
    "DGS10": 0.6,
    "DGS30": 0.4
})

# Test portfolio against multiple scenarios
test = portfolio.test(scenario)

# Access comprehensive metrics
print(test.summary_stats)
# Returns: Sharpe Ratio, Sortino Ratio, Calmar Ratio, VaR, CVaR, Total Return, Max Drawdown

# Plot time-series evolution
test.plot_evolution('pnl')
test.plot_evolution('returns')
test.plot_evolution('drawdown')
test.plot_evolution('portfolio_value')

# Plot distributions
test.plot_distribution('sharpe_ratio')
test.plot_distribution('total_return')
test.plot_distribution('max_drawdown')
```

## API Key Management

The SDK automatically manages API keys and settings in a local SQLite database:

```python
# First time: SDK will prompt for registration
client = SablierClient(api_url="https://your-backend.run.app")

# Save an API key with a custom name
client.save_api_key(
    api_key="sk_...",
    api_url="https://your-backend.run.app",
    description="production"
)

# List all saved keys
keys = client.list_api_keys()

# Use a specific key by name
client = SablierClient(api_url="https://your-backend.run.app")
# Will use the default key automatically
```

## Examples

See the `examples/` directory for detailed usage examples.

## License

MIT License
