# Sablier SDK

Python SDK for the Sablier Market Scenario Generator - Create scenario-conditioned synthetic financial data for portfolio testing and risk analysis.

## Installation

### From PyPI (Coming Soon)
```bash
pip install sablier-sdk
```

### From Source (Current Method)
Since the SDK is not yet on PyPI, install from the cloned repository:

```bash
# Clone the repository
git clone https://github.com/yourusername/market-scenario-generator.git
cd market-scenario-generator/sdk

# Install in editable mode (recommended)
pip install -e ".[all]"

# OR install with specific dependencies:
pip install -e ".[viz]"        # For visualization features
pip install -e ".[stats]"      # For statistical analysis
pip install -e ".[dev]"        # For development tools
```

### Requirements
- Python 3.8 or higher
- Core dependencies: requests, pandas, numpy, pydantic, python-dateutil, matplotlib

For detailed installation instructions, see [INSTALLATION.md](../INSTALLATION.md).

## Quick Start

```python
from sablier import SablierClient

# Initialize client
client = SablierClient(
    api_url="https://your-backend.run.app",
    api_key="your-api-key"
)

# Build a model
model = client.models.create(
    name="Market Volatility Model",
    features=[
        {"name": "S&P 500", "source": "yahoo"},
        {"name": "VIX", "source": "yahoo"},
        {"name": "10-Year Treasury", "source": "fred"}
    ],
    training_period={"start": "2020-01-01", "end": "2024-12-31"}
)

# Fetch and process data
model.fetch_data()

# Generate training samples
model.generate_samples(
    past_window=90,
    future_window=90
)

# Train the model
results = model.train()

# Create a scenario
scenario = model.create_scenario(
    name="High Volatility Scenario",
    simulation_date="2024-12-01"
)

# Run simulation
result = scenario.simulate(n_samples=1000)

# Access forecast data
forecast_data = result.get('forecast_windows', [])
print(f"Generated {len(forecast_data)} forecast samples")
```

## Features

- **Model Building**: Create and train statistical models for market scenario generation
- **Scenario Generation**: Define custom market scenarios
- **Synthetic Data**: Generate thousands of realistic market paths
- **Validation**: Comprehensive statistical validation tools
- **Portfolio Testing**: Test portfolios against synthetic scenarios
- **Template Projects**: Access pre-trained models for immediate use

## Using Template Projects

Template projects provide pre-configured models that you can use immediately without training your own models. This is perfect for getting started quickly or for standard market scenarios.

```python
from sablier import SablierClient

# Initialize client
client = SablierClient(
    api_url="https://your-backend.run.app",
    api_key="your_api_key"
)

# List projects (includes your projects + templates)
projects = client.projects.list()

# Find template project
template = next(p for p in projects if p.is_template)
print(f"Template: {template.name}")
print(f"  Is template: {template.is_template}")

# List models in the template
models = template.list_models()
model = models[0]
print(f"Model: {model.name}")
print(f"  Is shared: {model.is_shared}")

# Create scenario on template model
# The scenario is saved under YOUR user, not the template owner
scenario = model.create_scenario(
    name="My Custom Scenario",
    simulation_date="2024-01-01",
    n_samples=1000
)

# Run simulation
result = scenario.simulate()

# Your forecasts are private to you
# The template model remains shared and unchanged
print(f"Generated {len(result['forecast_windows'])} forecast samples")
```

### Benefits of Template Projects

- **Instant Access**: No need to train models from scratch
- **Pre-validated**: Templates are professionally configured and tested
- **Private Scenarios**: Your scenarios and forecasts remain private
- **No Training Costs**: Use pre-trained models without computational overhead
- **Learn by Example**: Study template configurations to build your own models

## Documentation

Full documentation available at: [docs.sablier.io](https://docs.sablier.io)

## Examples

See the `examples/` directory for detailed usage examples.

## License

MIT License
