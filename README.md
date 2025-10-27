# Sablier SDK

Python SDK for the Sablier Market Scenario Generator - Create scenario-conditioned synthetic financial data for portfolio testing and risk analysis.

## Installation

```bash
pip install sablier-sdk
```

For development:
```bash
pip install -e ".[all]"
```

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
    future_window=90,
    conditioning_features=["S&P 500", "VIX"],
    target_features=["10-Year Treasury"]
)

# Train the model
model.fit_encoding_models()
model.encode_samples()
results = model.train()

# Create a scenario
scenario = client.scenarios.create(
    model=model,
    name="High Volatility Scenario"
)

# Configure the scenario
scenario.fetch_past_data(start_date="2024-12-01", end_date="2025-03-31")
scenario.set_feature("VIX", components=[2.0, 1.5, 1.0, 0.5, 0.0])

# Generate synthetic paths
synthetic_data = scenario.generate_paths(n_samples=1000)

# Validate and analyze
validation = synthetic_data.validate_realism()
print(validation.summary())

# Export results
synthetic_data.to_csv("scenario_results.csv")
```

## Features

- **Model Building**: Create and train quantile regression forest models
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
