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

## Documentation

Full documentation available at: [docs.sablier.io](https://docs.sablier.io)

## Examples

See the `examples/` directory for detailed usage examples.

## License

MIT License
