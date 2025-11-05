# Sablier SDK

Sablier SDK is a Python toolkit for scenario‑conditioned synthetic financial data generation, portfolio testing, and risk analysis. It lets you simulate market regimes (e.g., risk‑off, inflation shocks), generate realistic multi‑asset paths, and evaluate portfolios under those scenarios.

---

## Why Synthetic Time-Series Data?

Synthetic data are statistically realistic series generated from learned patterns of historical data. In finance, this enables you to:
- Stress test beyond history: Generate regimes that are short or missing in real data and probe model robustness.
- Blend regimes on purpose: Mix features like COVID-style volatility with 2022-level inflation to craft targeted scenarios.
- Explore tail risk at scale: Sample thousands of paths to study distributional uncertainty, drawdowns, and path dependence.
- Enable forward-looking analysis: Move beyond backtesting to “fore-testing” — exploring how strategies might perform under unseen future conditions.
- Accelerate experimentation: Prototype, train, and validate AI models or trading strategies faster without costly data-licensing or collection.
- Overcome data scarcity: Create longer histories or regime-specific datasets that may not exist historically.

## Explainable Synthetic Data Generation

Sablier uses an interpretable blend of models to generate synthetic series you can actually understand. Instead of black-box models such as GANs, we decompose signals into meaningful factors, model their dependencies transparently, then recombine them to produce realistic new paths. The result is statistically sound, interpretable, and controllable synthetic data — enabling confidence and accountability in financial modeling.

---

## Key Capabilities

- Template Projects: Access pre-trained models immediately
- Scenario Generation: Define custom market scenarios with historical or synthetic conditions
- Synthetic Data: Generate thousands of realistic market paths
- Portfolio Testing: Test portfolios against synthetic scenarios with comprehensive metrics
- Visualization: Built-in plotting for scenarios, forecasts, and portfolio performance

---

## Installation

From PyPI (when available):
```bash
pip install sablier-sdk
```

For local development (editable install):
```bash
# Clone
git clone https://github.com/michu5696/sablier-sdk.git
cd sablier-sdk

# Install (editable)
pip install -e ".[all]"    # or: pip install -e .
```

Recommended: use a virtual environment (e.g., venv).

---

## Quickstart

At present, you can browse existing projects/models provided by the backend, then create scenarios and portfolios locally to test them. 
Note: In our next version, you’ll be able to create your own projects and train custom models directly within the SDK.

```python
from sablier import SablierClient

# 1) Initialize client (auto-registers an API key if none exists)
client = SablierClient(api_url="https://<your-api-url>")

# 2) List existing projects and pick one
projects = client.list_projects(include_templates=True)

# 3) List models in the project and pick template
models = project.list_models()
model = models[0]

# 4) Explore the model’s internal structure: identify available conditioning and target features.
# Conditioning features represent the factors expected to influence your model’s behavior.
# Target features represent the assets you aim to simulate.
conditioning_set = model.get_conditioning_set()
target_set = model.get_target_set()

# 5) Create a scenario on the selected model 
scenario = model.create_scenario(
    simulation_date="2022-06-15", # Default date for conditional features if nothing else specified
    name="Inflation + Hikes",
    feature_simulation_dates={
        # Use exact conditioning feature names available on your model
        "Consumer Price Index": "2022-06-01", # CPI YoY peak month
        "Federal Funds Rate": "2022-06-15", # fast 75bp liftoff phase
        "VIX Volatility Index": "2020-03-16", # modern vol shock
    }
)

# 6) Simulate forecasts (number of paths)
result = scenario.simulate(n_samples=100)

# 7) Access and visualize the forecast data
scenario.plot_forecasts(feature="30-Year Treasury Constant Maturity Rate",save=True, save_dir="./forecasts")

# 7) Create a portfolio from the model's target set (you CAN create portfolios)
portfolio = client.create_portfolio(
    name="Test Portfolio",
    target_set=model.get_target_set(),
    weights={
        "1-3 Year Treasury Bond ETF": 0.4,
        "3-7 Year Treasury Bond ETF": 0.3,
        "7-10 Year Treasury Bond ETF": 0.2,
        "20+ Year Treasury Bond ETF": 0.1
    },  
     capital=200000.0,
     description="US Treasury Portfolio"
)

# 8) Test the portfolio against the scenario
test = portfolio.test(scenario)

# 9) Review scenario results and summary outputs.
metrics = test.report_aggregated_metrics()

print(f"Sharpe (mean): {metrics['sharpe_distribution']['mean']:.3f}")
print(f"Total Return (mean): {metrics['return_distribution']['mean']:.2%}")
print(f"Max Drawdown (mean): {metrics['drawdown_distribution']['mean']:.2%}")

# Plot
test.plot_distribution('total_return')
test.plot_evolution('portfolio_value')
```

---

## API Key Management & Client

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

---

## Jupyter Template Notebook

A prebuilt Jupyter notebook named `Example.ipynb` is provided in the `testing/` folder of this repo to:
- Initialize the client and list projects/models/scenarios
- Create and simulate scenarios
- Build portfolios and run tests
- Plot distributions and time‑series (VaR/CVaR bands, drawdowns, etc.)

Launch:
```bash
cd testing
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
jupyter notebook "Template.ipynb"
```

---

## Common API Surface (First‑use Functions)

- Projects (browse existing)
  - `client.list_projects(include_templates=True)`
  - `client.get_project(name_or_index)`

- Models (read-only)
  - `project.list_models()`
  - `model.list_features()`
  - `model.get_target_set()`
  - `model.list_scenarios(verbose=True)`

- Scenarios (create & simulate)
  - `model.create_scenario(simulation_date, name, feature_simulation_dates={...})`
  - `scenario.simulate(n_samples=100)`
  - `scenario.plot_forecasts(feature=...)`
  - `scenario.plot_conditioning(feature=...)`

- Portfolios & Tests
  - `portfolio = client.create_portfolio(name, target_set, weights={...}, capital=..., constraint_type=...)`
  - `test = portfolio.test(scenario)`
  - `test.report_aggregated_metrics()` / `test.report_sample_metrics(sample_idx)`
  - `test.plot_distribution('total_return' | 'sharpe_ratio' | ...)`
  - `test.plot_evolution('pnl' | 'returns' | 'portfolio_value' | 'drawdown')`
  - `portfolio.compare_scenarios([scenario_a, scenario_b], labels=[...])`
  - `portfolio.plot_scenario_comparison([scenario_a, scenario_b], labels=[...])`

---

## Typical Workflows

- From scratch
  1) Select project from template
  2) Select model from template
  3) Create scenario → simulate → plot
  4) Create portfolio → test sccenarios → analyze metrics/plots

- Using existing resources
  - `projects = client.list_projects()` → pick one
  - `model = projects[0].list_models()[0]`
  - `scenarios = model.list_scenarios()` → pick one (avoid re‑simulating to keep results stable)
  - `portfolio = client.list_portfolios()[0]` → `portfolio.test(scenario)`

---

## Notes & Tips

- Metrics in this version are computed over a fixed 80-day simulation window, as the simulation period is currently set to 80 days from the execution date. Future releases will allow users to configure the simulation horizon dynamically.
- Select scenarios by name/ID rather than index; list order can change.
- For stable tails, use larger `n_samples` (e.g., ≥ 100) and avoid re‑simulation between comparisons.
- Portfolio tests are stored locally in `~/.sablier/portfolios.db`.

---

## License

MIT

---

