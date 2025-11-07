<div align="center">
  <img src="images/logo.png" alt="Sablier logo" width="320" />
  <h1>Sablier SDK</h1>
  <p>Open-source Python SDK for scenario-conditioned synthetic markets, portfolio testing, and risk analytics.</p>
  <p>
    <a href="https://pypi.org/project/sablier-sdk/">
      <img src="https://img.shields.io/pypi/v/sablier-sdk.svg?color=2274A5" alt="PyPI version" />
    </a>
    <a href="https://pypi.org/project/sablier-sdk/">
      <img src="https://img.shields.io/pypi/pyversions/sablier-sdk.svg" alt="Python versions" />
    </a>
    <a href="https://github.com/michu5696/sablier-sdk/stargazers">
      <img src="https://img.shields.io/github/stars/michu5696/sablier-sdk?style=social" alt="GitHub stars" />
    </a>
    <a href="LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg" alt="License: MIT" />
    </a>
  </p>
</div>

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

From PyPI :
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

At present, you can browse existing projects/models provided in our backend, then create scenarios and portfolios locally to test them. 
Note: In our next version, you’ll be able to create your own projects and train custom models directly within the SDK.

```python
from sablier import SablierClient

API_URL = "https://sablier-api-z4kedk7vaa-uc.a.run.app"
 
# 1) Initialize client. First time: SDK will prompt for registration
client = SablierClient(api_url=API_URL)

# Register with your e-mail address and you will receive an e-mail with your API Key
client.register_user(email="team@sablier.it", name="Alex Doherty", company="HSBC", role="Analyst")

# Save an API key with a custom name
client.save_api_key(
    api_key="sk_...",
    api_url=API_URL,
    description="default"
)

# List all saved keys
keys = client.list_api_keys()

# Use a specific key by name
client = SablierClient(api_url=API_URL) 
# Will use the default key automatically
# If you haven't saved the key, do:
# client = SablierClient(api_url=API_URL, api_key=YOUR_API_KEY)

# Check limits and usage
client.get_limits_and_usage()

# 2) List existing projects and pick one
projects = client.list_projects()

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
    simulation_date="2024-08-05",  # anchor date 
    name="S&P 500 drop",
    description=(
        "SP500 dropped by 160 points in a single day, 12th largest daily drop in history")
)

# 6) Simulate forecasts (number of paths)
scenario.simulate(n_samples=1000)

# 7) Access and visualize the forecast data
scenario.plot_forecasts(feature="1-3 Year Treasury Bond ETF",save=True, save_dir="./forecasts")
```

<div align="center">
  <img src="images/forecast.png" alt="Forecast visualization" style="max-width: 900px; width: 100%;" />
</div>

```python
# 8) Create a portfolio from the model's target set (you CAN create portfolios)
portfolio = client.create_portfolio(
    name="US Treasury Long Positions",
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

# 9) Test the portfolio against the scenario
test = portfolio.test(scenario)

# 9) Review scenario results and summary outputs.
metrics = test.report_aggregated_metrics()

# 10) Review scenario results and summary outputs.
metrics = test.show_aggregated_metrics()

# 10) Plot
test.plot_distribution('total_return')
test.plot_evolution('portfolio_value')

# 11) Compare multiple scenarios' effect on portfolio.
portfolio.compare_scenarios(scenario_a, scenario_b)
```

<div align="center">
  <img src="images/scenario_comparison.png" alt="Scenario comparison table" width="520" />
</div>

```python
# 12) Compare the same scenario on multiple portfolios.
scenario.compare_portfolios(portfolio_a, portfolio_b)
```

<div align="center">
  <img src="images/portfolio_comparison.png" alt="Portfolio comparison table" width="520" />
</div>

---

## Jupyter Template Notebook

A prebuilt Jupyter notebook named `Example.ipynb` is provided in this repo to:
- Initialize the client and list projects/models/scenarios
- Create and simulate scenarios
- Build portfolios and run tests
- Plot distributions and time‑series (VaR/CVaR bands, drawdowns, etc.)


---

## Common API Surface (First‑use Functions)

- Projects (browse existing)
  - `client.list_projects()`
  - `client.get_project(name_or_id)`

- Models (read-only)
  - `project.list_models()`
  - `model.list_features()`
  - `model.get_target_set()`
  - `model.list_scenarios()`

- Scenarios (create & simulate)
  - `model.create_scenario(simulation_date, name, feature_simulation_dates={...})`
  - `scenario.simulate(n_samples=1000)`
  - `scenario.plot_forecasts(feature=...)`
  - `scenario.plot_conditioning(feature=...)`

- Portfolios & Tests
  - `portfolio = client.create_portfolio(name, target_set, weights=[...], capital=..., constraint_type=...)`
  - `test = portfolio.test(scenario)`
  - `test.report_aggregated_metrics()` / `test.report_sample_metrics(sample_idx)`
  - `test.plot_distribution('total_return' | 'sharpe_ratio' | ...)`
  - `test.plot_evolution('pnl' | 'returns' | 'portfolio_value' | 'max_drawdown')`
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
  - `scenarios = model.list_scenarios()` → pick one (re-simulating scenarios uses fresh past data)
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

