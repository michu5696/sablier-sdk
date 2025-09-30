"""
Quickstart example for Sablier SDK

This example shows the basic workflow:
1. Create a model
2. Build and train it
3. Create a scenario
4. Generate synthetic paths
5. Validate the results
"""

from sablier import SablierClient

# Initialize client
client = SablierClient(
    api_url="http://localhost:8000",
    api_key="c511aa8b5a93703f55bc03c9dbd1b9cc3343a949346bca9ad7e250c99d06d775"
)

print("✅ Client initialized")
print(f"   API URL: {client.http.api_url}")

# Step 1: Create a model
print("\n📊 Creating model...")
model = client.models.create(
    name="Quickstart Model",
    description="Simple 3-feature model for demonstration",
    features=[
        {"name": "S&P 500", "source": "yahoo"},
        {"name": "VIX", "source": "yahoo"},
        {"name": "10-Year Treasury", "source": "fred"}
    ],
    training_period={"start": "2020-01-01", "end": "2024-12-31"}
)
print(f"✅ Model created: {model}")

# Step 2: Build the model (placeholder for now)
print("\n🔨 Building model...")
print("   [Fetching data...]")
print("   [Generating samples...]")
print("   [Training...]")
print("✅ Model built (placeholder)")

# Step 3: Create a scenario
print("\n🎯 Creating scenario...")
scenario = client.scenarios.create(
    model=model,
    name="High Volatility Test",
    description="Testing with elevated VIX"
)
print(f"✅ Scenario created: {scenario}")

# Step 4: Generate synthetic paths
print("\n🌊 Generating synthetic paths...")
synthetic_data = scenario.generate_paths(n_samples=1000)
print(f"✅ Generated: {synthetic_data}")

# Step 5: Analyze the synthetic data
print("\n📈 Analyzing synthetic data...")
print(f"   Source model: {synthetic_data.source_model.name}")
print(f"   Source scenario: {synthetic_data.source_scenario.name}")
print(f"   Features: {synthetic_data.features}")
print(f"   Metadata: {synthetic_data.metadata}")

print("\n🎉 Quickstart complete!")
print("\nNext steps:")
print("- Implement actual API calls")
print("- Add real data fetching")
print("- Implement validation methods")
print("- Add visualization")
