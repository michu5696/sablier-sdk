"""
Test listing and finding models via SDK
"""

from sablier import SablierClient

# Initialize client
client = SablierClient(
    api_url="http://localhost:8000",
    api_key="sk_82640081f0e8d97b54274dadbc42c2071e9e57de64321d73f596661b9eea450d"
)

print("✅ Client initialized\n")

# List all models
print("📋 Listing all models...")
models = client.models.list()

if models:
    print(f"\n✅ Found {len(models)} model(s):")
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model.name} (ID: {model.id})")
        print(f"      Status: {model._data.get('status')}, Stage: {model._data.get('current_stage')}")
else:
    print("   No models found")

# Try to find a specific model by name
print("\n🔍 Finding model by name...")
model = client.models.get_by_name("SDK Created Model")

if model:
    print(f"✅ Found model: {model.name}")
    print(f"   ID: {model.id}")
    print(f"   Created: {model._data.get('created_at')}")
else:
    print("❌ Model not found")

# Create a new one if none exist
if not models:
    print("\n📊 Creating a test model...")
    new_model = client.models.create(
        name="My First Model",
        description="Created via SDK"
    )
    print(f"✅ Created: {new_model.name}")

print("\n🎉 Model listing test complete!")
