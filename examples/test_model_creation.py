"""
Test model creation via SDK
"""

from sablier import SablierClient

# Initialize client with API key
client = SablierClient(
    api_url="http://localhost:8000",
    api_key="sk_82640081f0e8d97b54274dadbc42c2071e9e57de64321d73f596661b9eea450d",
    supabase_url="https://ttlahhqhtmqwyeqvbmbp.supabase.co"
)

print("✅ Client initialized")

# Create a model
print("\n📊 Creating model via SDK...")
model = client.models.create(
    name="SDK Created Model",
    description="This model was created using the Sablier SDK"
)

print(f"\n✅ Model created successfully!")
print(f"   ID: {model.id}")
print(f"   Name: {model.name}")
print(f"   Description: {model.description}")

# The model object already has all data - no need to call get()
# The get() method is for loading existing models in a new session

print("\n💡 Simulating a new session...")
print(f"   User saves model ID: {model.id}")

# Later, in a new session, load the existing model
print("\n📖 Loading existing model (as if in a new session)...")
existing_model = client.models.get(model.id)
print(f"✅ Model loaded: {existing_model.name}")
print(f"   Status: {existing_model._data.get('status')}")
print(f"   Stage: {existing_model._data.get('current_stage')}")

print("\n🎉 SDK model creation and retrieval test successful!")
