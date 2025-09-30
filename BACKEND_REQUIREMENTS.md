# Backend Requirements for SDK Support

## Overview

The SDK needs REST endpoints for model and scenario CRUD operations. Currently, these are handled directly by the frontend through Supabase, but the SDK needs backend endpoints for proper API access.

---

## Required Endpoints

### **1. Model Management**

#### **Create Model**
```
POST /api/v1/models
```

**Request:**
```json
{
  "user_id": "...",
  "name": "My Model",
  "description": "Model description",
  "selected_features": [
    {"name": "S&P 500", "source": "yahoo", "frequency": "daily"},
    {"name": "VIX", "source": "yahoo", "frequency": "daily"}
  ],
  "training_period": {
    "start": "2020-01-01",
    "end": "2024-12-31"
  }
}
```

**Response:**
```json
{
  "id": "model-uuid",
  "name": "My Model",
  "status": "created",
  "created_at": "2025-01-01T00:00:00Z"
}
```

**Implementation:**
- Insert row into `models` table
- Set status to `"created"`
- Return model data

---

#### **Get Model**
```
GET /api/v1/models/{model_id}
```

**Response:**
```json
{
  "id": "model-uuid",
  "name": "My Model",
  "description": "...",
  "status": "trained",
  "selected_features": [...],
  "training_period": {...},
  "feature_normalization_params": {...},
  "input_features": [...],
  "created_at": "...",
  "updated_at": "..."
}
```

**Implementation:**
- Query `models` table by ID
- Return full model data

---

#### **List Models**
```
GET /api/v1/models?limit=100&offset=0
```

**Response:**
```json
{
  "models": [
    {
      "id": "...",
      "name": "...",
      "status": "...",
      "created_at": "..."
    }
  ],
  "total": 150,
  "limit": 100,
  "offset": 0
}
```

---

### **2. Scenario Management**

#### **Create Scenario**
```
POST /api/v1/scenarios
```

**Request:**
```json
{
  "user_id": "...",
  "model_id": "...",
  "name": "VIX Spike Scenario",
  "description": "High volatility scenario"
}
```

**Response:**
```json
{
  "id": "scenario-uuid",
  "model_id": "...",
  "name": "VIX Spike Scenario",
  "current_step": "model-selection",
  "created_at": "..."
}
```

---

#### **Get Scenario**
```
GET /api/v1/scenarios/{scenario_id}
```

**Response:**
```json
{
  "id": "...",
  "model_id": "...",
  "name": "...",
  "description": "...",
  "current_step": "...",
  "fetched_past_data": [...],
  "normalized_fetched_past": [...],
  "encoded_future_conditioning": [...]
}
```

---

### **3. Data Fetching Endpoint**

#### **Fetch Historical Data**
```
POST /api/v1/data/fetch
```

**Request:**
```json
{
  "features": ["S&P 500", "VIX"],
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "processing_config": {
    "interpolation": {...}
  }
}
```

**Response:**
```json
{
  "data": {
    "S&P 500": [...],
    "VIX": [...]
  },
  "metadata": {
    "start_date": "...",
    "end_date": "...",
    "points_fetched": 365
  }
}
```

**Note:** This might already exist through edge functions, just needs to be exposed as an API endpoint.

---

## Implementation Priority

### **Phase 1 (Critical for SDK MVP):**
1. ✅ Model CRUD (Create, Get, List)
2. ✅ Scenario CRUD (Create, Get)
3. ⚠️ Data fetching (check if edge functions can be used)

### **Phase 2 (Already Exists):**
- ✅ Sample generation (`/api/v1/ml/generate-samples`)
- ✅ Fit encoding models (`/api/v1/ml/fit`)
- ✅ Encode samples (`/api/v1/ml/encode`)
- ✅ Train model (`/api/v1/ml/train-model`)
- ✅ Generate forecast (`/api/v1/ml/forecast`)
- ✅ Reconstruct (`/api/v1/ml/reconstruct`)

---

## Recommendation

**Create a new backend file:** `backend/endpoints/crud.py`

This will contain:
- Model CRUD operations
- Scenario CRUD operations
- Simple wrappers around Supabase queries
- Proper authentication and RLS

**Estimated time:** 1-2 hours

Should I create these CRUD endpoints for you?
