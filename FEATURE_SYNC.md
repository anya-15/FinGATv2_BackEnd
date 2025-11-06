# 🔄 Automatic Feature Synchronization

## Overview

The FinGAT predictor now **automatically synchronizes** with the latest trained model's feature selection. You never need to manually update feature dimensions again!

---

## How It Works

### **1. Training Phase**
When you train a new model with RL optimization:
```bash
python scripts/train_with_hybrid_rl.py
```

The training script:
- Optimizes feature selection (e.g., selects 9 out of 14 features)
- Saves the feature mask to: `rl_models/hybrid/YYYY-MM-DD_HH-MM-SS/best_features.npy`
- Updates the manifest: `rl_models/selected_runs/latest_manifest.json`

### **2. Prediction Phase**
When you make predictions (via API or script):
```bash
curl http://localhost:8000/api/v1/predict/now
```

The predictor **automatically**:
1. Loads the feature mask from `latest_manifest.json`
2. Applies the mask to select only the RL-optimized features
3. Passes the correct number of features to the model

**No manual intervention required!** ✅

---

## Key Features

### **✅ Auto-Detection**
- Predictor checks `latest_manifest.json` on first prediction
- Loads the feature mask specified in the manifest
- Caches the mask for subsequent predictions

### **✅ Smart Fallback**
- If no manifest exists → uses all features
- If no feature mask → uses all features
- If mask file missing → logs warning, uses all features

### **✅ Hot Reload**
- Reload features without restarting server
- Call `/api/v1/reload-features` endpoint after training
- Immediately uses new model's features

---

## API Endpoints

### **POST /api/v1/reload-features**

Reload the feature mask from the latest training run without restarting the server.

**Usage:**
```bash
curl -X POST http://localhost:8000/api/v1/reload-features
```

**Response:**
```json
{
  "status": "success",
  "message": "Feature mask reloaded from latest training run",
  "features_selected": 9,
  "total_features": 14,
  "feature_mask": [false, false, true, false, true, true, false, true, true, true, true, false, true, true],
  "timestamp": "2025-11-06T16:58:00.000000"
}
```

---

## Workflow Examples

### **Example 1: Train New Model**

```bash
# Step 1: Train with RL optimization
python scripts/train_with_hybrid_rl.py

# Output:
# ✓ Saved feature mask: rl_models/hybrid/2025-11-06_16-49-13/best_features.npy
# ✓ Updated latest manifest: rl_models/selected_runs/latest_manifest.json
# ✓ Selected features: 9/14

# Step 2: Reload features (if server is running)
curl -X POST http://localhost:8000/api/v1/reload-features

# Step 3: Make predictions (automatically uses new features)
curl http://localhost:8000/api/v1/predict/now
```

### **Example 2: Start Fresh Server**

```bash
# Step 1: Start server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Step 2: Make predictions (automatically loads latest features)
curl http://localhost:8000/api/v1/predict/now

# ✅ Predictor automatically:
# - Reads latest_manifest.json
# - Loads feature mask
# - Applies mask to features
# - Uses correct dimensions
```

---

## Technical Details

### **Feature Mask Storage**

**Location:** `rl_models/hybrid/YYYY-MM-DD_HH-MM-SS/best_features.npy`

**Format:** NumPy boolean array
```python
array([False, False, True, False, True, True, False, True, True, True, True, False, True, True])
# Meaning: Use features at indices [2, 4, 5, 7, 8, 9, 10, 12, 13]
```

### **Manifest Structure**

**Location:** `rl_models/selected_runs/latest_manifest.json`

```json
{
  "run_id": "2025-11-06_16-49-13",
  "features_path": "rl_models/hybrid/2025-11-06_16-49-13/best_features.npy",
  "hparams_path": "rl_models/hybrid/2025-11-06_16-49-13/best_hparams.json",
  "checkpoint_path": "C:\\Users\\hp\\FinGAT_Backend\\checkpoints\\fingat-hybrid-epoch=07-val_mrr=0.3333.ckpt",
  "notes": "Hybrid RL: features + hparams; final 50-epoch training"
}
```

### **Predictor Implementation**

**File:** `app/core/predictor.py`

**Key Methods:**
- `_load_feature_mask()` - Loads mask from manifest (called once per session)
- `reload_feature_mask()` - Forces reload (called via API endpoint)
- `predict_top_k()` - Applies mask before model inference

**Code Flow:**
```python
# 1. Load data (14 features)
data, metadata = self.data_loader.prepare_dataset()
x = data.x.to(self.device)  # Shape: [147, 14]

# 2. Auto-load and apply feature mask
self._load_feature_mask()
if self.feature_mask is not None:
    x = x[:, self.feature_mask]  # Shape: [147, 9]

# 3. Model inference (expects 9 features)
predictions = self.model(x, edge_index, ...)
```

---

## Benefits

### **🎯 Zero Manual Configuration**
- No need to edit `predictor.py` after training
- No need to match feature counts manually
- No dimension mismatch errors

### **🔄 Seamless Updates**
- Train new model → Features auto-sync
- Restart server → Loads latest features
- Hot reload → No downtime

### **🛡️ Error Prevention**
- Eliminates "normalized_shape" errors
- Prevents feature dimension mismatches
- Ensures model-data compatibility

### **📊 Transparency**
- Logs which features are selected
- API endpoint shows current mask
- Easy to verify configuration

---

## Troubleshooting

### **Issue: Features not syncing**

**Check manifest:**
```bash
cat rl_models/selected_runs/latest_manifest.json
```

**Verify feature mask exists:**
```bash
ls rl_models/hybrid/YYYY-MM-DD_HH-MM-SS/best_features.npy
```

**Force reload:**
```bash
curl -X POST http://localhost:8000/api/v1/reload-features
```

### **Issue: Still getting dimension errors**

**Check logs:**
```bash
# Look for:
# "✅ Loaded RL feature mask: X/Y features selected"
```

**Verify mask is applied:**
```python
# In predictor.py, check:
logger.info(f"Feature shape before mask: {x.shape}")
if self.feature_mask is not None:
    x = x[:, self.feature_mask]
logger.info(f"Feature shape after mask: {x.shape}")
```

### **Issue: Want to use all features**

**Option 1: Delete manifest**
```bash
rm rl_models/selected_runs/latest_manifest.json
```

**Option 2: Train without RL**
```bash
python scripts/train_model.py  # Uses all features
```

---

## Migration from Old System

### **Before (Manual)**
```python
# ❌ Had to manually edit predictor.py after each training
# ❌ Had to count features and update code
# ❌ Frequent dimension mismatch errors

# predictor.py (manual edit required)
x = data.x[:, [2, 4, 5, 7, 8, 9, 10, 12, 13]]  # Hardcoded!
```

### **After (Automatic)**
```python
# ✅ Automatically loads from manifest
# ✅ No code changes needed
# ✅ Zero configuration

# predictor.py (automatic)
self._load_feature_mask()
if self.feature_mask is not None:
    x = x[:, self.feature_mask]  # Auto-synced!
```

---

## Summary

**The feature synchronization system ensures that:**
1. ✅ Every training run saves its feature selection
2. ✅ Every prediction automatically uses the correct features
3. ✅ No manual configuration is ever needed
4. ✅ You can hot-reload features without restarting

**You can now train as many models as you want, and predictions will always use the right features automatically!** 🎉

---

**Last Updated:** November 6, 2025  
**Status:** ✅ Production Ready
