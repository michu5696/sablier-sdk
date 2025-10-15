"""
Validation helpers for Model class
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta


def validate_sample_generation_inputs(
    input_features: List[Dict],
    past_window: int, 
    future_window: int,
    target_features: List[str],
    conditioning_features: Optional[List[str]]
):
    """Validate sample generation inputs"""
    
    # 0. Check minimum window sizes (ensures PCA-ICA has enough dimensions)
    MIN_PAST_WINDOW = 30  # At least 30 days of history
    MIN_FUTURE_WINDOW = 30  # At least 30 days to predict (ensures PCA can fit properly)
    
    if past_window < MIN_PAST_WINDOW:
        raise ValueError(f"past_window must be at least {MIN_PAST_WINDOW} days (got {past_window})")
    
    if future_window < MIN_FUTURE_WINDOW:
        raise ValueError(f"future_window must be at least {MIN_FUTURE_WINDOW} days (got {future_window})")
    
    # 1. Check we have input features
    if not input_features:
        raise ValueError("No features configured. Call model.add_features() first or use DataCollection")
    
    all_feature_names = [f.get('display_name', f.get('name')) for f in input_features]
    
    # 2. Target features must be specified and valid
    if not target_features:
        raise ValueError("target_features is required. Specify which features to predict.")
    
    invalid_targets = [f for f in target_features if f not in all_feature_names]
    if invalid_targets:
        raise ValueError(f"Invalid target features: {invalid_targets}. Available: {all_feature_names}")
    
    # 3. Conditioning features must be valid
    if conditioning_features:
        invalid_conditioning = [f for f in conditioning_features if f not in all_feature_names]
        if invalid_conditioning:
            raise ValueError(f"Invalid conditioning features: {invalid_conditioning}. Available: {all_feature_names}")
        
        # Check for overlap
        overlap = set(target_features) & set(conditioning_features)
        if overlap:
            raise ValueError(f"Features cannot be both target and conditioning: {overlap}")
    
    # 4. All features must be assigned a type
    assigned_features = set(target_features)
    if conditioning_features:
        assigned_features.update(conditioning_features)
    else:
        assigned_features.update([f for f in all_feature_names if f not in target_features])
    
    unassigned = set(all_feature_names) - assigned_features
    if unassigned:
        raise ValueError(f"All features must be assigned as target or conditioning. Unassigned: {unassigned}")


def validate_splits(splits: Dict[str, Dict[str, str]], past_window: int, future_window: int):
    """Validate split date ranges and check for overlaps"""
    
    required_splits = ["training", "validation"]
    for split_name in required_splits:
        if split_name not in splits:
            raise ValueError(f"Missing required split: {split_name}")
        if "start" not in splits[split_name] or "end" not in splits[split_name]:
            raise ValueError(f"Split '{split_name}' must have 'start' and 'end' dates")
    
    # Parse dates
    train_start = datetime.strptime(splits["training"]["start"], "%Y-%m-%d")
    train_end = datetime.strptime(splits["training"]["end"], "%Y-%m-%d")
    val_start = datetime.strptime(splits["validation"]["start"], "%Y-%m-%d")
    val_end = datetime.strptime(splits["validation"]["end"], "%Y-%m-%d")
    
    # Test split is optional
    has_test = "test" in splits
    if has_test:
        test_start = datetime.strptime(splits["test"]["start"], "%Y-%m-%d")
        test_end = datetime.strptime(splits["test"]["end"], "%Y-%m-%d")
    
    # Check minimum split lengths
    sample_size = past_window + future_window
    MIN_SAMPLES_PER_SPLIT = 10  # At least 10 samples per split
    min_split_days = sample_size + (MIN_SAMPLES_PER_SPLIT * 10)  # Assuming stride ~10 days
    
    train_days = (train_end - train_start).days
    val_days = (val_end - val_start).days
    
    if train_days < min_split_days:
        raise ValueError(
            f"Training split too short! Got {train_days} days, need at least {min_split_days} days "
            f"(sample size {sample_size} + room for ~{MIN_SAMPLES_PER_SPLIT} samples)"
        )
    
    if val_days < sample_size:
        raise ValueError(f"Validation split too short! Got {val_days} days, need at least {sample_size} days (sample size)")
    
    if has_test:
        test_days = (test_end - test_start).days
        if test_days < sample_size:
            raise ValueError(f"Test split too short! Got {test_days} days, need at least {sample_size} days (sample size)")
    
    # Check temporal ordering
    if has_test:
        if not (train_start < train_end < val_start < val_end < test_start < test_end):
            raise ValueError("Splits must be temporally ordered: training -> validation -> test (no overlaps)")
    else:
        if not (train_start < train_end < val_start < val_end):
            raise ValueError("Splits must be temporally ordered: training -> validation (no overlaps)")
    
    # Check for date range overlaps
    if train_end >= val_start:
        raise ValueError(f"Training end ({train_end.date()}) must be before validation start ({val_start.date()})")
    
    if has_test and val_end >= test_start:
        raise ValueError(f"Validation end ({val_end.date()}) must be before test start ({test_start.date()})")
    
    # Check for sample overlap (critical!)
    sample_size = past_window + future_window
    
    # Gap between training and validation
    train_val_gap = (val_start - train_end).days
    if train_val_gap < sample_size:
        raise ValueError(
            f"Training and validation splits are too close! "
            f"Gap: {train_val_gap} days, Sample size: {sample_size} days. "
            f"A sample starting at training end could overlap into validation. "
            f"Minimum gap required: {sample_size} days"
        )
    
    # Gap between validation and test (warning only, contiguous is OK)
    if has_test:
        val_test_gap = (test_start - val_end).days
        if val_test_gap < sample_size:
            print(f"  ⚠️  Validation and test are close (gap: {val_test_gap} days, sample: {sample_size} days)")
            print(f"      Last validation samples may overlap into test period (acceptable)")
    
    print(f"  ✓ Splits validated: no data leakage")


def auto_generate_splits(
    start_date: str, 
    end_date: str, 
    sample_size: Optional[int] = None,
    train_pct: float = 0.8,
    val_pct: float = 0.2,
    test_pct: float = 0.0
) -> Dict[str, Dict[str, str]]:
    """
    Auto-generate splits from training period with proper gaps
    
    Creates splits with gap between training and validation to prevent data leakage.
    Test split is optional (set test_pct=0 to disable).
    
    Args:
        start_date: Training period start (YYYY-MM-DD)
        end_date: Training period end (YYYY-MM-DD)
        sample_size: Optional sample size (past + future windows) for calculating gaps
        train_pct: Training split percentage (default: 0.8 = 80%)
        val_pct: Validation split percentage (default: 0.2 = 20%)
        test_pct: Test split percentage (default: 0.0 = disabled)
        
    Returns:
        Dict with training, validation, and optionally test splits
    """
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days
    
    # If sample_size provided, reserve gap days
    # Otherwise, use default gap
    gap_days = sample_size if sample_size else 45  # Default 45 day gap
    
    # Allocate days using provided percentages
    usable_days = total_days - gap_days  # Reserve gap between train and val
    
    train_days = int(usable_days * train_pct)
    val_days = int(usable_days * val_pct)
    test_days = int(usable_days * test_pct) if test_pct > 0 else 0
    
    # Calculate split boundaries
    train_start = start
    train_end = start + timedelta(days=train_days - 1)
    
    # Add gap after training
    val_start = train_end + timedelta(days=gap_days + 1)
    
    # If test split is disabled, validation extends to the end
    if test_pct == 0:
        val_end = end
    else:
        val_end = val_start + timedelta(days=val_days - 1)
        # Test is contiguous with validation (no gap needed)
        test_start = val_end + timedelta(days=1)
        test_end = end
    
    result = {
        "training": {
            "start": train_start.strftime("%Y-%m-%d"),
            "end": train_end.strftime("%Y-%m-%d")
        },
        "validation": {
            "start": val_start.strftime("%Y-%m-%d"),
            "end": val_end.strftime("%Y-%m-%d")
        }
    }
    
    # Only include test split if requested
    if test_pct > 0:
        result["test"] = {
            "start": test_start.strftime("%Y-%m-%d"),
            "end": test_end.strftime("%Y-%m-%d")
        }
    
    return result


def validate_training_period(start_date: str, end_date: str) -> int:
    """
    Validate training period length
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        int: Total days in period
        
    Raises:
        ValueError: If period is too short
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days
    
    MIN_TRAINING_DAYS = 180  # At least 6 months for meaningful models
    if total_days < MIN_TRAINING_DAYS:
        raise ValueError(
            f"Training period too short! Got {total_days} days, need at least {MIN_TRAINING_DAYS} days "
            f"(~6 months minimum for meaningful time series models)"
        )
    
    return total_days
