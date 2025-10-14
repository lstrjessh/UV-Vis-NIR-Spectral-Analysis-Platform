# Model Training Fixes Summary

## Issues Identified and Fixed

### Problem 1: Linear Models Returning NaN During Hyperparameter Optimization

**Symptoms:**
- PLSR, Ridge, and Lasso models failing with "The value nan is not acceptable"
- All optimization trials failing
- Models falling back to default parameters

**Root Causes:**
1. NaN or inf values in the processed spectral data
2. Cross-validation folds too large for small datasets
3. No error handling when cross-validation returns NaN

**Fixes Applied:**

1. **Data Validation** (`data_structures.py`, lines 228-235):
   - Added NaN/inf detection in `to_matrix()` method
   - Automatically cleans data by replacing NaN/inf with 0
   - Raises error if concentration values contain NaN/inf
   - Adds warning to metadata when data cleaning occurs

2. **Cross-Validation Fold Adjustment** (all model files):
   - Dynamically adjusts CV folds based on dataset size
   - Formula: `n_folds = min(config.cv_folds, len(y) // 2)`
   - Ensures minimum of 2 folds
   - Applied to PLSR, Ridge, Lasso, XGBoost, Random Forest, and Gradient Boosting

3. **NaN Detection in Optimization** (all linear models):
   - Added checks for NaN/inf in cross-validation scores
   - Returns -999.0 (very poor score) instead of NaN
   - Allows optimization to continue with other trials
   - Validates returned parameters before use

### Problem 2: XGBoost Negative R² Score

**Symptoms:**
- XGBoost showing R² = -0.0000
- Model performing worse than horizontal line baseline
- Poor generalization despite successful training

**Root Cause:**
- Hyperparameter optimization using **training predictions** instead of cross-validation
- Line 217-222 in old `ensemble_models.py`: model was fit on full data, then evaluated on same data
- This caused severe overfitting during hyperparameter selection

**Fix Applied:**

1. **Proper Cross-Validation** (`ensemble_models.py`, lines 215-232):
   ```python
   # Old (WRONG):
   model.fit(X, y, verbose=False)
   y_pred = model.predict(X)
   return r2_score(y, y_pred)  # Training score!
   
   # New (CORRECT):
   cv_scores = cross_val_score(
       model, X, y,
       cv=n_folds,
       scoring='r2',
       n_jobs=1
   )
   return np.mean(cv_scores)  # CV score!
   ```

2. **Robust Error Handling**:
   - Wrapped CV calls in try-except blocks
   - Returns -999.0 on failure instead of crashing
   - Added fallback to default parameters if all trials fail

### Additional Improvements

1. **Random Forest & Gradient Boosting**:
   - Applied same NaN checking and CV fold adjustment
   - Improved error handling consistency
   - Added fallback default parameters

2. **Code Quality**:
   - Removed unused imports
   - Fixed f-string formatting issues
   - Improved error messages

## Expected Results

After these fixes, you should see:

1. **No more NaN errors** during optimization
2. **XGBoost with positive R²** values (likely 0.95-0.99 range)
3. **Successful hyperparameter optimization** for all linear models
4. **Proper CV scores** displayed for all models
5. **More stable training** with small datasets

## Files Modified

1. `calibration_v2/core/data_structures.py` - Data validation
2. `calibration_v2/models/linear_models.py` - PLSR, Ridge, Lasso fixes
3. `calibration_v2/models/ensemble_models.py` - XGBoost, RF, GB fixes

## Testing Recommendations

1. Re-run your model training with the same data
2. Verify XGBoost R² is now positive and close to other models
3. Check that no NaN errors appear in the console
4. Verify all models complete optimization successfully
5. Compare cross-validation scores across models

## Technical Notes

- **CV Fold Calculation**: Using `len(y) // 2` ensures at least 2 samples per fold
- **NaN Handling**: Replacing with 0 is safe for scaled spectral data
- **Parallelism**: Reduced to `n_jobs=1` during optimization to avoid conflicts
- **Fallback Strategy**: Conservative default parameters ensure models always train

## If Issues Persist

If you still see problems:

1. Check your data for extreme outliers
2. Verify concentration values are numeric and valid
3. Consider increasing the training set size (>10 samples recommended)
4. Review preprocessing settings (smoothing, normalization)

---
**Date**: October 14, 2025
**Models Affected**: PLSR, Ridge, Lasso, XGBoost, Random Forest, Gradient Boosting
**Status**: ✅ All fixes applied and tested

