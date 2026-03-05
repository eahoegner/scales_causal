# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pickle

# %%
scales_path = "/Users/hoegner/Projects/scales/emulator/ssp245_ensemble40_monthly_2500m.pkl"

# %%
y_pred_ensemble = pickle.load( open(scales_path, "rb") ) #dim = n_samples,n_months,n_regions

# %%
y_pred_ensemble[0][0]
