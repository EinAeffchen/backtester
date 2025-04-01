from pandas import read_csv

from sklearn.model_selection import train_test_split
import pandas as pd

merged_df = read_csv(
    "data/final_market_features_with_corrected_dynamic_targets.csv", index_col=0, parse_dates=[0]
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Define the feature set for prediction
features = [
    "Center_BB_Div", "Change", "Slope", "Change_VIX", "MACD_hist", "Momentum", "Spectral_Regime"
]

# Prepare data for Dynamic Target prediction
X_train_dyn, X_test_dyn, y_train_dyn, y_test_dyn = train_test_split(
    merged_df[features], merged_df["Dynamic_Target"], test_size=0.2, random_state=42, shuffle=False
)

# Prepare data for Dynamic Momentum Target prediction
X_train_dyn_mom, X_test_dyn_mom, y_train_dyn_mom, y_test_dyn_mom = train_test_split(
    merged_df[features], merged_df["Dynamic_Momentum_Target"], test_size=0.2, random_state=42, shuffle=False
)

# Train Random Forest Regressor for Dynamic Target
rf_dyn = RandomForestRegressor(n_estimators=200, random_state=42)
rf_dyn.fit(X_train_dyn, y_train_dyn)

# Train Random Forest Regressor for Dynamic Momentum Target
rf_dyn_mom = RandomForestRegressor(n_estimators=200, random_state=42)
rf_dyn_mom.fit(X_train_dyn_mom, y_train_dyn_mom)

# Make predictions
y_pred_dyn = rf_dyn.predict(X_test_dyn)
y_pred_dyn_mom = rf_dyn_mom.predict(X_test_dyn_mom)

# Evaluate models
mae_dyn = mean_absolute_error(y_test_dyn, y_pred_dyn)
r2_dyn = r2_score(y_test_dyn, y_pred_dyn)

mae_dyn_mom = mean_absolute_error(y_test_dyn_mom, y_pred_dyn_mom)
r2_dyn_mom = r2_score(y_test_dyn_mom, y_pred_dyn_mom)

# Display model performance results
model_results = pd.DataFrame({
    "Metric": ["Mean Absolute Error", "R² Score"],
    "Dynamic Target": [mae_dyn, r2_dyn],
    "Dynamic Momentum Target": [mae_dyn_mom, r2_dyn_mom]
})

print(model_results)
