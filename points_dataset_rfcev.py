import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFECV

# Code to perform recursive feature elimination with cross-validation to find best features for the points datasets


n_cores = joblib.cpu_count(only_physical_cores=True)

allData2021_2022_2023_normalisedPos = pd.read_csv("/dcs/22/u2208738/Project/AllData2021_2022_2023_normalisedPos.csv", index_col = 0)

labels = ["goals", "assists", "clean_sheet_kept", "saves_made", "bonus_scored", "mins_played"]
for label in labels:
  print(f"Factor :{label}")
  hgbm = GradientBoostingRegressor(learning_rate=0.025, min_samples_leaf = 300)
  rep_cv = RepeatedKFold(n_splits=5, n_repeats=5)
  rfecv = RFECV(estimator=hgbm, step=1, cv=rep_cv, scoring='neg_root_mean_squared_error', n_jobs = n_cores)
  y = allData2021_2022_2023_normalisedPos[label]
  X = allData2021_2022_2023_normalisedPos.drop(columns = ["total_points", "Name", "GW", "Team", "goals", "assists", "clean_sheet_kept", "saves_made", "bonus_scored", "mins_played", "Web Name", "Date", "fpl_id"])
  rfecv.fit(X, y)

  print(f"Optimal number of features : {rfecv.n_features_}")
  print(f"Selected features: {list(rfecv.support_)}\n")
  print(f"Feature Names :, {rfecv.get_feature_names_out()}")

