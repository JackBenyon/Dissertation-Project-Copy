import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFECV
n_cores = joblib.cpu_count(only_physical_cores=True)

# Code to perform recursive feature elimination with cross-validation to find best features for the position datasets

allData2021_2022_2023_normalisedPos = pd.read_csv("/dcs/22/u2208738/Project/AllData2021_2022_2023_normalisedPos.csv", index_col = 0)

gk_data = allData2021_2022_2023_normalisedPos[allData2021_2022_2023_normalisedPos["position"] == 1].sample(frac = 1).reset_index(drop=True)
def_data = allData2021_2022_2023_normalisedPos[allData2021_2022_2023_normalisedPos["position"] == 2].sample(frac = 1).reset_index(drop=True)
mid_data = allData2021_2022_2023_normalisedPos[allData2021_2022_2023_normalisedPos["position"] == 3].sample(frac = 1).reset_index(drop=True)
fwd_data = allData2021_2022_2023_normalisedPos[allData2021_2022_2023_normalisedPos["position"] == 4].sample(frac = 1).reset_index(drop=True)

all_data = [gk_data, def_data, mid_data, fwd_data]
pos = ["GK", "DEF", "MID", "FWD"]

count = 0
for data in all_data:
  print(f"Position :{pos[count]}")
  hgbm = GradientBoostingRegressor(learning_rate=0.025, min_samples_leaf = 300)
  rep_cv = RepeatedKFold(n_splits=5, n_repeats=5)
  rfecv = RFECV(estimator=hgbm, step=1, cv=rep_cv, scoring='neg_root_mean_squared_error', n_jobs = n_cores)
  y = data["total_points"]
  X = data.drop(columns = ["total_points", "Name", "GW", "Team", "goals", "assists", "clean_sheet_kept", "saves_made", "bonus_scored", "mins_played", "Web Name", "Date", "fpl_id"])
  rfecv.fit(X, y)

  print(f"Optimal number of features : {rfecv.n_features_}")
  print(f"Selected features: {list(rfecv.support_)}\n")
  print(f"Feature Names :, {rfecv.get_feature_names_out()}")
  count += 1
