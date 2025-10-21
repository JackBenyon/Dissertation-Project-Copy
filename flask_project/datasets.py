import pandas as pd

# Read all required datasets

allData2021_2022_2023_normalisedPos = pd.read_csv('data/AllData2021_2022_2023_normalisedPos.csv')
allData2024_normalisedPos = pd.read_csv('data/AllData2024_normalisedPos.csv')
smogn_data = pd.read_csv('data/smogn_data.csv', index_col = 0)

gk_train_data = pd.read_csv("data/gk_data_21_22_23.csv", index_col = 0)
def_train_data = pd.read_csv("data/def_data_21_22_23.csv", index_col = 0)
mid_train_data = pd.read_csv("data/mid_data_21_22_23.csv", index_col = 0)
fwd_train_data = pd.read_csv("data/fwd_data_21_22_23.csv", index_col = 0)

smogn_gk_train = pd.read_csv("data/smogn_gk_data_21_22_23.csv", index_col = 0)
smogn_def_train = pd.read_csv("data/smogn_def_data_21_22_23.csv", index_col = 0)
smogn_mid_train = pd.read_csv("data/smogn_mid_data_21_22_23.csv", index_col = 0)
smogn_fwd_train = pd.read_csv("data/smogn_fwd_data_21_22_23.csv", index_col = 0)

gk_test_data = pd.read_csv("data/gk_data_24.csv", index_col = 0)
def_test_data = pd.read_csv("data/def_data_24.csv", index_col = 0)
mid_test_data = pd.read_csv("data/mid_data_24.csv", index_col = 0)
fwd_test_data = pd.read_csv("data/fwd_data_24.csv", index_col = 0)


goals_train_data = pd.read_csv("data/goals_data_21_22_23.csv", index_col = 0)
goals_test_data = pd.read_csv("data/goals_data_24.csv", index_col = 0)
assists_train_data = pd.read_csv("data/assists_data_21_22_23.csv", index_col = 0)
assists_test_data = pd.read_csv("data/assists_data_24.csv", index_col = 0)
mins_train_data = pd.read_csv("data/mins_data_21_22_23.csv", index_col = 0)
mins_test_data = pd.read_csv("data/mins_data_24.csv", index_col = 0)
saves_train_data = pd.read_csv("data/saves_data_21_22_23.csv", index_col = 0)
saves_test_data = pd.read_csv("data/saves_data_24.csv", index_col = 0)
clean_sheet_train_data = pd.read_csv("data/clean_sheet_data_21_22_23.csv", index_col = 0)
clean_sheet_test_data = pd.read_csv("data/clean_sheet_data_24.csv", index_col = 0)
bonus_train_data = pd.read_csv("data/bonus_data_21_22_23.csv", index_col = 0)
bonus_test_data = pd.read_csv("data/bonus_data_24.csv", index_col = 0)
goals_conceded_train_data = pd.read_csv("data/goals_conceded_data_21_22_23.csv", index_col = 0)
goals_conceded_test_data = pd.read_csv("data/goals_conceded_data_24.csv", index_col = 0)

smogn_goals_train_data = pd.read_csv("data/smogn_goals_data_21_22_23.csv", index_col = 0)
smogn_assists_train_data = pd.read_csv("data/smogn_assists_data_21_22_23.csv", index_col = 0)
smogn_mins_train_data = pd.read_csv("data/smogn_mins_data_21_22_23.csv", index_col = 0)
smogn_saves_train_data = pd.read_csv("data/smogn_saves_data_21_22_23.csv", index_col = 0)
smogn_clean_sheet_train_data = pd.read_csv("data/smogn_clean_sheet_data_21_22_23.csv", index_col = 0)
smogn_bonus_train_data = pd.read_csv("data/smogn_bonus_data_21_22_23.csv", index_col = 0)