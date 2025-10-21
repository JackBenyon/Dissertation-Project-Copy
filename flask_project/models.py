import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from datasets import *

# This code is identical to the function in the jupyter notebook which predicts player points for a given gameweek

def gw_predict(gw, train_data, test_data, isSmogn, sentiment_type, llm_type):

  gb_reg = HistGradientBoostingRegressor(learning_rate=0.025, max_bins = 255, max_iter = 250, min_samples_leaf = 300)

  labels = train_data["total_points"]
  if(isSmogn):
      data = train_data.drop(columns = ["total_points", "goals", "assists", "bonus_scored", "clean_sheet_kept", "mins_played", "saves_made"])
  elif(sentiment_type == "sum"):
    data = train_data.drop(columns = ["total_points", "Name", "GW", "fpl_id", "Team", "index", "goals", "assists", "bonus_scored", "clean_sheet_kept", "mins_played", "saves_made", "Date", "Web Name",
                                      "Unnamed: 0", "Positive", "Neutral", "Negative", "Positive Mean", "Neutral Mean", "Negative Mean"])
  elif(sentiment_type == "mean"):
    data = train_data.drop(columns = ["total_points", "Name", "GW", "fpl_id", "Team", "index", "goals", "assists", "bonus_scored", "clean_sheet_kept", "mins_played", "saves_made", "Date", "Web Name",
                                      "Unnamed: 0", "Positive", "Neutral", "Negative", "Positive Sum", "Neutral Sum", "Negative Sum"])
  elif(llm_type == "small embedding"):
    data = train_data.drop(columns = ["total_points", "Name", "GW", "fpl_id", "Team", "index", "goals", "assists", "bonus_scored", "clean_sheet_kept", "mins_played", "saves_made","Date", "Web Name", "Player", "Good Form", "Bad Form", "Injured", "Unsure"])
    data = data.drop(data.columns[data.columns.str.contains('l_em')], axis=1)
  elif(llm_type == "large embedding"):
    data = train_data.drop(columns = ["total_points", "Name", "GW", "fpl_id", "Team", "index", "goals", "assists", "bonus_scored", "clean_sheet_kept", "mins_played", "saves_made","Date", "Web Name", "Player", "Good Form", "Bad Form", "Injured", "Unsure"])
    data = data.drop(data.columns[data.columns.str.contains('s_em')], axis=1)
  elif(llm_type == "prompt"):
    data = train_data.drop(columns = ["total_points", "Name", "GW", "fpl_id", "Team", "index", "goals", "assists", "bonus_scored", "clean_sheet_kept", "mins_played", "saves_made","Date", "Web Name", "Player"])
    data = data.drop(data.columns[data.columns.str.contains('s_em') or data.columns.str.contains('l_em')], axis=1)
  else:
    data = train_data.drop(columns = ["total_points", "Name", "GW", "fpl_id", "Team", "goals", "assists", "bonus_scored", "clean_sheet_kept", "mins_played", "saves_made", "Date", "Web Name"])

  gb_reg.fit(data, labels)

  gw_data = test_data[test_data["GW"] == gw]

  if(sentiment_type == "sum"):
    gw_data_pred = gw_data.drop(columns = ["total_points", "Name", "GW", "fpl_id", "Team", "index", "goals", "assists", "bonus_scored", "clean_sheet_kept", "mins_played", "saves_made", "Date", "Web Name",
                                           "Unnamed: 0", "Positive", "Neutral", "Negative", "Positive Mean", "Neutral Mean", "Negative Mean"])
  elif(sentiment_type == "mean"):
    gw_data_pred = gw_data.drop(columns = ["total_points", "Name", "GW", "fpl_id", "Team", "index", "goals", "assists", "bonus_scored", "clean_sheet_kept", "mins_played", "saves_made", "Date", "Web Name",
                                           "Unnamed: 0", "Positive", "Neutral", "Negative", "Positive Sum", "Neutral Sum", "Negative Sum"])
  elif(llm_type == "small embedding"):
    gw_data_pred = gw_data_pred.drop(columns = ["total_points", "Name", "GW", "fpl_id", "Team", "index", "goals", "assists", "bonus_scored", "clean_sheet_kept", "mins_played", "saves_made","Date", "Web Name", "Player", "Good Form", "Bad Form", "Injured", "Unsure"])
    gw_data_pred = gw_data_pred.drop(data.columns[data.columns.str.contains('l_em')], axis=1)
  elif(llm_type == "large embedding"):
    gw_data_pred = gw_data_pred.drop(columns = ["total_points", "Name", "GW", "fpl_id", "Team", "index", "goals", "assists", "bonus_scored", "clean_sheet_kept", "mins_played", "saves_made","Date", "Web Name", "Player", "Good Form", "Bad Form", "Injured", "Unsure"])
    gw_data_pred = gw_data_pred.drop(data.columns[data.columns.str.contains('s_em')], axis=1, inplace=True)
  elif(llm_type == "prompt"):
    gw_data_pred = gw_data_pred.drop(columns = ["total_points", "Name", "GW", "fpl_id", "Team", "index", "goals", "assists", "bonus_scored", "clean_sheet_kept", "mins_played", "saves_made","Date", "Web Name", "Player"])
    gw_data_pred = gw_data_pred.drop(data.columns[data.columns.str.contains('s_em') or data.columns.str.contains('l_em')], axis=1, inplace=True)
  else:
    gw_data_pred = gw_data.drop(columns = ["total_points", "Name", "GW", "Team", "fpl_id", "goals", "assists", "bonus_scored", "clean_sheet_kept","mins_played", "saves_made", "Date", "Web Name"])

  gw_pred = gb_reg.predict(gw_data_pred)

  pairs = {}
  for i in range(0, len(gw_pred)):
    # Accountz for double gameweeks
    if(gw_data.iloc[i]["fpl_id"] in pairs):
      pairs[gw_data.iloc[i]["fpl_id"]] += gw_pred[i]
    else:
      pairs[gw_data.iloc[i]["fpl_id"]] = gw_pred[i]

  sorted_pairs = dict(sorted(pairs.items(), key=lambda item: item[1], reverse = True))

  return sorted_pairs


def gw_predict_with_points_data(gw, isSmogn):

  def gw_predict(gw, train_data, test_data, label):

    gb_reg = HistGradientBoostingRegressor(learning_rate=0.1, max_iter=90, max_bins=220, random_state = 12)

    labels = train_data[label]
    #print(labels)
    if(isSmogn):
      data = train_data.drop(columns = [label])
    else:
      data = train_data.drop(columns = ["Name", "GW", "Team", "fpl_id", label])

    gb_reg.fit(data, labels)

    gw_data = test_data[(test_data["GW"] == gw)]

    gw_data_pred = gw_data.drop(columns = ["Name", "GW", "Team", "fpl_id", label])

    gw_pred = gb_reg.predict(gw_data_pred)

    pairs = {}
    for i in range(0, len(gw_pred)):
      pairs[gw_data.iloc[i]["fpl_id"]] = (gw_pred[i], gw_data.iloc[i]["position"])
    return pairs

  if(isSmogn):
    pred_goals = gw_predict(gw, smogn_goals_train_data, goals_test_data, "goals")
    pred_assists = gw_predict(gw, smogn_assists_train_data, assists_test_data, "assists")
    pred_mins = gw_predict(gw, smogn_mins_train_data, mins_test_data, "mins_played")
    pred_saves = gw_predict(gw, smogn_saves_train_data, saves_test_data, "saves_made")
    pred_clean_sheet = gw_predict(gw, smogn_clean_sheet_train_data, clean_sheet_test_data, "clean_sheet_kept")
    pred_bonus = gw_predict(gw, smogn_bonus_train_data, bonus_test_data, "bonus_scored")
  else:
    pred_goals = gw_predict(gw, goals_train_data, goals_test_data, "goals")
    pred_assists = gw_predict(gw, assists_train_data, assists_test_data, "assists")
    pred_mins = gw_predict(gw, mins_train_data, mins_test_data, "mins_played")
    pred_saves = gw_predict(gw, saves_train_data, saves_test_data, "saves_made")
    pred_clean_sheet = gw_predict(gw, clean_sheet_train_data, clean_sheet_test_data, "clean_sheet_kept")
    pred_bonus = gw_predict(gw, bonus_train_data, bonus_test_data, "bonus_scored")

  all_points = {}

  for player in pred_mins.keys():
    if(pred_mins[player][0] > 0 and pred_mins[player][0] < 60):
      all_points[player] = 1
    elif(pred_mins[player][0] >= 60):
      all_points[player] = 2
    else:
      all_points[player] = 0

  for player in pred_goals.keys():
    if(pred_goals[player][1] == 2):
      all_points[player] += 6*pred_goals[player][0]
    elif(pred_goals[player][1] == 3):
      all_points[player] += 5*pred_goals[player][0]
    elif(pred_goals[player][1] == 4):
      all_points[player] += 4*pred_goals[player][0]

  for player in pred_assists.keys():
    all_points[player] += 3*pred_assists[player][0]

  for player in pred_saves.keys():
    all_points[player] += pred_saves[player][0]//3

  for player in pred_clean_sheet.keys():
    if(pred_clean_sheet[player][1] == 1 or pred_clean_sheet[player][1] == 2):
      all_points[player] += 4*pred_clean_sheet[player][0]
    elif(pred_clean_sheet[player][1] == 3):
      all_points[player] += pred_clean_sheet[player][0]

  for player in pred_bonus.keys():
    all_points[player] += pred_bonus[player][0]

  sorted_points = dict(sorted(all_points.items(), key = lambda item : item[1], reverse = True))

  return sorted_points

def gw_predict_with_position_data(gw, isSmogn):

  def gw_predict(gw, train_data, test_data):

    gb_reg = HistGradientBoostingRegressor(learning_rate=0.1, max_iter=90, max_bins=220, random_state = 13)

    labels = train_data["total_points"]

    if(not isSmogn):
      data = train_data.drop(columns = ["total_points", "Name", "GW", "Team", "fpl_id"])
    else:
      data = train_data.drop(columns = ["total_points"])

    gb_reg.fit(data, labels)

    gw_data = test_data[((test_data["GW"] == gw) & (test_data["Team"] != "Liverpool") & (test_data["Team"] != "Everton")) | ((test_data["GW"] == gw-1) & ((test_data["Team"] == "Liverpool") | (test_data["Team"] == "Everton")))]

    gw_data.to_csv("/content/drive/MyDrive/Colab Notebooks/gw" + str(gw) + "_data.csv")

    gw_data_pred = gw_data.drop(columns = ["total_points", "Name", "GW", "Team", "fpl_id"])

    gw_pred = gb_reg.predict(gw_data_pred)

    pairs = {}
    for i in range(0, len(gw_pred)):
      pairs[gw_data.iloc[i]["fpl_id"]] = gw_pred[i]

    return pairs

  if(not isSmogn):
    gk_pred = gw_predict(gw, gk_train_data, gk_test_data)
    def_pred = gw_predict(gw, def_train_data, def_test_data)
    mid_pred = gw_predict(gw, mid_train_data, mid_test_data)
    fwd_pred = gw_predict(gw, fwd_train_data, fwd_test_data)

  else:
    gk_pred = gw_predict(gw, smogn_gk_train, gk_test_data)
    def_pred = gw_predict(gw, smogn_def_train, def_test_data)
    mid_pred = gw_predict(gw, smogn_mid_train, mid_test_data)
    fwd_pred = gw_predict(gw, smogn_fwd_train, fwd_test_data)

  sorted_all = dict(sorted((gk_pred | def_pred | mid_pred | fwd_pred).items(), key = lambda item : item[1], reverse = True))
  return sorted_all