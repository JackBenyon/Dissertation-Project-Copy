from pulp import *
from models import *


def sum_data(player_data):
  if(player_data.shape[0] < 2):
    return player_data
  else:
    ret = pd.DataFrame(columns = player_data.columns)
    ret.loc[0] = [player_data["Name"].iloc[0], player_data["position"].iloc[0],sum(player_data["total_points"]), sum(player_data["mins_played"])]
  return ret

# This function is the same as the optimal team function within the jupyter notebook and is used to get the opimal team for a given gameweek

def optimal_team(gw, pred_points, budget):

  # Get player prices and positions
  curr_prices = pd.read_csv(f"C:/Users/Jack Benyon/Documents/3YP/Code/flask_project/data/prices/prices_{gw}.csv", index_col = 0)[["id","now_cost", "photo"]]
  all_prices = pd.read_csv(f"C:/Users/Jack Benyon/Documents/3YP/Code/flask_project/data/prices/all_prices.csv", index_col = 0).sort_values(by = ["gw"])
  positions = allData2024_normalisedPos[allData2024_normalisedPos["GW"]==gw][["Name", "fpl_id", "position", "Team", "total_points", "mins_played"]]

  # Accounts for if a player has two games in one gameweek
  for player in positions.itertuples():
    if(positions[positions["fpl_id"] == player.fpl_id].shape[0] > 1):
      positions.loc[positions["fpl_id"] == player.fpl_id, "total_points"] = positions.loc[positions["fpl_id"] == player.fpl_id, "total_points"].sum()
      positions.loc[positions["fpl_id"] == player.fpl_id, "mins_played"] = positions.loc[positions["fpl_id"] == player.fpl_id, "mins_played"].sum()
      positions = positions.drop_duplicates(subset = ["fpl_id"], keep = "first")

  n = positions.shape[0]
  # Dictionary mapping player id to their cost
  cost = dict(zip(curr_prices["id"], curr_prices["now_cost"]))
  #Dictionary mapping player id to their position
  pos = dict(zip(positions["fpl_id"], positions["position"]))
  #Dictionary mapping player id to their club
  club = dict(zip(positions["fpl_id"], positions["Team"]))
  all_clubs = ["Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Chelsea","Crystal Palace", "Everton", "Fulham", "Ipswich", "Leicester", "Liverpool", "Manchester City", "Manchester United", "Newcastle United", "Nottingham Forest", "Southampton", "Tottenham", "West Ham", "Wolverhampton Wanderers"]
  prob = LpProblem("FPL_Team", LpMaximize)
  # Binary variable for each player representing whether they are chosen in the team (1) or not (0)
  player_vars = [pulp.LpVariable("x{}".format(i), cat='Binary') for i in range(n)]
  # Binary variable for each player representing whether they are chosen in the starting 11 (1) or not (0)
  starting_player_vars = [pulp.LpVariable("y{}".format(i), cat='Binary') for i in range(n)]
  # Binary variable for each player representing whether they are chosen as captain (1) or not (0)
  captain_player_vars = [pulp.LpVariable("z{}".format(i), cat = 'Binary') for i in range(n)]
  # Binary variable for each player representing whether they are chosen as vice captain (1) or not (0)
  vice_captain_player_vars = [pulp.LpVariable("w{}".format(i), cat = 'Binary') for i in range(n)]

  # Dictionary mapping player variables to played fpl ids
  var_id_map = dict(zip([i for i in range(n)], positions["fpl_id"]))


  # Objective function to balance maximising the starting 11 and bench
  prob += (lpSum([(pred_points[var_id_map[i]] * (0.99 * starting_player_vars[i] + captain_player_vars[i] + 0.01 * vice_captain_player_vars[i] + 0.01 * player_vars[i])) for i in range(0,n)]), "Total points per player",)

  # Total team cost must not exceed the budget
  prob += (lpSum([cost[var_id_map[j]] * player_vars[j] if var_id_map[j] in cost else all_prices[all_prices["id"] == var_id_map[j]]["now_cost"].iloc[0] * player_vars[j] for j in range(0,n)]) <= budget, "Total Cost constraint",) # Accounts for differences in the price and player datasets
  # Starting 11 must be 11 players
  prob += (lpSum(starting_player_vars) == 11, "Starting 11 Number Requirement",)
  # Total team must be 15 players
  prob += (lpSum(player_vars) == 15, "Team Number Requirement",)
  # One captain and one vice captain
  prob += (lpSum(captain_player_vars) == 1, "Captain Number Requirement",)
  prob += (lpSum(vice_captain_player_vars) == 1, "Vice Captain Number Requirement",)

  # Position Constraints for starting 11
  prob += (lpSum(starting_player_vars[i] for i in range(0,n) if pos[var_id_map[i]] == 1) == 1, "Goalkeeper Number Requirement",)
  prob += (lpSum(starting_player_vars[i] for i in range(0,n) if pos[var_id_map[i]] == 2) >= 3, "Defender Number Requirement LB",)
  prob += (lpSum(starting_player_vars[i] for i in range(0,n) if pos[var_id_map[i]] == 2) <= 5, "Defender Number Requirement UB",)
  prob += (lpSum(starting_player_vars[i] for i in range(0,n) if pos[var_id_map[i]] == 3) >= 2, "Midfielder Number Requirement LB",)
  prob += (lpSum(starting_player_vars[i] for i in range(0,n) if pos[var_id_map[i]] == 3) <= 5, "Midfielder Number Requirement UB",)
  prob += (lpSum(starting_player_vars[i] for i in range(0,n) if pos[var_id_map[i]] == 4) >= 1, "Forward Number Requirement LB",)
  prob += (lpSum(starting_player_vars[i] for i in range(0,n) if pos[var_id_map[i]] == 4) <= 3, "Forward Number Requirement UB",)

  # Position Constraints for the entire team
  prob += (lpSum(player_vars[i] for i in range(0,n) if pos[var_id_map[i]] == 1) == 2, "Goalkeeper Number Requirement Starting 11",)
  prob += (lpSum(player_vars[i] for i in range(0,n) if pos[var_id_map[i]] == 2) == 5, "Defender Number Requirement Starting 11",)
  prob += (lpSum(player_vars[i] for i in range(0,n) if pos[var_id_map[i]] == 3) == 5, "Midfielder Number Requirement Starting 11",)
  prob += (lpSum(player_vars[i] for i in range(0,n) if pos[var_id_map[i]] == 4) == 3, "Forward Number Requirement Starting 11",)

  # Number of players from each team constraint
  for i in range(0, 20):
    prob += (lpSum(player_vars[j] for j in range(0,n) if club[var_id_map[j]] == all_clubs[i]) <= 3, "Team Requirement" + str(i),)

  # All players in the starting 11 must also be chosen in the team
  for i in range(0, n):
    prob += (player_vars[i] - starting_player_vars[i] >= 0, "Starting 11 constraint" + str(i),)

  # Captain and vice captain must be in the starting 11. Vice captain cannot be captain either.
  for i in range(0, n):
    prob += (starting_player_vars[i] - captain_player_vars[i] >= 0, "Captain constraint" + str(i),)
    prob += (starting_player_vars[i] - vice_captain_player_vars[i] >= 0, "Vice Captain constraint" + str(i),)
    prob += (captain_player_vars[i] + vice_captain_player_vars[i] <= 1.5, "Captain can't be vice captain constraint" + str(i),)

  prob.writeLP("FPL_Team.lp")
  prob.solve()

  
  goalkeepers, defenders, midfielders, forwards, bench = [], [], [], [], []
  team_points = 0
  budget_used = 0
  gk_bench = []
  zero_mins_players = []
  captain_played = False
  vice_captain_points = 0
  for v in range(0,n):
    # If the player is chosen to be in the starting 11
    if(starting_player_vars[v].varValue == 1):
      curr_player = sum_data(positions[positions["fpl_id"] == var_id_map[v]][["Name", "position", "total_points", "mins_played"]])
      team_points += int(curr_player["total_points"].iloc[0])
      budget_used += cost[var_id_map[v]] if var_id_map[v] in cost else all_prices[all_prices["id"] == var_id_map[v]]["now_cost"].iloc[0]

      if(curr_player["position"].iloc[0] == 1):
        if(captain_player_vars[v].varValue == 1):
          team_points += int(curr_player["total_points"].iloc[0])
          if(curr_player["mins_played"].iloc[0] != 0):
            captain_played = True
          goalkeepers.append((curr_player["Name"].iloc[0], round(pred_points[var_id_map[v]],2)*2, int(curr_player["total_points"].iloc[0])*2, "image_" + str(var_id_map[v]) + ".png", "Captain"))
        elif(vice_captain_player_vars[v].varValue == 1):
          goalkeepers.append((curr_player["Name"].iloc[0], round(pred_points[var_id_map[v]],2), int(curr_player["total_points"].iloc[0]), "image_" + str(var_id_map[v]) + ".png", "Vice Captain"))
          vice_captain_points = curr_player["total_points"].iloc[0]
        else:
          goalkeepers.append((curr_player["Name"].iloc[0], round(pred_points[var_id_map[v]],2), int(curr_player["total_points"].iloc[0]), "image_" + str(var_id_map[v]) + ".png", "Not Captain"))
        if(curr_player["mins_played"].iloc[0] == 0):
          zero_mins_players.append((1,1))
      elif(curr_player["position"].iloc[0] == 2):
        if(captain_player_vars[v].varValue == 1):
          team_points += int(curr_player["total_points"].iloc[0])
          defenders.append((curr_player["Name"].iloc[0], round(pred_points[var_id_map[v]],2)*2, int(curr_player["total_points"].iloc[0])*2, "image_" + str(var_id_map[v]) + ".png", "Captain"))
          if(curr_player["mins_played"].iloc[0] != 0):
              captain_played = True
        elif(vice_captain_player_vars[v].varValue == 1):
          defenders.append((curr_player["Name"].iloc[0], round(pred_points[var_id_map[v]],2), int(curr_player["total_points"].iloc[0]), "image_" + str(var_id_map[v]) + ".png", "Vice Captain"))
          vice_captain_points = curr_player["total_points"].iloc[0]
        else:
          defenders.append((curr_player["Name"].iloc[0], round(pred_points[var_id_map[v]],2), int(curr_player["total_points"].iloc[0]), "image_" + str(var_id_map[v]) + ".png", "Not Captain"))
        if(curr_player["mins_played"].iloc[0] == 0):
          zero_mins_players.append((2,len(defenders)-1))
      elif(curr_player["position"].iloc[0] == 3):
        if(captain_player_vars[v].varValue == 1):
          team_points += int(curr_player["total_points"].iloc[0])
          midfielders.append((curr_player["Name"].iloc[0], round(pred_points[var_id_map[v]],2)*2, int(curr_player["total_points"].iloc[0])*2, "image_" + str(var_id_map[v]) + ".png", "Captain"))
          if(curr_player["mins_played"].iloc[0] != 0):
              captain_played = True
        elif(vice_captain_player_vars[v].varValue == 1):
          midfielders.append((curr_player["Name"].iloc[0], round(pred_points[var_id_map[v]],2), int(curr_player["total_points"].iloc[0]), "image_" + str(var_id_map[v]) + ".png", "Vice Captain"))
          vice_captain_points = curr_player["total_points"].iloc[0]
        else:
          midfielders.append((curr_player["Name"].iloc[0], round(pred_points[var_id_map[v]],2), int(curr_player["total_points"].iloc[0]), "image_" + str(var_id_map[v]) + ".png", "Not Captain"))
        if(curr_player["mins_played"].iloc[0] == 0):
          zero_mins_players.append((3,len(midfielders)-1))
      else:
        if(captain_player_vars[v].varValue == 1):
          team_points += int(curr_player["total_points"].iloc[0])
          forwards.append((curr_player["Name"].iloc[0], round(pred_points[var_id_map[v]],2)*2, int(curr_player["total_points"].iloc[0])*2, "image_" + str(var_id_map[v]) + ".png", "Captain"))
          if(curr_player["mins_played"].iloc[0] != 0):
              captain_played = True
        elif(vice_captain_player_vars[v].varValue == 1):
          forwards.append((curr_player["Name"].iloc[0], round(pred_points[var_id_map[v]],2), int(curr_player["total_points"].iloc[0]), "image_" + str(var_id_map[v]) + ".png", "Vice Captain"))
          vice_captain_points = curr_player["total_points"].iloc[0]
        else:
          forwards.append((curr_player["Name"].iloc[0], round(pred_points[var_id_map[v]],2), int(curr_player["total_points"].iloc[0]), "image_" + str(var_id_map[v]) + ".png", "Not Captain"))
        if(curr_player["mins_played"].iloc[0] == 0):
          zero_mins_players.append((4,len(forwards)-1))

    # If the player is not in the starting 11 but is in the team
    elif(starting_player_vars[v].varValue == 0 and player_vars[v].varValue == 1):
      curr_player = positions[positions["fpl_id"] == var_id_map[v]][["Name", "position", "total_points", "mins_played"]]
      budget_used += cost[var_id_map[v]] if var_id_map[v] in cost else all_prices[all_prices["id"] == var_id_map[v]]["now_cost"].iloc[0]
      # Goalkeeper must be the first position in the bench
      if(curr_player["position"].iloc[0] == 1):
        gk_bench = [(curr_player["Name"].iloc[0], round(pred_points[var_id_map[v]],2), int(curr_player["total_points"].iloc[0]), "image_" + str(var_id_map[v]) + ".png", curr_player["mins_played"].iloc[0], curr_player["position"].iloc[0])]
      else:
        bench.append((curr_player["Name"].iloc[0], round(pred_points[var_id_map[v]],2), int(curr_player["total_points"].iloc[0]), "image_" + str(var_id_map[v]) + ".png", curr_player["mins_played"].iloc[0], curr_player["position"].iloc[0]))

  # If the captin does not play then the vice captain is used as the captain
  if(not captain_played):
    team_points += vice_captain_points

  all_players = [goalkeepers.copy(), defenders.copy(), midfielders.copy(), forwards.copy()]

  bench.sort(key = lambda x: x[1], reverse = True)
  bench = gk_bench + bench


  #Make team substitutions if necessary
  bench_available = [0,1,2,3]
  bench_after_swaps = bench.copy()
  for i in range(0, len(zero_mins_players)):
    if(i > 3):
      # Max 4 subs allowed
      break
    # If player being subbed is a goalkeeper than swap with goalkeeper on the bench, if they played
    if(zero_mins_players[i][0] == 1 and bench[0][4] > 0):
      player_subbed_out = all_players[zero_mins_players[i][0]-1][zero_mins_players[i][1]-1]
      all_players[0] = bench[0]
      bench_after_swaps[0] = player_subbed_out
      team_points += bench[0][2]
      bench_available.remove(0)
    # If the player being subbed is a defender
    elif(zero_mins_players[i][0] == 2):
      for j in bench_available:
        if(bench[j][4] > 0):
          if(bench[j][5] == 2):
            all_players[1][zero_mins_players[i][1]-1] = bench[j]
          elif(bench[j][5] == 3 and len(defenders) > 3):
            all_players[2].append(bench[j])
          elif(bench[j][5] == 4 and len(defenders) > 3):
            all_players[3].append(bench[j])
          else:
            continue
          player_subbed_out = all_players[zero_mins_players[i][0]-1][zero_mins_players[i][1]-1]
          bench_after_swaps[j] = player_subbed_out
          team_points += bench[j][2]
          bench_available.remove(j)
          break


    # If the player being subbed is a midfielder
    elif(zero_mins_players[i][0] == 3):
      for j in bench_available:
        if(bench[j][4] > 0):
          if(bench[j][5] == 2 and len(midfielders) > 2):
            all_players[1].append(bench[j])
          elif(bench[j][5] == 3):
            all_players[2][zero_mins_players[i][1]-1] = bench[j]
          elif(bench[j][5] == 4 and len(midfielders) > 2):
            all_players[3].append(bench[j])
          else:
            continue
          player_subbed_out = all_players[zero_mins_players[i][0]-1][zero_mins_players[i][1]-1]
          bench_after_swaps[j] = player_subbed_out
          team_points += bench[j][2]
          bench_available.remove(j)
          break

    # If the player being subbed is a forward
    elif(zero_mins_players[i][0] == 4):
      for j in bench_available:
        if(bench[j][4] > 0):
          if(bench[j][5] == 2 and len(forwards) > 1):
            all_players[1].append(bench[j])
          elif(bench[j][5] == 3 and len(forwards) > 1):
            all_players[2].append(bench[j])
          elif(bench[j][5] == 4):
            all_players[3][zero_mins_players[i][1]-1] = bench[j]
          else:
            continue
          player_subbed_out = all_players[zero_mins_players[i][0]-1][zero_mins_players[i][1]-1]
          bench_after_swaps[j] = player_subbed_out
          team_points += bench[j][2]
          bench_available.remove(j)
          break

  return round(value(prob.objective),2), team_points, goalkeepers, defenders, midfielders, forwards, bench