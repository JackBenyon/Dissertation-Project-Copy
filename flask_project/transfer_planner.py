from pulp import * 
from models import *
from get_team import get_team
from datasets import *

# Creates the data for the gameweeks that are to generate the transfer recommendations for, using the data from the most recent gameweek and fixture specific data for each game

def create_data(start, num_weeks, train, test):
    all_points_predictions = pd.DataFrame(columns=["fpl_id", "Points Prediction", "GW"])
    prev_gw_stats = test[test["GW"] == start]
    for i in range(0, num_weeks):
        x = test[test["GW"] == start+i][["fpl_id", "GW", "was_home", "win_prob", "draw_prob"]]
        for player in x.itertuples():
            if(player[1] not in prev_gw_stats["fpl_id"].values):
                x = x.drop(player[0])
        x.index = prev_gw_stats.index
        x.drop(columns =  ["fpl_id"], inplace = True)
        prev_gw_stats.loc[:,["GW", "was_home", "win_prob", "draw_prob"]] = x
        points_predictions = gw_predict(start+i, train, prev_gw_stats, False, None, None)
        points_predictions_df = pd.DataFrame(points_predictions.items(), columns = ["fpl_id", "Points Prediction"])
        points_predictions_df["GW"] = start+i
        if(all_points_predictions.empty):
            all_points_predictions = points_predictions_df
        else:
            all_points_predictions = pd.concat([all_points_predictions, points_predictions_df])

    return all_points_predictions

# Predictions start in gameweek 'start' and are made for 'num_weeks' -1 weeks after   
# Note that this function only works for num_weeks  = 1
def transfer_planner(fpl_id, start, num_weeks, curr_free_transfers_p):

    curr_team = get_team(fpl_id,start)
    player_ids = []
    starting_player_ids = []
    for pos in curr_team:
        if(type(pos) == list):
            for player in pos:
                player_ids.append(player[0])
                if(player[2] == 1):
                    starting_player_ids.append(player[0])

    curr_bank_p = curr_team[6]
    curr_team_p = player_ids
    curr_starting_team_p = starting_player_ids

    points_predictions = create_data(start, num_weeks, allData2021_2022_2023_normalisedPos, allData2024_normalisedPos)

    curr_prices = pd.read_csv(f"data/prices/prices_{start}.csv", index_col = 0)[["id","now_cost", "photo"]]
    all_prices = pd.read_csv(f"data/prices/all_prices.csv", index_col = 0).sort_values(by = ["gw"])
    positions = allData2024_normalisedPos[allData2024_normalisedPos["GW"].isin([i for i in range(start, start+num_weeks)])][["Name", "fpl_id", "position", "Team", "total_points", "mins_played"]]
    
    n = points_predictions[points_predictions["GW"] == start].shape[0]

    #Dictionary mapping player id to their cost in the current gameweek
    cost = dict(zip(curr_prices["id"], curr_prices["now_cost"]))
    #Dictionary mapping player id to their position
    pos = dict(zip(positions["fpl_id"], positions["position"]))
    #Dictionary mapping player id to their club
    club = dict(zip(positions["fpl_id"], positions["Team"]))
    all_clubs = ["Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Chelsea","Crystal Palace", "Everton", "Fulham", "Ipswich", "Leicester", "Liverpool", "Manchester City", "Manchester United", "Newcastle United", "Nottingham Forest", "Southampton", "Tottenham", "West Ham", "Wolverhampton Wanderers"]
    prob = LpProblem("FPL_Team", LpMaximize)

    # Dictionary mapping each variable to the corresponding player id and vice versa
    var_id_map = dict(zip([i for i in range(n)], points_predictions[points_predictions["GW"] == start]["fpl_id"]))
    id_var_map = dict(zip(points_predictions[points_predictions["GW"] == start]["fpl_id"], [i for i in range(n)]))

    # Binary array representing the current team of players and the current starting 11
    curr_team = np.zeros((num_weeks, n))
    curr_starting_team = np.zeros((num_weeks,n))
    curr_bank = np.zeros(num_weeks)
    curr_bank[0] = curr_bank_p

    for i in range(0, 15):
        curr_team[0][id_var_map[curr_team_p[i]]] = 1
    for i in range(0, 11):
        curr_starting_team[0][id_var_map[curr_starting_team_p[i]]] = 1

    # Variables representing if a player has been transfered out or not

    free_transfer_in_vars = [[pulp.LpVariable(f"t{str(i)}_{str(j)}", cat='Binary') for i in range(n)] for j in range(num_weeks)]
    paid_transfer_in_vars = [[pulp.LpVariable(f"pt{str(i)}_{str(j)}", cat='Binary') for i in range(n)] for j in range(num_weeks)]
    transfer_out_vars = [[pulp.LpVariable(f"to{str(i)}_{str(j)}", cat='Binary') for i in range(n)] for j in range(num_weeks)]

    # Store the number of free transfers based on the number of transfers made in the previous week
    curr_free_transfers = [curr_free_transfers_p]
    for i in range(1, num_weeks):
        curr_free_transfers.append(curr_free_transfers[i-1] + 1 - sum(free_transfer_in_vars[i-1]))

    # Update the team for the next week based on the transfers made
    next_week_team = [curr_team[0].copy()]
    for i in range(1, num_weeks+1):
        next_week_team.append([
            (next_week_team[i-1][j] + lpSum(free_transfer_in_vars[i-1][j] + paid_transfer_in_vars[i-1][j] - transfer_out_vars[i-1][j]))
            for j in range(n)
        ])

    # Update the budget for the next week based on the transfers made
    curr_bank = [curr_bank_p]
    for i in range(1, num_weeks):
        total_transfer_cost = sum(
        ((free_transfer_in_vars[i-1][j] + paid_transfer_in_vars[i-1][j] - transfer_out_vars[i-1][j]) * 
        (cost[var_id_map[j]] if var_id_map[j] in cost else all_prices[all_prices["id"] == var_id_map[j]]["now_cost"].iloc[0]))
        for j in range(n)
        )
        curr_bank.append(curr_bank[i-1] - total_transfer_cost)

    # Binary variables representing the players in the starting 11 for the next week
    next_week_start = [[pulp.LpVariable(f"s{str(i)}_{str(j)}", cat='Binary') for j in range(n)] for i in range(num_weeks)]

    # Positional constraints for each week
    for i in range(num_weeks):
        prob += sum(next_week_team[i+1][j] for j in range(n) if pos[var_id_map[j]] == 1) == 2
        prob += sum(next_week_team[i+1][j] for j in range(n) if pos[var_id_map[j]] == 2) == 5
        prob += sum(next_week_team[i+1][j] for j in range(n) if pos[var_id_map[j]] == 3) == 5
        prob += sum(next_week_team[i+1][j] for j in range(n) if pos[var_id_map[j]] == 4) == 3

        prob += lpSum(next_week_start[i][j] for j in range(n) if pos[var_id_map[j]] == 1) == 1
        prob += lpSum(next_week_start[i][j] for j in range(n) if pos[var_id_map[j]] == 2) >= 3
        prob += lpSum(next_week_start[i][j] for j in range(n) if pos[var_id_map[j]] == 2) <= 5
        prob += lpSum(next_week_start[i][j] for j in range(n) if pos[var_id_map[j]] == 3) >= 2
        prob += lpSum(next_week_start[i][j] for j in range(n) if pos[var_id_map[j]] == 3) <= 5
        prob += lpSum(next_week_start[i][j] for j in range(n) if pos[var_id_map[j]] == 4) >= 1
        prob += lpSum(next_week_start[i][j] for j in range(n) if pos[var_id_map[j]] == 4) <= 3

    # Team constraints for each week
    for i in range(0, 20):
        prob += sum(next_week_team[j+1][k] for k in range(0,n) for j in range(num_weeks) if club[var_id_map[k]] == all_clubs[i]) <= 3

    # Captain and vice-captain constraints for each week
    captain_player_vars = [[pulp.LpVariable(f"z{str(i)}_{str(j)}", cat = 'Binary') for j in range(n)] for i in range(num_weeks)]
    vice_captain_player_vars = [[pulp.LpVariable(f"w{str(i)}_{str(j)}", cat = 'Binary') for j in range(n)] for i in range(num_weeks)]

    for i in range(num_weeks):
        # One captain 
        prob += lpSum(captain_player_vars[i]) == 1
        # One vice-captain
        prob += lpSum(vice_captain_player_vars[i]) == 1
        # Number of free transfers used is at most the number of free transfers available
        prob += lpSum(free_transfer_in_vars[i]) <= curr_free_transfers[i]
        # 15 player must be chosen in the team
        prob += sum(next_week_team[i+1]) == 15
        # 11 players must be chosen in the starting 11
        prob += lpSum(next_week_start[i]) == 11

    for i in range(0, num_weeks):
        for j in range(n):
            prob += next_week_team[i+1][j] <= 1 
            prob += next_week_team[i+1][j] >= 0
            # Each starting player must be in the team
            prob += next_week_team[i+1][j] - next_week_start[i][j] >= 0
            # A player cant be transferred in and out at the same time
            prob += free_transfer_in_vars[i][j] + transfer_out_vars[i][j] + paid_transfer_in_vars[i][j] <= 1
            # Captain must be in the starting 11
            prob += next_week_start[i][j] - captain_player_vars[i][j] >= 0
            # Vice-captain must be in the starting 11
            prob += next_week_start[i][j] - vice_captain_player_vars[i][j] >= 0
            # A player can only be captain or vice-captain, not both
            prob += captain_player_vars[i][j] + vice_captain_player_vars[i][j] <= 1.5

    for i in range(1, num_weeks):
        for j in range(0, n):
            # A player cannot be transfered out and chosen in the next weeks team
            prob += transfer_out_vars[i][j] + next_week_team[i][j] <= 1.5
            prob += next_week_team[i][j] + (free_transfer_in_vars[i][j] + paid_transfer_in_vars[i][j]) <= 1

    next_gw = [points_predictions[points_predictions["GW"] == i] for i in range(start, start+num_weeks)]
    next_gw_predictions = [dict(zip(entry["fpl_id"], entry["Points Prediction"])) for entry in next_gw]

    # Can only transfer in a player if they are affordable
    total_transfer_in_cost = [sum((paid_transfer_in_vars[i][j] + free_transfer_in_vars[i][j]) * cost[var_id_map[j]] if var_id_map[j] in cost else all_prices[all_prices["id"] == var_id_map[j]]["now_cost"].iloc[0] * (paid_transfer_in_vars[i][j] + free_transfer_in_vars[i][j]) for j in range(n)) for i in range(num_weeks)]
    total_transfer_out_cost = [sum(transfer_out_vars[i][j] * cost[var_id_map[j]] if var_id_map[j] in cost else all_prices[all_prices["id"] == var_id_map[j]]["now_cost"].iloc[0] * transfer_out_vars[i][j] for j in range(n)) for i in range(num_weeks)]

    # Total transfer costs must be subject to budget constraints
    for i in range(0, num_weeks):
        prob += total_transfer_out_cost[i] + curr_bank[i] - total_transfer_in_cost[i] >= 0

    # Objective function to maximize the expected points gained from the transfers made
    prob += lpSum((((0.99 * next_week_start[i][j] + captain_player_vars[i][j] + 0.001 * vice_captain_player_vars[i][j] + 0.01 * next_week_team[i+1][j]) * next_gw_predictions[i][var_id_map[j]]) - 4 * paid_transfer_in_vars[i][j]) for i in range(num_weeks) for j in range(n))
    
    prob.writeLP("FPL_Team.lp")
    prob.solve(PULP_CBC_CMD(msg=0))

    transfers_in = [[] for i in range(num_weeks)]
    transfer_out = [[] for i in range(num_weeks)]

    sub_in = []
    sub_out = [] 

    # Output the results of the optimization : player transfers, captain and vice-captain choices, expected points gained, and the final team

    for i in range(0, num_weeks):
        expected_gain = 0
        print(f"\n---Gameweek {start+i+1}---")
        print("Budget :",  curr_bank[i].value() if type(curr_bank[i]) != float else curr_bank[i])
        print("-----Transfers In-----")
        for j in range(0, n):
            if((free_transfer_in_vars[i][j].varValue) == 1):
                print(positions[positions["fpl_id"] == var_id_map[j]]["Name"].iloc[0] + " : " + str(round(next_gw_predictions[i][var_id_map[j]],2)))
                transfers_in[i].append(positions[positions["fpl_id"] == var_id_map[j]]["Name"].iloc[0])
                if(next_week_start[i][j].varValue == 1):
                    expected_gain += next_gw_predictions[i][var_id_map[j]]
            elif((paid_transfer_in_vars[i][j].varValue) == 1):
                print(positions[positions["fpl_id"] == var_id_map[j]]["Name"].iloc[0] + " : " + str(round(next_gw_predictions[i][var_id_map[j]],2)))
                transfers_in[i].append(positions[positions["fpl_id"] == var_id_map[j]]["Name"].iloc[0])
                if(next_week_start[i][j].varValue == 1):
                    expected_gain += next_gw_predictions[i][var_id_map[j]] - 4

        print("\n-----Transfers Out-----")
        for j in range(0, n):
            if((transfer_out_vars[i][j].varValue) == 1 and curr_starting_team[i][j] == 1):
                print(positions[positions["fpl_id"] == var_id_map[j]]["Name"].iloc[0] + " : " + str(round(next_gw_predictions[i][var_id_map[j]],2)))
                transfer_out[i].append(positions[positions["fpl_id"] == var_id_map[j]]["Name"].iloc[0])
                expected_gain -= next_gw_predictions[i][var_id_map[j]]

        swapped_out, swapped_in = [], []
        for j in range(0,n):
            if(curr_starting_team[i][j] == 1 and next_week_start[i][j].varValue == 0 and transfer_out_vars[i][j].varValue == 0 and i==0):
                swapped_out.append(positions[positions["fpl_id"] == var_id_map[j]]["Name"].iloc[0] + " : " + str(round(next_gw_predictions[i][var_id_map[j]],2)))
                expected_gain -= next_gw_predictions[i][var_id_map[j]]
            elif(curr_starting_team[i][j] == 0 and next_week_start[i][j].varValue == 1 and free_transfer_in_vars[i][j].varValue == 0 and paid_transfer_in_vars[i][j].varValue == 0 and i==0):
                swapped_in.append(positions[positions["fpl_id"] == var_id_map[j]]["Name"].iloc[0] + " : " + str(round(next_gw_predictions[i][var_id_map[j]],2)))
                expected_gain += next_gw_predictions[i][var_id_map[j]]
            elif(next_week_start[i-1][j].varValue == 1 and next_week_start[i][j].varValue == 0 and transfer_out_vars[i][j].varValue == 0):
                swapped_out.append(positions[positions["fpl_id"] == var_id_map[j]]["Name"].iloc[0] + " : " + str(round(next_gw_predictions[i][var_id_map[j]],2)))
                expected_gain -= next_gw_predictions[i][var_id_map[j]]
            elif(next_week_start[i-1][j].varValue == 0 and next_week_start[i][j].varValue == 1 and free_transfer_in_vars[i][j].varValue == 0 and paid_transfer_in_vars[i][j].varValue == 0):
                swapped_in.append(positions[positions["fpl_id"] == var_id_map[j]]["Name"].iloc[0] + " : " + str(round(next_gw_predictions[i][var_id_map[j]],2)))
                expected_gain += next_gw_predictions[i][var_id_map[j]]
        
        sub_in.append(swapped_in)
        sub_out.append(swapped_out)

        print("\n-----Swapped Out-----")
        for player in swapped_out:
            print(player)
        print("\n-----Swapped In-----")
        for player in swapped_in:
            print(player)

        print(f"\n-----Expected Points Gain-----\n{round((expected_gain),2)}")

        print(f"\n----Captain-----")
        for j in range(0, n):
            if((captain_player_vars[i][j].varValue) == 1):
                print(positions[positions["fpl_id"] == var_id_map[j]]["Name"].iloc[0] + " : " + str(round(next_gw_predictions[i][var_id_map[j]] * 2,2)))

        print(f"\n----Vice Captain-----")
        for j in range(0, n):
            if((vice_captain_player_vars[i][j].varValue) == 1):
                print(positions[positions["fpl_id"] == var_id_map[j]]["Name"].iloc[0] + " : " + str(round(next_gw_predictions[i][var_id_map[j]] ,2)))

        goalkeepers, defenders, midfielders, forwards, bench = [], [], [], [], []
        team_points = 0
        budget_used = 0
        gk_bench = []
        zero_mins_players = []
        for v in range(0,n):
            if(next_week_start[i][v].varValue == 1):
                curr_player = positions[positions["fpl_id"] == var_id_map[v]][["Name", "position", "total_points", "mins_played"]]
                team_points += int(curr_player["total_points"].iloc[0])
                budget_used += cost[var_id_map[v]] if var_id_map[v] in cost else all_prices[all_prices["id"] == var_id_map[v]]["now_cost"].iloc[0]

                if(curr_player["position"].iloc[0] == 1):
                    if(captain_player_vars[i][v].varValue == 1):
                        team_points += int(curr_player["total_points"].iloc[0])
                        goalkeepers.append((curr_player["Name"].iloc[i], round(next_gw_predictions[i][var_id_map[v]],2)*2, int(curr_player["total_points"].iloc[i])*2, "image_" + str(var_id_map[v]) + ".png", "Captain"))
                    else:
                        goalkeepers.append((curr_player["Name"].iloc[i], round(next_gw_predictions[i][var_id_map[v]],2), int(curr_player["total_points"].iloc[i]), "image_" + str(var_id_map[v]) + ".png", "Not Captain"))
                    if(curr_player["mins_played"].iloc[0] == 0):
                        zero_mins_players.append((1,1))
                elif(curr_player["position"].iloc[0] == 2):
                    if(captain_player_vars[i][v].varValue == 1):
                        team_points += int(curr_player["total_points"].iloc[0])
                        defenders.append((curr_player["Name"].iloc[i], round(next_gw_predictions[i][var_id_map[v]],2)*2, int(curr_player["total_points"].iloc[i])*2, "image_" + str(var_id_map[v]) + ".png", "Captain"))
                    else:
                        defenders.append((curr_player["Name"].iloc[i], round(next_gw_predictions[i][var_id_map[v]],2), int(curr_player["total_points"].iloc[i]), "image_" + str(var_id_map[v]) + ".png", "Not Captain"))
                    if(curr_player["mins_played"].iloc[0] == 0):
                        zero_mins_players.append((2,len(defenders)-1))
                elif(curr_player["position"].iloc[0] == 3):
                    if(captain_player_vars[i][v].varValue == 1):
                        team_points += int(curr_player["total_points"].iloc[0])
                        midfielders.append((curr_player["Name"].iloc[i], round(next_gw_predictions[i][var_id_map[v]],2)*2, int(curr_player["total_points"].iloc[i])*2, "image_" + str(var_id_map[v]) + ".png", "Captain"))
                    else:
                        midfielders.append((curr_player["Name"].iloc[i], round(next_gw_predictions[i][var_id_map[v]],2), int(curr_player["total_points"].iloc[i]), "image_" + str(var_id_map[v]) + ".png", "Not Captain"))
                    if(curr_player["mins_played"].iloc[0] == 0):
                        zero_mins_players.append((3,len(midfielders)-1))
                else:
                    if(captain_player_vars[i][v].varValue == 1):
                        team_points += int(curr_player["total_points"].iloc[0])
                        forwards.append((curr_player["Name"].iloc[i], round(next_gw_predictions[i][var_id_map[v]],2)*2, int(curr_player["total_points"].iloc[i])*2, "image_" + str(var_id_map[v]) + ".png", "Captain"))
                    else:
                        forwards.append((curr_player["Name"].iloc[i], round(next_gw_predictions[i][var_id_map[v]],2), int(curr_player["total_points"].iloc[i]), "image_" + str(var_id_map[v]) + ".png", "Not Captain"))
                    if(curr_player["mins_played"].iloc[0] == 0):
                        zero_mins_players.append((4,len(defenders)-1))

            elif(next_week_team[i+1][v].value() == 1 and next_week_start[i][v].varValue == 0):
                curr_player = positions[positions["fpl_id"] == var_id_map[v]][["Name", "position", "total_points", "mins_played"]]
                budget_used += cost[var_id_map[v]] if var_id_map[v] in cost else all_prices[all_prices["id"] == var_id_map[v]]["now_cost"].iloc[0]
                # Goalkeeper must be the first position in the bench
                if(curr_player["position"].iloc[0] == 1):
                    gk_bench = [(curr_player["Name"].iloc[i], round(next_gw_predictions[i][var_id_map[v]],2), int(curr_player["total_points"].iloc[i]), "image_" + str(var_id_map[v]) + ".png", curr_player["mins_played"].iloc[0], curr_player["position"].iloc[0])] 
                else:
                    bench.append((curr_player["Name"].iloc[i], round(next_gw_predictions[i][var_id_map[v]],2), int(curr_player["total_points"].iloc[i]), "image_" + str(var_id_map[v]) + ".png", curr_player["mins_played"].iloc[0], curr_player["position"].iloc[0]))

        bench.sort(key = lambda x: x[1], reverse = True)
        bench = gk_bench + bench

        print(f"\n-----Starting 11 for GW {start+i+1} -----")
        
        print([gk[0] for gk in goalkeepers])
        print([defs[0] for defs in defenders])
        print([mid[0] for mid in midfielders])
        print([fw[0] for fw in forwards])
        print("\n-----Bench-----")
        print([bk[0] for bk in bench])    

    return start, transfers_in, transfer_out, sub_in, sub_out, round(expected_gain,2)


transfer_planner(34283, 20, 1, 1)