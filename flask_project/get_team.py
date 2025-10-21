import requests
from datasets import *

# Gets the user's FPL team at the end of a given gameweek

fpl_id_map = dict(zip(allData2024_normalisedPos["fpl_id"], allData2024_normalisedPos["Name"]))

def get_team(team_id, gw):
    gk, defs, mid, fwd, subs = [], [], [], [], []
    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/"
    response = requests.get(url)
    data = response.json()
    for player in data["picks"]:
        if(player["element_type"] == 1 and player["position"] < 12):
            gk.append((player["element"], fpl_id_map[player["element"]], 1))
        elif(player["element_type"] == 2 and player["position"] < 12):
            defs.append((player["element"], fpl_id_map[player["element"]], 1))
        elif(player["element_type"] == 3 and player["position"] < 12):
            mid.append((player["element"], fpl_id_map[player["element"]], 1))
        elif(player["element_type"] == 4 and player["position"] < 12):
            fwd.append((player["element"], fpl_id_map[player["element"]], 1))
        else:
            if(player["element"] in fpl_id_map):
                subs.append((player["element"], fpl_id_map[player["element"]], 0))
    curr_value = data["entry_history"]["value"]
    curr_bank = data["entry_history"]["bank"]
    return gk, defs, mid, fwd, subs, curr_value/10, curr_bank/10

    