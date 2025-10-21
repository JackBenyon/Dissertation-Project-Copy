import asyncio
from twikit import Client
from twikit import utils
from twikit import errors
import pandas as pd
from datetime import timedelta, datetime
from sklearn.svm import SVC, LinearSVC
import numpy as np
import os 
from dotenv import find_dotenv, load_dotenv

# File which stored all the login details and open ai keys

dot_env_path = find_dotenv()
load_dotenv(dot_env_path) 

#Account 1 
USERNAME1 = os.getenv("USERNAME1")
EMAIL1 = os.getenv("EMAIL1")
PASSWORD1 = os.getenv("PASSWORD1")

# Account 3 
USERNAME2 = os.getenv("USERNAME2")
EMAIL2 = os.getenv("EMAIL2")
PASSWORD2 = os.getenv("PASSWORD2")

# Account 3
USERNAME3 = os.getenv("USERNAME3")
EMAIL3 = os.getenv("EMAIL3")
PASSWORD3 = os.getenv("PASSWORD3")

# Account 4
USERNAME4 = os.getenv("USERNAME4")
EMAIL4 = os.getenv("EMAIL4")
PASSWORD4 = os.getenv("PASSWORD4")

# Account 5
USERNAME5 = os.getenv("USERNAME5")
EMAIL5 = os.getenv("EMAIL5")
PASSWORD5 = os.getenv("PASSWORD5")

# Account 6
USERNAME6 = os.getenv("USERNAME6")
EMAIL6 = os.getenv("EMAIL6")
PASSWORD6 = os.getenv("PASSWORD6")

# Account 7
USERNAME7 = os.getenv("USERNAME7")
EMAIL7 = os.getenv("EMAIL7")
PASSWORD7 = os.getenv("PASSWORD7")

# Reading the datasets
data_21_22_23 = pd.read_csv("C:/Users/Jack Benyon/Documents/3YP/Code/flask_project/data/AllData2021_2022_2023_normalisedPos.csv", index_col=0)
dataset_23 = pd.read_csv("C:/Users/Jack Benyon/Documents/3YP/Code/flask_project/data/AllData2023_normalisedPos.csv", index_col=0)
dataset_24 = pd.read_csv("C:/Users/Jack Benyon/Documents/3YP/Code/flask_project/data/AllData2024_normalisedPos.csv", index_col=0)

# Function to get the tweets for a specific gameweek and team
async def get_posts(gw, team, account_number, client, season):
    twitter_data = pd.DataFrame(columns = ["Player", "Content", "Start Date", "End Date", "Tweet Date" , "fpl_id", "GW", "Likes", "Followers", "Comments", "Verified"])

    print("GW ", gw)

    if(season == 2024):
        data = dataset_24[(dataset_24["GW"] == gw) & (dataset_24["Team"] == team)].copy()
    elif(season == 2023):
        data = dataset_23[(dataset_23["GW"] == gw) & (dataset_23["Team"] == team)].copy()

    unique_dates = data["Date"].unique()
    
    if(data.shape[0] == 0):
        # Team has no match in the gameweek
        print("Blank GW")
    
    else:
        # If a team has multiple matches in a gameweek, we need to filter the data to only include the earliest date
        if(unique_dates.size > 1):
            converted_dates = [datetime.fromisoformat(date.split("Z")[0]) for date in unique_dates]
            earliest_date = min(converted_dates)
            earliest_date_str = earliest_date.isoformat() + "Z"
            data = data[data["Date"] == earliest_date_str]

        # For gameweek 5 onwards we need filter the data to only include players who are predicted to score over zero points. This reduces the number of posts needed to be collected.
        if(gw > 4):
            model_lin = LinearSVC(loss = 'squared_hinge', max_iter = 500)
            train_data = data_21_22_23[data_21_22_23["Team"] != team]
            train_labels = [0 if i < 1 else 1 for i in list(train_data["total_points"])]
            train = train_data.drop(columns = ["Name", "Web Name","Date", "GW", "Team", "fpl_id", "goals", "assists", "clean_sheet_kept", "saves_made", "bonus_scored", "mins_played", "total_points"])
            model_lin.fit(train, train_labels)
            test_data = data.drop(columns = ["Name", "Web Name","GW", "Date", "Team", "fpl_id", "goals", "assists", "clean_sheet_kept", "saves_made", "bonus_scored", "mins_played", "total_points"])
            pred = model_lin.predict(test_data)
            data["Predicted Zero"] = np.array(pred)
            filtered_data = data[data["Predicted Zero"] == 1]
        else:
            filtered_data = data

        print("logged in to account", account_number)

        # Iterate over all players to collect posts for
        for row in filtered_data.to_dict(orient = "records"):
            curr_date = row["Date"]
            date_format = datetime.fromisoformat(curr_date.split("Z")[0])
            end_date = str((date_format - timedelta(days = 1)).date())

            if(gw > 1):
                if(season == 2024):
                    prev_data = dataset_24[(dataset_24["GW"] == gw-1) & (dataset_24["Team"] == team)].copy()

                    if(prev_data.empty): #Accounts for previous gameweek not having a match
                        prev_data = dataset_24[(dataset_24["GW"] == gw-2) & (dataset_24["Team"] == team)].copy()
                elif(season == 2023):
                    prev_data = dataset_23[(dataset_23["GW"] == gw-1) & (dataset_23["Team"] == team)].copy()

                    if(prev_data.empty): #Accounts for previous gameweek not having a match
                        prev_data = dataset_23[(dataset_23["GW"] == gw-2) & (dataset_23["Team"] == team)].copy()

                prev_date = prev_data["Date"]

                # Posts only collected between the end of the previous gameweek and the start of the current gameweek
                prev_date_format = datetime.fromisoformat(prev_date.iloc[0].split("Z")[0])
                start_date = str((prev_date_format + timedelta(days = 1)).date())
            else:
                prev_date_format = date_format
                start_date = str((prev_date_format - timedelta(days = 7)).date())

            fpl_id = row["fpl_id"]
            player_web_name = row["Web Name"]
           
            # Some of the web names were not accurate
            if(player_web_name == "Mario Jr."):
                player_web_name = "Lemina"
            elif(player_web_name == "Matheus N."):
                player_web_name = "Nunes"
            elif(player_web_name == "J.Otto"):
                player_web_name = "Jonny"
            elif(player_web_name == "B.Traore"):
                player_web_name = "Boubacar Traore"
            elif("." in player_web_name):
                player_web_name = player_web_name.split(".")[1]

            # Search for player posts

            tweets = await client.search_tweet(f'"{player_web_name}" (#fpl) lang:en since:{start_date} until:{end_date}', product='Top', count = 20)
            for tweet in tweets:
                # Store the data from each post
                body = tweet.text
                likes = tweet.favorite_count
                comments = tweet.reply_count
                followers = tweet.user.followers_count
                verified = tweet.user.verified
                date = tweet.created_at
                twitter_data.loc[twitter_data.shape[0]] = ([player_web_name, body, start_date, end_date, date,fpl_id, gw, likes, followers, comments, verified])
            
            # Wait for 3 seconds to avoid rate limiting        
            await asyncio.sleep(3)

        # Store all the collected posts
        twitter_data.to_csv(f"C:/Users/Jack Benyon/Documents/3YP/Code/{team.lower()}_twitter_{season}-{season+1}/twitter_data_gw_{gw}_{team}_{int(str(season)[-2:])}_{int(str(season)[-2:])+1}.csv", index = False)

async def main():
    client = Client('en-UK')

    await client.login(
        auth_info_1=USERNAME6 ,
        auth_info_2=EMAIL6,
        password=PASSWORD6,
    )

    await get_posts(1, "Chelsea", 20, client, 2023)

if __name__ == "__main__":
   asyncio.run(main())

