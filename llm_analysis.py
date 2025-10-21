from openai import OpenAI
import os
import pandas as pd
from dotenv import find_dotenv, load_dotenv

# Reading the datasets containing the collected X posts

chelsea_23_24 = "C:/Users/Jack Benyon/Documents/3YP/Code/chelsea_twitter_2023-2024"
chelsea_24_25 = "C:/Users/Jack Benyon/Documents/3YP/Code/chelsea_twitter_2024-2025"

wolves_23_24 = "C:/Users/Jack Benyon/Documents/3YP/Code/wolverhampton wanderers_twitter_2023-2024"
wolves_24_25 = "C:/Users/Jack Benyon/Documents/3YP/Code/wolverhampton wanderers_twitter_2024-2025"

arsenal_23_24 = "C:/Users/Jack Benyon/Documents/3YP/Code/arsenal_twitter_2023-2024"
arsenal_24_25 = "C:/Users/Jack Benyon/Documents/3YP/Code/arsenal_twitter_2024-2025"

# This will not work since the .env file is not submiited as part of the project (it contains login details and api keys)
dot_env_path = find_dotenv()
load_dotenv(dot_env_path) 

# Note that part of the preprocess function is from the example provided on https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

def preprocess(text):
    new_text = []
    # Remove new lines in the tweet
    for t in text.split(" "):
        if("\n" in t):
            t = ".".join(t.split("\n"))
        new_text.append(t)

    # Remove links, hashtags and tokenize @ mentions
    x =  " ".join(new_text)
    new_text_2 = []
    for t in x.split(" "):

        t = '@user' if t.startswith('@') and len(t) > 1 else t
        
        if("http" in t):
            t = t.split("http")[0]

        if(not t.startswith("#")):
            new_text_2.append(t)
    return " ".join(new_text_2)

def get_embeddings(dir):

    client = OpenAI(api_key = os.getenv("GPT_API_KEY"))
    
    for file in sorted(os.listdir(dir)):

        if("llm_analysis" in file):
            continue

        all_players = pd.DataFrame(columns = ["Player", "fpl_id", "GW", "Embedding", "Embedding Reduced", "GPT Response"])

        gw = file.split("_")[3]   # Some files contain data from multiple gameweeks
        print(gw)
        curr = pd.read_csv(dir + "/" + file)
        curr = curr[curr["GW"] == int(gw)]
        unique_ids = curr["fpl_id"].unique()

        for i in range(0, len(unique_ids)):
            curr_player = curr[curr["fpl_id"] == unique_ids[i]]
            curr_player_name = curr_player.iloc[0]["Player"]
            curr_player_id = curr_player.iloc[0]["fpl_id"]

            for row in curr_player.itertuples():
                player_output = [0,0,0,"","",""]
                player_output[0] = curr_player_name
                player_output[1] = curr_player_id
                player_output[2] = gw

                text = preprocess(row[2])

                # Get the large embedding vector (1536 dimensions)
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input = text
                )
                player_output[3] = response.data[0].embedding

                # Get the small embedding vector (256 dimensions)
                response2 = client.embeddings.create(
                    model="text-embedding-3-small",
                    input = text,
                    dimensions= 256
                )

                player_output[4] = response2.data[0].embedding

                # Using the gpt-4o-mini model to categorise the tweets'
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": f"You will be provided with a tweet about {player_output[0]}, a football player. I want you to explain to me if you think he is currently injured (A), in bad form (B), if he is playing well (C) or if you are unsure (D). Tweets may contain information related to fantasy football, such as a player being transfered in or out of someone's team."},
                        {
                            "role": "user",
                            "content": text
                        }
                    ]
                )

                player_output[5] = completion.choices[0].message

                all_players = pd.concat([pd.DataFrame([player_output], columns = all_players.columns), all_players], ignore_index = True)  

        # Store the analysis
        all_players.to_csv(dir + f"/llm_analysis/llm_analysis_gw_{gw}.csv")

