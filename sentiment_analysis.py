from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from dotenv import find_dotenv, load_dotenv
import os
from scipy.special import softmax
import numpy as np
import pandas as pd 

# This will not work since the .env file is not submiited as part of the project (it contains login details and api keys)
dot_env_path = find_dotenv()
load_dotenv(dot_env_path) 

# Read required datasets

dataset_23 = pd.read_csv("C:/Users/Jack Benyon/Documents/3YP/Code/flask_project/data/AllData2023_normalisedPos.csv", index_col=0)
dataset_24 = pd.read_csv("C:/Users/Jack Benyon/Documents/3YP/Code/flask_project/data/AllData2024_normalisedPos.csv", index_col=0)

chelsea_23_24 = "C:/Users/Jack Benyon/Documents/3YP/Code/chelsea_twitter_2023-2024_w_sentiment"
chelsea_24_25 = "C:/Users/Jack Benyon/Documents/3YP/Code/chelsea_twitter_2024-2025_w_sentiment"

wolves_23_24 = "C:/Users/Jack Benyon/Documents/3YP/Code/wolves_twitter_2023-2024_w_sentiment"
wolves_24_25 = "C:/Users/Jack Benyon/Documents/3YP/Code/wolves_twitter_2024-2025_w_sentiment"

arsenal_23_24 = "C:/Users/Jack Benyon/Documents/3YP/Code/arsenal_twitter_2023-2024_w_sentiment"
arsenal_24_25 = "C:/Users/Jack Benyon/Documents/3YP/Code/arsenal_twitter_2024-2025_w_sentiment"


# Perform sentiment analysis on the player posts that were collected
# Note that part of this code is from the example provided on https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest


def sentiment_analysis(dir):

    # Preprocess text for @ mentions and links in the tweets
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

    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    # Tokenizer class using the pretrained vocabulary
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # Configuration class 
    config = AutoConfig.from_pretrained(MODEL)
    # Imoport the roberta model  
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    all_sentiment = pd.DataFrame(columns = ["Player","fpl_id", "GW", "Likes", "Followers", "Comments", "Verified", "Positive", "Neutral", "Negative", "Positive Sum", "Neutral Sum", "Negative Sum", "Positive Mean", "Neutral Mean", "Negative Mean", "Positive Majority", "Neutral Majority", "Negative Majority"])

    for file in os.listdir(dir):
    
        gw = file.split("_")[3]
        curr = pd.read_csv(dir + "/" + file)
        curr = curr[curr["GW"] == int(gw)]

        unique_ids = curr["fpl_id"].unique()

        for i in range(0, len(unique_ids)):
            curr_player_data = ["Name", "ID", "GW", [], [], [], [], [], [] ,[], 0,0,0,0,0,0,0,0,0]
            curr_player = curr[curr["fpl_id"] == unique_ids[i]]
            x = curr_player.iloc[0]

            curr_player_data[0] = x["Player"]
            curr_player_data[1] = x["fpl_id"]
            curr_player_data[2] = x["GW"]

            for row in curr_player.itertuples():
                curr_player_data[3].append(row[8])
                curr_player_data[4].append(row[9])
                curr_player_data[5].append(row[10])
                curr_player_data[6].append(row[11])

                # Preprocess the text to tokensize any @ mentions and links
                text = row[2]
                text = preprocess(text)
                # Tokenize the rest of the text
                encoded_input = tokenizer(text, return_tensors='pt')
                # Perform the sentiment analysis
                output = model(**encoded_input)
                # Get the sentiment scores
                scores = output[0][0].detach().numpy()
                # Convert scores to probabilities
                scores = softmax(scores)

                # Rank the sentiments by most to least likely
                ranking = np.argsort(scores)
                ranking = ranking[::-1]
                highest_sentiment = config.id2label[ranking[0]]
                
                # Store the most likely sentiment score

                if(highest_sentiment == "positive"):
                    curr_player_data[16] += 1
                elif(highest_sentiment == "neutral"):
                    curr_player_data[17] += 1
                elif(highest_sentiment == "negative"):
                    curr_player_data[18] += 1
                for i in range(scores.shape[0]):
                    s = scores[ranking[i]]
                    curr_player_data[7+i].append(s)

            curr_player_data[3] = sum(curr_player_data[3])
            curr_player_data[4] = sum(curr_player_data[4])
            curr_player_data[5] = sum(curr_player_data[5])
            curr_player_data[10] = sum(curr_player_data[7])
            curr_player_data[11] = sum(curr_player_data[8])
            curr_player_data[12] = sum(curr_player_data[9])
            curr_player_data[13] = np.mean(curr_player_data[7])
            curr_player_data[14] = np.mean(curr_player_data[8])
            curr_player_data[15] = np.mean(curr_player_data[9])

            all_sentiment = pd.concat([pd.DataFrame([curr_player_data], columns = all_sentiment.columns), all_sentiment], ignore_index=True)
    all_sentiment.to_csv(dir + "/sentiment.csv")


# Functions to create the sentiment datasets

def add_sentiment(dir):
    old_sentiment = pd.read_csv(dir + "/old/sentiment.csv")
    new_sentiment = pd.read_csv(dir + "/sentiment_2.csv")
    sentiment = pd.concat([old_sentiment, new_sentiment], axis = 0, ignore_index=True)
    sentiment.to_csv(dir + "/sentiment.csv")

def merge_sentiment(dir, data, sentiment_data):
    merged_all = data.merge(sentiment_data.drop(columns=["Player", "Verified"]), on = ["fpl_id", "GW"], how = "left")
    merged_no_nan = data.merge(sentiment_data.drop(columns=["Player", "Verified"]), on = ["fpl_id", "GW"], how = "inner")
    merged_all = merged_all.fillna(0, axis = 1)
    merged_all.to_csv(dir + "/merged_sentiment_all.csv")
    merged_no_nan.to_csv(dir + "/merged_sentiment_no_nan.csv")

# merge_sentiment(chelsea_23_24, dataset_23[dataset_23["Team"] == "Chelsea"], pd.read_csv(chelsea_23_24 + "/sentiment.csv"))
# merge_sentiment(chelsea_24_25, dataset_24[dataset_24["Team"] == "Chelsea"], pd.read_csv(chelsea_24_25 + "/sentiment.csv"))
# merge_sentiment(wolves_23_24, dataset_23[dataset_23["Team"] == "Wolverhampton Wanderers"], pd.read_csv(wolves_23_24 + "/sentiment.csv"))
# merge_sentiment(wolves_24_25, dataset_24[dataset_24["Team"] == "Wolverhampton Wanderers"], pd.read_csv(wolves_24_25 + "/sentiment.csv"))
# merge_sentiment(arsenal_23_24, dataset_23[dataset_23["Team"] == "Arsenal"], pd.read_csv(arsenal_23_24 + "/sentiment.csv"))
# merge_sentiment(arsenal_24_25, dataset_24[dataset_24["Team"] == "Arsenal"], pd.read_csv(arsenal_24_25 + "/sentiment.csv"))


