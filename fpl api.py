import requests, json
from pprint import pprint
base_url = 'https://fantasy.premierleague.com/api/'
r = requests.get(base_url+'bootstrap-static/').json()
pprint(r,indent=2,depth=1, compact=True)
output = open("output.txt","w")
json.dump(r, output)