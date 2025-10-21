from flask import Flask, render_template, request, redirect, url_for, session
from models import *
from choose_team import optimal_team
from get_team import get_team
from transfer_planner import transfer_planner
import os


# This code defines the structure of the web app and the function that are called when each button is pressed


app = Flask(__name__)
app.secret_key = os.urandom(24)
@app.route('/', methods = ["GET","POST"])
def index(): 
    return render_template('home.html')

@app.route('/transfer', methods = ["GET","POST"])
def transfer():

    session.setdefault("fpl_id", None)
    session.setdefault("gw", None)

    if 'output' not in session:
        session['output'] = ''
    if 'output2' not in session:
        session['output2'] = ''

    if request.method == 'POST':
        if(request.form.get("button") == "Import Team"):
            form = request.form
            session['fpl_id'] = int(form["FPL ID"])
            session['gw'] = int(form["GW"])
            team = get_team(int(form["FPL ID"]), int(form["GW"]))
            session['output'] = team

        elif(request.form.get("button") == "Generate Transfer Recommendations"):
            form2 = request.form
            transfers = transfer_planner(session.get("fpl_id"), session.get("gw"), int(form2["Lookahead period"]), int(form2["Number of free transfers"]))
            session['output2'] = transfers
        
    return render_template('transfer.html', output = session.get("output"), output2 = session.get("output2"))


@app.route('/team_intro', methods = ["GET","POST"])
def team_intro():
    output = ""
    if request.method == 'POST':
        form = request.form
        output = generate_team(form)
        return redirect(url_for('team', form = form, output = output))
    return render_template('base.html', output = output)

@app.route('/team', methods = ["GET","POST"])
def team():
    output = ""
    if(request.method == 'POST'):
        form = request.form
        output = generate_team(form)
    
    elif(request.method == 'GET'):
        form = request.args.get('form')
        output = request.args.getlist('output')
        for i in range(0, len(output)):
            output[i] = eval(output[i])
    
    return render_template('new_index.html', output = output)

def generate_team(form):

    predictions = gw_predict(int(form["gw"]), allData2021_2022_2023_normalisedPos, allData2024_normalisedPos, False, None, None)
    opt = optimal_team(int(form["gw"]),predictions, int(form["bud"]))
    weekly_average = [57, 69, 64, 51, 58, 50, 46, 36, 54, 39, 49, 49, 60, 58, 50, 47, 60, 51, 66, 60, 55, 46, 59, 87, 74, 62, 53, 52, 40, 44]

    predicted_points = opt[1]
    total_points = opt[0]
    goalkeepers, defenders, midfielders, forwards, bench = opt[2], opt[3], opt[4], opt[5], opt[6]

    average_points = weekly_average[int(form["gw"])-1]
        
    return goalkeepers, defenders, midfielders, forwards, round(predicted_points,2), total_points, average_points, bench
