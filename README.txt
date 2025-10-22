## Code for my third year dissertation project : An Evaluation of Machine Learning Techniques and Social Media Analysis to Predict an Optimal Fantasy Football Team

All the datasets created during the project have been uploaded as part of the submission in the "datasets" folder. 

The "Poject_Code_FINAL" jupyter notebook contains all the code that was used to create datasets and develop and test the machine learning models

Some additional python files are included which show the code for getting posts from X, performing the X posts analysis methods (sentiment analysis, vector embeddings, LLM categorisation) and the code for recursive feature elimination

Note that the directories of the any datasets which are read using pd.read_csv() function will need to be changed to whatever the local directory of the downloaded dataset is 

Instructions for running the web app :

Change the directories of all of the datasets in the datasets.py file to the local directories of the data in the flask_project/data directory 
Navigate to the flask_project directory and run the command : flask --app web_app run
Click on the link to the web app (it will look something like http://127.0.0.1:5000 )

To test the transfer recommendation feature, the fpl ID of 34283 can be used. "Gameweek" can be any value from 1 to 30. Lookahead period can only take value one. Number of free transfers can be any integer between 1 and 5. 

To test the optimal team generator feature, any gameweek from 1 to 30 can be used and budgets ranging from 70 to 110.
