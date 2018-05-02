# CS2803-final-project- NFL Analysis

## File Descriptions
- nfl_data.csv: The original dataset from which we gathered all the data. Came from Keggle, can be found [here](https://www.kaggle.com/maxhorowitz/nflplaybyplay2009to2016).
- DataViz.ipynb: A Jupyter Notebook containing data visualizations of run/pass trends in the NFL since 2009, made with matplotlib. As suspected, even within
this short period of time the NFL as a whole shifted ~3% toward passing.
- nfl_classifiers.py: Python module which has functions to create scikit-learn classifiers and regressors on the NFL data. The classifiers
were used to try to classify a play as either run or pass based on quarter, down, distance, time left in the game, and the point differential 
relative to the offensive team (i.e. if the offensive team is winning by 7, it would be 7.0. If they are losing by 7, it would be -7.0.)
The accuracy scores typically range from 65% to 68%, with 68% being the absolute upper bound we got. The regressors were used to 
train models to determine the win probability of a team given the time left in the game and the point differential relative to that team. 
These worked much better, typically getting around 95% to 97% accuracy. The functions return a tuple of the trained model and its average accuracy
based on test data. An example use of the function could be this: `clf, score = build_sklearn_randforest_classifier('new_run_pass.csv')`. This will
give you an sklearn RandomForestClassifier trained on data from new_run_pass.csv.
- new_run_pass.csv: This is the data used to train the classifiers to determine run or pass based on game situation. This csv was 
created from the original nfl_data.csv using Pandas DataFrame queries. 
- play_year.csv: This is the file the visualizations were created from. This csv was 
created from the original nfl_data.csv using Pandas DataFrame queries. 
- winprob.csv: This is the file the regressors were trained on to determine win probability of a team based on game situation. This csv was 
created from the original nfl_data.csv using Pandas DataFrame queries. 

