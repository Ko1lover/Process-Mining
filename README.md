# Predictive Process Mining
This entire project is based on the process mining of insights with regard to the bank application procedures. 

For this project our main task was at first to predict the next event ('concept:name'), then we had to predict the time when the predicted next event will occur. Lastly we had to a bit improve previous solutions and combine them into the whole trace prediction (both time an events provided some amount of lags and other information)

## How to run the code
In order to run all of the files in the repository please follow first 3 steps, the rest are descriptions of what you can do with the data. Use the requirements.txt in order to install all the necessary libraries::
1. Download the data from any of these links and use it in the preprocessing.ipynb file (for this project we used 2012 data, however 2017 and 2018 will work just fine as well):
    - BPI Challenge 2012: [can be found here](https://data.4tu.nl/articles/_/12689204/1)
    - BPI Challenge 2017: [can be found here](https://data.4tu.nl/articles/_/12696884/1)
    - BPI Challenge 2018: [can be found here](https://data.4tu.nl/articles/_/12688355/1)

2. Make sure you have the Split_function.py file since this is the one which contains proper train_test_split 
solely made for this process mining project (gets rid of overlapping events in train and test)

3. Run the general_preprocessing.ipynb with the data you had downloaded. This file would do some basic data cleaning such that you wouldn't put too much data in your memory

4. Now if you want a quick way to see the way our models run and predict the whole trace you can run the 'Demo_version.ipynb' file which takes on average ~70 seconds. As well as grpahics to showcase the performance of each of the models (In demo version Random Forest with hyperopt to optimize the hyperparameters for models is used for both)
    - If you want to compare XGBoost and Random Forest in the event prediction or the time prediction or both you would first have to run Event_forecasting.ipynb, then Time_forecasting.ipynb these 2 files would create and save 20 models per file and only then the trace_prediciton.ipynb file to check performances of these models on the chosen data.
    <i>After that at the end of the trace_prediciton.ipynb file you would be able to see 2 graphics</i>
    - First one tells you about the accuracy of prediciton of the whole trace
    - Seconds one tells you about the mean absolute errors of prediciton of the time for each event in the whole trace in hours

5. If you would want to see what insights we had found from the data you would need to go to Data_exploration.ipynb or the Visualizaiton.ipynb.
- Data exploration file shows interesting insights found about the data
-  Visualization shows information regarding how does the train test split looks like with sensitive data (and how would it look in the customly made nested k-fold cross validation for the time sensitive data)


Hope you have a great day, and if you have any suggestions regarding our project feel free to send an email to anseka.@gmail.com with your suggestions, which we would implement in the future. Because I want to finalize this project aside of the course, and will have to think of some ideas in order to make it better.
