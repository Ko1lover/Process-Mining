This entire project is based on the Process Mining of business events. 

Our goal in this project was to predict at first the next event, then the time till the next event, 
and eventually combine that into the whole suffix prediction

In order to properly run all of the files you will need to follow this guidline:
1. Download the data from any of these links and use it in the preprocessing.ipynb file:
  - Least amount of data: https://data.4tu.nl/articles/_/12689204/1
  - Modest amount of data: https://data.4tu.nl/articles/_/12696884/1
  - Largest amount of data: https://data.4tu.nl/articles/_/12688355/1

2. Make sure you have the Split_function.py file since this is the one which contains proper train_test_split 
solely made for this process mining project (gets rid of overlapping events in train and test)

3. Run the general_preprocessing.ipynb with the data you had downloaded. This file would do some basic data cleaning such that you wouldn't put too much data in your memory

4. Now you can run the Demo_version.ipynb file to look how the trace prediction and time forecasting for each event in the trace had been done. As well as grpahics to showcase the performance of each of the models (In demo version Random Forest with hyperopt is used for both)
4.1  Or if you want to compare XGBoost and Random Forest in trace prediction or the time prediction you would first have to run Event_forecasting.ipynb, then Time_forecasting.ipynb and only then the trace_prediciton.ipynb file to check performances of these models on the chosen data.
After that at the end of the trace_prediciton.ipynb file you would be able to see 2 graphics
4.2.1 First one tells you about the accuracy of prediciton of the whole trace
4.2.2 Seconds one tells you about the mean absolute errors of prediciton of the time for each event in the whole trace in hours

5. If you would want to see what insights we had found from the data you would need to go to Data_exploration.ipynb or the Visualizaiton.ipynb.
5.1 Data exploration file shows interesting insights found about the data
5.2 Visualization shows information regarding how does the train test split looks like with sensitive data (and how would it look in the customly made nested k-fold cross validation for the time sensitive data)


If you would want to show off the models you see in our files in front of your friends and tell them what an amazing job we had done, 
however the time it takes for all of the files is too large. You could use our demo_version.ipynb which runs on average ~70 seconds. 

Hope you have a great day, and if you have any suggestions regarding our project feel free to send an email to anseka.@gmail.com with your suggestions, which we would implement in the future. Because I want to finalize this project aside of the course, and will have to think of some ideas in order to make it better.
