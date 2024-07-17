This entire project is based on the Process Mining of business events. 

Our goal in this project was an attempt to predict at first the next event, then the time till the next event, 
and eventually combine all of these models to get the whole suffix prediction

In order to properly run all of the files you will need to follow this guidline:
1. Make sure you have the Split_function.py file since this is the one which contains proper train_test_split 
solely made for the process mining (gets rid of overlapping events in train and test)
2. Run the PROCESS_MINING_ML_MODELS.ipynb which would run create 10 models for RandomForestClassifier, and 10 models for XGBoostClassifier 
and will save those models in the same director you have all of the uploaded files. These models are for next event prediction
3. Run the Time_forecasting.ipynb which would run create 10 models for RandomForestRegressor, and 10 models for XGBoostRegressor
and will save those models in the same director you have all of the uploaded files. These models are for the next time prediciton
4. Run the trace_prediction.ipynb which would take the models you had created in 2 files above and will recurrently use them in order to predict the whole suffix
After that at the end of the trace_prediciton.ipynb file you would be able to see 2 graphics
1. First one tells you about the accuracy of prediciton of the whole trace
2. Seconds one tells you about the mean absolute errors of prediciton of the time for each event in the whole trace in hours

If you would want to show off the models you see in our files in front of your friends and tell them what an amazing job we had done, 
however the time it takes for all of the files is too large. You could use our demo_version.ipynb which runs on average ~70 seconds. 

Hope you have a great day, and if you have any suggestions regarding our project feel free to send an email to anseka.@gmail.com with your suggestions, 
which we would implement in the future. Because I want to finalize this project aside of the course, and will have to think of some ideas in order to make it better.