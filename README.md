<Phase 1>
We successfully downloaded all the data from 5 stations and conducted basic EDA for all 5 stations.

We also built a random forest model to identify the window size (in LSTM model) and we found out that the next hour temperature can mostly be predicted by two previous hours temperature data. Thus, our window size will be 2-4 hours. 

<Phase 2>
We successfully built our first model using Gated Recurrent Unit (GRU). Our model incorporated five different stations and used five different variables, which include temperature, humidity, pressure, wind speed, and wind direction, for each stations (thus, 25 variables total for one time step). 

We shifted the dataframe 24 hours so that our model can predict the temperature of philadelphia station 24 hours ahead, using 25 variables from five stations. We used one layer GRU, and we used bath_generator to split the dataset with the defined batch_size, which helped us maximizing the work-load of GPU. It took almost 35 mins to train one model with GPU.

For validation, we used Mean Square Error (MSE). We also used callback early stopping to avoid unnecessary training. 

In conclusion, our model was good at predicting the general trend (when to go up and when to go down), but it was not good at predicting the peak. For our final phase, we will improve this by incorporating the time label (hour, day, month....)

[final report]


We added a distance metric to our EDA to give an idea of how far each variable (at each station) is from our target station's data (KPHL). First, we scaled the data between 0-1 by using a min-max scaler, then we  took the absolute value of the difference between each observation and the corresponding observation of our target class; we averaged the absolute differences over each column to get an average distance that each observation is from the target observations. 

We successfully built a model using time labels included. Surprisingly, when we included time variable, the accuracy went up increasingly. When the model predicts the temperature with only temperature, the accuracy was higher than the model trained with temperature, humidity, wind speed, and pressure. However, when we included wind direction variable, we got even higher accuracy than the model trained with temperature. We also looked at the time and made a plot that shows the error rate by hour (0 to 23). For both train and testing model, 3pm to 8pm have higher error compare to other hours. In addition, we built different models that predict 12 hours, 24 hours, and 36 hours ahead. As we guessed, the accuracy was highest when we predict 12 hours ahead and lowest when we predict 36 hours ahead. 

Moreover, to check the feature importance, we decided to build random forest. Surprisingly, when we predicted the temperature of philadelphia station, the lancaster station was most influential in predicting the temperature of philadelphia statuion. Also, we figured out that previous temperature is most important variable predicting the temperature in 24 hours.