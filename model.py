import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data_frame=pd.read_csv('nomalized_data.csv')
print(data)

# Splitting the data into train and test sections
x_train, x_test, y_train, y_test = train_test_split(data.drop('Employees',axis = 1), data['Profit'], test_size = 0.20, random_state = 7)

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import metrics

# creating and training the model
regressor= LinearRegression()
regressor.fit (x_train, y_train) 
y_predicted= regressor.predict(x_test)

# Comparing weather the predicted data produced is similar to the test dataset
print(x_test)
cross_check_data= pd.Dataframe(y_test, y_predicted)
print(cross_check_data)

# Graphical representation of the datapoints
plt.scatter(x_train, y_train)
plt.plot(x_train, y_predicted, c='g')
plt.title("Representation of data-points before and after training")
plt.xlabel("investments in diff depts")
plt.ylael("Profits")
plt.legend()
plt.grid(True,color='y')
plt.show()

# Displaying accuracy scores 
acc = metrics.accuracy_score(y_test, y_predicted))
print(acc)
cnf_matrix = metrics.confusion_matrix(y_test, y_predicted)
print(cnf_matrix)

# Enable the user to make live prediction
ip=[]
print("predict the profit if the resources allocated are in the way of ... \n")
print("Production_design, Production_development, Product_testing, Marketing, Sales")

ip = int(input())
live_prediction = regressor.predict([ip])
print(live_prediction)

