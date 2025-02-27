#-------------------------------------------------------------------------
# AUTHOR: Brandon Chang
# FILENAME: naive_bayes.py
# SPECIFICATION: Output classification confidence
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
#--> add your Python code here
X = []
Y = []

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
outlook_map = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temperature_map = {'Cool': 1, 'Mild': 2, 'Hot': 3}
humidity_map = {'Normal': 1, 'High': 2}
wind_map = {'Weak': 1, 'Strong': 2}
play_map = {'Yes': 1, 'No': 2}

with open("weather_training.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        X.append([outlook_map[row[1]], temperature_map[row[2]], humidity_map[row[3]], wind_map[row[4]]])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
with open("weather_training.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        Y.append(play_map.get(row[5], 0))

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
test_data = []
test_labels = []

with open("weather_test.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        test_data.append([outlook_map.get(row[1], 0), temperature_map.get(row[2], 0), humidity_map.get(row[3], 0), wind_map.get(row[4], 0)])
        test_labels.append(row)  # Store full row for output
        
#Printing the header os the solution
#--> add your Python code here
print("Day", "Outlook", "Temperature", "Humidity", "Wind", "PlayTennis", "Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for i, test_sample in enumerate(test_data):
    probabilities = clf.predict_proba([test_sample])[0]
    prediction = clf.predict([test_sample])[0]
    confidence = max(probabilities)
    
    if confidence >= 0.75:
        play_tennis = "Yes" if prediction == 1 else "No"
        print(test_labels[i][0], test_labels[i][1], test_labels[i][2], test_labels[i][3], test_labels[i][4], play_tennis, round(confidence, 2))


