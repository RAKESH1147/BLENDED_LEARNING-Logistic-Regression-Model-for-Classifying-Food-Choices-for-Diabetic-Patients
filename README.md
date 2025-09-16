# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load Data Import and prepare the dataset to initiate the analysis workflow.


2.Explore Data Examine the data to understand key patterns, distributions, and feature relationships.


3.Select Features Choose the most impactful features to improve model accuracy and reduce complexity.


4.Split Data Partition the dataset into training and testing sets for validation purposes.


5.Scale Features Normalize feature values to maintain consistent scales, ensuring stability during training.


6.Train Model with Hyperparameter Tuning Fit the model to the training data while adjusting hyperparameters to enhance performance.


7.Evaluate Model Assess the model’s accuracy and effectiveness on the testing set using performance metrics.

## Program:
```
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by:  Rakesh K S
RegisterNumber:  212224040264
```
``` py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
df=pd.read_csv("food_items .csv") # Corrected filename
print('Name: Rakesh K S')
print('Reg. No: 212224040264')
print("Dataset Overview")
print(df.head())
print("\ndatset Info")
print(df.info())

X_raw=df.iloc[:, :-1]
y_raw=df.iloc[:, -1:]
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw.values.ravel())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)

# Define the parameters before initializing the model
penalty='l2'
multi_class='multinomial'
solver='lbfgs'
max_iter=1000

# The model was not trained and y_pred was not defined, adding model training
model = LogisticRegression(penalty=penalty, multi_class=multi_class, solver=solver, max_iter=max_iter)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print('Name: Rakesh K S')
print('Reg. No: 212224040264')
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
```
## Output:
<img width="799" height="680" alt="image" src="https://github.com/user-attachments/assets/69b841aa-ccf4-47ee-805a-e957ff382943" />
<img width="504" height="554" alt="image" src="https://github.com/user-attachments/assets/a3ebae19-7710-4237-89cf-210a974edd27" />
<img width="1330" height="525" alt="image" src="https://github.com/user-attachments/assets/44cbbdf1-4830-4e9d-bff1-872e03f3d9cd" />


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
