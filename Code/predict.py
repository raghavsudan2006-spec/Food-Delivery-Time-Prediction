# ---------------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score

# ----------------------------------------------------------------------------------

data = pd.read_csv('train.csv')

# ----------------------------------------------------------------------------------

def only_num(x):
    num_list = ['1','2','3','4','5','6','7','8','9','0']

    for word in str(x):
        if word not in num_list:
            x = x.replace(word,"")

    return int(x)

# ----------------------------------------------------------------------------------

x = data.drop(columns= ['Time_taken(min)'],axis = 1)
y = data["Time_taken(min)"].apply(only_num)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model_cv = joblib.load('model.joblib')

y_pred = model_cv.predict(x_test)

r2 = r2_score(y_test,y_pred)
print(r2)

# -----------------------------------------------------------------------------------