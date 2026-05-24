# ---------------------------------------------------------------------------------
import time
start = time.time()
print("Code starts")

import main
from xgboost import XGBRegressor
from sklearn.model_selection import KFold,RandomizedSearchCV,train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

print("Exporting done in ",time.time() - start ,'sec')

# ------------------------------------------------------------------------------------

data = pd.read_csv("../Data/train.csv")
print("Readed CSV",time.time() - start)

# ------------------------------------------------------------------------------------

def only_num(x):
    num_list = ['1','2','3','4','5','6','7','8','9','0']

    for word in x:
        if word not in num_list:
            x = x.replace(word,"")

    return int(x)

# --------------------------------------------------------------------------------------

x = data.drop(columns= ['Time_taken(min)','ID',
                        'Delivery_person_ID'],axis = 1)
y = data["Time_taken(min)"].apply(only_num)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# ---------------------------------------------------------------------------------------

pipe = Pipeline([
    ('Normalizer', main.MissingNormalizer()),
    ('Filling_NaN',main.Filling_NaN()),
    ('feature_engineering', main.Datetime()),
    ('drop_columns', main.delete_columns()),
    ('encoding',main.encoding()),
    ('cluster',main.Making_clusters()),
    ('standarization',main.Standarization()),
    ('model',XGBRegressor())
])

# ----------------------------------------------------------------------------------------

CV = KFold(n_splits=3,shuffle=True,random_state=42)

model_cv = RandomizedSearchCV(
    pipe, 
    param_distributions={"model__n_estimators":[100,200,300,400,500,600,700,800],
              "model__max_depth":[4,7,10],
              "model__learning_rate":[0.01,0.1,0.2],
              "model__min_child_weight":[1,2,3,4,5,6,7,8,9,10],
              "model__reg_alpha":[0.01,0.05,0.1],
              "model__reg_lambda":[1,2,3,4,5]},
              
    cv=CV,
    random_state=42,
    n_iter=100,
    n_jobs=-1,
    scoring='r2'
)

model_cv.fit(x_train,y_train)

# ----------------------------------------------------------------------------------------

with open("model1.joblib",'wb')as f:
    joblib.dump(model_cv,f)

# ----------------------------------------------------------------------------------------