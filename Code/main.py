# -------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import KFold,RandomizedSearchCV,train_test_split
from sklearn.pipeline import Pipeline
import joblib

# ----------------------------------------------------------------------------------
class MissingNormalizer(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.missing_patterns = ['NaN ','conditions NaN']

        self.num_col = ['Delivery_person_Age',
                        'Delivery_person_Ratings',
                        'multiple_deliveries']
        
        self.cat_col = ['Weatherconditions',
                        'Road_traffic_density',
                        'Festival','City',
                        'Type_of_order',
                        'Type_of_vehicle',
                        'Time_Orderd']

    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        
        # For Numerical Columns
        for col in self.num_col:
            X[col] = X[col].apply(lambda x:np.nan if x in self.missing_patterns else float(x))
        
        # For Categorical Columns
        for col in self.cat_col:
            X[col] = X[col].replace(self.missing_patterns,np.nan)

        return X

# ----------------------------------------------------------------------------------------
    
class Filling_NaN(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.num_col = ['Delivery_person_Age',
           'Delivery_person_Ratings',
           'multiple_deliveries']

        self.cat_col = ['Weatherconditions',
           'City',
           'Type_of_order',
           'Type_of_vehicle',
           'Festival',
           'Road_traffic_density']

    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()

        for col in self.num_col:
            X[col] = X[col].fillna(X[col].median())
        
        for col in self.cat_col:
            X[col] = X[col].fillna(X[col].mode()[0])
        
        return X
    
Cleaning_steps = Pipeline([
    ('Normalizer', MissingNormalizer()),
    ('Filling_NaN',Filling_NaN())])

# ------------------------------------------------------------------------------------
    
class Datetime(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.datetime_col = ['Time_Orderd',
                        'Order_Date',
                        'Time_Order_picked']

    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        
        for col in self.datetime_col:
            if col == "Time_Orderd":
                X[col] = pd.to_datetime(X[col],errors='coerce')

                X['Order_Hour'] = X['Time_Orderd'].dt.hour
                X['Order_minute'] = X['Time_Orderd'].dt.minute
                X['Order_second'] = X['Time_Orderd'].dt.second

                X['Order_Hour'] = X['Order_Hour'].fillna(X['Order_Hour'].median())
                X['Order_minute'] = X['Order_minute'].fillna(X['Order_minute'].median())
                X['Order_second'] = X['Order_second'].fillna(X['Order_second'].median())

            elif col == "Order_Date":
                X[col] = pd.to_datetime(X[col])

                X['Order_day'] = X['Order_Date'].dt.day
                X['Order_month'] = X['Order_Date'].dt.month
                X['Order_year'] = X['Order_Date'].dt.year
            
            else:
                X[col] = pd.to_datetime(X[col])

                X['Picked_hour'] = X['Time_Order_picked'].dt.hour
                X['Picked_minute'] = X['Time_Order_picked'].dt.minute
                X['Picked_second'] = X['Time_Order_picked'].dt.second

        return X
    
# --------------------------------------------------------------------------------------

class delete_columns(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.drop_columns = ['Order_Date',
                        'Time_Order_picked',
                        'Time_Orderd']

    def fit(self,X,y = None):
        return self


    def transform(self,X):
        X = X.copy()
        X = X.drop(columns = self.drop_columns)

        return X
    
# ----------------------------------------------------------------------------------------

class Standarization(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.num_col = ['Delivery_person_Age',
           'Delivery_person_Ratings',
           'multiple_deliveries',
           'Restaurant_latitude',
           'Restaurant_longitude',
           'Delivery_location_latitude',
           'Delivery_location_longitude',
           'Vehicle_condition',
           'Order_Hour',
           'Order_minute',
           'Order_second',
           'Order_month',
           'Order_year',
           'Order_day',
           'Picked_hour',
           'Picked_minute',
           'Picked_second',
           'Festival',
           'Road_traffic_density']
        
        self.standarization = StandardScaler()
        
    def fit(self,X,y=None):
        self.standarization.fit(X[self.num_col])
        return self

    def transform(self,X):
        X = X.copy()

        x_std = self.standarization.transform(X[self.num_col])

        x_std1 = pd.DataFrame(x_std,columns = self.num_col)
        
        X = X.drop(columns = self.num_col)
        
        X = X.reset_index(drop = True)
        x_std1 = x_std1.reset_index(drop = True)

        X = pd.concat([X,x_std1],axis = 1)
        
        return X
# ------------------------------------------------------------------------------------------
class encoding(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.cat_col = ['Weatherconditions',
           'City',
           'Type_of_order',
           'Type_of_vehicle']
        
        self.cat1_col = ['Festival',
           'Road_traffic_density']
        
        self.one_hot_encoder = OneHotEncoder(sparse_output=False,handle_unknown="ignore")

        self.label_encoder = {}

    def fit(self,X,y=None):
        self.one_hot_encoder.fit(X[self.cat_col])
        
        for col in self.cat1_col:
            Label_encoder = LabelEncoder()
            Label_encoder.fit(X[col])

            self.label_encoder[col] = Label_encoder

        return self
       
    def transform(self,X):
        encoded_columns = self.one_hot_encoder.transform(X[self.cat_col])
        encoded_data = pd.DataFrame(encoded_columns,
                                    columns=self.one_hot_encoder.get_feature_names_out())
        
        X = X.drop(columns=["City","Type_of_vehicle","Weatherconditions","Type_of_order"])
        X = X.reset_index(drop=True)
        encoded_data = encoded_data.reset_index(drop=True)
        X = pd.concat([X,encoded_data],axis=1)

        for col in self.cat1_col:
            X[col] = self.label_encoder[col].transform(X[col])

        return X
    
# ------------------------------------------------------------------------------------
    
class Making_clusters(BaseEstimator,TransformerMixin):
    def __init__(self,n_clusters = 3):
        self.n_clusters = n_clusters

        self.kmeans = KMeans(n_clusters = self.n_clusters,
                             random_state= 42)
        
        self.columns = ['Restaurant_latitude',
                          'Restaurant_longitude',
                          'Delivery_location_latitude',
                          'Delivery_location_longitude']

    def fit(self,X,y =None):
        self.kmeans.fit(X[self.columns])
        return self

    def transform(self,X):
        X = X.copy()

        X['cluster'] = self.kmeans.predict(X[self.columns])

        return X
    
# ---------------------------------------------------------------------------------