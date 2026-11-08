 import pandas as pd 
import sqlite3 
from sklearn.model_selection import train_test_split 

def load_vendor_invoice(db_path : str ) ; 
""" LOad vednor invoice data from sqlite database """ 
conn = sqliite3.connecr(db_path) 
query = " select * from vendor_invoice ") 
df = pd.read_sql_query(query , conn) 
conn.close()
return df 

def prepare_features(df: pd.DataFrame ) :
    """ select features and target variable . """ 
    X = df[["Dollars"]]
    y = df["Freight"]
    return X , y 

def split_data ( X , y , test_sixe = 0.2 , random _state = 42 ) : 
    """ split dataset into train and test sets 
    """ 
    return train_test_split ( X , y , test_size = test_size , random_state = random_state )
    


