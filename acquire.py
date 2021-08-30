#Aquire functions for telco churn classification project

import pandas as pd
import numpy as np
import os
from env import host, user, password

###################### Acquire Telco Data ######################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    
    
def new_telco_data():
    '''
    This function reads the telco data from the Codeup db into a df,
    write it to a csv file, and returns the df.
    '''
    # Create SQL query.
    sql_query = """
    SELECT 
	    customer_id, churn, dependents, device_protection, gender, 
        monthly_charges, multiple_lines, paperless_billing, partner,  
        phone_service,tenure, online_backup, online_security, senior_citizen,
        streaming_tv, streaming_movies, tech_support, total_charges,
	    i.internet_service_type_id AS 'internet_service_type_id', internet_service_type,
	    ct.contract_type_id AS 'contract_type_id', contract_type,
	    p.payment_type_id AS 'payment_type_id', payment_type
    FROM customers AS c
    JOIN contract_types AS ct ON ct.`contract_type_id` = c.contract_type_id
    JOIN internet_service_types AS i ON i.internet_service_type_id = c.internet_service_type_id
    JOIN payment_types AS p ON p.payment_type_id = c.payment_type_id;
    """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    
    return df



def get_telco_data():
    '''
    This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('telco_churn_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('telco_churn_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_telco_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('telco_churn_df.csv')
        
    return df