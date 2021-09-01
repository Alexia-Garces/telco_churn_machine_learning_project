import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# helper functions used to prepare telco_churn data.

#data splitting
def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on churn.
    return train, validate, test DataFrames.
    '''
    train, validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[churn])
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate[churn])
    return train, validate, test

def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test

#prep function for exploring
def prep_telco_explore(df):
    """
    This functions takes in the telco churn dataframe and retuns the cleaned and prepped dataset
    to use when doing exploratory data analysis
    """
    # list of columns to be dropped
    columns_to_drop = ['customer_id']
    
    # drops columns listed above
    df = df.drop(columns=columns_to_drop)
    
    #add autopay column
    df['autopay'] = (((df['payment_type'] == "Credit card (automatic)") == True) | ((df['payment_type'] == "Bank transfer (automatic)") == True)).astype(int)
    
    # Replace rows with no value in the total_charges column
    df.total_charges = pd.to_numeric(df.total_charges, errors='coerce').astype('float64')
    df.total_charges = df.total_charges.fillna(value="0")

    return df

#encoding function
def encode(df):
    '''
    Takes a dataframe and returns a new dataframe with encoded categorical variables
    '''
    label_encoder = LabelEncoder()
    for x in df.columns:
        df[x] = label_encoder.fit_transform(df[x])
    return df

#prep function for modeling
def prep_telco_model(df):
    '''
    This function takes in a dataframe and returns the cleaned, encoded and split data.
    Adds autopay column and encodes categoricals.
    Use this function before modeling.
    returns train, validate, test
    '''
    
    # list of columns to be dropped
    #columns_to_drop_model = ['gender','payment_type_id', 'contract_type_id', 'internet_service_type_id']
    
    # drops columns listed above
    #df = df.drop(columns=columns_to_drop_model)
    
    # Replace rows with no value in the total_charges column
    df.total_charges = pd.to_numeric(df.total_charges, errors='coerce').astype('float64')
    df.total_charges = df.total_charges.fillna(value="0")
    
    #add numeric values to categoricals
    df["partner"] = df.partner == "Yes"
    df['partner'] = (df['partner']).astype(int)

    df["dependents"] = df.dependents == "Yes"
    df['dependents'] = (df['dependents']).astype(int)

    df["phone_service"] = df.phone_service == "Yes"
    df['phone_service'] = (df['phone_service']).astype(int)

    df["streaming_tv"] = df.streaming_tv == "Yes"
    df['streaming_tv'] = (df['streaming_tv']).astype(int)

    df["streaming_movies"] = df.streaming_movies == "Yes"
    df['streaming_movies'] = (df['streaming_movies']).astype(int)

    df["paperless_billing"] = df.paperless_billing == "Yes"
    df['paperless_billing'] = (df['paperless_billing']).astype(int)

    df["churn"] = df.churn == "Yes"
    df['churn'] = (df['churn']).astype(int)

    df["multiple_lines"] = df.multiple_lines == "Yes"
    df['multiple_lines'] = (df['multiple_lines']).astype(int)

    df["online_security"] = df.online_security == "Yes"
    df['online_security'] = (df['online_security']).astype(int)

    df["online_backup"] = df.online_backup == "Yes"
    df['online_backup'] = (df['online_backup']).astype(int)

    df["device_protection"] = df.device_protection == "Yes"
    df['device_protection'] = (df['device_protection']).astype(int)

    df["tech_support"] = df.tech_support == "Yes"
    df['tech_support'] = (df['tech_support']).astype(int)
    
    #drop redundant columns
    df = df.drop(columns =['payment_type_id', 'contract_type_id', 'internet_service_type_id', 'gender', 'customer_id', 'total_charges'])
    
    #make a dummy df, and combining it back to the original df. Dropping redundant columns again.
    dummy_df = pd.get_dummies(df[['internet_service_type', 'contract_type', 'payment_type']], drop_first=False)
    #rename columns
    dummy_df = dummy_df.rename(columns={'internet_service_type_DSL': 'dsl',
                                   'internet_service_type_Fiber optic': 'fiber_optic',
                                   'internet_service_type_None': 'no_internet',
                                   'contract_type_Month-to-month': 'monthly_contract',
                                   'contract_type_One year': 'one_year',
                                   'contract_type_Two year': 'two_year',
                                   'payment_type_Bank transfer (automatic)': 'auto_bank_transfer',
                                   'payment_type_Credit card (automatic)': 'auto_credit_card',
                                   'payment_type_Electronic check': 'electronic_check',
                                   'payment_type_Mailed check': 'mailed_check'})
    df = pd.concat([df, dummy_df], axis =1)
    
    
    #add autopay column
    df['autopay'] = (((df['payment_type'] == "Credit card (automatic)") == True) | ((df['payment_type'] == "Bank transfer (automatic)") == True)).astype(int)
    
    #drop payment_type column
    df = df.drop(columns=['payment_type', 'internet_service_type', 'contract_type'])
    
    # split into train validate and test 
    train, validate, test = train_validate_test_split(df, target='churn', seed=123)
    
    return train, validate, test