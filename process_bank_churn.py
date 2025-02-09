from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTENC



def clean_data(raw_df):
    raw_df.drop(['Surname', 'CustomerId'], axis=1, inplace=True)
    X=raw_df.iloc[:,:-1]
    y=raw_df.Exited
    input_cols=raw_df.iloc[:, :-1].columns
    target_col=raw_df.iloc[:,-1].name
    numeric_cols=X.select_dtypes(exclude='object').columns.tolist()
    categorical_cols=X.select_dtypes(include='object').columns.tolist()
    return X, y, input_cols, target_col, numeric_cols, categorical_cols

def split_data(X, y):
    X_train, X_val, y_train, y_val=train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    train_inputs=X_train
    val_inputs=X_val
    train_targets=y_train
    val_targets=y_val
    return train_inputs, val_inputs, train_targets, val_targets
    
    

# Кодування категоріальних колонок
def encode_cat(train_inputs, val_inputs, categorical_cols):
    encoder=OneHotEncoder()
    encoder.fit(train_inputs[categorical_cols])
    geo_code_train=encoder.transform(train_inputs[categorical_cols]).toarray()
    geo_code_val=encoder.transform(val_inputs[categorical_cols]).toarray()
    categories=np.concatenate((encoder.categories_[0],encoder.categories_[1]))
    train_inputs[categories]=geo_code_train
    val_inputs[categories]=geo_code_val
    return encoder, categories


# Масштабування числови колонок для приведення всіх значень до одного масштабу, оскільки різні колонки містять значення в різних діапазонах, що може негативно вплинути на
# сходження алгоритму
def scale_num(train_inputs, val_inputs, numeric_cols):
    scaler=MinMaxScaler()
    scaler.fit(train_inputs[numeric_cols])
    train_inputs[numeric_cols]=scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols]=scaler.transform(val_inputs[numeric_cols])
    return scaler

def create_full_inputs(train_inputs, val_inputs, numeric_cols, categories):
    cols=np.concatenate((numeric_cols, categories))
    X_trains=train_inputs[cols]
    X_vals=val_inputs[cols]
    return X_trains, X_vals
    
    
def class_balance(X_trains, train_targets):
    cat_cols = ['NumOfProducts','HasCrCard', 'IsActiveMember', 'France', 'Germany', 'Spain', 'Female', 'Male']
    smenc = SMOTENC(random_state=42, categorical_features=cat_cols)
    X_train_smenc, y_train_smenc = smenc.fit_resample(X_trains, train_targets)
    return X_train_smenc, y_train_smenc
    
    
def preprocess_data(raw_df):
    X, y, input_cols, target_col, numeric_cols, categorical_cols=clean_data(raw_df)
    train_inputs, val_inputs, train_targets, val_targets=split_data(X,y)
    encoder, categories=encode_cat(train_inputs, val_inputs, categorical_cols)
    scaler=scale_num(train_inputs, val_inputs, numeric_cols)
    X_trains, X_vals=create_full_inputs(train_inputs, val_inputs, numeric_cols, categories)
    X_train_smenc, y_train_smenc = class_balance(X_trains, train_targets)
    result={'X_train':X_train_smenc,
    'train_targets':y_train_smenc,
    'X_val':X_vals,
    'val_targets':val_targets,
    'input_cols':input_cols,
    'scaler':scaler,
    'encoder':encoder,
    'categorical_cols':categorical_cols,
    'numeric_cols':numeric_cols,
    'categories':categories
    }
    return result
    
def preprocess_new_data(test_df, encoder, scaler, categorical_cols, numeric_cols, categories):
    test_df.drop(['Surname', 'CustomerId'], axis=1, inplace=True)
    geo_code_test=encoder.transform(test_df[categorical_cols]).toarray()
    test_df[categories]=geo_code_test
    test_df[numeric_cols]=scaler.transform(test_df[numeric_cols])
    cols=np.concatenate((numeric_cols, categories))
    test_inputs=test_df[cols]
    return test_inputs
    
    