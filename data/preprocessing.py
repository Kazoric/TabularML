import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

# def preprocessing(config):
#     train_file = config.train_file
#     test_file = config.test_file

#     df_train = pd.read_csv(train_file, index_col='id')
#     df_test = pd.read_csv(test_file, index_col='id')

#     mapping = {'No': 0, 'Yes': 1}
#     df_train['Churn'] = df_train['Churn'].map(mapping)

#     df_train['origin'] = 'train'
#     df_test['origin'] = 'test'
#     df_all = pd.concat([df_train, df_test], axis=0)

#     categorical_features = config.categorical_features
#     numeric_features = config.numeric_features
#     target = config.target

#     config.cat_cardinalities = [
#         df_train[col].nunique()
#         for col in config.categorical_features
#     ]

#     config.target_dim = df_train[target].nunique()

#     for col in categorical_features:
#         df_all[col] = df_all[col].astype('category').cat.codes

#     df_train[categorical_features] = df_all.loc[df_all['origin'] == 'train', categorical_features]
#     df_test[categorical_features] = df_all.loc[df_all['origin'] == 'test', categorical_features]
    
#     scaler = StandardScaler()
#     df_train[numeric_features] = scaler.fit_transform(df_train[numeric_features])
#     df_test[numeric_features] = scaler.transform(df_test[numeric_features])

#     X_cat = torch.tensor(df_train[categorical_features].values, dtype=torch.int64)
#     X_num = torch.tensor(df_train[numeric_features].values, dtype=torch.float32)
#     y = torch.tensor(df_train[target].values, dtype=torch.int64).view(-1)

#     X_cat_train, X_cat_valid, X_num_train, X_num_valid, y_train, y_valid = train_test_split(
#         X_cat, X_num, y, test_size=0.2, random_state=42
#     )

#     X_cat_test = torch.tensor(df_test[categorical_features].values, dtype=torch.int64)
#     X_num_test = torch.tensor(df_test[numeric_features].values, dtype=torch.float32)

#     return X_cat_train, X_cat_valid, X_num_train, X_num_valid, y_train, y_valid, X_cat_test, X_num_test


def preprocessing(config):
    train_file = config.train_file
    test_file = config.test_file

    df_train = pd.read_csv(train_file, index_col='id')
    df_test = pd.read_csv(test_file, index_col='id')

    mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    df_train['Irrigation_Need'] = df_train['Irrigation_Need'].map(mapping)

    df_train['origin'] = 'train'
    df_test['origin'] = 'test'
    df_all = pd.concat([df_train, df_test], axis=0)

    categorical_features = config.categorical_features
    numeric_features = config.numeric_features
    target = config.target

    config.cat_cardinalities = [
        df_train[col].nunique()
        for col in config.categorical_features
    ]

    config.target_dim = df_train[target].nunique()

    for col in categorical_features:
        df_all[col] = df_all[col].astype('category').cat.codes

    df_train[categorical_features] = df_all.loc[df_all['origin'] == 'train', categorical_features]
    df_test[categorical_features] = df_all.loc[df_all['origin'] == 'test', categorical_features]
    
    scaler = StandardScaler()
    df_train[numeric_features] = scaler.fit_transform(df_train[numeric_features])
    df_test[numeric_features] = scaler.transform(df_test[numeric_features])

    X_cat = torch.tensor(df_train[categorical_features].values, dtype=torch.int64)
    X_num = torch.tensor(df_train[numeric_features].values, dtype=torch.float32)
    y = torch.tensor(df_train[target].values, dtype=torch.int64).view(-1)

    X_cat_train, X_cat_valid, X_num_train, X_num_valid, y_train, y_valid = train_test_split(
        X_cat, X_num, y, test_size=0.2, random_state=42
    )

    X_cat_test = torch.tensor(df_test[categorical_features].values, dtype=torch.int64)
    X_num_test = torch.tensor(df_test[numeric_features].values, dtype=torch.float32)

    return X_cat_train, X_cat_valid, X_num_train, X_num_valid, y_train, y_valid, X_cat_test, X_num_test