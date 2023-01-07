import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# Project Configurations
class Configs:
    def __init__(self, debug=False) -> None:
        self.path_prefix = "./tabular-playground-series-aug-2022"
        self.train_path = "./tabular-playground-series-aug-2022/train.csv"
        self.test_path = "./tabular-playground-series-aug-2022/test.csv"
        self.store_path = "./model"
        self.submission_path = "./tabular-playground-series-aug-2022/sample_submission.csv"
        self.is_debug = debug

"""
Main function for inferencing
"""
def main():
    # load configuration
    configs = Configs(debug=False)
    # Read Data
    train = pd.read_csv(f"{configs.train_path}")
    test = pd.read_csv(f"{configs.test_path}")
    submission = pd.read_csv(f"{configs.submission_path}")
    data = data_preprocessing(train, test, configs)
    train = data[data.failure.notnull()]
    test = data[data.failure.isnull()].drop(['failure'], axis=1)
    # select features
    select_feature = [
        'loading', 
        'area', 'count_null',
        'm3_missing', 'm5_missing', 
        'measurement_1', 'measurement_2', 'measurement_17'
    ]
    # Init and train model
    lrmodel = LRModel(train, select_feature, configs)
    lrmodel.train(test_data=test)
    # get results
    submission['lr0'], submission['lr1'], submission['lr2'], submission['lr3'] = lrmodel.get_results()
    submission['failure'] = submission['lr0'] * lrmodel.lr_effects[0] + \
        submission['lr1'] * lrmodel.lr_effects[1] + \
        submission['lr2'] * lrmodel.lr_effects[2] +\
        submission['lr3'] * lrmodel.lr_effects[2]
    # write to csv
    submission[['id', 'failure']].to_csv('109550129_submission.csv', index=False)




"""
Utilities modified from train scripts
"""
def corr_selection(data, test):
    full_dict = {}
    # Append additional attributes
    col = [col for col in test.columns if 'measurement' not in col] \
        + ['loading', 'm3_missing', 'm5_missing', 'count_null', 'area']
    correlations = []
    selected =[]
    for m_num in range(3,18):
        correlation = data.drop(col, axis=1).corr()[f'measurement_{m_num}']
        correlation = np.absolute(correlation).sort_values(ascending=False)
        # selection with correlation sum
        correlations.append(np.round(np.sum(correlation[1:5]), 5))
        selected.append(f'measurement_{m_num}')
    selected_col = pd.DataFrame()
    selected_col['selected'] = selected
    selected_col['correlation'] = correlations
    selected_col = selected_col.sort_values(by='correlation',ascending=False).reset_index(drop=True)
    for i in range(7):
        target_cols = 'measurement_' + selected_col.iloc[i,0][12:] # selection for next best
        fill_dict ={}
        for x in data.product_code.unique() : 
            # Compute correlation for all product code
            correlation = data[data.product_code == x].drop(col, axis=1).corr()[target_cols]
            correlation = np.absolute(correlation).sort_values(ascending=False)
            target_cols_dic = {}
            target_cols_dic[target_cols] = correlation[1:5].index.tolist()
            fill_dict[x] = target_cols_dic[target_cols]
        full_dict[target_cols] = fill_dict
    return full_dict

def data_preprocessing(train, test, configs):
    # Aggregate orb create new values by discussions mentioned in report
    data = pd.concat([train, test])
    data['area'] = data['attribute_2'] * data['attribute_3']
    data['count_null'] = data.isnull().sum(axis=1)
    data['m3_missing'] = data['measurement_3'].isnull().astype(np.int8)
    data['m5_missing'] = data['measurement_5'].isnull().astype(np.int8)
    data['loading'] = np.log1p(data['loading'])
    # locate for all measurement
    features = [f for f in test.columns if f.startswith('measurement') or f=='loading']
    full_dict = corr_selection(data, test)
    for code in data.product_code.unique():
        # Fill with LinearRegression
        for m_col in list(full_dict.keys()):
            corress_data = data[data.product_code==code]
            column = full_dict[m_col][code]
            tmp_train = corress_data[column + [m_col]].dropna(how='any')
            tmp_test = corress_data[
                (corress_data[column].isnull().sum(axis=1)==0) & (corress_data[m_col].isnull())
            ]
            huber_regressor = joblib.load(f"{configs.store_path}/Huber_{code}_{m_col}") # dump for inference
            data.loc[(data.product_code==code)                  # Imputation on selected missing fields
                     & (data[column].isnull().sum(axis=1)==0)
                     & (data[m_col].isnull()), m_col] = huber_regressor.predict(tmp_test[column])
        # Fill with KNN
        imputer = joblib.load(f"{configs.store_path}/imputer_{code}")
        data.loc[data.product_code==code, features] = imputer.fit_transform(data.loc[data.product_code==code, features])
    return data

class LRModel:
    def __init__(self, data, features, configs):
        self.x = data.drop(['failure'], axis=1)
        self.y = data['failure'].astype(int)
        self.configs = configs
        self.features = features
        self.n_splits = 5 # KFold Spilt
        self.data_len = len(data)
        self.lr_auc, self.lr_preds = 0, 0
        # evaluation of model
        self.x_trains = []
        self.train_aucs = []
        self.train_accs = []
        self.oof_aucs = []
        self.oof_accs = []
        self.feats_evaluations = []
        self.lr_effects = [0.3, 0.3, 0.2, 0.2]
        self.lr_results = []        # prediction results
        # construct feats subset to train multiple LR
        self.subsets = []
        self._construct_subset()
        
    def _construct_subset(self): 
        # fixed chosen subset for better results
        self.subsets.append(self.features)
        tmp_list = [0, 1, 5, 6, 7]
        self.subsets.append([self.features[x] for x in tmp_list])
        tmp_list = [0, 2, 3, 4, 6, 7]
        self.subsets.append([self.features[x] for x in tmp_list])
        tmp_list = [0, 5, 7]
        self.subsets.append([self.features[x] for x in tmp_list])

    def _scale(self, train_data, val_data, test_data, feats):
        # function for cross validation, scaling the data of x_train, x_val, x_test by selected feature
        scaler = StandardScaler()                                 # Gaussion distribussion
        # transform datas with specific feats
        scaled_train = scaler.fit_transform(train_data[feats])  
        scaled_val = scaler.transform(val_data[feats])
        scaled_test = scaler.transform(test_data[feats])
        # make return value
        rtn_train = train_data.copy()
        rtn_val = val_data.copy()
        rtn_test = test_data.copy()
        # replace specific feats with standardize datas
        rtn_train[feats] = scaled_train
        rtn_val[feats] = scaled_val
        rtn_test[feats] = scaled_test
        return rtn_train, rtn_val, rtn_test
    
    def train(self, test_data):
        # train multiple LRs
        for i_feat, feats in enumerate(self.subsets):
            lr_target = np.zeros(len(test_data))
    
            # KFold
            kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=0)
            for _, (train_idx, valid_idx) in enumerate(kfold.split(self.x, self.y)):
                x_train, x_val = self.x.iloc[train_idx], self.x.iloc[valid_idx]
                x_test = test_data.copy()
                x_train, x_val, x_test = self._scale(x_train, x_val, x_test, feats)
                # train linear regressor
                lr = joblib.load(f"{self.configs.store_path}/LRModel_{i_feat}_{_}")  # dump model
                lr_target += lr.predict_proba(x_test[feats])[:, 1] / self.n_splits
            
            # Store results
            self.lr_results.append(lr_target)

    def get_results(self):
        return self.lr_results


if __name__ == '__main__':
    main()