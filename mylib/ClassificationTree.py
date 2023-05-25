import pandas as pd
import numpy as np
def gini(df):
    if(len(df)==0):
        return 0
    else:
        num_classes = [np.sum(df.iloc[:,-1] == clas) for clas in set(df.iloc[:,-1])]
        return 1.0 - sum((class_num/len(df.iloc[:,-1]))**2 for class_num in num_classes)
def split(df, column_ind, value):
    left, right = [], []
    for string_ind in range(df.shape[0]):
        st = df.iloc[string_ind]
        if st[column_ind]<value:
            left.append(st)
        else:
            right.append(st)
    return pd.DataFrame(left), pd.DataFrame(right)
def best_split(df):
    gini_best, value_best = np.inf, np.inf
    col_ind_best, split_best = np.inf, None
    for column_ind in range(df.shape[1]-1):
        for row_ind in range(df.shape[0]):
            split_ = split(df, column_ind, df.iloc[row_ind, column_ind])
            gini_ = gini(split_[0])+ gini(split_[1])
            if gini_ < gini_best:
                gini_best, value_best = gini_, df.iloc[row_ind, column_ind]
                col_ind_best, split_best = column_ind, split_
    return dict({"col_index":col_ind_best, "value":value_best, "nodes":split_best})
def leaf(node):
    classes = np.zeros(len(set(node.iloc[:,-1]))+1)
    for row in node.iloc[:,-1]:
        classes[int(row)]+=1
    return list(classes).index(max(classes))
def recursive_split(node, max_depth, depth):
    left, right = node["nodes"]
    del node['nodes']
    if len(left)==0 or len(right)==0:
        node["left"] = node["right"] = leaf(pd.concat([left, right]))
        return
    if depth >= max_depth:
        node['left'], node['right'] = leaf(left), leaf(right)
        return
    node['right'] = best_split(right)
    recursive_split(node['right'], max_depth, depth+1)
    node['left'] = best_split(left)
    recursive_split(node['left'], max_depth, depth+1)
def predict(node, row):
    if row[node['col_index']]<node['value']:
        if isinstance(node['left'],dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'],dict):
            return predict(node['right'], row)
        else:
            return node['right']
        
class ClassificationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
    def fit(self, df):
        node = best_split(df)
        recursive_split(node, self.max_depth, 0)
        self.node = node   
    def predict(self,df_test):
        predictions = []
        for row_ind in range(df_test.shape[0]):
            predictions.append(predict(self.node, df_test.iloc[row_ind,:]))
        return predictions