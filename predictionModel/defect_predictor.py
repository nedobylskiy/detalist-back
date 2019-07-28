import matplotlib
matplotlib.use('Agg')

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
import theano.tensor as tt
import pickle as pkl
import pandas as pd

sns.set(style="darkgrid")

model, trace = None, None
x_train, y_train = None, None
mask_num = 0

train_columns = ['nuts_hardness', 'reliability_index', 'width_diff', 'diam_diff']


    # global x_train, y_train
    # global model, trace
with open('X_train.pkl', 'rb') as f:
    X_train = pkl.load(f)
with open('y_train.pkl', 'rb') as f:
    y_train = pkl.load(f)
with pm.Model() as loaded_model:
    Xmu = pm.Normal('Xmu', mu=0, sd=10, shape=(1,len(train_columns)))
    X_modeled = pm.Normal('X', mu=Xmu, sd=1., observed=X_train) # 

    intercept = pm.Normal('Intercept', 0, sd=20)
    coefs = [pm.Normal(colname, 0, sd=5) for colname in train_columns]
    
    y_prob = pm.math.sigmoid(intercept + sum([coefs[i] * X_modeled[:,i] for i in range(len(train_columns))]))
    y = pm.Bernoulli('y', y_prob, observed=y_train)
    trace = pm.load_trace('traces')
model = loaded_model
    
def predict_for_row(row):
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    vals = np.array([row[train_columns].tolist()]*(len(train_columns)+1))
    masks = np.triu(np.ones((len(train_columns)+1, len(train_columns))))
    return predict_with_masks(vals, masks)

def predict_with_masks(vals, masks):
    global mask_num
    with model:
        X_mask = pm.Normal(f'X_mask{mask_num}', mu=Xmu, sd=1., shape=vals.shape)
        Xpred = tt.squeeze(X_mask)*masks + vals*(1-masks)
        y_prob = pm.math.sigmoid(intercept + sum([coefs[i] * Xpred[:,i] for i in range(len(train_columns))]))
        y = pm.Bernoulli(f'y{mask_num}', y_prob, shape=(len(vals)))
        mask_num += 1
        ppc = pm.sample_posterior_predictive(trace, model=model, vars=[y], samples=300)        
    return ppc[f'y{mask_num-1}']
    
def make_plot(ppc):
    plt.figure(figsize=(10,7))
    plt.errorbar(np.arange(5), ppc.mean(axis=0), ppc.std(axis=0)**2, marker='^')
    plt.xticks(np.arange(5), ['empty'] + train_columns)
    plt.yticks(np.linspace(0, 1, 10))
    plt.savefig('res.png')

if __name__ == '__main__':
    row = pd.DataFrame([{'nuts_hardness': 2.460573,  'reliability_index': 8.34, 'width_diff': 10, 'diam_diff': 0}])
    ppc = predict_for_row(row)
    make_plot(ppc)
    
    
    