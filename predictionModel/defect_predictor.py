import matplotlib
matplotlib.use('Agg')

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
import theano.tensor as tt
import pickle as pkl
import pandas as pd
import logging
from io import BytesIO
import base64
import json

from flask import Flask, request, session, g, redirect, url_for, \
     abort, render_template, flash, jsonify

app = Flask(__name__)
app.config.from_object(__name__)

sns.set(style="darkgrid")

model, trace = None, None
x_train, y_train = None, None
mask_num = 0

read_columns = ['Твёрдость гайки', 'Диаметр шины', 'Ширина шины', 'Диаметр диска', 'Ширина диска', 'Эластичность шины', 'Индекс надёжности']
train_columns = ['Твёрдость гайки', 'Индекс надёжности', 'Разница в ширине', 'Разница в диаметре']
logging.basicConfig(format='%(asctime)-15s: [%(levelname)s] %(message)s', level=logging.INFO)
### LOAD MODEL
logging.info('Loading model...')
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
logging.info('Loading is finished')

def parse_tree(tree):
    res = {}
    for elname, el in tree.items():
        if isinstance(el, dict):
            res = {**res, **parse_tree(el)}
        elif elname in read_columns:
            res[elname] = el
    return res
logging.info(pd.DataFrame([{'a':1}]))
@app.route('/', methods=['GET', 'POST'])
def index():
    tree = json.loads(request.form['tree'])
    params = parse_tree(tree)
    row = pd.DataFrame([params], columns = read_columns)
    row['Разница в ширине'] = abs(row['Ширина шины'] - row['Ширина диска'] - 10)
    row['Разница в диаметре'] = abs(row['Диаметр шины'] - row['Диаметр диска'] - 200)
    logging.info(row)
    ppc = predict_for_row(row) 
    return make_plot(ppc)


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
    my_stringIObytes = BytesIO()
    plt.savefig(my_stringIObytes, format='png')
    my_stringIObytes.seek(0)
    my_base64_data = base64.b64encode(my_stringIObytes.read())
    return my_base64_data

if __name__ == '__main__':
    app.run()
    
    
    