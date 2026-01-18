## --------- Imports ---------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, \
Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, \
GradientBoostingRegressor

import scipy.io
data = scipy.io.loadmat('data.mat')
times = data['times']
dfof = data['dfof']
configs = data['configs']
roi_used = data['roi_used']

# Nishal Added
# for i in range(3):
#   dfof[0, i] = (dfof[0, i] - np.mean(dfof[0, i], 0)) / np.std(dfof[0, i], 0)

## ---------- Show traces ----------
def show_traces(ind):
  n = len(ind)
  fig, ax = plt.subplots(n,1, sharex=True, sharey=True)
  for i in ind:
    ax[i].plot(dfof[0,0][:,i])
    ax[i].set_ylabel(i)
  plt.tight_layout()
  plt.subplots_adjust(hspace=0)

## ---------- Plot interesting configs, roi ----------
# config, roi
# 15, 22
# 15, 23
# 27, 31

def show_stim(config, roi):
  config = config-1
  roi = roi-1
  plt.clf()
  plt.gcf().subplots(1,3,sharey=True)
  for session_id in range(3):
    plt.subplot(1, 3, session_id+1)
    plt.title(f'Session{session_id+1}')
    for i in range(8):
      a = times[0,session_id][i,config] - 0 - 1
      b = times[0,session_id][i,config] + 40 - 1
      plt.plot(dfof[0,session_id][a:b, roi], 'k')
    plt.ylabel('dfof')
    plt.xlabel('frames')

## ---------- Extract pre post pairs ----------
def get_pre_post(config,roi):
  config = config-1
  roi = roi-1
  pre = lambda ts: np.mean(ts[0:10])
  post = lambda ts: np.mean(ts[10:40])
  X = np.empty([24,110])
  y = np.empty(24);
  n = 0;
  for session_id in range(3):
    for i in range(8):
      a = times[0,session_id][i,config] - 0 - 1
      b = times[0,session_id][i,config] + 40 - 1
      ts = lambda r: dfof[0,session_id][a:b,r]
      X[n,:] = [pre(ts(r)) for r in range(110)]
      y[n] = post(ts(roi))
      n = n+1
  return X, y

## ---------- Show Model ----------
def show_model(X,y,model,f=None):
  plt.scatter(X,y)
  Xc = np.linspace(*plt.xlim(),100)[:,np.newaxis]
  Xf = X if f is None else f(X)
  Xcf = Xc if f is None else f(Xc)
  model.fit(Xf,y)
  yc = model.predict(Xcf)
  plt.plot(Xc,yc)
  plt.title('Score : ' + str(model.score(Xf,y)))
  
## -------- Leave one out ---------
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error

def leaveoneout(model,X,y):
  loo = LeaveOneOut()
  predictions = []
  actuals = []
  for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions.append(y_pred[0])
    actuals.append(y_test[0])
  return actuals, predictions
  
def get_mse_r2(actuals, predictions):
  mse = mean_squared_error(actuals, predictions)
  r2 = 1 - mse / np.var(actuals)
  return mse, r2

def plot_actuals_predictions(actuals, predictions, label=''):
  mse, r2 = get_mse_r2(actuals, predictions)
  # Plot actual and predicted values
  plt.plot(actuals, predictions, '.', label=f'{label}, R2: {r2}')
  plt.plot(np.sort(actuals), np.sort(actuals), 'k')
  plt.legend()
  plt.xlabel('actuals')
  plt.ylabel('predictions')

## ---------- Get Post-Post Pairs ----------
# Nishal's ground-breaking hypothesis
def get_post_post(config,roi):
  config = config-1
  roi = roi-1
  pre = lambda ts: np.mean(ts[0:10])
  post = lambda ts: np.mean(ts[10:40])
  X = np.empty([24,110-1])
  y = np.empty(24);
  n = 0;
  for session_id in range(3):
    for i in range(8):
      a = times[0,session_id][i,config] - 0 - 1
      b = times[0,session_id][i,config] + 40 - 1
      ts = lambda r: dfof[0,session_id][a:b,r]
      X[n,:] = [post(ts(r)) for r in range(110) if r!=roi]
      y[n] = post(ts(roi))
      n = n+1
  return X, y

## ---------- Find best model ----------
def get_values(config, roi, pp):
  if pp == 'pre-post':
    X, y = get_pre_post(config, roi)
    x = X[:,roi-1,np.newaxis] # for [1-1]
  if pp == 'post-post':
    X, y = get_post_post(config, roi)
    x = y
  return x, X, y

def get_best_model(model_fun, config, roi, pp, spec=[1,10], plot=False, all_in=True, model_name=''):
  x, X, y = get_values(config, roi, pp)
  all_r2 = []
  all_p = []
  if spec is not None:
    ran = np.arange(*spec)
  else:
    ran = [None]
  for p in ran:
    if p is None:
      model = model_fun()
    else:
      model = model_fun(p)
    if not all_in:
      X = x
    ac, pred = leaveoneout(model, X, y)
    if plot:
      label = f'Parameter : {p}'
      plot_actuals_predictions(ac, pred, label)
    mse, r2 = get_mse_r2(ac, pred)
    all_r2.append(r2)
    all_p.append(p)
  best = np.argmax(all_r2)
  best_p = all_p[best]
  best_r2 = all_r2[best]
  if plot:
    plt.title(f'{model_name} [All - 1] [{config},{roi}] [{pp}]')
    plt.figure()
    plt.plot(all_p,all_r2)
    plt.scatter([best_p],[best_r2],c='r')
    plt.xlabel('Parameter')
    plt.ylabel('$R^2$')
    plt.title(f'{model_name} [All - 1] [{config},{roi}] [{pp}]')
    plt.show()
  return best_p, np.round(best_r2,3)
