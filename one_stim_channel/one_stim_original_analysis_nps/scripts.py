## ----- Imports -----
import sys
sys.path.append('.')
from mylib import *
from sklearn.model_selection import train_test_split, \
cross_val_score

good_pairs = [(3,26),(3,22),(7,31),(8,26),(8,31),(9,26),(9,27),(9,31),(11,26),
              (12,26),(13,22),(14,22),(14,23),
              (15,22),(15,23),(23,21),(24,21),(26,31),(27,31),(28,26),(28,31),
              (29,26),(29,31),(30,26),(30,27),(30,31)]

## ----- Load Pre-Post -----
config, roi = 27, 31
X, y = get_pre_post(config, roi)
x = X[:,roi-1,np.newaxis] # for [1-1]
pp = 'pre - post'

## ----- Load Post-Post -----
X, y = get_post_post(config, roi)
pp = 'post - post'

## ==================== TEST ====================

## ----- Linear [1-1] -----
model = LinearRegression()
show_model(x,y,model)

## ---------- Linear [All - 1] ----------
# Not useful in underetmined case. Not unique.
model = LinearRegression()
model.fit(X,y)
model.score(X,y)



## ==================== CROSS VALIDATION ====================

## --------- Linear - cross validation [All - 1] ---------
# Not useful in underdetermined case. Not unique.
model = LinearRegression()
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean score:", scores.mean())
print("These are random values as it is undetermined case")

## ---------- Polynomial - cross validation [1-1] ----------
print('Polynomial Fitting, single roi input')
for i in range(1,10):
  poly = PolynomialFeatures(degree=i)
  model = LinearRegression()
  xp = poly.fit_transform(x)
  model.fit(xp,y)
  cs = cross_val_score(model, xp, y, cv=5)
  print(i, ':', model.score(xp,y), ':', cs.mean())

## ---------- PLS - cross validation [All - 1] ----------
print('PLS Regression, all roi input')
for i in range(2,10):
  model = PLSRegression(n_components=i)
  model.fit(X,y)
  cs = cross_val_score(model, X, y, cv=5)
  print(i, ':', model.score(X,y), ':', cs.mean())

## ---------- Ridge - cross validation [All - 1] ----------
model = Ridge()
model.fit(X,y)
model.score(X,y)
cross_val_score(model,X,y,cv=5)

## ---------- Lasso - cross validation [All - 1] ----------
model = Lasso(alpha=0.001)
model.fit(X,y)
model.score(X,y)
cross_val_score(model,X,y,cv=5)



## ==================== LEAVE ONE OUT ====================

## Linear [1 - 1]
plt.figure()
label = 'own ROI, linear'
model = LinearRegression()
ac, pred = leaveoneout(model, X, y)
plot_actuals_predictions(ac, pred, label)

# Linear [All - 1]
# Undertermined. Not useful.
label = 'all ROIs, linear'
model = LinearRegression()
leaveoneout(model, X, y, label=label)

# Ridge [All - 1], sweep alpha
get_best_model(Ridge, config, roi, 'pre-post', [0.1,2,0.1], True)

## Lasso [All - 1], sweep alpha
get_best_model(Lasso, config, roi, 'pre-post', [0.001,0.01,0.001], True)

## PLS [All - 1], sweep alpha
get_best_model(PLSRegression, config, roi, 'pre-post', [1,10], True)

## Try configs, rois
def compare_models(config, roi):
  _, r2_lin = get_best_model(LinearRegression, config, roi, 'pre-post', None, False, False)
  _, r2_ridge = get_best_model(Ridge, config, roi, 'pre-post', [0.1,2,0.1], False)
  _, r2_lasso = get_best_model(Lasso, config, roi, 'pre-post', [0.001,0.01,0.001], False)
  _, r2_pls = get_best_model(PLSRegression, config, roi, 'pre-post', [1,10], False)

  _, r2_ridge1 = get_best_model(Ridge, config, roi, 'post-post', [0.1,2,0.1], False)
  _, r2_lasso1 = get_best_model(Lasso, config, roi, 'post-post', [0.001,0.01,0.001], False)
  _, r2_pls1 = get_best_model(PLSRegression, config, roi, 'post-post', [1,10], False)
  print(f'Model       Pre-Post \t Post-Post')
  print('R2 Lin11 : ', r2_lin, '\t', 'NA')
  print('R2 Ridge : ', r2_ridge, '\t', r2_ridge1)
  print('R2 Lasso : ', r2_lasso, '\t', r2_lasso1)
  print('R2 PLS   : ', r2_pls, '\t', r2_pls1)
#  #return [r2_ridge, r2_lasso, r2_pls]
#  return [r2_lin, r2_ridge]
  
## =====
all_r2 = []
for roi in range(20,30):
  for config in range(31):
    all_r2.append(compare_models(config,roi))
    
## =====
lin11_log = []
ridge_all1_log = []

for roi in range(20,30):
  for config in range(31):    
    _, r2_lin = get_best_model(LinearRegression, config, roi, 'pre-post', None, False, False)
    _, r2_ridge = get_best_model(Ridge, config, roi, 'pre-post', [0.1,2,0.1], False)
    lin11_log += [r2_lin]
    ridge_all1_log += [r2_ridge]

plt.figure()   
plt.scatter(lin11_log, ridge_all1_log)
plt.plot([0, 1], [0, 1], 'k')

## ====
lin11_log = []
ridge_all1_log = []
lasso_all1_log = []

for config, roi in good_pairs:
    _, r2_lin = get_best_model(LinearRegression, config, roi, 'pre-post', None, False, False)
    _, r2_ridge = get_best_model(Ridge, config, roi, 'pre-post', [0.1,2,0.1], False)
    _, r2_lasso = get_best_model(Lasso, config, roi, 'pre-post', [0.001,0.01,0.001], False)

    lin11_log += [r2_lin]
    ridge_all1_log += [r2_ridge]
    lasso_all1_log += [r2_lasso]
    
plt.figure()
plt.scatter(lin11_log, ridge_all1_log)
plt.plot([0, 1], [0, 1], 'k')
plt.xlabel('lin 11')
plt.ylabel('ridge')

plt.figure()
plt.scatter(ridge_all1_log, lasso_all1_log)
plt.plot([0, 1], [0, 1], 'k')
plt.xlabel('ridge')
plt.ylabel('lasso')

plt.figure()
plt.scatter(lin11_log, np.max(np.array([ridge_all1_log, lasso_all1_log]), axis=0))
plt.plot([0, 1], [0, 1], 'k')
plt.xlabel('lin 11')
plt.ylabel('best of ridge/lasso')


## ====

compare_models(27,31)

## ==================== GOOD PARAMETERS ====================

# For config, roi = 15, 23
# Ridge [All - 1] : alpha = 2
# Lasso [All - 1] : alpha = 0.0001
# PLS [All - 1] : Components = 

# For config, roi = 27, 31 [pre - post]
# Ridge [All - 1] : alpha = 0.06, r2 = 0.5
# Lasso [All - 1] : alpha = 0.002, r2 = 0.38
# PLS [All - 1] : Components = 5, r2 = 0.238

# For config, roi = 27, 31 [post - post]
# Ridge [All - 1] : alpha = 0.7, r2 = 0.277
# Lasso [All - 1] : alpha = 0.003, r2 = 0.246
# PLS [All - 1] : Components = 1, r2 = 0.21

##