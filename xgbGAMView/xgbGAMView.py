# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import json
import os
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.neighbors import KernelDensity
from sklearn.base import BaseEstimator

class xgbGAMView(BaseEstimator):
    def __init__(self, *, param = {}):
        assert "max_depth" not in param, 'max_depth cannot be inserted! Please remove max_depth from param.'
        assert 'objective' not in param or param['objective'] in ['reg:squarederror', 'binary:logitraw', 'survival:cox'], 'objective must be "reg:squarederror", "binary:logitraw" or "survival:cox" but is ' + param['objective']
        param['max_depth'] = 1
        self.param = param
        self.beta = None # This is the parameter that will be used to make a soft prediction
        self.is_fitted_ = False

    def set_beta(self, beta):
        assert beta is None or isinstance(beta, (int, float, dict)) , 'beta must be int, float, dict or None but is ' + type(beta).__name__
        self.beta = beta
        return self

    def fit(self, X, y, train_param = {}):
        # Convert X to DataFrame if necessary
        if type(X) == np.ndarray:
          n_columns = np.shape(X)[1]
          column_names = [f'f{i}' for i in range(n_columns)]
          X = pd.DataFrame(X, columns=column_names)
        # This is where the model is trained
        self.X = X
        self.y = y
        dmatrix_X = xgb.DMatrix(X, label=y)
        self.model = xgb.train(self.param, dmatrix_X, **train_param)
        self.is_fitted_ = True
        # Extract trees to DataFrame
        self.trees_df = self.model.trees_to_dataframe()
        # Save base score
        config = self.model.save_config()
        self.config_dict = json.loads(config)
        self.base_score = float(self.config_dict.get('learner').get('learner_model_param').get('base_score'))
        self.obj = self.config_dict.get('learner').get('objective').get('name')
        # find h0 for survival analysis
        if self.obj == 'survival:cox':
          self.h0 = None
          n = X.shape[1]
          Z = np.zeros((1, n))
          beta_in = self.beta
          beta_tmp_none = None
          self.set_beta(beta_tmp_none)
          pred_xgb = self.predict(Z)
          beta_tmp_dict = {}
          self.set_beta(beta_tmp_dict)
          pred_model = self.predict(Z)
          h0 = pred_xgb / np.exp(pred_model)
          self.h0 = h0
          self.set_beta(beta_in)
        return self
    
    def predict(self, X):
        # This is where the model predicts
        # The parameter beta is used to make a soft version of the prediction.
        # If beta is None, the model will use the hard prediction
        if self.beta == None:
          if type(X) == np.ndarray:
            column_names = self.X.columns
            X = pd.DataFrame(X, columns=column_names)
          X_dmatrix = xgb.DMatrix(X)
          booster_type = self.config_dict.get('learner').get('gradient_booster').get('name')
          if booster_type == 'dart':
            n_trees = self.model.num_boosted_rounds()
            y_pred = self.model.predict(X_dmatrix ,iteration_range = (0,n_trees))
          else:
            y_pred = self.model.predict(X_dmatrix)
          if self.obj == 'survival:cox':
            if self.h0 != None:
              y_pred = np.log(y_pred / self.h0)
          return y_pred
        else:
          # This is the calculation for smooth prediction: smooth_score = no_score + (yes_score - no_score) * sigmoid(beta * (split_value - current_value))
          if type(X) == pd.core.frame.DataFrame:
            X = X.values
          scores = np.zeros(X.shape)
          i = 0
          const_score = 0
          trees = self.trees_df
          while (i < len(trees)):
            tree_feature = trees.loc[i, 'Feature']
            if tree_feature == 'Leaf':
              const_score += trees.loc[i , 'Gain']
              i = i + 1
              continue
            split = trees.loc[i,'Split']
            yes_score = trees.loc[i + 1, 'Gain']
            no_score = trees.loc[i + 2, 'Gain']
            i = i + 3
            column_indx = self.X.columns.get_loc(tree_feature)
            curr = X[: , column_indx]
            delta = split - curr
            if type(self.beta) == dict:
              if tree_feature in self.beta:
                beta = self.beta[tree_feature]
              else:
                beta = None
            else:
              beta = self.beta
            if beta == None:
              sigmoid_score = delta > 0
            else:
              x = beta * delta
              sigmoid_score = 1/(1 + np.exp(-x))
            smooth_score = no_score + (yes_score - no_score) * sigmoid_score
            smooth_score = np.nan_to_num(smooth_score, nan = no_score) # if value is missing, its score will be the no_score
            scores[: , column_indx] += smooth_score
          smooth_pred = np.sum(scores, axis=1)
          smooth_pred += const_score
          smooth_pred += self.base_score
          return smooth_pred
    
    def feature_contribution(self, X):
      features_names = self.X.columns
      n_tot_features = len(features_names)
      if type(X) == pd.core.frame.DataFrame:
        X = X.values
      assert np.shape(X) == (1, n_tot_features) , 'Feature shape mismatch, expected: (1, ' + str(n_tot_features) + '), got: ' + str(np.shape(X))
      A = np.zeros((1, n_tot_features))
      offset = self.predict(A)[0]
      contribution = pd.DataFrame(np.full((2,n_tot_features), np.nan) , columns = features_names, index = ['Value','Contribution'])
      for i in range(n_tot_features):
        A = np.zeros((1, n_tot_features))
        curr_value = X[0, i]
        A[0, i] = curr_value
        curr_contribution = self.predict(A)
        curr_contribution -= offset
        contribution.iloc[0,i] = curr_value
        contribution.iloc[1,i] = curr_contribution
      return contribution
    
    def plot(self, *, name = 'xgbGAMView', bandwidth = 0.2, features = [], ctg_or_cnt = {}, n_density_samples = 200):
        # This is where the model plots are generated
        # use assert if model is not trained
        assert self.is_fitted_ == True, 'Model is not trained.'
        # Set plots path
        if (not(os.path.exists("plots"))):
            os.mkdir("plots")
        plots_path = os.path.join("plots", name)
        if (not(os.path.exists(plots_path))):
            os.mkdir(os.path.join("plots", name))
        # Initial settings - GAM
        n_tot_features = len(self.X.columns)
        if features == []:
          features_names = self.X.columns
        else:
          features_names = features
        n_features = len(features_names)
        A = np.zeros((1, n_tot_features))
        offset = self.predict(A)[0]
        all_preds = []
        all_vals = []
        all_density = []
        all_types = []
        max_density = 0
        # Estimate the contribution of each feature - GAM
        for i in range(n_features):
            # extract uniuqe values and counts
            curr_name = features_names[i]
            curr_feat = self.X[curr_name]
            uniq = curr_feat.value_counts().sort_index()
            vals = uniq.index
            counts = uniq.values
            # set feature type - categorical or continuous
            if ctg_or_cnt == 'Categorical':
              feat_type = 'Categorical'
            elif ctg_or_cnt == 'Continuous':
              feat_type = 'Continuous'
            elif type(ctg_or_cnt) is dict:
              if curr_name in ctg_or_cnt:
                feat_type = ctg_or_cnt[curr_name]
                assert feat_type in ['Categorical','Continuous'] , 'ctg_or_cnt['+curr_name+'] must be "Categorical" or "Continuous" but is ' + ctg_or_cnt[curr_name]
              else:
                if len(vals) <= 10:
                  feat_type = 'Categorical'
                else:
                  feat_type = 'Continuous'
            else:
              assert False, 'Invalid ctg_or_cnt'
            # create density function
            if feat_type == 'Categorical':
              density = counts / sum(counts)
              mx_density = max(density)
            elif feat_type == 'Continuous':
              if type(bandwidth) == dict:
                if curr_name in bandwidth:
                  curr_bandwidth = bandwidth[curr_name]
                else:
                  curr_bandwidth = 0.2
              else:
                curr_bandwidth = bandwidth
              density = KernelDensity(kernel = 'gaussian', bandwidth = curr_bandwidth)
              density.fit(pd.Series.to_numpy(curr_feat).reshape(-1, 1))
              mx_density = max(np.exp(density.score_samples(pd.Series.to_numpy(curr_feat).reshape(-1, 1))))
            # GAM process
            A = np.zeros((len(vals), n_tot_features))
            indx = self.X.columns.get_loc(curr_name)
            A[:, indx] = vals
            pred = self.predict(A)
            pred = [p - offset for p in pred]
            # update variables
            all_preds.append(pred)
            all_vals.append(vals)
            all_density.append(density)
            all_types.append(feat_type)
            if mx_density > max_density:
              max_density = mx_density
        # General settings - plotting
        max_pred = max(max(pred) for pred in all_preds)
        min_pred = min(min(pred) for pred in all_preds)
        delta_y = max_pred - min_pred
        ylim_min, ylim_max = (min_pred - delta_y * 0.1, max_pred + delta_y * 0.1)
        norm = Normalize(vmin=0, vmax=max_density)
        # Set white to steelblue gradient
        cdict = {'red':   [(0.0,  1.0, 1.0),
                               (1.0,  0.275, 0.275)],
                 'green': [(0.0,  1.0, 1.0),
                               (1.0, 0.51, 0.51)],
                 'blue':  [(0.0,  1.0, 1.0),
                               (1.0,  0.706, 0.706)]}
        # Create the colormap
        steelbluemap = LinearSegmentedColormap('steelblue', cdict, 256)
        #
        for i in range(n_features):
            pred = all_preds[i]
            vals = all_vals[i]
            density = all_density[i]
            feat_type = all_types[i]
            max_x = max(vals)
            min_x = min(vals)
            delta_x = max_x - min_x
            xlim_min, xlim_max = (min_x - delta_x * 0.05, max_x + delta_x * 0.05)
            # Generate plot
            if (not(np.any(pred))):
               continue
            fig, ax = plt.subplots(figsize=(10, 5))
            vals = pd.Series.to_numpy(vals)
            pred = np.array(pred)
            if feat_type == 'Continuous':
              # Sampling for density
              x_density = np.linspace(xlim_min, xlim_max, n_density_samples)
              y_density = np.exp(density.score_samples(x_density.reshape(-1, 1)))
              # plot
              plt.imshow(y_density.reshape(1,-1), aspect='auto', cmap=steelbluemap, norm=norm, extent=[xlim_min, xlim_max, ylim_min, ylim_max])
              plt.colorbar(label='Density')
              threshold = 0
              plt.plot(vals, pred, color='black',linestyle='--', linewidth=0.4)
              plt.plot(vals[pred > threshold], pred[pred > threshold],marker='o', markersize=2, linestyle='none', color='forestgreen')
              plt.plot(vals[pred <= threshold], pred[pred <= threshold], marker='o', markersize=2, linestyle='none', color='maroon')
              plt.xlim(xlim_min, xlim_max)
            elif feat_type == 'Categorical':
              colors = steelbluemap(norm(density))
              plt.bar(range(len(vals)), pred, color = colors, edgecolor = "black",linewidth=0.5)
              plt.xticks(ticks=range(len(vals)), labels = vals)
              plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=steelbluemap), label='Density', ax=ax)
            plt.ylim(ylim_min, ylim_max)
            plt.title(features_names[i], fontsize=15)
            plt.axhline(0, color='black', linestyle='-' , linewidth=0.5)
            # Save plot
            plt.savefig(os.path.join(plots_path, features_names[i] + ".png"))
        return(offset)
