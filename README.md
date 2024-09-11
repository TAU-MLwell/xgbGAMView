## `class xgbGAMView (* , param={})`

### Description:
> `class xgbGAMView` utilizes the Generalized Additive Model (GAM) technique to establish an interpretable model and offers an easy-to-understand visual presentation of the model's predictive mechanism. The GAM technique combines predictors linearly, while allowing each predictor to have a non-linear relationship with the response variable. This flexibility enables the exploration of complex relationships within datasets, revealing trends and providing local explanations for predictions. `class xgbGAMView` employs decision tree algorithm implemented through "XGBoost", an optimized distributed gradient boosting library, and is designed for flexible modeling of regression, binary classification, and survival analysis tasks. The visual presentation, represented by figures of the features' shape functions, allows for easy examination and comparison of how different features influence the predicted values.

### Parameters:
> **param :** *dict, optional (default={})*
> Dictionary of hyperparameters for the XGBoost model. The following conditions must be met:
> 'max_depth' should not be in the dictionary. The maximum depth is automatically set to 1.
> 'objective' must be one of ['reg:squarederror', 'binary:logitraw', 'survival:cox'] if specified.
>   
> Note:  
> While 'survival:cox' in XGBoost returns predictions on hazard ratio scale (i.e., as HR = exp(marginal_prediction) in the proportional hazard function h(t) = h0(t) * HR), `xgbGAMView` returns only the marginal_prediction.

### Methods:
> #### `set_beta(beta)`:  
>>  Sets the beta parameter used for smooth shape function predictions.  
>> The model is based on decision trees with depth of 1, means that each tree has single node with single condition. Each tree has a "yes score," which is the score added to the predicted value if the data meets its condition, and a "no score", which is added if the data does not meet the condition. Therefore, data points with similar values may have same contributions if they meet the same conditions, leading to shape functions that are piecewise constant. In order to obtain smooth shape functions, when beta is set to a value other than `None`, the scores are being interpolated with sigmoid:  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*smooth score* = *no score* + (*yes score* - *no score*) ⋅ sigmoid(*beta* ⋅ Δ)  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Δ = *split value* - *current value* &nbsp;&nbsp;&nbsp;&nbsp; *beta* = constant
>> - *smooth score* - the score added to the predicted value after the interpolation
>> - *no score* - the score obtained by the tree if the data does not meet the condition
>> - *yes score* - the score obtained by the tree if the data meets the condition
>> - *beta* - constant that controls the steepness of the sigmoid curve
>> - *split value* - the value of the tree's condition
>> - *current value* - the value of the relevant feature in the given data
>> ##### Parameters:
>>> **beta :** *int, float, dict, or None*  
>>> Specifies the beta constant for each feature.  
>>> If a float or an int is provided, it applies to all features.  
>>> If a dictionary is provided, it should map feature names to desired beta value.  
>>> If `None` there will not be an interpolation with sigmoid.
>>> 
>> ##### Returns:
>>> Returns the updated instance.

> #### `fit(X, y, train_param={})`:
>> Train the model on the provided data.
>> ##### Parameters:
>>> **X :** *array-like or pandas.DataFrame*
>>> Training data.
>>>
>>> **y :** *array-like*
>>> Target values.
>>>
>>> **train_param :** *dict, optional (default={})*
>>> Additional parameters for training the XGBoost model.
>> ##### Returns:
>>> Returns the instance itself.

> #### `predict(X)`:
>> Make predictions using the trained model.
>> ##### Parameters:  
>>> **X :** *array-like or pandas.DataFrame*  
>>> Input data for making predictions.  
>> ##### Returns:  
>>> The predicted values: *numpy.ndarray*  

> #### `feature_contribution(X)`:  
>> Calculate the contribution of each feature to the prediction for data X.  
>> ##### Parameters:  
>>> **X :** *array-like or pandas.DataFrame*  
>>> Input data for calculating feature contributions.  
>> ##### Returns:  
>>> DataFrame containing the value and corresponding contribution of each feature: *pandas.DataFrame*  

> #### `plot(*, name='xgbGAMView', bandwidth=0.2, features=[], ctg_or_cnt={}, n_density_samples=200)`:
>> Generate plots for each feature showing its shape function, represents its contribution to the prediction relative to an offset value.
>> The plots will also indicate the density distribution of the data.
>>
>> **Plots presentation:**  
>> - Categorical features are represented using bar graphs, while continuous features are depicted with scatter plots.  
>> - In scatter plots: green marks - positive contribution, red marks - negative contribution.
>>   
>> **Density presentation:**  
>> - In scatter plots, the background color intensity varies according to the density of the feature values.  
>> - For bar graphs, the density is reflected by the color of the bars.  
>> - The density's color scale is standardized across all plotted features, ensuring consistent interpretation of density levels.
>>   
>> **Scaling**  
>> - y axis limits in all the plots are standardized across all plotted features for easy comparison.  
>> 
>> ##### Parameters:  
>>> **name :** *str, optional (default='xgbGAMView')*  
>>> The name of the directory where plots will be saved.  
>>>   
>>> **bandwidth :** *float or dict, optional (default=0.2)*  
>>> The bandwidth used for Kernel Density Estimation (KDE) to estimate data density when plotting continuous features.
>>> If a float is provided, it applies to all features.  
>>> If a dictionary is provided, it should map feature names to desired bandwidth. 
>>>  
>>> **features :** *list of str, optional (default=[])*  
>>> List of feature names to be plotted. If empty, all features are plotted.  
>>>  
>>> **ctg_or_cnt :** *dict or str, optional (default={})*  
>>> Specifies whether each feature is 'Categorical' or 'Continuous'.  
>>> If a string is provided, it applies to all features.  
>>> If a dictionary is provided, it should map feature names to 'Categorical' or 'Continuous'.
>>> As default, if feature contains 10 unique values or less it will be considered as 'Categorical'; Else, 'Continuous'.
>>>  
>>> **n_density_samples :** *int, optional (default=200)*  
>>> Number of samples used for Kernel Density Estimation (KDE) to represent density distribution when plotting continuous features.
>>>  
>> ##### Returns:  
>>> Shape functions figures.

            
