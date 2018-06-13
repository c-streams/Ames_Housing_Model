# Ames_Housing_Model

Background

Ames, Iowa is consistently rated as one of the best places to live in the US. Home to Iowa State University, Ames has a population of 66,000 and is 30 miles north of Des Moines (Figure 1). Ames boasts comprehensive medical care, ample parks and recreation, a stable economy, and an emerging cultural scene [1]. As the city continues to experience growth, there is value in modeling home prices in the area to help prospective builders, sellers, and buyers understand the impact of particular property features  on the sell price of a home.



Figure 1: Scenes of Ames, IA 

Dataset

I was provided a dataset with 79 property features on 2,051 homes sold in Ames from 2006 - 2010. These features included information such as sale price, neighborhood, lot area, building type, year built, overall quality, and garage type.  A detailed data dictionary can be found on my GitHub. At first glance, it becomes obvious that the features are very specific. For example, there is no total bathroom column. Instead there are four separate features for full bath, half bath, basement full bath, and basement half bath. Since this is the case, I will later use feature engineering to consolidate columns and reduce the number of features I will use for modeling. Reducing the number of features considered for modeling will help prevent overfitting.

Data Cleaning 

The dataset required a substantial amount of cleaning. It became apparent that many of the categorical features used NaN as a value. For these features, NaN was mapped to NA (not applicable). For these features, I updated their NaN values with an appropriate string value. For the majority of the remaining NaN values, I was able to infer appropriate values based on other features. I dropped rows for the few NaNs I was unable to infer.

Many of the columns were not accurately represented as either numeric or categorical. I changed non-ordinal columns to categorical (such as month sold and MS Subclass) and I changed ordinal categorical columns to numerics (such as quality scales for pool, fireplaces, garages, etc.).

Year values are tricky since they are not entirely ordinal. It doesn't make sense to input 0  for missing data. I made the year sold categorical since we are only dealing with 5 years, but I will engineer new features to handle year built and year remodeled. I dropped year garage built since there is not an appropriate way to represent missing data (homes with no garage) and there are other garage features that I believe will be more important. 

Feature Engineering 

In order to reduce the number of features and make more general property features, I created a handful of new features. I consolidated bathroom, square footage, and garage features into total bathrooms, total square footage, and has garage features and then dropped the original columns. In order to address year values, I created an age, is remodeled, and is new column to replace the original columns. The additional motivation for this feature engineering is because I believe that these new features represent more realistic factors that people consider when they price a home.

Exploratory Data Analysis 

I first looked at the distribution of sale price, our target variable. From the histogram in Figure 2, its becomes apparent that its distribution is skewed right. This makes sense, since less people are buying expensive homes. Many of the other features are also skewed, which is something that I will address later.



Figure 2: Sale Price distribution 

I next identified the features with the strongest linear relationship with sale price by calculating  Pearson's correlation values (Figure 3).  Overall quality, total square footage, exterior quality, kitchen quality, total bathrooms, basement quality, age, fireplace quality, and masonry veneer area have the strongest linear relationship with sale price. From these top features, we can see that many of them are related, such as exterior quality and overall quality. This is the first hint that our data might have a high degree of multicollinearity, which I will investigate later.



Figure 3: Top 9 most strongly correlated features with Sale Price 

I next investigated these top features further and eliminated outliers where present. I plotted the relationship between sale price and overall quality and sale price and total square footage in Figures 4 and 5 below. From the figures we can see that there are indeed clear linear relationships. 



Figure 4: Box plot of overall quality score vs sale price 





Figure 5: Scatter plot of total square footage vs sale price 

As I mentioned earlier, there are two main issues with the data: skewness and multicollinearity. In order to deal with the former I tested the skew of each numeric feature. We generally want to keep skew scores between -1 and 1. Anything outside this range is considered highly skewed. I took the log of any feature with an absolute skew score above 0.7. Taking the log will normalize a skewed distribution and make it more appropriate for use in a linear regression. In order to handle the multicollinearity, or linearity between features, I compared highly correlated features (r > 0.6), and dropped the feature with the lower correlation to Sale Price. I will also use a regularization model to further reduce multicollinearity when I'm modeling. 

Model Selection 

After correcting for skew, I dummied the categorical data, and then scaled the data since we have different units represented in the dataset. Next I made sure that the columns were the same in the train and test dataframes, as it is possible there are categorical values in the test that are not present in the train.

Once the data was finally prepped, I did k-fold cross validation on unfitted models to get a baseline root mean squared error (RMSE) on linear regression, Lasso regularization, Ridge regularization, and Elastic Net regularization (Figure 6).  Of the four models, Lasso had the lowest baseline RMSE with Elastic Net (high L1 ratio) close behind. I decided to model both to see which performed best.  I suspect that Lasso will perform better since it has an aggressive approach to multicollinearity by eliminating features from the model. 



Figure 6: Formula for root mean squared error 



Modeling

I fitted the training data to both a LassoCV and ElasticNetCV (L1 = 0.9) models and calculated RMSE with the test data. The LassoCV model had a slightly lower RMSE (< $19,000) on the testing data than ElasticNetCV. Figure 7 shows the top 10 features used in the model and the strength of their beta coefficient in the linear model. Total square footage and overall quality are again the top features followed by lot area and overall condition. There is a negative relationship with age, which is also to be expected. It also appears that one of the neighborhoods, Northridge Heights, is a desirable place to live in Ames.

Figure 7: Top 10 features used to predict home price in Lasso model 



Model Evaluation

The Lasso model performs best on cheaper homes and becomes less accurate on expensive ones (Figure 8). Lower representation of high end homes in the training data likely accounts for this difference. This model can now be used to predict future home sale prices in Ames, IA given certain property features.



Figure 8: Actual vs predicted sale price

Next Steps

In summary, Figure 9 lays out the top 10 indicators for home price in Ames, Iowa that my model uses. This model is helpful to those in real estate as well as builders and sellers, who want to better understand the housing market in Ames.



Figure 9: Top 10 indicators of home price in Lasso model 

For next steps, I would like to further refine my model in a few different ways. I would do additional feature engineering to consolidate and reduce number of features. My final model used around 80 features, which is still relatively high, indicating that the model might be too complex. I would also like to optimize algorithm hyperparameters using a GridSearch. Lastly I would like to explore using different algorithms such as XGBoost.   



References 

[1] https://www.cityofames.org/about-ames


