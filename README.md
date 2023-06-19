# Edible Mushroom Analysis
## Scenario
The practice of mushroom picking is still enjoyed by many people, but wild mushrooms can be very dangerous to eat without knowing exactly which are safe, and which are poisonous. Therefore, building a model to help in the identification of poisonous mushrooms, based on the features of the mushrooms themselves, may help people to stay safe and well. The dataset presents data on a variety of features representing the physical and environment characteristics of mushroom species. Your task is to build a classification model capable of predicting the class of the mushrooms (i.e., whether they are poisonous or edible), and to attempt to identify which features are most indicative of poisonous mushrooms.
## Data overview
First, we read in the dataset and view
```
mushroom <- read.csv('mushrooms.csv')
View(mushroom)
```
After View the mushroom dataset and the dataset description, we can get the following info
Table 1: Dataset info
|Data Set Characteristics:	Multivariate	|Number of Instances:	8124|
|-|-|
|Attribute Characteristics:	Categorical|	Number of Attributes:	22|

For more specified detail, we use:
```
summary(mushroom)
```
<img width="452" alt="image" src="https://github.com/ntg2208/Edible-Mushroom-Analysis/assets/25520448/687bc777-9bd1-471d-b088-16691b64ff0a">


Because all of the variables in the dataset are categorical variable, we will convert them to factor to make the analysis easier.
```
mushroom <- mushroom %>% map_df(function(.x) as.factor(.x))
summary(mushroom)
```
<img width="452" alt="image" src="https://github.com/ntg2208/Edible-Mushroom-Analysis/assets/25520448/05819ce6-d02f-413d-b69c-b0ba0c59a36e">

From observation, we can see that veil.type has only 1 type of value, it will not have any contributed to our prediction and analysis, for more convenience we will remove the veil.type column
```
mushroom$veil.type <- NULL
```
In other to make it easier for modelling and analysis, we will need to convert every column into factor, change the levels of each column into full word instead of letter, for the response variable, we will need to keep response variable “class” as numeric with edible = 1, poison = 0
```
library(tidyverse)
mushroom <- mushroom %>% map_df(function(.x) as.factor(.x))

levels(mushroom$cap.shape) <- c("bell", "conical", "flat", "knobbed", "sunken", "convex")
levels(mushroom$cap.color) <- c("buff", "cinnamon", "red", "gray", "brown", "pink","green", "purple", "white", "yellow")
levels(mushroom$cap.surface) <- c("fibrous", "grooves", "scaly", "smooth")

// ... Repeat for other 18 features

mushroom$class <- ifelse(mushroom$class == "e", 1, 0)
```

## Exploratory Data Analysis
Exploratory data analysis (EDA) is used by data scientists to analyse and investigate data sets and summarize their main characteristics, often employing data visualization methods. It helps determine how best to manipulate data sources to get the answers you need, making it easier for data scientists to discover patterns, spot anomalies, test a hypothesis, or check assumptions. (IBM Cloud, 2020)
### Missing data
Evaluating the quality of the dataset, we will visualize the missingness of the data
```
library(naniar)
vis_miss(mushroom)
```
From Figure 3 we can see that, our mushroom dataset has 100% data presentation, we do not need to do any missing data imputation.

### Independent variables characteristics
We will analyse and comments about the distribution of every value in each column
```
freq = table(mushroom$cap.shape, dnn = 'cap.shape')
count = as.data.frame(freq, responseName = 'count')

ggplot(data=count, aes(x=cap.shape, y=count)) +
  geom_bar(stat="identity", fill="steelblue")+ 
  ggtitle("Mushroom's cap_shape distribution") +
  xlab("cap_shape") + ylab("count") + 
  theme_minimal()
```
<img width="419" alt="image" src="https://github.com/ntg2208/Edible-Mushroom-Analysis/assets/25520448/3c6ac513-3120-42da-b1ec-26fb6c8139b1">

cap.shape has 4 types: bell, conical, flat, knobbed, sunken, convex. Most of the mushroom has flat and convex cap shape


<img width="210" alt="image" src="https://github.com/ntg2208/Edible-Mushroom-Analysis/assets/25520448/965a57bb-a44b-4800-aaf1-64a9996830dd">

cap.surface variable has 4 characteristic, most of mushroom have the cap.surface: fibrous, scaly, and smooth while very little number of mushrooms having grooves cap surface



<img width="419" alt="image" src="https://github.com/ntg2208/Edible-Mushroom-Analysis/assets/25520448/c997e153-82e8-4418-946c-60e08ef495b7">

class describe whether the mushroom is edible or poisons (edible= 1, poisons =0). In our dataset, the number of mushrooms in each class is balance
 

### Relationship between independent variables and dependent variables
Now we will look at the relationship between mushroom features and its class.
```
x <- subset(mushroom, select=c('cap.shape', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(cap.shape,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()
```


<img width="201" alt="image" src="https://github.com/ntg2208/Edible-Mushroom-Analysis/assets/25520448/72fda6b6-785e-4499-b6d7-53b2bf8c2d3a">

From the graph above and APENDIX, we can have some observation:
	Odor has the clearest separation, most of the mushroom which has smell: almond, anise and does not have smell are edible. Other than that, are mostly had poisonous
	All the mushroom which has buff gill colors are poisonous while most of the mushroom with red gill color are safe to eat
	Most of the mushroom which lived as abundant, clustered, and numerous are edible
	Mushroom that has spore print color either chocolate or white is poisonous, in contrast, most of mushroom that has spore print color black, and brown are edible
### Relationship between independent variables
In this section, we will examine some common feature that are easy to be recognized
```
library(ggplot2)
ggplot(mushroom, aes(x = cap.surface, y = cap.color, col = as.factor(class))) + 
  geom_jitter(alpha = 0.5) + 
  scale_color_manual(breaks = c("1", "0"), 
                     values = c("green", "red"))
```


<img width="199" alt="image" src="https://github.com/ntg2208/Edible-Mushroom-Analysis/assets/25520448/94f1a67d-f162-42e8-93f6-eb74815b1e4e">

From above graph, we can see that, all the mushroom with fibrous cap surface, colored red and cinnamon are edible. Stay away from smoothed cap mushroom unless they are purple or green.
  
All the mushroom with red gill or green, purple cap are edible. Stay away from mushroom with buff gill. If you have a yellow cap mushroom, make sure they have black or brown gill, those are safer to eat
## Train/test split
Before we can begin to build any models, we must therefore split our dataset into two smaller subsets. The ratio of training to testing data will vary depending on several factors, including the size of the dataset itself (larger datasets allow for splits more heavily biased in favour of the training data, whereas smaller datasets may have to be more even to ensure that there are enough observations within the testing set), and the types of algorithms that is to be used. The most common splits are 80/20

To perform this split, a random number approach will be used. To achieve this within R, we will allocate a pseudo-random value of either 1 or 0 to every observation within the dataset, with the probability of 0.8 for a 1, and 0.2 (or more accurately, 1 - 0.8), for 0. The first line of the code below assigns the seed value. This sets the seed that is used for the pseudo-random number generation and ensures that the experiment is reproducible when the same seed value is used. The second line performs the value allocation, by creating a new feature called “train”, representing whether an observation will be included within the training data, or the testing data. After done splitting we will remove the “train” feature.
```
set.seed(42)
mushroom[,"train"] <- ifelse(runif(nrow(mushroom))<0.8, 1, 0)
trainset <- mushroom[mushroom$train == "1",]
testset <- mushroom[mushroom$train == "0",]

trainset$train <- NULL
testset$train <- NULL

test_data <- testset[-22]
```
## Model
### XGBoost
We will use gradient boosting to perform classification between edible and poison mushroom. To achieve this, we will be using XGBoost, a powerful boosting algorithm used in many leading-edge applications

XGBoost is short for eXtreme Gradient Boosting package. It supports various objective functions, including regression, classification, and ranking. The package is made to be extendible, so that users are also allowed to define their own objective functions easily. (dmlc, 2022)
Before using XGBoost, we need to convert our training, testing data into matrices, and save training label for later use in xgboost model
```
library(xgboost)

train_matrix <- model.matrix(class~., trainset)
test_matrix <- model.matrix(class~., testset)

train_labels <- trainset$class
```
Now we apply to xgboost model with eat=0.1 (learning rate) nrounds=5 (number of iterations through all training data), max_depth = 3 (maximum depth of a tree), objective = “binary:logistic” (learning objective, using logistic regression for binary classification), verbose = 0 (not echoing the training process)
```
xgbmodel <- xgboost(data = train_matrix, label = train_labels, 
                    eta = 0.1, nrounds = 2, max_depth = 3, 
	 objective = "binary:logistic", 
                    verbose=0)
```
After training the model, we will test xgboost model on test dataset, because binary logistic regression used in the learning objective output as probability of each class, we will need to convert probability to class with threshold=0.5
```
xgb_prob <- predict(xgbmodel, test_matrix, type = "response")
xgb_pred <- ifelse(xgb_prob>0.5, 1, 0)
```
Examining the confusion matrix
```
table(predicted = xgb_pred, actual = testset$class)
```


<img width="69" alt="image" src="https://github.com/ntg2208/Edible-Mushroom-Analysis/assets/25520448/2b2bd190-b49a-4d59-8525-0ae5f1e611d3">

From above confusion matrix, we can observe that, the xgboost model is good, it can correctly identify the edible and the poison in most of the cases. There are 26 mis-classified as poison, while 1 cases mis-classified as edible
In other to evaluate the model, we also need to calculate the Precision, Recall, and Area Under the Curve (AUC)
Precision is the fraction of relevant instances among the retrieved instances
```
Precision=TP/(TP+FN)=822/(822+1)=99.88%
```
Recall is the fraction of relevant instances that were retrieved
```
Recall=TP/(TP+FN)=822/(822+26)=96.93%
```
The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve (Bhandari, 2020). AUC has 2 axes on x-axis is TruePositiveRate (TPR) and on the y-axis is the FalsePositiveRate (FPR)
```
library(ModelMetrics)
library(pROC)

par(pty = "s")
roc(testset$class, xgb_prob, plot = TRUE, col = "red", legacy.axes = TRUE, 
    xlab = "False Positive Rate", ylab = "True Positive Rate", 
			lwd = 2, print.auc = TRUE)
```
<img width="227" alt="image" src="https://github.com/ntg2208/Edible-Mushroom-Analysis/assets/25520448/c9d53ac7-279b-4da9-92e7-b10c77bc88c7">

In the mushroom problem, we need to build a classifier that can classify that mushroom is edible or not. Because our model could be used by anybody even with people who do not have any knowledge about mushroom, and the mis-classified can cause very serious problem. For example, what if there is a mushroom that are poison but mis-classified as edible, it can cause someone poison or even dead. So, we need to improve our model in the way that reduce the number of False Positive cases as much as possible. Reduce False Positive will lead to increase Precision and AUC, so the main object of us is now trying to optimise the model so that it has the highest Precision or AUC
In other to do that, first we will try to change other parameter, we will increase the max_depth of trees, and number of iterations through data during training process
```
xgbmodel <- xgboost(data = train_matrix, label = train_labels, 
                    eta = 0.1, nrounds = 15, max_depth = 5, 
		     objective = "binary:logistic", verbose=0)
```
We changed to number of iterations to 10 (nrounds=15), maximum depth of the trees to 5 (max_depth=5). After test on testset, we get the following confusion matrix

Precision=100%, AUC=1.0 The model has been improved by increase the number of iterations through whole dataset and increase the maximum depth of each tree. With deeper trees, the model can learn more complex data, and by going through the data multiple time, the model has more time to learn the whole dataset.

We can see that, after increased number of iteration and the depth of the trees, we can get maximum Precision and AUC of our classification model as in Figure 32.

To examine why the model has a very good performance and what features that has the most contribution to the XGBoost model
```
importance_matrix = xgb.importance(colnames(train_matrix), model = xgbmodel)
xgb.plot.importance(importance_matrix[1:5,])
```


<img width="167" alt="image" src="https://github.com/ntg2208/Edible-Mushroom-Analysis/assets/25520448/5c186231-0172-42ac-adab-431d10a96440">

The chart above is list of features that has highest information gain to the XGBoost model. We can see that feature odor=none have the highest, it is because in the EDA steps above, odor has the clearest separation between edible and poison. clubbed stalk root and having bruise also have a significant high information gain and help to model to classify whether the mushroom is poisonous or not. From the analysis earlier, most of the mushroom with no smell are not poisonous.

In other to make sure the model is not having a risk of being overfitted, that means the model only learn to memorise the dataset not the general of the data, this is a bad sign, because if in real life, we feed the model with the case that it has never seen before the model could return the wrong prediction, which is not what we want. To overcome this, we will use k-fold cross validation method to train and evaluate the model.

Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample (Brownlee, 2018). Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data. That is, to use a limited sample to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model.

It is a popular method because it is simple to understand and because it generally results in a less biased or less optimistic estimate of the model skill than other methods, such as a simple train/test split.

In cross-validation, we will split the whole dataset into k-fold, so we will use the whole mushroom data for the input of model. We will need to convert data into xgb.DMatrix, with target variable is class.

We will use xgb.cv for XGBoost cross-validation. data used for training is the whole dataset which was converted to xgb.DMatrix, nfold=5 (split the dataset into 5 folds, use 4 folds for training, 1 for testing), nrounds=15, max_depth=5, eta=0.1, nthread=2 (how much resource use for training). metrics= list("auc","map"), (use auc and mean average precision for model evaluation).
```
data <- model.matrix(class~., mushroom)
dtrain <- with(mushroom, xgb.DMatrix(data, label = class))
cv <- xgb.cv(data=dtrain, nrounds = 15, nthread = 2, nfold = 5, metrics = list("auc","map"),
             max_depth = 5, eta = 0.1, objective = "binary:logistic", verbose = 0)
```
After training through 15 epochs, we will evaluate cross-validation model based on its mean of the test_auc_mean and test_map_mean
```
> mean(cv[["evaluation_log"]][["test_auc_mean"]])

[1] 0.9988209
> mean(cv[["evaluation_log"]][["test_map_mean"]])
[1] 0.9995036
```
So, we get mean(AUC)=0.999 and mean(map)=1.0, the cross-validation model is performing good, and very similar to the earlier model with AUC=1.0 but it is safer to use the cross-validated model, because it used cross-validation and prevent model from being overfitted.
### Support Vector Machine
The Support Vector Machine (SVMs) is a supervised learning algorithm mostly used for classification, but it can be used also for regression (Gandhi, 2018). The main idea is that based on the labelled data (training data) the algorithm tries to find the optimal hyperplane which can be used to classify new data points. In two dimensions the hyperplane is a simple line.
Before starting to train new svm, we will create new variable use to store training data, and convert the target variable class into factor, so it will be suitable for svm
```
library(e1071)

svm_trainset <- trainset
svm_trainset$class <- as.factor(svm_trainset$class)
```
We fed the training data into svm model with kernel=’radial’
```
svm_model <- svm(class~., data = svm_trainset, kernel = "radial")
```
We will use the model to predict on testset and get the confusion matrix
```
svm_pred <- predict(svm_model, newdata = test_data, type = "response")
table(predicted = svm_pred, actual = testset$class)
```

From the confusion matrix, we can calculate the Precision and Recall as follow
```
Precision=TP/(TP+FP)=848/(848+5)=99.41%
Recall=TP/(TP+FN)=848/(848+0)=100%
```
And AUC graph


<img width="167" alt="image" src="https://github.com/ntg2208/Edible-Mushroom-Analysis/assets/25520448/fd377485-1f16-419b-b372-221e00e89ff0">

The SVM’s results is good, it has 100% recall, and 99.41% precision and 0.997 auc. But when setting up the model, we use the ‘radial’ kernel, next step we will show the results of trying to change the kernel to see whether it can improve the results
Table 2: SVM kernel performances
|Kernel|	Precision|	Recall	|AUC|
|-|-|-|-|
|radial|	99.41%|	100%|	0.997|
|linear|	100%|	100%	1.0|
|polynomial|	89.09%|	99.17%|	0.930|
|sigmoid|	97.8%|	100%|	0.988|

As table above has shown, most of the svm kernel can classify mushroom with high Precision, Recall and AUC. But in as mentioned above, our focus metric is Precision and AUC, so we can choose svm with linear kernel is our best option.
The model can perform well with linear kernel because some of the features from the dataset are linearly separated. For example, the following plot on Figure 36 has shown that odor, stalk root feature can clearly separate between poisonous and edible. All the mushroom smell like creosote, foul, pungent, spicy, fishy are poisonous. For the stalk root features, unless it has rooted stalk root, it will have some chance of being poisonous in other cases.


<img width="167" alt="image" src="https://github.com/ntg2208/Edible-Mushroom-Analysis/assets/25520448/d0bf5039-1089-4928-af45-d6a18c094768">


### Compare and Conclusion
As mentioned above, because it will be terrible if our models miss classify poison mushroom as edible, it is more importance than miss classify edible into poison, so our focused metric will be model’s precision and AUC. We get the following table

Table 3: Models performance summary
|Model|	Parameters|	Precision|	AUC|
|-|-|-|-|
|xgboost	 | eta = 0.1, nrounds = 15, max_depth = 5, objective = "binary:logistic"	|100%	|1.0|
|Cross-validated xgboost	|nrounds = 15, nthread = 2, nfold = 5, metrics = list("auc","map"), max_depth = 5, eta = 0.1, objective = "binary:logistic"	|100%	|0.999|
|SVM|	Kernel=’radial’	|99.41%|	0.997|
|SVM|	Kernel=’linear’	|100%	|1.0|

We are able to build a classification model that help to identify whether the mushroom is edible or poisonous, with the highest AUC of 1.0 in SVM using linear kernel, and 0.99 when we use cross-validated xgboost model.

The models can achieve very high AUC as well as Recall due to the quality of the mushroom dataset, the mushroom dataset has more than 8000 rows while only have around 22 columns and the data very linearly separated.

In conclusion, poisonous mushroom usually has

- Smell like creosote, foul, spicy, fishy, and pungent.
- Stalk root are missing or bulbous.
- If the mushroom has equal stalk root, you need to check if it has pungent smell or not, all the mushroom with pungent smell are poisonous.
 
For general properties that can be identified by eyes, poisonous mushroom usually has:

- Smooth cap, unless they have colored green or purple
- Buffed gill or greened gill
 
### Bibliography
Bhandari, A. (2020, 06 16). AUC-ROC Curve in Machine Learning Clearly Explained. Retrieved from AnalyticsVidhya: https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/
Brownlee, J. (2018). A Gentle Introduction to k-fold Cross-Validation. Retrieved from Machine Learning Mastery: https://machinelearningmastery.com/k-fold-cross-validation/
dmlc, X. (2022). XGBoost R Tutorial. Retrieved from XGBoost: https://xgboost.readthedocs.io/en/stable/R-package/xgboostPresentation.html
Gandhi, R. (2018). Support Vector Machine — Introduction to Machine Learning Algorithms. Retrieved from Towards Data Science: https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
IBM Cloud, E. (2020, 08 25). What is exploratory data analysis? Retrieved from IBM: https://www.ibm.com/uk-en/cloud/learn/exploratory-data-analysis

