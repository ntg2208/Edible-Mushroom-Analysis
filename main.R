# Read in the dataset
library(tidyverse)
mushroom <- read.csv('mushrooms.csv')
View(mushroom)
str(mushroom)
# table(mushroom$veil.color)
## We make each variable as a factor
mushroom <- mushroom %>% map_df(function(.x) as.factor(.x))
summary(mushroom)

str(mushroom)

mushroom$veil.type <- NULL

## We redefine each of the category for each of the variables
# levels(mushroom$class) <- c(0, 1)
levels(mushroom$cap.shape) <- c("bell", "conical", "flat", "knobbed", "sunken", "convex")
levels(mushroom$cap.color) <- c("buff", "cinnamon", "red", "gray", "brown", "pink", 
                                "green", "purple", "white", "yellow")
levels(mushroom$cap.surface) <- c("fibrous", "grooves", "scaly", "smooth")
levels(mushroom$bruises) <- c("no", "yes")
levels(mushroom$odor) <- c("almond", "creosote", "foul", "anise", "musty", "none", "pungent", "spicy", "fishy")
levels(mushroom$gill.attachment) <- c("attached", "free")
levels(mushroom$gill.spacing) <- c("close", "crowded")
levels(mushroom$gill.size) <- c("broad", "narrow")
levels(mushroom$gill.color) <- c("buff", "red", "gray", "chocolate", "black", "brown", "orange", 
                                 "pink", "green", "purple", "white", "yellow")
levels(mushroom$stalk.shape) <- c("enlarging", "tapering")
levels(mushroom$stalk.root) <- c("missing", "bulbous", "club", "equal", "rooted")
levels(mushroom$stalk.surface.above.ring) <- c("fibrous", "silky", "smooth", "scaly")
levels(mushroom$stalk.surface.below.ring) <- c("fibrous", "silky", "smooth", "scaly")
levels(mushroom$stalk.color.above.ring) <- c("buff", "cinnamon", "red", "gray", "brown", "pink", 
                                             "green", "purple", "white", "yellow")
levels(mushroom$stalk.color.below.ring) <- c("buff", "cinnamon", "red", "gray", "brown", "pink", 
                                             "green", "purple", "white", "yellow")
# levels(mushroom$veil.type) <- "partial"
levels(mushroom$veil.color) <- c("brown", "orange", "white", "yellow")
levels(mushroom$ring.number) <- c("none", "one", "two")
levels(mushroom$ring.type) <- c("evanescent", "flaring", "large", "none", "pendant")
levels(mushroom$spore.print.color) <- c("buff", "chocolate", "black", "brown", "orange", 
                                        "green", "purple", "white", "yellow")
levels(mushroom$population) <- c("abundant", "clustered", "numerous", "scattered", "several", "solitary")
levels(mushroom$habitat) <- c("wood", "grasses", "leaves", "meadows", "paths", "urban", "waste")

library(naniar)
vis_miss(mushroom)

# mushroom$class <- as.numeric(mushroom$class)
mushroom$class <- ifelse(mushroom$class == "e", 1, 0)

# independent variable analysis 
#-----------
library(ggplot2)

# plot(mushroom$cap.shape)
freq = table(mushroom$class , dnn = 'class')
count = as.data.frame(freq, responseName = 'count')

ggplot(data=count, aes(x=class, y=count)) +
  geom_bar(stat="identity", fill="steelblue")+ 
  ggtitle("Mushroom's class distribution") +
  xlab("class") + ylab("count") + 
  theme_minimal()

plot(mushroom$cap.shape, ylab='count', xlab='cap.shape')
plot(mushroom$cap.surface, ylab='count', xlab='cap.surface')
plot(mushroom$cap.color, ylab='count', xlab='cap.color')

plot(mushroom$bruises, ylab='count', xlab='bruise')
plot(mushroom$odor, ylab='count', xlab='odor')

plot(mushroom$gill.attachment, ylab='count', xlab='gill.attachment')
plot(mushroom$gill.spacing, ylab='count', xlab='gill.spacing')
plot(mushroom$gill.size, ylab='count', xlab='gill.size')
plot(mushroom$gill.color, ylab='count', xlab='gill.color')

plot(mushroom$stalk.shape, ylab='count', xlab='stalk.shape')
plot(mushroom$stalk.root, ylab='count', xlab='stalk.root')
plot(mushroom$stalk.color.above.ring, ylab='count', xlab='stalk.color.above.ring')
plot(mushroom$stalk.color.below.ring, ylab='count', xlab='stalk.color.below.ring')
plot(mushroom$stalk.surface.above.ring, ylab='count', xlab='stalk.surface.above.ring')
plot(mushroom$stalk.surface.below.ring, ylab='count', xlab='stalk.surface.below.ring')

plot(mushroom$veil.color, ylab='count', xlab='veil.color')

plot(mushroom$ring.number, ylab='count', xlab='ring.number')
plot(mushroom$ring.type, ylab='count', xlab='ring.type')

plot(mushroom$spore.print.color, ylab='count', xlab='spore.print.color')

plot(mushroom$population, ylab='count', xlab='population')

plot(mushroom$habitat, ylab='count', xlab='habitat')

# between independent variable
# relationship between independent variable and dependent variable
#----------
x <- subset(mushroom, select=c('cap.shape', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(cap.shape,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('cap.surface', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(cap.surface,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('cap.color', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(cap.color,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('bruises', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(bruises,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('odor', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(odor,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('gill.attachment', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(gill.attachment,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('gill.spacing', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(gill.spacing,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('gill.size', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(gill.size,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('gill.color', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(gill.color,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('stalk.shape', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(stalk.shape,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('stalk.root', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(stalk.root,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('stalk.surface.above.ring', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(stalk.surface.above.ring,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('stalk.surface.below.ring', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(stalk.surface.below.ring,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('stalk.color.above.ring', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(stalk.color.above.ring,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('stalk.color.below.ring', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(stalk.color.below.ring,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('veil.color', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(veil.color,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('ring.number', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(ring.number,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('stalk.color.below.ring', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(stalk.color.below.ring,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('ring.type', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(ring.type,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('spore.print.color', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(spore.print.color,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('population', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(population,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()

x <- subset(mushroom, select=c('habitat', 'class'))
tbl <- table(x)
ggplot(as.data.frame(tbl, responseName = 'count'), aes(habitat,count, color = class, fill=class)) + 
  geom_bar(stat="identity", position = "dodge") + theme_minimal()
# Independent variable together
#----------------------
library(ggplot2)
ggplot(mushroom, aes(x = odor, y = bruises, col = as.factor(class))) + 
  geom_jitter(alpha = 0.5) + 
  scale_color_manual(breaks = c("1", "0"), 
                     values = c("green", "red"))


# Split into only edible and poisson
#------------
edible <- filter(mushroom, class == "1" )
summary(edible)
poisson <- filter(mushroom, class == "0" )
summary(poisson)

# Modeling
# -----------------------------------------------------------------
set.seed(42)
mushroom[,"train"] <- ifelse(runif(nrow(mushroom))<0.8, 1, 0)
trainset <- mushroom[mushroom$train == "1",]
testset <- mushroom[mushroom$train == "0",]

trainset$train <- NULL
testset$train <- NULL

test_data <- testset[-22]
# -----------------------------------------------------------------
library(rpart)
library(rpart.plot)

model_tree <- rpart(class~., data = trainset, method = "class")

model_tree
rpart.plot(model_tree, extra = 106)

test_data <- testset[-22]

tree_pred <- predict(model_tree, newdata = test_data, type = "class")

table(predicted = tree_pred, actual = testset$class)
mean(tree_pred==testset$class)
# ---------------------
penalty_matrix <- matrix(c(0, 1, 10, 0), byrow = TRUE, nrow = 2)
model_tree_penalty <- rpart(edibility ~ ., data = train_mushroom, method = "class", 
                            parms = list(loss = penalty_matrix))

caret::confusionMatrix(data=predict(model_tree_penalty, type = "class"), 
                       reference = train_mushroom$edibility, 
                       positive="edible")

model_tree <- rpart(edibility ~ ., data = train_mushroom, 
                    method = "class", cp = 0.00001)

printcp(model_tree)

plotcp(model_tree)

model_tree$cptable[which.min(model_tree$cptable[, "xerror"]), "CP"]

bestcp <- round(model_tree$cptable[which.min(model_tree$cptable[, "xerror"]), "CP"], 4)
model_tree_pruned <- prune(model_tree, cp = bestcp)

rpart.plot(model_tree_pruned, extra = 104, box.palette = "GnBu", 
           branch.lty = 3, shadow.col = "gray", nn = TRUE)

caret::confusionMatrix(data=predict(model_tree_pruned, type = "class"), 
                       reference = train_mushroom$edibility, 
                       positive="edible")

test_tree <- predict(model_tree, newdata = test_mushroom)
caret::confusionMatrix(data = predict(model_tree, newdata = test_mushroom, type = "class"), 
                       reference = test_mushroom$edibility, 
                       positive = "edible")
# ---------------------------------------------------
library(party)
library(ModelMetrics)
library(pROC)

forest_model <- cforest(class~., data = trainset, 
                        control = cforest_unbiased(mtry = 10, ntree = 50))

rf_prob <- predict(forest_model, newdata = testset, type = "response")

rf_pred <- ifelse(rf_prob>0.5, 1, 0)

table(predicted = rf_pred, actual = testset$class)

mean(rf_pred==testset$class)*100

ForestVarImp <- varimp(forest_model)
barplot(ForesrVarImp)

par(pty = "s")
f1Score(actual=testset$class, predicted = rf_pred, cutoff = 0.5)
mcc(actual=testset$class, predicted = as.numeric(rf_pred), cutoff = 0.5)
roc(testset$class, rf_prob, plot = TRUE, col = "red", legacy.axes = TRUE, 
    xlab = "False Positive Rate", ylab = "True Positive Rate", 
    lwd = 2, print.auc = TRUE)

tmp = as.data.frame(ForestVarImp)
df <- cbind(features = rownames(tmp), tmp)
rownames(df) <- 1:nrow(df)
barplot(df$ForestVarImp~df$features)

#-------------------------------------------------------
library(e1071)

svm_trainset <- trainset
svm_trainset$class <- as.factor(svm_trainset$class)

svm_model <- svm(class~., data = svm_trainset, kernel = "radial")
svm_model <- svm(class~., data = svm_trainset, kernel = "polynomial")
summary(svm_model)

svm_pred <- predict(svm_model, newdata = test_data, type = "response")

table(predicted = svm_pred, actual = testset$class) 
mean(svm_pred==testset$class)

par(pty = "s")
roc(testset$class, as.numeric(svm_pred), plot = TRUE, col = "red", legacy.axes = TRUE, 
    xlab = "False Positive Rate", ylab = "True Positive Rate", lwd = 2, print.auc = TRUE)


f1Score(actual=testset$class, predicted = svm_pred, cutoff = 0.5)
mcc(actual=testset$class, predicted = as.numeric(rf_pred), cutoff = 0.5)

# ----------------------------------------------
library(xgboost)

# mushroom <- rs_function(mushroom)

# mushroom$class <- class_response
train_matrix <- model.matrix(class~., trainset)
test_matrix <- model.matrix(class~., testset)

train_labels <- trainset$class


xgbmodel <- xgboost(data = train_matrix, label = train_labels, 
                    eta = 0.1, nrounds = 15, max_depth = 5, objective = "binary:logistic", 
                    verbose=0)

xgb_prob <- predict(xgbmodel, test_matrix, type = "response")
xgb_pred <- ifelse(xgb_prob>0.5, 1, 0)

table(predicted = xgb_pred, actual = testset$class)
mean(xgb_pred==testset$class)

par(pty = "s")
roc(testset$class, xgb_prob, plot = TRUE, col = "red", legacy.axes = TRUE, 
    xlab = "False Positive Rate", ylab = "True Positive Rate", lwd = 2, print.auc = TRUE)

importance_matrix = xgb.importance(colnames(train_matrix), model = xgbmodel)
importance_matrix
xgb.plot.importance(importance_matrix[1:5,], xlab='Information Gain')
# ---------------------
data <- model.matrix(class~., mushroom)
dtrain <- with(mushroom, xgb.DMatrix(data, label = class))
cv <- xgb.cv(data=dtrain, nrounds = 15, nthread = 2, nfold = 5, metrics = list("auc","map"),
             max_depth = 5, eta = 0.1, objective = "binary:logistic", verbose = 0)
print(cv)
mean(cv[["evaluation_log"]][["test_auc_mean"]])
mean(cv[["evaluation_log"]][["test_map_mean"]])
print(cv, verbose=TRUE)
