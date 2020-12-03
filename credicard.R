library(tidyverse)
library(randomForest)
library(rpart)
library(rpart.plot)
library(rattle)
library(caTools)
library(tictoc)
library(imbalance)
library(ROCR)
library(caret)


####Prepare data
total <- rbind(read_csv("C:\\Users\\guozh\\Desktop\\fraudTrain.csv")[,-1], read_csv("C:\\Users\\guozh\\Desktop\\fraudTest.csv")[,-1])

total$trans_month <- format(as.Date(total$trans_date_trans_time), format="%b")
total$owner_age <- as.numeric(
    format(as.Date(total$trans_date_trans_time), format="%Y")) - 
    as.numeric(format(as.Date(total$dob), format="%Y"))

cols <- c("is_fraud", "trans_month", "category", "gender", "state")
total[cols] <- lapply(total[cols], factor) 

####Train and Test, 75:25
set.seed(6969)
total$spl <- sample.split(seq_len(nrow(total)), SplitRatio = 0.75)

data.train <- subset(total, spl == TRUE)[,-25]
data.test <- subset(total, spl == FALSE)[,-25]

# write_csv(data.train, "fraudTrain_real.csv")
# write_csv(data.test, "fraudTest_real.csv")

data.train.lite <- data.train %>% select(is_fraud, trans_month, category, amt, gender, owner_age)

data.test.lite <- data.test %>% select(is_fraud, trans_month, category, amt, gender, owner_age)

####RF
memory.limit(size = 500000)

tic("RF")
forest <- randomForest(data = data.train.lite, is_fraud ~., ntree = 200)
toc()
forest$confusion
importance(forest)
plot(forest$votes[,2], col =  data.train.lite$is_fraud)

forest.t <- predict(forest, type='class' , newdata = data.test.lite)
table(actual = data.test.lite$is_fraud , predicted = forest.t)

month <-ggplot(data.train.lite, aes(x=weight)) + 
    geom_histogram(color="black", fill="white")

#### adjusted parameter
# forest <- randomForest(data = data.train.lite, is_fraud ~., ntree = 200, cutoff=c(.99,.01))

#### Reg

# reg = glm(is_fraud ~ trans_month + category + amt + gender + owner_age + state, family = 'binomial', data = data.train)
# 
# summary(reg)
# 
# reg.pred.train = predict(reg, type ='response')
# reg.pred.train.stats = ifelse(reg.pred.train > 0.5, '1', '0')
# table(actual = data.train$is_fraud , predicted = reg.pred.train.stats)
# 
# 
# predict_p_reg_machine_t = predict(reg_machine , type ='response', newdata = heart_test)
# predict_stat_reg_machine_t = ifelse(predict_p_reg_machine_t > 0.5, 'diagnosis_Yes', 'diagnosis_No')
# table(actual = heart_test$result , predicted = predict_stat_reg_machine_t)

#### Tree

tree = rpart(is_fraud ~ trans_month + category + amt + gender + owner_age + state, method = 'class', cutoff=list(c(0.5, 0.5)), data = data.train)

split.fun <- function(x, labs, digits, varlen, faclen)
{
    # replace commas with spaces (needed for strwrap)
    labs <- gsub(",", " ", labs)
    for(i in 1:length(labs)) {
        # split labs[i] into multiple lines
        labs[i] <- paste(strwrap(labs[i], width=30), collapse="\n")
    }
    labs
}

rpart.plot(tree, split.fun=split.fun)





tree.pred <- predict(tree, type ='class')
actual <- data.train$is_fraud

pred <- prediction(as.numeric(tree.pred), as.numeric(actual))

# Recall-Precision curve             
RP.perf <- performance(pred, "prec", "rec")
plot(RP.perf)

# ROC curve
ROC.perf <- performance(pred, "tpr", "fpr")
plot (ROC.perf)

#caret
precision <- posPredValue(tree.pred, actual, positive="1")
recall <- sensitivity(tree.pred, actual, positive="1")



#
measurePrecisionRecall <- function(predict, actual_labels){
    precision <- sum(predict & actual_labels) / sum(predict)
    recall <- sum(predict & actual_labels) / sum(actual_labels)
    fmeasure <- 2 * precision * recall / (precision + recall)
    
    cat('precision:  ')
    cat(precision * 100)
    cat('%')
    cat('\n')
    
    cat('recall:     ')
    cat(recall * 100)
    cat('%')
    cat('\n')
    
    cat('f-measure:  ')
    cat(fmeasure * 100)
    cat('%')
    cat('\n')
}
measurePrecisionRecall(as.numeric(tree.pred), as.numeric(actual))




###
table(actual = data.train$is_fraud , predicted = tree.pred)

predicted_status_tree_manual_t = predict(tree_manual, type='class', newdata = heart_test)
table(actual = heart_test$result , predicted = predicted_status_tree_manual_t)



####oversampling 
total.OS <- total[,-25]
total.OS <- total.OS %>% select(-is_fraud,is_fraud)
set.seed(9560)

total.OS <- upSample(x = total.OS[, -ncol(total.OS)],
                     y = total.OS$is_fraud)                         
table(total.OS$Class) 


total.OS$spl <- sample.split(seq_len(nrow(total.OS)), SplitRatio = 0.75)

data.train.OS <- subset(total.OS, spl == TRUE)[,-25]
data.test.OS <- subset(total.OS, spl == FALSE)[,-25]

# Tree with oversample data
tree.up = rpart(Class ~ trans_month + category + amt + gender + owner_age + state, method = 'class', data = data.train.OS)

rpart.plot(tree, split.fun=split.fun)

#Training E
tree.up.pred <- predict(tree.up, type ='class')
table(actual = data.train.OS$Class , predicted = tree.up.pred)

measurePrecisionRecall(as.numeric(tree.up.pred), as.numeric(data.train.OS$Class))


predicted_status_tree_manual_t = predict(tree.up, type='class', newdata = data.test.OS)
table(actual = data.test.OS$Class, predicted = predicted_status_tree_manual_t)

# data.train.OF$Class <- "positive"
# data.train.OF$Class[data.train.OF$is_fraud == 0] <- "negative"
# data.train.OF$Class<- as.factor(data.train.OF$Class)
# 
# newRACOG <- racog(data.train.OF, numInstances = 100)
# 
# data.train.OFed <- oversample(data = data.train.OF, ratio = 1, method = "RACOG")


split.fun <- function(x, labs, digits, varlen, faclen)
{
    # replace commas with spaces (needed for strwrap)
    labs <- gsub(",", " ", labs)
    for(i in 1:length(labs)) {
        # split labs[i] into multiple lines
        labs[i] <- paste(strwrap(labs[i], width=30), collapse="\n")
    }
    labs
}