#resetting all the variables in the workspace
rm(list=ls())

#load data
cat("loading data...\n")
load("./data/trainSet.rda")
load("./data/testSet.rda")

#====put data into matrix====
X <- as.matrix(X)
y <- as.matrix(y)
Xtest <- as.matrix(Xtest)


#set some useful values
num_labels=10
lambda=0.1

#====learn Theta====
source("./functions/oneVsAll.R")
cat("Training One-vs-All Logistic Regression...\n")
all_theta <- oneVsAll(X,y,num_labels,lambda)

# #Pause
# cat("Program paused. Press enter to continue...")
# readline()

#====Predict One Vs All=====
source("./functions/predictOneVsAll.R")
cat("predicting train set...\n")
pred <- predictOneVsAll(all_theta,X)
cat("Training set accuracy: ",mean(as.numeric(pred==y)),"\n")

#====Predict test Set====
cat("predicting test set...\n")
pred <-  predictOneVsAll(all_theta,Xtest)
#replace 10 by 0 in pred
cat("replacing 10 by 0 in predictions...\n")
pred[pred==10] <- 0
#export pred
cat("exporting predictions into file...\n")
save(pred,file="predictions1.rda")
#end
cat("THE END\n")

