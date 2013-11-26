#Closing all the plots and resetting all the variables in the workspace
#graphics.off()
rm(list = ls())

#====load data====
#source("./data/dataProcessedLinearAge/trainAgeReplaced.R")

#====Load functions====
source("./functions/sigmoid.R")
source("./functions/costFunctionReg.R")
source("./functions/gradFunctionReg.R")

#=====Read data====
#load X,y,Xval,yval,Xtest
load("./logistic regression/trainValidationTestSet.rda")


#====Compute Cost====
#setup  some useful values
m=nrow(X)
n=ncol(X)

#====Add intercept term====
X <- cbind(matrix(1,m),X)
Xval <- cbind(matrix(1,nrow(Xval)),Xval)
Xtest <- cbind(matrix(1,nrow(Xtest)),Xtest)

#====Initialize fitting parameters====
#initialize theta and lambda
initial_theta=matrix(0,n+1,1)
lambda=0

#====initial cost====
cat("initial cost :",costFunctionReg(initial_theta,X,y,0),"\n")
#Pause
# cat("Program paused. Press enter to continue...")
# readline()

#====learn theta====
cat("learning theta...","\n")
result <- optim(par=initial_theta,fn=costFunctionReg,gr=gradFunctionReg,X=X,y=y,lambda=lambda,
                method="BFGS")
theta <- result$par

#====Train accuracy====
p <- sigmoid(X%*%theta)>=0.5
trainAccuracy <- mean(p==y)*100
cat('Train accuracy: ', trainAccuracy,"\n")
cat('Train error: ', costFunctionReg(theta,X,y,0),"\n")

#====CV accuracy====
p <- sigmoid(Xval%*%theta)>=0.5
cvAccuracy <- mean(p==yval)*100
cat('CV accuracy: ', cvAccuracy,"\n")
cat('CV error: ', costFunctionReg(theta,Xval,yval,0),"\n")

# #Pause
# cat("Program paused. Press enter to continue...")
# readline()

# #====Learning curve====
# cat("Plotting learning curve...")
# source("./functions/learningCurveLogistic.R")
# #lambda=0
# error_train<- learningCurveLogistic(X,y,Xval,yval,lambda)$error_train
# error_val <- learningCurveLogistic(X,y,Xval,yval,lambda)$error_val
# 
# #====plot learning curve====
# plot(2:m,error_train,
#      type="l",
#      lwd=1,
#      col="blue",
#      main="Learning curve for logistic regression",
#      ylab="Error",
#      xlab="Number of training examples",
#      cex.lab=0.9,
#      xlim=c(0,m),
#      ylim=range(0,1.5))
# 
# lines(2:m,error_val,col="green",lwd=1)
# legend(600,0.2,c("Train","Cross Validation"),col=c("blue","green"), cex=0.5, xjust=1, lwd=1)
# 
# #Pause
# cat("Program paused. Press enter to continue...")
# readline()

# #====validation for selecting lambda====
# cat("plotting validation curve...","\n")
# source("./functions/validationCurveLogistic.R")
# vc <- validationCurveLogistic(X,y,Xval,yval)
# error_train <- vc$error_train
# error_val <- vc$error_val
# lambda_vec <- vc$lambda_vec
# 
# #====plot validaion curve====
# plot(lambda_vec,error_train,
#      type="l",
#      lwd=1,
#      col="blue",
#      main="validation curve",
#      ylab="Error",
#      xlab="lambda",
#      cex.lab=0.9,
#      xlim=c(0,max(lambda_vec)),
#      ylim=c(0.3,0.6))
# 
# lines(lambda_vec,error_val,col="green",lwd=1)
# # # legend(400,150,c("Train","Cross Validation"),col=c("blue","green"), cex=0.5, xjust=1, lwd=1)
# 
# cat("Lambda\t\tTrain error\t\tValidation error\n")
# for(i in 1:length(lambda_vec)){
#   cat(lambda_vec[i],"\t\t\t")
#   cat(error_train[i],"\t\t\t")
#   cat(error_val[i],"\n")
# }

#====Predict Test Set====
# predictions <- as.numeric(sigmoid(Xtest%*%theta)>=0.5)
# dump("predictions",file="predictions2.R")


