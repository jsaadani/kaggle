#Closing all the plots and resetting all the variables in the workspace
graphics.off()
rm(list = ls())

#====Load functions====
source("./functions/sigmoid.R")
source("./functions/costFunctionReg.R")
source("./functions/gradFunctionReg.R")

#=====Read data====
#load X and y
source("trainSet.R")
#load Xval and yval
source("cvSet.R")
#load Xtest
source("testSet.R")

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

#====learn theta====
#costFunctionReg(initial_theta,X,y,lambda)
result <- optim(par=initial_theta,fn=costFunctionReg,gr=gradFunctionReg,X=X,y=y,lambda=lambda,
                method="BFGS")
theta <- result$par
cost <- result$value

#====Train accuracy====
p <- sigmoid(X%*%theta)>=0.5
trainAccuracy <- mean(p==y)*100
sprintf('Train accuracy: %f', trainAccuracy)
sprintf('Train error: %f', costFunctionReg(theta,X,y,lambda))

#====CV accuracy====
# p <- sigmoid(Xval%*%theta)>=0.5
# cvAccuracy <- mean(p==yval)*100
# sprintf('CV accuracy: %f', cvAccuracy)
# sprintf('Cross Validation error: %f', costFunctionReg(theta,Xval,yval,lambda))

#====Predict Test Set====
predictions <- as.numeric(sigmoid(Xtest%*%theta)>=0.5)
dump("predictions",file="predictions.R")


