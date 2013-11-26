#Closing all the plots and resetting all the variables in the workspace
graphics.off()
#rm(list = ls())

#Loading functions
source("./functions/sigmoid.R")
source("./functions/costFunctionReg.R")
source("./functions/gradFunctionReg.R")

#Reading data
source("./data/dataProcessed.R")

#splitting data into training set and cross validation set
set.seed(1)
rand <- sample(nrow(dataTrain))
randData <- dataTrain[rand,]
trainSet <- randData[1:500,]
cvSet <- randData[501:714,]

#reading training data into matrix
y <- as.matrix(trainSet[,1])
X <- as.matrix(trainSet[,2:ncol(trainSet)])
yval <- as.matrix(cvSet[,1])
Xval <- as.matrix(cvSet[,2:ncol(cvSet)])

#====Compute Cost====
#setup  some useful values
m=nrow(X)
n=ncol(X)

#Initialize fitting parameters
initial_theta=matrix(0,n,1)
lambda=0

#====learning theta====
costFunctionReg(initial_theta,X,y,lambda)
result <- optim(par=initial_theta,fn=costFunctionReg,gr=gradFunctionReg,X=X,y=y,lambda=lambda,
                method="BFGS")
theta <- result$par
cost <- result$value

#====Train accuracy====
p <- sigmoid(X%*%theta)>=0.5
trainAccuracy <- mean(p==y)*100
sprintf('Train accuracy: %f', trainAccuracy)

#====CV accuracy====
p <- sigmoid(Xval%*%theta)>=0.5
cvAccuracy <- mean(p==yval)*100
sprintf('CV accuracy: %f', cvAccuracy)

#====Test accuracy: Train set with NA====
data <- read.csv("./data/train.csv")
set.seed(2)
rand <- sample(nrow(data))
randData <- data[rand,]
trainSet <- randData[1:500,]
cvSet <- randData[501:714,]

mean(data$age, na.rm=TRUE)



dataTest <- read.csv("./data/test.csv")
str(dataTest)
sum(is.na(dataTest$Age))/418
