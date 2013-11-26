#Closing all the plots and resetting all the variables in the workspace
graphics.off()
rm(list = ls())

#====reading data====
data <- read.csv("./data/train.csv")

#====missing values====
good<- complete.cases(data)
cData <- data[good,]
trainData <- cData[,c("survived","pclass","sex","sibsp","parch","fare","embarked","age")]
trainData$sex <- as.numeric(trainData$sex)-1
trainData$embarked <- as.numeric(trainData$embarked)

#====split train set and CV set====
set.seed(2)
rand <- sample(nrow(trainData))
randData <- trainData[rand,]
c <- floor(0.7*nrow(randData))#determines where to cut
trainSet <- randData[1:c,]#70% train set
cvSet <- randData[(c+1):nrow(randData),]#30% CV set

#====Normalize====
#normalize train set
source("./functions/featureNormalize.R")
normalizeTrain <- featureNormalize(trainSet)
trainSet <- normalizeTrain$norm
mu <- normalizeTrain$mu
sigma <- normalizeTrain$sigma
colnames(trainSet) <- c("survived","pclass","sex","sibsp","parch","fare","embarked","age")
#normalize validation set
source("./functions/normVal.R")
cvSet <- normVal(cvSet,mu,sigma)
colnames(cvSet) <- c("survived","pclass","sex","sibsp","parch","fare","embarked","age")
#====define X,y,Xval,yval====
X <- trainSet[,c("survived","pclass","sex","sibsp","parch","fare","embarked")]
y <- trainSet[,c("age")]

Xval <- cvSet[,c("survived","pclass","sex","sibsp","parch","fare","embarked")]
yval <- cvSet[,c("age")]

#====put data into matrix====
X <- as.matrix(X)
y <- as.matrix(y)
Xval <- as.matrix(Xval)
yval <- as.matrix(yval)

#====Load functions====
source("./functions/linearRegCostFunction.R")
source("./functions/linearRegGradFunction.R")

#====setup  some useful values====
m=nrow(X)
n=ncol(X)

#====Add intercept term====
X <- cbind(matrix(1,nrow(X)),X)
Xval <- cbind(matrix(1,nrow(Xval)),Xval)

#====Initialize fitting parameters====
#initialize theta and lambda
initial_theta <- matrix(0,n+1,1)
lambda=0

#====learn theta====
#costFunctionReg(initial_theta,X,y,lambda)
result <- optim(par=initial_theta,fn=linearRegCostFunction,gr=linearRegGradFunction,X=X,y=y,lambda=lambda,
                method="BFGS")
theta <- result$par
cost <- result$value

#====Training error====
sprintf('Train error: %f', linearRegCostFunction(theta,X,y,lambda))

#====Validation error====

sprintf('Validation error: %f', linearRegCostFunction(theta,Xval,yval,lambda))

#====Learning curve====
source("./functions/learningCurve.R")
source("./functions/trainLinearReg.R")
lambda=0
error_train<- learningCurve(X,y,Xval,yval,lambda)$error_train
error_val <- learningCurve(X,y,Xval,yval,lambda)$error_val

#====plot learning curve====
plot(2:m,error_train,
     type="l",
     lwd=1,
     col="blue",
     main="Learning curve for linear regression",
     ylab="Error",
     xlab="Number of training examples",
     cex.lab=0.9,
     xlim=c(0,m),
     ylim=range(error_val))

lines(2:m,error_val,col="green",lwd=1)
legend(400,1.5,c("Train","Cross Validation"),col=c("blue","green"), cex=0.5, xjust=1, lwd=1)


