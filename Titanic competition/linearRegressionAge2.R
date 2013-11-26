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

#====define X,y,Xval,yval====
X <- trainSet[,c("pclass","sibsp")]
y <- trainSet[,c("age")]
Xval <- cvSet[,c("pclass","sibsp")]
yval <- cvSet[,c("age")]

#====Map Features====
source("./functions/mapFeature.R")
X <- mapFeature(X[,1],X[,2])
Xval <- mapFeature(Xval[,1],Xval[,2])

#====Normalize====
#=normalize train set
source("./functions/featureNormalize.R")
normalizeTrain <- featureNormalize(X)
X <- normalizeTrain$norm
mu <- normalizeTrain$mu
sigma <- normalizeTrain$sigma
#=normalize validation set
source("./functions/normVal.R")
Xval <- normVal(Xval,mu,sigma)

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

#====initial cost====
cat("initial cost :",linearRegCostFunction(initial_theta,X,y,lambda),"\n")
#Pause
cat("Program paused. Press enter to continue...")
readline()

#====Gradient Descent====
source("./functions/gradientDescent.R")
cat("Running gradient descent...","\n")
par(mfrow=c(2,3))

# for(alpha in c(0.01,0.03,0.1,0.3,1,3)){
  
#set some gradient settings
iterations=400
alpha=0.1
#learn theta
gd <- gradientDescent(X,y,initial_theta,alpha,iterations)
theta <- gd$theta
J_history <- gd$J_hist
#====Print Convergence Graph====
cat("plotting data...","\n")
#par(mfrow=c(2,2))
plot(1:nrow(J_history),J_history,
     type="l",
     lwd=1,
     col="blue",
     main=paste("Convergence Graph -Alpha= ",alpha,sep=""),
     ylab="Cost",
     xlab="Number of iterations",
     cex.lab=0.9,
     xlim=c(0,nrow(J_history)),
     ylim=c(0,500))
#}
#Pause
cat("Program paused. Press enter to continue...")
readline()

#====Print theta and Final cost====
cat("Theta found by gradient descent: ","\t")
cat("Theta:",theta,"\n")
cat("Final Cost:",linearRegCostFunction(theta,X,y,0))

#====learn theta====
linearRegCostFunction(initial_theta,X,y,lambda)
result <- optim(par=initial_theta,fn=linearRegCostFunction,gr=linearRegGradFunction,X=X,y=y,lambda=1,
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
     ylim=c(0,300))

lines(2:m,error_val,col="green",lwd=1)
legend(400,200,c("Train","Cross Validation"),col=c("blue","green"), cex=0.5, xjust=1, lwd=1)


