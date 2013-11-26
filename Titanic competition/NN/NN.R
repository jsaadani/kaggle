#remove all variables from workspace
rm(list=ls())

#load functions
source("sigmoid.R")
source("predict.R")
source("nnCostFunction.R")
source("nnGradFunction.R")

#====load data====
#load X,y,Xval,yval,Xtest
cat("loading training data X and y \n")
load("NN_data3.rda")

#====NN_parameters====
input_layer_size=8
hidden_layer_size=32
num_labels=1

#Initializing parameters
source("randInitializeWeights.R")
initial_Theta1 <-randInitializeWeights(input_layer_size,hidden_layer_size)
initial_Theta2 <-randInitializeWeights(hidden_layer_size,num_labels)

#unroll parameters
initial_nn_params <- matrix(c(initial_Theta1,initial_Theta2),ncol=1)

#====check regularization====
#  lambda=0
#  J=nnCostFunction(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda)
#  cat("Cost with initial parameters (without regularization): ",J,"\n")

# #pause
# cat("Program paused. Press enter to continue...\n")
# readline()
#====Training NN====
cat("Training neural networks...\n")
lambda=0
result <- optim(par=initial_nn_params,fn=nnCostFunction,gr=nnGradFunction,
                input_layer_size=input_layer_size,
                hidden_layer_size=hidden_layer_size,
                num_labels=num_labels,
                X=X,y=y,lambda=lambda,
                method="BFGS",
                control=list(maxit=1000))
nn_params <- result$par

#reshape nn_params into Theta1 and Theta2
Theta1 <- nn_params[1:(hidden_layer_size*(input_layer_size+1))]
dim(Theta1) <- c(hidden_layer_size,(input_layer_size+1))
Theta2 <- nn_params[(hidden_layer_size*(input_layer_size+1)+1):length(nn_params)]
dim(Theta2) <- c(num_labels,(hidden_layer_size+1))

cat("Program pause. Press enter to continue...\n")
readline()

#predict
pred <- predict(X, Theta1,Theta2)
cat("Training set Accuracy: ",mean(as.numeric(pred==y))*100,"\n")
cat("Train error: ",nnCostFunction(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,
                                   X,y,0),"\n")

pred <- predict(Xval,Theta1,Theta2)
cat("Validation set Accuracy: ",mean(as.numeric(pred==yval))*100,"\n")
cat("CV error: ",nnCostFunction(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,
                                   Xval,yval,0),"\n")

# #====Learning curve====
# cat("Plotting learning curve...\n")
# source("learningCurveNN.R")
# lambda=0
# error_train<- learningCurveNN(X,y,Xval,yval,lambda,
#                                     input_layer_size,
#                                     hidden_layer_size,
#                                     num_labels)$error_train
# error_val <- learningCurveNN(X,y,Xval,yval,lambda,
#                              input_layer_size,
#                              hidden_layer_size,
#                              num_labels)$error_val
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
#      ylim=range(0,1))
# 
# lines(2:m,error_val,col="green",lwd=1)

# #====validation for selecting lambda====
# cat("plotting validation curve...","\n")
# source("validationCurveNN.R")
# vc <- validationCurveNN(X,y,Xval,yval,
#                         input_layer_size,
#                         hidden_layer_size,
#                         num_labels)
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
#      ylim=c(0.3,0.5))
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

#test set
# pred <- as.numeric(predict(Xtest,Theta1,Theta2))
# save(pred,file="predictions_NN_2.rda")