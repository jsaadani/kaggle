#remove all variables from workspace
rm(list=ls())

#load functions
source("sigmoid.R")
source("predict2.R")
source("nnCostFunction2.R")
source("nnGradFunction2.R")

#====load data====
#load X,y,Xval,yval,Xtest
cat("loading training data X and y \n")
load("NN_data2.rda")#80% train data, 20% validation data

#====NN_parameters====
input_layer_size=7
hidden_layer_size_1=14
hidden_layer_size_2=14
num_labels=1

#Initializing parameters
source("randInitializeWeights2.R")
initial_Theta1 <-randInitializeWeights(input_layer_size,hidden_layer_size_1)
initial_Theta2 <-randInitializeWeights(hidden_layer_size_1,hidden_layer_size_2)
initial_Theta3 <-randInitializeWeights(hidden_layer_size_2,num_labels)

#unroll parameters
initial_nn_params <- matrix(c(initial_Theta1,initial_Theta2,initial_Theta3),ncol=1)

#====check regularization====
 lambda=0
 J=nnCostFunction(initial_nn_params,input_layer_size,hidden_layer_size_1,
                  hidden_layer_size_2,num_labels,X,y,lambda)
 cat("Cost with initial parameters (without regularization): ",J,"\n")

#pause
cat("Program paused. Press enter to continue...\n")
readline()
#====Training NN====
cat("Training neural networks...\n")
lambda=0.1
result <- optim(par=initial_nn_params,fn=nnCostFunction,gr=nnGradFunction,
                input_layer_size=input_layer_size,
                hidden_layer_size_1=hidden_layer_size_1,
                hidden_layer_size_2=hidden_layer_size_2,
                num_labels=num_labels,
                X=X,y=y,lambda=lambda,
                method="BFGS",
                control=list(maxit=1000))
nn_params <- result$par

#reshape nn_params into Theta1 and Theta2

len_Theta1=hidden_layer_size_1*(input_layer_size+1)
len_Theta2=hidden_layer_size_2*(hidden_layer_size_1+1)
len_Theta3=num_labels*(hidden_layer_size_2+1)

Theta1 <- nn_params[1:len_Theta1]
dim(Theta1) <- c(hidden_layer_size_1,input_layer_size+1)
Theta2 <- nn_params[(1+len_Theta1):(len_Theta1+len_Theta2)]
dim(Theta2) <- c(hidden_layer_size_2,(hidden_layer_size_1+1))
Theta3 <- nn_params[(1+len_Theta1+len_Theta2):length(nn_params)]
dim(Theta3) <- c(num_labels,(hidden_layer_size_2+1))

cat("Program pause. Press enter to continue...\n")
readline()

#Train accuracy
pred <- predict(X, Theta1,Theta2,Theta3)
cat("Training set Accuracy: ",mean(as.numeric(pred==y)),"\n")
cat("Training error: ",nnCostFunction(nn_params,input_layer_size,
                                      hidden_layer_size_1,hidden_layer_size_2,
                                      num_labels,X,y,0),"\n")

#Cross Validation accuracy
pred <- predict(Xval,Theta1,Theta2,Theta3)
cat("Validation set Accuracy: ",mean(as.numeric(pred==yval)),"\n")
cat("Training error: ",nnCostFunction(nn_params,input_layer_size,
                                      hidden_layer_size_1,hidden_layer_size_2,
                                      num_labels,Xval,yval,0),"\n")


#test set
pred <- as.numeric(predict(Xtest,Theta1,Theta2,Theta3))
save(pred,file="predictions_NN_3.rda")