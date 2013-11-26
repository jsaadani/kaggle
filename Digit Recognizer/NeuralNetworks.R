#remove all variables from workspace
rm(list=ls())

#load functions
source("./functions/sigmoid.R")
source("./functions/predict.R")
source("./functions/nnCostFunction.R")
source("./functions/nnGradFunction.R")

#load data
cat("loading data...\n")
# load("./data/trainSet.rda")
# load("./data/testSet.rda")
load("ex4data1.rda")

X <- as.matrix(X)
y <- as.matrix(y)
#Xtest <- as.matrix(Xtest)

#====Params NN====
#layers sizes
input_layer_size=400
hidden_layer_size=25
num_labels=10

#Initialize parameters
cat("Initialize parameters...\n")
source("./functions/randInitializeWeights.R")
initial_Theta1 <-randInitializeWeights(input_layer_size,hidden_layer_size)
initial_Theta2 <-randInitializeWeights(hidden_layer_size,num_labels)

#unroll parameters
initial_nn_params <- matrix(c(initial_Theta1,initial_Theta2),ncol=1)

#====Training NN====
cat("Training neural networks...\n")
lambda=1
result <- optim(par=initial_nn_params,fn=nnCostFunction,gr=nnGradFunction,
                input_layer_size=input_layer_size,
                hidden_layer_size=hidden_layer_size,
                num_labels=num_labels,
                X=X,y=y,lambda=lambda,
                method="BFGS")
nn_params <- result$par

#reshape nn_params into Theta1 and Theta2
cat("Reshape parameters...")
Theta1 <- nn_params[1:(hidden_layer_size*(input_layer_size+1))]
dim(Theta1) <- c(hidden_layer_size,(input_layer_size+1))
Theta2 <- nn_params[(hidden_layer_size*(input_layer_size+1)+1):length(nn_params)]
dim(Theta2) <- c(num_labels,(hidden_layer_size+1))

#predict
pred <- predict(X, Theta1,Theta2)
cat("Training set Accuracy: ",mean(as.numeric(pred==y)),"\n")
#export predictions
# cat("replacing 10 by 0 in predictions...\n")
# pred[pred==10] <- 0
# save(pred,file="predictions.rda")


