nnCostFunction <- function(nn_params,
                           input_layer_size,
                           hidden_layer_size,
                           num_labels,
                           X,y,lambda){
  
  #load functions
  #source("./functions/sigmoid.R")
  
  #reshape nn_params into Theta1 and Theta2
  Theta1 <- nn_params[1:(hidden_layer_size*(input_layer_size+1))]
  dim(Theta1) <- c(hidden_layer_size,(input_layer_size+1))
  Theta2 <- nn_params[(hidden_layer_size*(input_layer_size+1)+1):length(nn_params)]
  dim(Theta2) <- c(num_labels,(hidden_layer_size+1))
  
  #setup some useful values
  m=nrow(X)
  
  #variables to return
  J=0
  
  #Cost Function
  #Layer 2
  X <- cbind(matrix(1,m),X)#add bias unit
  z2 <-Theta1%*%t(X)
  a2 <- sigmoid(z2)
  
  #Layer 3
  a2 <- rbind(matrix(1,1,m),a2)#add bias unit
  z3 <- Theta2%*%a2
  a3 <- sigmoid(z3) #dimension: num_labels*m
  h <- a3
  for(i in 1:m){
    for(k in 1:num_labels){
      J=J-(y[i]==k)*log(h[k,i])-(1-(y[i]==k))*log(1-h[k,i])
    }
  }
  J=J/m
  
  #Regularized cost function
  Theta1_reg <- Theta1[,2:ncol(Theta1)]
  Theta2_reg <- Theta2[,2:ncol(Theta2)]
  
  cost_Theta1=sum(Theta1_reg^2)
  cost_Theta2=sum(Theta2_reg^2)
  
  J=J+(lambda/(2*m))*(cost_Theta1+cost_Theta2)
  
return(J)
  
}