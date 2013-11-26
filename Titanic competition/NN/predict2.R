predict <- function(X,Theta1,Theta2,Theta3){
  
  #load functions
  source("sigmoid.R")
  
  #useful values
  m=nrow(X)
  
  #Layer 2
  X <- cbind(matrix(1,m),X)#add bias unit
  z2 <-Theta1%*%t(X) 
  a2 <- sigmoid(z2)
  
  #Layer 3
  a2 <- rbind(matrix(1,1,m),a2)#add bias unit
  z3 <- Theta2%*%a2
  a3 <- sigmoid(z3) #dimension: num_labels*m
  
  #layer 4
  a3 <- rbind(matrix(1,1,m),a3)#add bias unit
  z4 <- Theta3%*%a3
  a4 <- sigmoid(z4)
  
  h <- matrix(a4,ncol=1)
  
  #predictions
  pred <-  h>=0.5
    #apply(a3,2,which.max)
    
return (pred)
}
