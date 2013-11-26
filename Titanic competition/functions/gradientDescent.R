gradientDescent <- function(X,y,theta,alpha,num_iters){
  
  source("./functions/linearRegCostFunction.R")
  
  
  m=nrow(y)
  n=ncol(X)
  J_history <- matrix(0,num_iters,1)
  
  
  for(iter in 1:num_iters){
    
    predictions=X%*%theta
    temp=theta
    for(i in 1:n){
      temp[i] <- theta[i]-(alpha/m)*sum((predictions-y)*X[,i])
    }
    theta=temp
    J_history[iter] <- linearRegCostFunction(theta,X,y,lambda=0)
  }
  
  gradientDescent <- list("theta"=theta,"J_hist"=J_history)
  return(gradientDescent)
  
}