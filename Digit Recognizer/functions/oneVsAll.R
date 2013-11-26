oneVsAll <- function(X,y,num_labels,lambda){
  
  #load functions
  source("./functions/gradFunctionReg.R")
  source("./functions/costFunctionReg.R")
  source("./functions/sigmoid.R")
  
  #put data into matrix
  X <- as.matrix(X)
  y <- as.matrix(y)
  
  
  #some useful variables
  m=nrow(X)
  n=ncol(X)
  lambda=lambda
  
  #variable to return
  all_theta <- matrix(0,num_labels,n+1)
  
  #add the intercept term
  X <- cbind(matrix(1,m),X)
  
  #set initial theta
  initial_theta <- matrix(0,n+1,1)
  c <- 1:num_labels
  
  #====learn theta====
  for (i in c){
  result <- optim(par=initial_theta,fn=costFunctionReg,gr=gradFunctionReg,X=X,y=(y==i),lambda=lambda,
                  method="BFGS")
  all_theta[i,] <- result$par
  #cost <- result$value
  }
  return(all_theta)
}