mapFeature <- function(X1,X2){

#X1=X[,1]
#X2=X[,2]
mat <- matrix(1,length(X1))
  degree=6
  for(i in 1:degree){
    for(j in 0:i){
      mat <- cbind(mat,X1^(i-j)*(X2^j))
    }
  }
mat <- mat[,2:ncol(mat)]
return(mat)
}