##########################################################
#' RESCAL Model
#' @param X response data, which is a three-way array of size n by n by k
#' @param k number of relations
#' @param n number of entities
#' @param r decomposition rank 
#' @param max_iter max number of iterations
#' @return fitted parameters
#' @references Nickel, Maximilian, Volker Tresp, and Hans-Peter Kriegel. "A three-way model for collective learning on multi-relational data.", Icml. 2011.
#' @useDynLib ptf
#' @importFrom Rcpp evalCpp
#' @export
#' @name Rescal
Rescal <- function(X,k,n,r,max_iter=500){
  X_Xt_k <- lapply(1:k,function(i) X[[i]]+t(X[[i]]))
  sum_XXt <- Reduce('+', X_Xt_k)
  A_initial <- eigen(sum_XXt)$vectors[,1:r] 
  w_initial <- vector(mode="list",length=k)
  for (i in 1:k){
    w_initial[[i]] <- matrix(0,r,r)
  }
  A <- A_initial
  w <- w_initial
  iter <- 1
  
  while(iter < max_iter){
    # update w
    Z <- kronecker(A,A)
    tmp <- kronecker(t(A)%*%A,t(A)%*%A)
    for (j in 1:k){ 
      w[[j]] <- matrix(solve(tmp,t(Z)%*%as.vector(X[[j]])),r,r)
    }
  
  
    # update A
    tmp2 <- lapply(1:k,function(j) X[[j]]%*%A%*%t(w[[j]])+t(X[[j]])%*%A%*%w[[j]])
    part1 <- Reduce('+', tmp2)
    AtA <- t(A)%*%A
    B <- lapply(1:k,function(j) w[[j]]%*%AtA%*%t(w[[j]]))
    C <- lapply(1:k,function(j) t(w[[j]])%*%AtA%*%w[[j]])
    part2 <- Reduce("+",B)+Reduce("+",C)
    A <- t(solve(t(part2),t(part1)))
    
    iter <- iter+1
  }
   
  parameter <- list(A=A,w=w,iter=iter)
  return(parameter)
  }
  
  