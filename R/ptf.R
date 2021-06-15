##########################################################
#' Fit a Probit Tensor Factorization Model
#' @param X  response data, which is a three-way array of size n by n by k
#' @param k  number of relations
#' @param n  number of entities
#' @param r  decomposition rank 
#' @param max_iter max number of iterations
#' @param tol tolerance of absolute change in likelihood
#' @param tol_M tolerance of absolute change in the M step
#' @param iter_M_max max number of iterations for M step
#' @param print_option whether print loss for each iteration or not
#' @return fitted parameters 
#' @references  @references Ye Liu, 2021. Computational Methods for Complex Models with Latent Structure. 
#' PhD thesis with link at https://repository.lib.ncsu.edu/bitstream/handle/1840.20/37507/etd.pdf?sequence=1
#' @examples 
#' n <- 20
#' k <- 10
#' r <- 3
#' p <- c(n, n, k)
#' X <- array(rnorm(prod(p)),dim=p)
#' X_binary <- ifelse(X < -1.5,1,0)
#' X_binary_with_missing <- X_binary
#' num_missing <- 200
#' missing_index <- data.frame(x1=sample(1:n,num_missing,replace=TRUE), 
#'                            x2=sample(1:n,num_missing,replace=TRUE),  
#'                            x3=sample(1:k,num_missing,replace=TRUE))
#' for(i in 1:num_missing){
#'  X_binary[missing_index[i,1],
#'           missing_index[i,2],
#'           missing_index[i,3]] <- NA
#'}
#'result <- ptf(X_binary_with_missing,k,n,r,print_option=FALSE) 

#' @useDynLib ptf
#' @importFrom Rcpp evalCpp
#' @importFrom stats rnorm
#' @export
ptf <- function(X,k,n,r=0,max_iter=1000,tol=1e-8,tol_M=0.00001,iter_M_max=2,print_option=TRUE){
  
  if(r==0){ # which means the user did not specify r, then use default
    message("r is not specified, using n^{1/3} as the rank")
    r = round(n^{1/3})
  }
  
  # initialize 
  X_Xt_K <- lapply(1:k,function(i) X[,,i]+t(X[,,i]))
  sum_XXt <- Reduce('+', X_Xt_K)
  A_initial <- eigen(sum_XXt)$vectors[,1:r] 
  w_initial <-array(data=0,c(r,r,k))

  for (i in 1:k){
    w_initial[,,i] <- matrix(rnorm(r^2,0,0.1),r,r)
  }
  
  A <- A_initial
  w <- w_initial
  iter <- 1
  X_F2 <- (sum(sapply(1:k, function(j) norm(X[,,j],"F"))))^2
  Z <- Expectation(X,A,w,k)
  l <- loss(Z,w,A,k)/X_F2
  while(iter < max_iter){
    Z <- Expectation(X,A,w,k)
    iter_M <- 0
    old_l <- l
    l <- loss(Z,w,A,k)/X_F2
    while(iter_M < iter_M_max){
      iter_M <- iter_M+1
      w <- UpdateW(Z,w,A,k)
      A <- UpdateA(Z,w,A,k)
      }
    l <- loss(Z,w,A,k)/X_F2
    iter <- iter+1
    if(print_option){
      message(paste0('loss', l))
      }
    
    if(l-old_l<tol & iter>100) break
  }
  parameter <- list(A=A,w=w,iter=iter,loglik=l)
  return(parameter)
}

