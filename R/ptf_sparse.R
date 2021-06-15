##########################################################
#' Fit a Probit RESCAL model (sparse representation). 
#' @param df  a four-column dataframe with columns (1) ent1 (2) ent2
#' (3) relation (4) true, which is an indicator of whether there is such a relation from ent1 to ent2 
#' @param n  number of entities
#' @param k  number of relations
#' @param r  decomposition rank 
#' @param max_iter max number of iterations
#' @param tol tolerance of absolute change in ,elihood
#' @param tol_M tolerance of absolute change in the M step
#' @param iter_M_max max number of iterations for M step
#' @param print_option whether print log-likelihood for each iteration or not
#' @return fitted parameters
#' @references Ye Liu, 2021. Computational Methods for Complex Models with Latent Structure. 
#' PhD thesis with link at https://repository.lib.ncsu.edu/bitstream/handle/1840.20/37507/etd.pdf?sequence=1
#' @examples 
#' n <- 20
#' k <- 10
#' r <- 3
#' A <- matrix(runif(n*r,0,1),n,r)
#' mean.k <- runif(k,-0.1,-0.01)
#' W <- lapply(1:k, function(i) matrix(rnorm(r*r,mean.k[i],1),r,r))
#' Z <- lapply(1:k, function(i) A%*%W[[i]]%*%t(A))
#' df <- data.frame(expand.grid(0:(n-1),0:(n-1),0:(k-1)))
#' df <-  df[sample(nrow(df),2000),]
#' colnames(df) <- c('ent1','ent2','relation')
#' awa <- function(row,A,W){
#'   inx1 <- row[1] + 1
#'   inx2 <- row[2] + 1
#'   rel <- row[3] + 1
#'   out <- A[inx1,] %*% W[[rel]] %*% A[inx2,]
#'}
#' z <- apply(df,1,function(row) awa(row,A,W))
#' df$true <- ifelse(z>0,1,0)
#' result2 <- ptf_sparse(df,n,k,r)
#' 
#' @useDynLib ptf
#' @importFrom Rcpp evalCpp
#' @importFrom Matrix sparseMatrix
#' @importFrom plyr count
#' @importFrom rARPACK eigs_sym
#' @export
ptf_sparse <- function(df,n,k,r=0,max_iter=500,tol=1e-8,tol_M=0.00001,iter_M_max=2,print_option=TRUE){
  if(r==0){ # which means the user did not specify r, then use default
    message("r is not specified, using n^{1/3} as rank")
    r = round(n^{1/3})
  }
  
  # get initial values by eigen decomposition
  df.true <- df[df$true==1,]
  counts <- count(df.true, c('ent1','ent2'))
  sumX <- sparseMatrix(i= counts[,1]+1, j= counts[,2]+1, x = counts[,3],
                       dims = c(n,n), symmetric=FALSE)
  sumX <- as.matrix(sumX)
  sum_XXt <- sumX+t(sumX)
  
  eigens <- eigs_sym(sum_XXt,r,which='LM')$vectors
  A_initial <- as.matrix(eigens)
  w_initial <- lapply(1:k, function(i) matrix(rnorm(r^2,0,0.5),r,r))
  w_initial <- array(as.numeric(unlist(w_initial)), dim=c(r,r,k))
  
  A <- A_initial
  w <- w_initial
  df_matrix <- as.matrix(df)
  #E_step_result <- E_step(A,array(as.numeric(unlist(w)), dim=c(r,r,k)),k,n,df_matrix)
  E_step_result <- E_step(A,w,k,n,df_matrix)
  Phi <- E_step_result$Phi
  m <- E_step_result$m
  svdA <- svd(A)
  v <- svdA$v
  sigma <- svdA$d
 
  l <- obs_loglik(Phi,df_matrix)
  iter <- 0
  while(iter < max_iter){
    iter_M <- 0
    old_l <- l
    while(iter_M < iter_M_max){
      ##update w
      inv <- beta_beta_inv(v,sigma,r)
      MA <- Calculate_MA(A,m,df_matrix,r,k,0)
      MtA <- Calculate_MA(A,m,df_matrix,r,k,1)
      BetaM <- Calculate_BetaM(A,m,df_matrix,r,k,MA)
      
      alpha2 <- sigma^2 %*% t(sigma^2)
      w <- lapply(1:k, function(i) vectorize(w[,,i],axis=0)+ 
                      vectorize(v%*%(t(v) %*% t(A)%*% MA[,,i]%*%v/alpha2) %*% t(v),axis=0))
      
      w <- lapply(1:k, function(i) matrix(w[[i]],r,r))
      w <- array(as.numeric(unlist(w)), dim=c(r,r,k))
      
      ##update A
      tmp <- update_A(A,w,m,df_matrix,k,MA,MtA)
      first <- tmp$first_part
      second <- tmp$second_part
      c <- chol(second)
      inv_second <- chol2inv(c)
      A <- first%*%inv_second
      svd_result <- svd(A)
      v <- svd_result$v
      sigma <- svd_result$d
      iter_M <- iter_M + 1
    }
    iter <- iter+1
    #E_step_result <- E_step(A,array(as.numeric(unlist(w)), dim=c(r,r,k)),k,n,df_matrix)
    E_step_result <- E_step(A,w,k,n,df_matrix)
    Phi <- E_step_result$Phi
    m <- E_step_result$m
    l <- obs_loglik(Phi,df_matrix[,c('ent1','ent2','relation','true')])
    if(print_option){
      message(paste0('log-likelihood:',l))
    }
    rel_change <- (l-old_l)/(1+old_l)
    if((abs(rel_change)<tol) & (iter>100)) break
  }
  parameter <- list(A=A,w=w,iter=iter,loglik=l)
  return(parameter)
}

