// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

#include <stdio.h>
#include <iostream>
#include <RcppArmadillo.h>
#include <math.h>  // for -INFINITY, NAN, isnan()

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

//
// [[Rcpp::depends(RcppArmadillo)]]


///////////////////////////////////////
// [[Rcpp::export]]
arma::mat get_varphi(arma::mat s, arma::vec a){
    int n = a.n_rows;
    int p = s.n_cols + 1;
    
    arma::mat result(n, p * 2);
    int i,j = 0;
    
    for(i=0; i<n; i++){
        for(j=0; j<p; j++){
            if(j==0){
                result(i, 0) = 1;
                result(i, p) = a(i);
            }
            else{
                result(i, j) = s(i, j-1);
                result(i, p+j) = s(i, j-1) * a(i);
            }
        }
    }
    return result;
}


///////////////////////////////////////
// [[Rcpp::export]]
arma::mat get_varphi_next(arma::mat snew, arma::vec decisionparm){
    int n = snew.n_rows;
    int p = snew.n_cols + 1;
    arma::vec quant;
    int anext;
    arma::mat result(n, p * 2);
    
    arma::vec row(p);
    row.zeros();
    
    int i,j = 0;
    
    for(i=0; i<n; i++){
        
        for(j=1; j<p; j++){
            row(j) = snew(i, j-1);
        }
        
        quant = row.t() * decisionparm;
        if(quant(0) > 0){
            anext = 1;
        }
        else{
            anext = 0;
        }
        
        for(j=0; j<p; j++){
            if(j==0){
                result(i, 0) = 1;
                result(i, p) = anext;
            }
            else{
                result(i, j) = snew(i, j-1);
                result(i, p+j) = snew(i, j-1) * anext;
            }
        }
        
    }
    return result;
}

///////////////////////////////////////
// [[Rcpp::export]]
arma::mat get_delta(arma::mat varphi, arma::mat varphi_next,
                    arma::vec theta0, arma::vec reward, double gamma){
    int n = varphi.n_rows;
    arma::vec result(n);
    result =  reward + gamma * (varphi_next * theta0) - varphi * theta0;
    return result;
}

 


/////////////////////////
// [[Rcpp::export]]
arma::vec maei_c_cont(arma::mat s, arma::mat snew, arma::vec reward, arma::vec a,
                      double gamma, arma::vec theta0, arma::vec stepsize,
                 double pow1, double pow2, int maxit, double tol, bool verbose){
    int n = s.n_rows;
    int p = s.n_cols + 1;
    int iter;
    double c1, c2, err;
    arma::vec decisionparm(p);
    
    arma::vec w0(p*2);
    arma::vec theta(p*2);
    arma::vec w(p*2);
    
    arma::vec summand1(p*2);
    arma::vec summand2(p*2);
    arma::vec temp(p*2);
    arma::vec tempunit;
    
    arma::mat regmat(p*2, p*2);
    regmat.zeros();
    int i, j;
    
    for(i=0; i<p*2; i++){
        regmat(i,i) = 0.0001;
    }
    
    for(i=0; i<p; i++){
        decisionparm(i) = theta0(p+i);
    }
    
    arma::mat varphi = get_varphi(s, a);
    arma::mat varphi_next = get_varphi_next(snew, decisionparm);
    arma::vec delta = get_delta(varphi, varphi_next, theta0, reward, gamma);
    
    arma::mat part1 = arma::inv(varphi.t() * varphi + regmat);
    
    arma::mat part2 = varphi.t() * delta;
    
    w0 = part1 * part2;
    
    for(iter=2; iter<maxit; iter++){
        c1 = pow(iter, pow1);
        c2 = pow(iter, pow2);
        
        varphi = get_varphi(s, a);
        varphi_next = get_varphi_next(snew, decisionparm);
        delta = get_delta(varphi, varphi_next, theta0, reward, gamma);
        
        summand1.zeros();
        summand2.zeros();
        for(i=0; i<n; i++){
            tempunit = varphi.row(i) * w0;
            
            temp = delta(i) * varphi.row(i).t() - gamma * tempunit(0) * varphi_next.row(i).t();
            
            summand1 = summand1 * i / (i+1) + temp / (i+1);
            
            temp = delta(i) - tempunit(0) * varphi.row(i).t();
            
            summand2 = summand2 * i / (i+1) + temp / (i+1);
            
        }
        
        for(j=0; j<p*2; j++){
            theta(j) = theta0(j) + stepsize(j) * c1 * summand1(j);
            w(j) = w0(j) + stepsize(j) * c2 * summand2(j);
        }
        
        
        err = 0.0;
        for(j=0; j<p*2; j++){
            err += (theta(j) - theta0(j)) * (theta(j) - theta0(j));
        }
        err = sqrt(err);
        if(err <= tol) break;
        
        theta0 = theta;
        for(j=0; j<p; j++){
            decisionparm(j) = theta0(p+j);
        }
        w0 = w;
        
        if( (verbose==TRUE) & (iter % 10 == 0)) Rcpp::Rcout << iter << ", " << err << ":" << theta.t() << "\n";
    }
    
    return theta;
}

/////////////////////////
// [[Rcpp::export]]
arma::vec vectorize(arma::mat mat, int axis){
    int row = mat.n_rows;
    int col = mat.n_cols;
    int i,j;
    arma::vec result(row*col);
    for(i=0;i<row;i++){
        for(j=0;j<col;j++){
            if(axis==0)  result(i+j*row) = mat(i,j);  // by column
            else result(j+i*col) = mat(i,j); //by row
        }
    }
    return result;
}

/////////////////////////
// [[Rcpp::export]]
arma::mat vec2mat(arma::vec vec, int row, int col, int axis){
    int i,j;
    arma::mat result(row, col);
    for(i=0; i<row; i++){
        for(j=0; j<col; j++){
            if(axis==0)  result(i,j) = vec(i+j*row) ; //by column
            else  result(i,j) = vec(j+i*col);     // by row
        }
    }
    return result;
}

///////////////////////////////////////
// [[Rcpp::export]]
arma::vec colsum(arma::mat matrix){
    long row = matrix.n_rows;
    int col = matrix.n_cols;
    long i;
    int j;
    arma::vec columnsum(col);
    
    for(j=0; j<col; j++) columnsum(j) = 0;
    for(i=0; i<row; i++){
        for(j=0; j<col; j++){
            columnsum(j) += matrix(i,j);
        }
    }
    return columnsum;
}

///////////////////////////////////////
// [[Rcpp::export]]
arma::vec rowsum(arma::mat matrix){
    long row = matrix.n_rows;
    int col = matrix.n_cols;
    long i;
    int j;
    arma::vec rowsum(row);
    
    for(i=0; i<row; i++) rowsum(i) = 0;
    for(j=0; j<col; j++){
        for(i=0; i<row; i++){
            rowsum(i) += matrix(i,j);
        }
    }
    return rowsum;
}

//////////////////////////////////
// [[Rcpp::export]]
double matsum(arma::mat X){
    int i,j;
    int rows = X.n_rows;
    int cols = X.n_cols;
    double out = 0;
    
    for(i=0;i<rows;i++){
        for(j=0;j<cols;j++){
            out += X(i,j);
        }
    }
    return out;
}

//////////////////////////////////
// [[Rcpp::export]]
double Frobenius(arma::mat X){
    int i,j;
    int rows = X.n_rows;
    int cols = X.n_cols;
    double out = 0;
    
    for(i=0;i<rows;i++){
        for(j=0;j<cols;j++){
            out += X(i,j)*X(i,j);
        }
    }
    return out;
}

//////////////////////////////////////////////////////////////////////////////////
//subset a big matrix by the beginning and ending of row/col index
// [[Rcpp::export]]
arma::mat subsetmatrix(arma::mat rawmat, arma::vec rowindex, arma::vec colindex){
    int row = rowindex(1) - rowindex(0) + 1;
    int col = colindex(1) - colindex(0) + 1;
    arma::mat result(row,col);
    int i,j;
    for(i=0;i<row;i++){
        for(j=0;j<col;j++){
            result(i,j) = rawmat(rowindex(0)+i,colindex(0)+j);
        }
    }
    return result;
}


/***
 //////////////////////////////////////////////////////////////////////////////////
 // [[Rcpp::export]]
 
 arma::mat sparse_mul(arma::mat m,arma::cube aw,arma::mat index,int r, int n, int transpose){
 int P = index.n_rows;
 int l,i,j,s,d,t;
 
 arma::mat out(n,r);
 out.zeros();
 if(transpose==1){
 for(l=0;l<P;l++){
 j = index(l,0);
 i = index(l,1);
 s = index(l,2);
 d = index(l,3);
 for(t=0;t<r;t++){
 //out(i,t) += m(j,i,s) * aw(j,t,s);
 out(i,t) += m(l,0) * aw(j,t,s);
 }
 }
 }
 else{
 for(l=0;l<P;l++){
 i = index(l,0);
 j = index(l,1);
 s = index(l,2);
 d = index(l,3);
 for(t=0;t<r;t++){
 //out(i,t) += m(i,j,s) * aw(j,t,s);
 out(i,t) += m(l,0) * aw(j,t,s);
 }
 }
 }
 
 return out;
 }
 */


//////////////////////////////////
// [[Rcpp::export]]
arma::mat beta_beta_inv(arma::mat V, arma::vec d,int r){
    arma::mat kron_V = kron(V,V);
    int r_sq = r*r;
    arma::mat inv(r_sq,r_sq);
    arma::vec d_sq_inv(r_sq);
    
    inv.zeros();
    int l,g,t;
    for(l=0;l<r;l++){
        for(g=0;g<r;g++){
            d_sq_inv[l*r+g] = 1/(d(l)*d(l)*d(g)*d(g));
        }
    }
    for(l=0;l<r_sq;l++){
        for(g=0;g<=l;g++){
            for(t=0;t<r_sq;t++){
                inv(l,g) += kron_V(l,t) * kron_V(g,t) * d_sq_inv(t);
            }
            inv(g,l) = inv(l,g);
        }
    }
    return inv;
}


//////////////////////////////////
// [[Rcpp::export]]
arma::cube Calculate_MA(arma::mat A,arma::mat M, arma::mat index,int r, int k, int transpose){
    ///if transpose=1 calculate t(M)%*%A ; if =0 M%*%A
    int P = index.n_rows;
    int n = A.n_rows;
    arma::cube out(n,r,k);
    out.zeros();
    int l,i,j,s;
    double m;
    if(transpose==1){
        for(l=0;l<P;l++){
            i = index(l,1);
            j = index(l,0);
            s = index(l,2);
            m = M(l,0);
            out.slice(s).row(i) += m * A.row(j);
        }
    }
    else{
        for(l=0;l<P;l++){
            i = index(l,0);
            j = index(l,1);
            s = index(l,2);
            m = M(l,0);
            out.slice(s).row(i) += m * A.row(j);
        }
    }
    return out;
}

//////////////////////////////////
// [[Rcpp::export]]
arma::cube Calculate_BetaM(arma::mat A, arma::mat M, arma::mat index,int r, int k, arma::cube MA){
    //int P = index.n_rows;
    //int n = A.n_rows;
    int l;
    //int l,i,j,s;
    //double m;
    arma::cube beta_M(r*r,1,k);
    beta_M.zeros();
    
    arma::mat tmp(r,r);
    for(l=0;l<k;l++){
        tmp = A.t() * MA.slice(l);
        beta_M.slice(l) = vectorize(tmp,0);
    }
    return beta_M;
}

//////////////////////////////////
// [[Rcpp::export]]
arma::cube update_W(arma::cube w, arma::mat inv, arma::cube beta_m,int k){
    int r = w.n_rows;
    arma::cube out(r,r,k);
    out.zeros();
    int l;
    
    for(l=0;l<k;l++){
        out.slice(l) = w.slice(l) + vec2mat(inv * beta_m.slice(l),r,r,0);
    }
    
    return out;
}



//////////////////////////////////
// [[Rcpp::export]]
Rcpp::List  update_A(arma::mat A, arma::cube w, arma::mat m, arma::mat index,int k,arma::cube MA,arma::cube MtA){
    int r = w.n_rows;
    int n = A.n_rows;
    arma::mat second_part(r,r);
    arma::mat B(r,r);
    arma::mat C(r,r);
    arma::mat AW(n,r);
    arma::mat AW_T(n,r);
    arma::mat first_part(n,r);
    int l;
    //int t;
    first_part.zeros();
    second_part.zeros();
    for(l=0;l<k;l++){
        AW = A * w.slice(l);
        AW_T = A * w.slice(l).t();
        B = AW_T.t() * AW_T;
        C = AW.t() * AW;
        first_part += A * B + A * C + MA.slice(l) * (w.slice(l).t()) + MtA.slice(l)*w.slice(l);
        second_part += B + C;
    }
    //arma::mat output = first_part * arma::inv(second_part);
    return Rcpp::List::create(Rcpp::Named("first_part")=first_part,
                              Rcpp::Named("second_part")=second_part);
}




//////////////////////////////////
// [[Rcpp::export]]
Rcpp::List E_step(arma::mat A, arma::cube w, int K, int n, arma::mat index){
    int P = index.n_rows;
    int d,i,j,s,l;
    arma::mat miu(1,1);
    double Phi, den, num;
    //arma::cube Phi_array(n,n,K);
    //arma::cube m(n,n,K);
    arma::mat Phi_array(P,1);
    arma::mat m(P,1);
    Phi_array.zeros();
    m.zeros();
    
    for(l=0;l<P;l++){
        i = index(l,0);
        j = index(l,1);
        s = index(l,2);
        d = index(l,3);
        miu = A.row(i) * w.slice(s) * A.row(j).t();
        if(miu(0)>7){
            miu(0)=7;
        }
        else if(miu(0)<-7){
            miu(0)=-7;
        }
        //Rcpp::Rcout<<miu<<std::endl;
        
        //phi = R::dnorm(-miu(0), 0, 1, FALSE);
        //Phi = R::pnorm(-miu(0), 0, 1,TRUE, FALSE);

	Phi = R::pnorm(-miu(0), 0, 1,TRUE, FALSE);
        //Phi_array(i,j,s) = Phi;
        num = (2*d-1) * R::dnorm(-miu(0), 0, 1, FALSE);
        den = d + (1-2*d)*Phi;
        //m(i,j,s) = num/den;
        m(l,0) = num/den;
        //Rcpp::Rcout<<"ok"<<std::endl;
        Phi_array(l,0) = Phi;
        
    }
    return Rcpp::List::create(Rcpp::Named("Phi")=Phi_array,
                              Rcpp::Named("m")=m);
}


//////////////////////////////////
// [[Rcpp::export]]
double obs_loglik(arma::mat Phi,arma::mat index){
    //int i,j,s,l,d;
    int d,l;
    int P = index.n_rows;
    double out = 0;
    for(l=0;l<P;l++){
        //i = index(l,0);
        //j = index(l,1);
        //s = index(l,2);
        d = index(l,3);
        out += log(d+(1-2*d)*(Phi(l,0)));
    }
    return out;
}



//////////////////////////////////
// [[Rcpp::export]]
Rcpp::List E_step_proj(arma::mat A, arma::cube w, int K, int n, arma::mat index){
    int P = index.n_rows;
    int d,i,j,s,l;
    arma::mat miu(1,1);
    double Phi, den, num;
    //arma::cube Phi_array(n,n,K);
    //arma::cube m(n,n,K);
    arma::mat Phi_array(P,1);
    arma::mat m(P,1);
    Phi_array.zeros();
    m.zeros();
    
    for(l=0;l<P;l++){
        i = index(l,0);
        j = index(l,1);
        s = index(l,2);
        d = index(l,3);
        miu = A.row(i) * w.slice(s) * A.row(j).t();
        if(miu(0)>7){
            miu(0)=7;
        }
        else if(miu(0)<-7){
            miu(0)=-7;
        }
        //Rcpp::Rcout<<miu<<std::endl;
        
        //phi = R::dnorm(-miu(0), 0, 1, FALSE);
        Phi = R::pnorm(-miu(0), 0, 1,TRUE, FALSE);
        //Phi_array(i,j,s) = Phi;
        num = (2*d-1) * R::dnorm(-miu(0), 0, 1, FALSE);
        den = d + (1-2*d)*Phi;
        //m(i,j,s) = num/den;
        m(l,0) = num/den;
        //Rcpp::Rcout<<"ok"<<std::endl;
        Phi_array(l,0) = Phi;
        
    }
    return Rcpp::List::create(Rcpp::Named("Phi")=Phi_array,
                              Rcpp::Named("m")=m);
}


/*
 //////////////////////////////////
 // [[Rcpp::export]]
 arma::mat Sparse_E_step(arma::mat A, arma::cube w, int K, int n, arma::mat index,double miu_sparse){
 int P = index.n_rows;
 int d,i,j,s,l;
 arma::mat miu(1,1);
 double Phi, den, num;
 //arma::cube Phi_array(n,n,K);
 //arma::cube m(n,n,K);
 arma::mat m(P,1);
 m.zeros();
 
 for(l=0;l<P;l++){
 i = index(l,0);
 j = index(l,1);
 s = index(l,2);
 d = index(l,3);
 miu = A.row(i) * w.slice(s) * A.row(j).t();
 
 if(miu(0)>7){
 miu(0)=7;
 }
 else if(miu(0)<-7){
 miu(0)=-7;
 }
 //phi = R::dnorm(-miu(0), 0, 1, FALSE);
 Phi = R::pnorm(-miu(0), 0, 1,TRUE, FALSE);
 //Phi_array(i,j,s) = Phi;
 num = (2*d-1) * R::dnorm(-miu(0), 0, 1, FALSE);
 den = d + (1-2*d)*Phi;
 //m(i,j,s) = num/den;
 m(l,0) = num/den + miu(0) - miu_sparse;
 
 }
 //Rcpp::Rcout<<"ok"<<std::endl;
 return m;
 }
 */

//////////////////////////////////
// [[Rcpp::export]]
Rcpp::List Sparse_E_step(arma::mat A, arma::cube w, int K, int n, arma::mat index,double miu_sparse){
    int P = index.n_rows;
    int d,i,j,s,l;
    arma::mat Phi_array(P,1);
    arma::mat miu(1,1);
    double Phi, den, num;
    //arma::cube Phi_array(n,n,K);
    //arma::cube m(n,n,K);
    arma::mat m(P,1);
    m.zeros();
    
    for(l=0;l<P;l++){
        i = index(l,0);
        j = index(l,1);
        s = index(l,2);
        d = index(l,3);
        miu = A.row(i) * w.slice(s) * A.row(j).t();
        
        if(miu(0)>7){
            miu(0)=7;
        }
        else if(miu(0)<-7){
            miu(0)=-7;
        }
        //phi = R::dnorm(-miu(0), 0, 1, FALSE);
        Phi = R::pnorm(-miu(0), 0, 1,TRUE, FALSE);
        //Phi_array(i,j,s) = Phi;
        num = (2*d-1) * R::dnorm(-miu(0), 0, 1, FALSE);
        den = d + (1-2*d)*Phi;
        //m(i,j,s) = num/den;
        m(l,0) = num/den + miu(0) - miu_sparse;
        Phi_array(l,0) = Phi;
        
    }
    //Rcpp::Rcout<<"ok"<<std::endl;
    return Rcpp::List::create(Rcpp::Named("Phi")=Phi_array,
                              Rcpp::Named("m")=m);
}

//////////////////////////////////
// [[Rcpp::export]]
arma::mat inv_beta_j(arma::mat A,arma::mat inv, double miu_sparse){
    int r = A.n_cols;
    int n = A.n_rows;
    arma::mat AI(r,1);
    
    int i,j;
    double sum = 0;
    for(i=0;i<r;i++){
        sum = 0;
        for(j=0;j<n;j++){
            sum += A(j,i);
        }
        AI(i,0) = sum;
    }
    arma::mat out = miu_sparse * vec2mat(inv * kron(AI,AI),r,r,0);
    return out;
}


//////////////////////////////////
// [[Rcpp::export]]
arma::cube Sparse_update_W(arma::cube w, arma::mat inv, arma::cube beta_m,int k,arma::mat inv_beta_j){
    int r = w.n_rows;
    arma::cube out(r,r,k);
    out.zeros();
    int l;
    
    for(l=0;l<k;l++){
        out.slice(l) = vec2mat(inv * beta_m.slice(l),r,r,0) + inv_beta_j;
    }
    
    return out;
}

//////////////////////////////////
// [[Rcpp::export]]
arma::mat J_AW(arma::mat AW,double miu_sparse){
    int r = AW.n_cols;
    int n = AW.n_rows;
    arma::mat tmp(1,r);
    arma::mat out(n,r);
    
    int i,j;
    double sum = 0;
    for(i=0;i<r;i++){
        sum = 0;
        for(j=0;j<n;j++){
            sum += miu_sparse * AW(j,i);
        }
        tmp(0,i) = sum;
    }
    for(i=0;i<n;i++){
        out.row(i) = tmp.row(0);
    }
    return out;
}





//////////////////////////////////
// [[Rcpp::export]]
Rcpp::List  Sparse_update_A(arma::mat A, arma::cube w, arma::mat m, arma::mat index,int k,arma::cube MA,arma::cube MtA,double miu_sparse){
    int r = w.n_rows;
    int n = A.n_rows;
    arma::mat second_part(r,r);
    arma::mat B(r,r);
    arma::mat C(r,r);
    arma::mat AW(n,r);
    arma::mat AW_T(n,r);
    arma::mat first_part(n,r);
    int l;
    //int t;
    first_part.zeros();
    second_part.zeros();
    for(l=0;l<k;l++){
        AW = A * w.slice(l);
        AW_T = A * w.slice(l).t();
        B = AW_T.t() * AW_T;
        C = AW.t() * AW;
        first_part += MA.slice(l) * (w.slice(l).t()) + MtA.slice(l)*w.slice(l)
        + J_AW(AW_T,miu_sparse) + J_AW(AW,miu_sparse);
        second_part += B + C;
    }
    //arma::mat output = first_part * arma::inv(second_part);
    return Rcpp::List::create(Rcpp::Named("first_part")=first_part,
                              Rcpp::Named("second_part")=second_part);
}





//////////////////////////////////
// [[Rcpp::export]]
arma::cube Expectation(arma::cube X, arma::mat A, arma::cube w, int K){
    
    int n = A.n_rows;
    //int r = A.n_cols;
    
    arma::cube out(n,n,K);
    arma::mat miu(n,n);
    double den;
    double num;
    
    int i,j,s;
    for(s=0;s<K;s++){
        miu = A * w.slice(s) * A.t();
        for(i=0; i<n; i++){
            for(j=0;j<n;j++){
                if(miu(i,j)>7){
                    miu(i,j)=7;
                }
                else if(miu(i,j)<-7){
                    miu(i,j)=-7;
                }
                num = (2*X(i,j,s)-1) * R::dnorm(-miu(i,j), 0, 1, FALSE);
                den = X(i,j,s) + (1-2*X(i,j,s))*R::pnorm(-miu(i,j),0,1,TRUE,FALSE);
                out(i,j,s) = miu(i,j) + num/den;
            }
        }
    }
    
    return out;
    
}


//////////////////////////////////
// [[Rcpp::export]]
double loss(arma::cube X, arma::cube w, arma::mat A, int k){
    int s;
    int n = A.n_rows;
    arma::cube out(n,n,k);
    for(s=0;s<k;s++){
        out.slice(s) = A * w.slice(s) * A.t();
    }
    
    int z;
    double result=0;
    
    for(z=0;z<k;z++){
        result += Frobenius( X.slice(z) - out.slice(z));
    }
    return result;
}

//////////////////////////////////
// [[Rcpp::export]]
arma::mat UpdateA(arma::cube exp, arma::cube w, arma::mat A, int k){
    int s;
    int n = A.n_rows;
    int r = w.n_rows;
    arma::cube temptensor(n,r,k);
    arma::mat tempmat(n,r);
    arma::mat part1(n,r);
    arma::mat ATA = A.t()*A;
    arma::mat part2(r,r);
    
    part1.zeros();
    part2.zeros();
    
    for(s=0;s<k;s++){
        tempmat = exp.slice(s) * A * w.slice(s).t() + exp.slice(s).t() * A * w.slice(s);
        part1 += tempmat;
        part2 += w.slice(s) * ATA * w.slice(s).t() + w.slice(s).t() * ATA * w.slice(s);
    }
    arma::mat invmat=inv(part2);
    arma::mat Anew = (invmat * part1.t()).t();
    return Anew;
}

//////////////////////////////////
// [[Rcpp::export]]
arma::cube UpdateW(arma::cube exp, arma::cube w, arma::mat A, int k){
    int s;
    int r = w.n_rows;
    arma::vec wnew_vec(r*r);
    arma::cube wnew(r,r,k);
    arma::mat ATA_inv = inv(A.t()*A);
    arma::mat invmat = kron(ATA_inv,ATA_inv);
    arma::mat Z = kron(A,A);
    arma::mat tempmat = invmat * Z.t();
    
    
    for(s=0;s<k;s++){
        wnew_vec = tempmat * vectorize(exp.slice(s),0);
        wnew.slice(s) = vec2mat(wnew_vec,r,r,0);
    }
    return wnew;
}

