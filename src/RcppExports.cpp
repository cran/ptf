// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// get_varphi
arma::mat get_varphi(arma::mat s, arma::vec a);
RcppExport SEXP _ptf_get_varphi(SEXP sSEXP, SEXP aSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type s(sSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a(aSEXP);
    rcpp_result_gen = Rcpp::wrap(get_varphi(s, a));
    return rcpp_result_gen;
END_RCPP
}
// get_varphi_next
arma::mat get_varphi_next(arma::mat snew, arma::vec decisionparm);
RcppExport SEXP _ptf_get_varphi_next(SEXP snewSEXP, SEXP decisionparmSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type snew(snewSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type decisionparm(decisionparmSEXP);
    rcpp_result_gen = Rcpp::wrap(get_varphi_next(snew, decisionparm));
    return rcpp_result_gen;
END_RCPP
}
// get_delta
arma::mat get_delta(arma::mat varphi, arma::mat varphi_next, arma::vec theta0, arma::vec reward, double gamma);
RcppExport SEXP _ptf_get_delta(SEXP varphiSEXP, SEXP varphi_nextSEXP, SEXP theta0SEXP, SEXP rewardSEXP, SEXP gammaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type varphi(varphiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type varphi_next(varphi_nextSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta0(theta0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type reward(rewardSEXP);
    Rcpp::traits::input_parameter< double >::type gamma(gammaSEXP);
    rcpp_result_gen = Rcpp::wrap(get_delta(varphi, varphi_next, theta0, reward, gamma));
    return rcpp_result_gen;
END_RCPP
}
// maei_c_cont
arma::vec maei_c_cont(arma::mat s, arma::mat snew, arma::vec reward, arma::vec a, double gamma, arma::vec theta0, arma::vec stepsize, double pow1, double pow2, int maxit, double tol, bool verbose);
RcppExport SEXP _ptf_maei_c_cont(SEXP sSEXP, SEXP snewSEXP, SEXP rewardSEXP, SEXP aSEXP, SEXP gammaSEXP, SEXP theta0SEXP, SEXP stepsizeSEXP, SEXP pow1SEXP, SEXP pow2SEXP, SEXP maxitSEXP, SEXP tolSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type s(sSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type snew(snewSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type reward(rewardSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta0(theta0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type stepsize(stepsizeSEXP);
    Rcpp::traits::input_parameter< double >::type pow1(pow1SEXP);
    Rcpp::traits::input_parameter< double >::type pow2(pow2SEXP);
    Rcpp::traits::input_parameter< int >::type maxit(maxitSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(maei_c_cont(s, snew, reward, a, gamma, theta0, stepsize, pow1, pow2, maxit, tol, verbose));
    return rcpp_result_gen;
END_RCPP
}
// vectorize
arma::vec vectorize(arma::mat mat, int axis);
RcppExport SEXP _ptf_vectorize(SEXP matSEXP, SEXP axisSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type mat(matSEXP);
    Rcpp::traits::input_parameter< int >::type axis(axisSEXP);
    rcpp_result_gen = Rcpp::wrap(vectorize(mat, axis));
    return rcpp_result_gen;
END_RCPP
}
// vec2mat
arma::mat vec2mat(arma::vec vec, int row, int col, int axis);
RcppExport SEXP _ptf_vec2mat(SEXP vecSEXP, SEXP rowSEXP, SEXP colSEXP, SEXP axisSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type vec(vecSEXP);
    Rcpp::traits::input_parameter< int >::type row(rowSEXP);
    Rcpp::traits::input_parameter< int >::type col(colSEXP);
    Rcpp::traits::input_parameter< int >::type axis(axisSEXP);
    rcpp_result_gen = Rcpp::wrap(vec2mat(vec, row, col, axis));
    return rcpp_result_gen;
END_RCPP
}
// colsum
arma::vec colsum(arma::mat matrix);
RcppExport SEXP _ptf_colsum(SEXP matrixSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type matrix(matrixSEXP);
    rcpp_result_gen = Rcpp::wrap(colsum(matrix));
    return rcpp_result_gen;
END_RCPP
}
// rowsum
arma::vec rowsum(arma::mat matrix);
RcppExport SEXP _ptf_rowsum(SEXP matrixSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type matrix(matrixSEXP);
    rcpp_result_gen = Rcpp::wrap(rowsum(matrix));
    return rcpp_result_gen;
END_RCPP
}
// matsum
double matsum(arma::mat X);
RcppExport SEXP _ptf_matsum(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(matsum(X));
    return rcpp_result_gen;
END_RCPP
}
// Frobenius
double Frobenius(arma::mat X);
RcppExport SEXP _ptf_Frobenius(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(Frobenius(X));
    return rcpp_result_gen;
END_RCPP
}
// subsetmatrix
arma::mat subsetmatrix(arma::mat rawmat, arma::vec rowindex, arma::vec colindex);
RcppExport SEXP _ptf_subsetmatrix(SEXP rawmatSEXP, SEXP rowindexSEXP, SEXP colindexSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type rawmat(rawmatSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type rowindex(rowindexSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type colindex(colindexSEXP);
    rcpp_result_gen = Rcpp::wrap(subsetmatrix(rawmat, rowindex, colindex));
    return rcpp_result_gen;
END_RCPP
}
// beta_beta_inv
arma::mat beta_beta_inv(arma::mat V, arma::vec d, int r);
RcppExport SEXP _ptf_beta_beta_inv(SEXP VSEXP, SEXP dSEXP, SEXP rSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type V(VSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type d(dSEXP);
    Rcpp::traits::input_parameter< int >::type r(rSEXP);
    rcpp_result_gen = Rcpp::wrap(beta_beta_inv(V, d, r));
    return rcpp_result_gen;
END_RCPP
}
// Calculate_MA
arma::cube Calculate_MA(arma::mat A, arma::mat M, arma::mat index, int r, int k, int transpose);
RcppExport SEXP _ptf_Calculate_MA(SEXP ASEXP, SEXP MSEXP, SEXP indexSEXP, SEXP rSEXP, SEXP kSEXP, SEXP transposeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat >::type M(MSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type index(indexSEXP);
    Rcpp::traits::input_parameter< int >::type r(rSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< int >::type transpose(transposeSEXP);
    rcpp_result_gen = Rcpp::wrap(Calculate_MA(A, M, index, r, k, transpose));
    return rcpp_result_gen;
END_RCPP
}
// Calculate_BetaM
arma::cube Calculate_BetaM(arma::mat A, arma::mat M, arma::mat index, int r, int k, arma::cube MA);
RcppExport SEXP _ptf_Calculate_BetaM(SEXP ASEXP, SEXP MSEXP, SEXP indexSEXP, SEXP rSEXP, SEXP kSEXP, SEXP MASEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat >::type M(MSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type index(indexSEXP);
    Rcpp::traits::input_parameter< int >::type r(rSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type MA(MASEXP);
    rcpp_result_gen = Rcpp::wrap(Calculate_BetaM(A, M, index, r, k, MA));
    return rcpp_result_gen;
END_RCPP
}
// update_W
arma::cube update_W(arma::cube w, arma::mat inv, arma::cube beta_m, int k);
RcppExport SEXP _ptf_update_W(SEXP wSEXP, SEXP invSEXP, SEXP beta_mSEXP, SEXP kSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type w(wSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type inv(invSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type beta_m(beta_mSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    rcpp_result_gen = Rcpp::wrap(update_W(w, inv, beta_m, k));
    return rcpp_result_gen;
END_RCPP
}
// update_A
Rcpp::List update_A(arma::mat A, arma::cube w, arma::mat m, arma::mat index, int k, arma::cube MA, arma::cube MtA);
RcppExport SEXP _ptf_update_A(SEXP ASEXP, SEXP wSEXP, SEXP mSEXP, SEXP indexSEXP, SEXP kSEXP, SEXP MASEXP, SEXP MtASEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::cube >::type w(wSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type m(mSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type index(indexSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type MA(MASEXP);
    Rcpp::traits::input_parameter< arma::cube >::type MtA(MtASEXP);
    rcpp_result_gen = Rcpp::wrap(update_A(A, w, m, index, k, MA, MtA));
    return rcpp_result_gen;
END_RCPP
}
// E_step
Rcpp::List E_step(arma::mat A, arma::cube w, int K, int n, arma::mat index);
RcppExport SEXP _ptf_E_step(SEXP ASEXP, SEXP wSEXP, SEXP KSEXP, SEXP nSEXP, SEXP indexSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::cube >::type w(wSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type index(indexSEXP);
    rcpp_result_gen = Rcpp::wrap(E_step(A, w, K, n, index));
    return rcpp_result_gen;
END_RCPP
}
// obs_loglik
double obs_loglik(arma::mat Phi, arma::mat index);
RcppExport SEXP _ptf_obs_loglik(SEXP PhiSEXP, SEXP indexSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type Phi(PhiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type index(indexSEXP);
    rcpp_result_gen = Rcpp::wrap(obs_loglik(Phi, index));
    return rcpp_result_gen;
END_RCPP
}
// E_step_proj
Rcpp::List E_step_proj(arma::mat A, arma::cube w, int K, int n, arma::mat index);
RcppExport SEXP _ptf_E_step_proj(SEXP ASEXP, SEXP wSEXP, SEXP KSEXP, SEXP nSEXP, SEXP indexSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::cube >::type w(wSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type index(indexSEXP);
    rcpp_result_gen = Rcpp::wrap(E_step_proj(A, w, K, n, index));
    return rcpp_result_gen;
END_RCPP
}
// Sparse_E_step
Rcpp::List Sparse_E_step(arma::mat A, arma::cube w, int K, int n, arma::mat index, double miu_sparse);
RcppExport SEXP _ptf_Sparse_E_step(SEXP ASEXP, SEXP wSEXP, SEXP KSEXP, SEXP nSEXP, SEXP indexSEXP, SEXP miu_sparseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::cube >::type w(wSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type index(indexSEXP);
    Rcpp::traits::input_parameter< double >::type miu_sparse(miu_sparseSEXP);
    rcpp_result_gen = Rcpp::wrap(Sparse_E_step(A, w, K, n, index, miu_sparse));
    return rcpp_result_gen;
END_RCPP
}
// inv_beta_j
arma::mat inv_beta_j(arma::mat A, arma::mat inv, double miu_sparse);
RcppExport SEXP _ptf_inv_beta_j(SEXP ASEXP, SEXP invSEXP, SEXP miu_sparseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat >::type inv(invSEXP);
    Rcpp::traits::input_parameter< double >::type miu_sparse(miu_sparseSEXP);
    rcpp_result_gen = Rcpp::wrap(inv_beta_j(A, inv, miu_sparse));
    return rcpp_result_gen;
END_RCPP
}
// Sparse_update_W
arma::cube Sparse_update_W(arma::cube w, arma::mat inv, arma::cube beta_m, int k, arma::mat inv_beta_j);
RcppExport SEXP _ptf_Sparse_update_W(SEXP wSEXP, SEXP invSEXP, SEXP beta_mSEXP, SEXP kSEXP, SEXP inv_beta_jSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type w(wSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type inv(invSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type beta_m(beta_mSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type inv_beta_j(inv_beta_jSEXP);
    rcpp_result_gen = Rcpp::wrap(Sparse_update_W(w, inv, beta_m, k, inv_beta_j));
    return rcpp_result_gen;
END_RCPP
}
// J_AW
arma::mat J_AW(arma::mat AW, double miu_sparse);
RcppExport SEXP _ptf_J_AW(SEXP AWSEXP, SEXP miu_sparseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type AW(AWSEXP);
    Rcpp::traits::input_parameter< double >::type miu_sparse(miu_sparseSEXP);
    rcpp_result_gen = Rcpp::wrap(J_AW(AW, miu_sparse));
    return rcpp_result_gen;
END_RCPP
}
// Sparse_update_A
Rcpp::List Sparse_update_A(arma::mat A, arma::cube w, arma::mat m, arma::mat index, int k, arma::cube MA, arma::cube MtA, double miu_sparse);
RcppExport SEXP _ptf_Sparse_update_A(SEXP ASEXP, SEXP wSEXP, SEXP mSEXP, SEXP indexSEXP, SEXP kSEXP, SEXP MASEXP, SEXP MtASEXP, SEXP miu_sparseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::cube >::type w(wSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type m(mSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type index(indexSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type MA(MASEXP);
    Rcpp::traits::input_parameter< arma::cube >::type MtA(MtASEXP);
    Rcpp::traits::input_parameter< double >::type miu_sparse(miu_sparseSEXP);
    rcpp_result_gen = Rcpp::wrap(Sparse_update_A(A, w, m, index, k, MA, MtA, miu_sparse));
    return rcpp_result_gen;
END_RCPP
}
// Expectation
arma::cube Expectation(arma::cube X, arma::mat A, arma::cube w, int K);
RcppExport SEXP _ptf_Expectation(SEXP XSEXP, SEXP ASEXP, SEXP wSEXP, SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::cube >::type w(wSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(Expectation(X, A, w, K));
    return rcpp_result_gen;
END_RCPP
}
// loss
double loss(arma::cube X, arma::cube w, arma::mat A, int k);
RcppExport SEXP _ptf_loss(SEXP XSEXP, SEXP wSEXP, SEXP ASEXP, SEXP kSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type w(wSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    rcpp_result_gen = Rcpp::wrap(loss(X, w, A, k));
    return rcpp_result_gen;
END_RCPP
}
// UpdateA
arma::mat UpdateA(arma::cube exp, arma::cube w, arma::mat A, int k);
RcppExport SEXP _ptf_UpdateA(SEXP expSEXP, SEXP wSEXP, SEXP ASEXP, SEXP kSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type exp(expSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type w(wSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    rcpp_result_gen = Rcpp::wrap(UpdateA(exp, w, A, k));
    return rcpp_result_gen;
END_RCPP
}
// UpdateW
arma::cube UpdateW(arma::cube exp, arma::cube w, arma::mat A, int k);
RcppExport SEXP _ptf_UpdateW(SEXP expSEXP, SEXP wSEXP, SEXP ASEXP, SEXP kSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cube >::type exp(expSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type w(wSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    rcpp_result_gen = Rcpp::wrap(UpdateW(exp, w, A, k));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_ptf_get_varphi", (DL_FUNC) &_ptf_get_varphi, 2},
    {"_ptf_get_varphi_next", (DL_FUNC) &_ptf_get_varphi_next, 2},
    {"_ptf_get_delta", (DL_FUNC) &_ptf_get_delta, 5},
    {"_ptf_maei_c_cont", (DL_FUNC) &_ptf_maei_c_cont, 12},
    {"_ptf_vectorize", (DL_FUNC) &_ptf_vectorize, 2},
    {"_ptf_vec2mat", (DL_FUNC) &_ptf_vec2mat, 4},
    {"_ptf_colsum", (DL_FUNC) &_ptf_colsum, 1},
    {"_ptf_rowsum", (DL_FUNC) &_ptf_rowsum, 1},
    {"_ptf_matsum", (DL_FUNC) &_ptf_matsum, 1},
    {"_ptf_Frobenius", (DL_FUNC) &_ptf_Frobenius, 1},
    {"_ptf_subsetmatrix", (DL_FUNC) &_ptf_subsetmatrix, 3},
    {"_ptf_beta_beta_inv", (DL_FUNC) &_ptf_beta_beta_inv, 3},
    {"_ptf_Calculate_MA", (DL_FUNC) &_ptf_Calculate_MA, 6},
    {"_ptf_Calculate_BetaM", (DL_FUNC) &_ptf_Calculate_BetaM, 6},
    {"_ptf_update_W", (DL_FUNC) &_ptf_update_W, 4},
    {"_ptf_update_A", (DL_FUNC) &_ptf_update_A, 7},
    {"_ptf_E_step", (DL_FUNC) &_ptf_E_step, 5},
    {"_ptf_obs_loglik", (DL_FUNC) &_ptf_obs_loglik, 2},
    {"_ptf_E_step_proj", (DL_FUNC) &_ptf_E_step_proj, 5},
    {"_ptf_Sparse_E_step", (DL_FUNC) &_ptf_Sparse_E_step, 6},
    {"_ptf_inv_beta_j", (DL_FUNC) &_ptf_inv_beta_j, 3},
    {"_ptf_Sparse_update_W", (DL_FUNC) &_ptf_Sparse_update_W, 5},
    {"_ptf_J_AW", (DL_FUNC) &_ptf_J_AW, 2},
    {"_ptf_Sparse_update_A", (DL_FUNC) &_ptf_Sparse_update_A, 8},
    {"_ptf_Expectation", (DL_FUNC) &_ptf_Expectation, 4},
    {"_ptf_loss", (DL_FUNC) &_ptf_loss, 4},
    {"_ptf_UpdateA", (DL_FUNC) &_ptf_UpdateA, 4},
    {"_ptf_UpdateW", (DL_FUNC) &_ptf_UpdateW, 4},
    {NULL, NULL, 0}
};

RcppExport void R_init_ptf(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
