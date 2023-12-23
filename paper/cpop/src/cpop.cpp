

#include "cpop_impl.h"

#include <Rcpp.h>
using namespace Rcpp;

#include <vector>

// [[Rcpp::export]]
std::vector<double> coeffupdate(const std::vector<double>& coeffs,
				const std::vector<double>& S,
				const std::vector<double>& SJ,
				const std::vector<double>& SS,
				const int& taustar,
				const double& sigsquared,
				const double& beta,
				const int& nrow,
				const int& ncol)
{
  std::vector<double> coeffnew(nrow*ncol);
  coeffupdate_orig_impl(&coeffs[0],&S[0],&SJ[0],&SS[0],&taustar,&sigsquared,&beta,&nrow,&coeffnew[0]);
  return coeffnew; 
}


// [[Rcpp::export]]
std::vector<int> prune(const std::vector<double>& x,const int& nrows)
{
  std::vector<int> result(nrows);
  prune2R_orig_impl(&x[0],&nrows,&result[0]);   
  return result;
}



