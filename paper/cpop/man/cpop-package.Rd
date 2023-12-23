\name{cpop-package}
\alias{cpop-package}
\alias{cpop}
\docType{package}
\title{
  Method for detecting changes-in-slope.
}
\description{
  Implements the CPOP algorithm to detect changes in slope for
  univariate data. It assumes the data is of a continuous
  piecewise-linear function plus additive noise. The algorithm estimates
  this function, and the corresponding changes in slope, by minimising a
  the residual sum of squares plus a penalty that depends linearly on
  the number of changepoints.
}
\details{
  Assume we have data, \eqn{y_{1:n}}, ordered by some covariate, such as
  time. The CPOP algorithm fits a model
  \eqn{y_{i}=\phi_{i}+\epsilon_{i}}
  for \eqn{i=1,\ldots,n}, under the assumption that \eqn{\phi_{1:n}} corresponds
  to the value of a continuous piecewise linear function. We assume that
  the \eqn{\epsilon_{i}} have mean 0 and variance \eqn{\sigma^2}. The CPOP
  algorithm is most appropriate if the noise distribution is close to Gaussian.

  The CPOP algorithm finds the best estimate of \eqn{\phi_{1:n}} under the a
  criteria that measures fit to the data by the residual sum of squares,
  and penalises for every change-in-slope. Formally this corresponds to
  finding \eqn{\hat{\phi}_{1:n}} that minimises
  \eqn{
\frac{1}{\sigma^2}\sum_{i=1}^n (y_{i}-\hat{\phi}_{i})^2 + \beta \sum_{i=2}^{n-1}
  |\beta_{i+1}-2\beta_{i}+\beta_{i-1}|_{0}
  }
  where \eqn{|\beta_{i+1}-2\beta_{i}+\beta_{i-1}|_{0}} is 1 if
  \eqn{\beta_{i+1}-2\beta_{i}+\beta_{i-1} \neq 0} and 0 otherwise. The sum of
  these terms just calculates the number of locations at which there is
  a change-in-slope. The user chosen constant, \eqn{\beta}, is a penalty for
  adding a changepoint. Choice of \eqn{\beta} is important, with larger
  values leading to fewer estimated changepoints.
  
}
\references{
Maidstone, Fearnhead and Letchford (2017), Detecting changes in slope
with an L_0 penalty. arXiv:1701.01672
}
\keyword{ package }

