                                       
#' Runs the CPOP algorithm for a range of penalties. The CPOP algorithm finds the best set of changepoints for fitting a change-in-slope model for a single penalty value.
#' This function uses the CROPS algorithm with CPOP to find all best sets of segmentations for penalty values in a user specified interval.
#'
#' @param  y A vector containing the data.
#' @param sigsquared Estimate of variance of the residuals (as used by CPOP).
#' @param min_pen Minimum penalty value.
#' @param max_pen Maximum penalty value. CPOP will find the best segmentation for all penalty values in the interval [min_pen,max_pen].
#' @param PRINT If TRUE, the CROPS algorithm will print its progress.
#'
#' @return
#' The output is a list of segmentations found for different penalty values.
#' \item{summary}{Table summarising the beta values, the corresponding number of changes and the residual sum of squares.}
#' \item{segmentations}{A list of the changepoints for each of the different segmentations.}
#' 
#' @references
#' Haynes, Eckley and Fearnhead (2017). Computationally efficient changepoint detection for a range of penalties. Journal of Computational and Graphical Statistics, 26(1), 134-143.
#' 
#' Maidstone, Fearnhead and Letchford (2017), Detecting changes in slope with an L_0 penalty. arXiv:1701.01672
#'
#' @examples
#' ## Simulate data
#' slope=c(rep(0,49),rep(0.1,50),rep(-0.2,50),rep(0,50)) ## slope
#' mu=cumsum(c(0,slope)) ## underlying piecewise-linear mean
#' n=length(mu)
#' y=rnorm(n,mu) ## data
#' ## run CPOP for penalty values in the interval [log(n),2*log(n)]
#' out=CROPS.CPOP(y,1,min_pen=log(n),max_pen=2*log(n),PRINT=TRUE) 
#' out[[1]] ## summary
#' out[[2]][[1]] ## one optimal set of changepoints
#' 
#' @export
CROPS.CPOP <- function(y,sigsquared,min_pen=5,max_pen=20,PRINT=TRUE) {

  NCALC=0
  pen_interval <- c(min_pen,max_pen)
  if (length(dim(y)) == 0){
    n <- length(y)
  }
  else{
    n <- dim(y)[1]
  }
  
  test_penalties <- NULL
  numberofchangepoints <- NULL
  penal <- NULL
  overall_cost <- array()
  segmentations <- NULL
  b_between <- array()
  
  count <- 0 
  
  while (length(pen_interval) > 0){
    
    new_numcpts <- array()
    new_penalty <- array()
    new_cpts <- array()
    
    for (b in 1:length(pen_interval)) {
      
     ans<-CPOP(y,pen_interval[b],sigsquared)
     resultingcpts <- c(ans[[2]])
      new_numcpts[b] <- length(resultingcpts)
      cost.test <- array()
      new_cpts[b] <- list(resultingcpts)
     new_penalty[b] <- ans[[1]]-(new_numcpts[b]-2)*pen_interval[b]
    }
    
    
    
    if (count == 0){
      if(PRINT==T){
      print(paste("Maximum number of runs of algorithm = ", new_numcpts[1] - new_numcpts[2] + 2, sep = ""))}
      count <- count + length(new_numcpts)
      if(PRINT==T){
      print(paste("Completed runs = ", count, sep = ""))}
    }
    
    else{
      count <- count + length(new_numcpts)
      if(PRINT==T){
      print(paste("Completed runs = ", count, sep = ""))}
    }
    
    ## Add the values calculated to the already stored values
    test_penalties <- unique((sort(c(test_penalties,pen_interval))))
    new_numcpts <- c(numberofchangepoints,new_numcpts)
    new_penalty <- c(penal,new_penalty)
    
    new_cpts <- c(segmentations,new_cpts)
    numberofchangepoints <- -sort(-new_numcpts) ##can use sort to re-order
    penal <- sort(new_penalty)
    
    ls <- array()
    
    for (l in 1:length(new_cpts)){
      ls[l] <- length(new_cpts[[l]])
    }
    
    
    ls1 <- sort(ls,index.return = T, decreasing = T)
    ls1 <- ls1$ix
    
    
    segmentations <- new_cpts[c(ls1)]
    
    pen_interval <- NULL
    tmppen_interval <- NULL
    lastchangelike <-   NULL
    numchangecpts <-   NULL
    lastchangecpts <-  NULL
    
    for (i in 1:(length(test_penalties)-1)){
      if(abs(numberofchangepoints[i]-numberofchangepoints[i+1])>1){ ##only need to add a beta if difference in cpts>1
        j <- i+1
        tmppen_interval <- (penal[j] - penal[i]) * ((numberofchangepoints[i] - numberofchangepoints[j])^-1)
        pen_interval <- c(pen_interval, tmppen_interval )
      }
    }
    
    
    if(length(pen_interval)>0){
      for(k in length(pen_interval):1){
        if(min(abs(pen_interval[k]-test_penalties))<1e-8) {
          pen_interval=pen_interval[-k]
        }
      }
    }
  }
  
  ##PRUNE VALUES WITH SAME num_cp
  for(j in length(test_penalties):2){
    if(numberofchangepoints[j]==numberofchangepoints[j-1]){
      numberofchangepoints=numberofchangepoints[-j]
      test_penalties=test_penalties[-j]
      penal=penal[-j]
      segmentations = segmentations[-j]
    }
  }
  
  
  
  ###calculate beta intervals
  nb=length(test_penalties)
  beta.int=rep(0,nb)
  beta.e=rep(0,nb)
  for(k in 1:nb){
    if(k==1){
      beta.int[1]=test_penalties[1]
    }else{
      beta.int[k]=beta.e[k-1]
    }
    if(k==nb){
      beta.e[k]=test_penalties[k]
    }else{
      beta.e[k]=(penal[k]-penal[k+1])/(numberofchangepoints[k+1]-numberofchangepoints[k])
    }
    
  }
  
  return(list(summary=rbind(test_penalties,beta.int,numberofchangepoints,penal),segmentations=segmentations))
}


