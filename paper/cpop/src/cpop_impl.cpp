


#include <math.h>
#include <stdio.h>
#include <limits.h>
#include "R.h"


#include "Rcpp.h"
// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
using namespace Rcpp;

// need to put const n this
void prune2R_orig_impl(const double* x, const int* nrows, int* Sets)
// void prune2R_orig_impl(const *x, int *nrows, int *Sets) ---- original
{
  
  int i;
  double tcurr= -DBL_MAX;
  int whichfun=0;
  double minA=*(x+2), maxB=*(x+3), minC=*(x+4);
    for ( i = 1 ; i < *nrows ; i++ ) 
    {
      if ( *(x+5*i+2) < minA ) 
      {
        minA = *(x+5*i+2);
	maxB = *(x+5*i+3);
	minC = *(x+5*i+4);
        whichfun = i;
      }
      else if(*(x+5*i+2) == minA){
	if(*(x+5*i+3) > maxB){
	  maxB = *(x+5*i+3);
	  minC = *(x+5*i+4);
	  whichfun = i;
	}
	else if(*(x+5*i+3) == maxB){
	  if(*(x+5*i+4) < minC){
	    minC = *(x+5*i+4);
	    whichfun = i;
	  }
	}
      }
    }

   
 int logicint[*nrows]; /*1 indicates NA*/
 for(i=0;i<*nrows;i++){
   logicint[i]=1;
   *(Sets+i)=0;
 }
 *(Sets+whichfun)=1;
 int sum=*nrows;
    while(sum>0){
      double intercepts[*nrows];
      logicint[whichfun]=0;
      intercepts[whichfun]=0;
      for (i=0;i<*nrows;i++){
	if(logicint[i]!=0){
	  double A=*(x+5*i+2)-*(x+5*whichfun+2);
	  double B=*(x+5*i+3)-*(x+5*whichfun+3);
	  double C=*(x+5*i+4)-*(x+5*whichfun+4); /*creates the diffcoeff function*/
	  double disc=B*B-4*A*C;
	  if(disc<0){
	    intercepts[i]=0/*NA*/;
	    logicint[i]=0;
	  }
	  else{
	    if(A==0){
	      if(B==0){
		intercepts[i]=0;
		logicint[i]=0;
		}
	      else if((-C/B)>tcurr){
		intercepts[i]=-C/B;
		logicint[i]=1;
	      }
	      else{
		intercepts[i]=0/*NA*/;
		logicint[i]=0;
	      }
	    }
	    else{
	      double vec2a=(-B-sqrt(disc))/(2*A);
	      double vec2b=(-B+sqrt(disc))/(2*A);
	      if(vec2a<=tcurr && vec2b<=tcurr){
		intercepts[i]=0/*NA*/;
		logicint[i]=0;
	      }
	      else if(vec2a<=tcurr && vec2b>tcurr){
		intercepts[i]=vec2b;
		logicint[i]=1;
	      }
	      else if((vec2a>tcurr && vec2b<=tcurr) || (vec2a>tcurr && vec2b>tcurr && vec2a<vec2b)){
		intercepts[i]=vec2a;
		logicint[i]=1;
	      }
	      else{
		intercepts[i]=vec2b;
		logicint[i]=1;
	      }
	    }
	  }
	}
      }
      int j;
      sum=0;
      int whichfunnew;
      double minimum=LONG_MAX;
      for(j=0;j<*nrows;j++){
	if(logicint[j]!=0){
	  sum=sum+logicint[j];
	  if ( intercepts[j] < minimum ) {
	    minimum = intercepts[j];
	    whichfunnew = j;
	  }
	}
      }
      logicint[whichfun]=1;
      if(sum!=0){
	tcurr = minimum;
	whichfun = whichfunnew;
	*(Sets+whichfun)=1;
      }
    }
}

void coeffupdate_orig_impl(const double* coeffs,
			   const double* S,
			   const double* SJ,
			   const double* SS,
			   const int* taustar,
			   const double* sigsquared,
			   const double* beta,
			   const int* nrows,
			   double* coeffnew)
// extern "C"   void coeffupdate(double *coeffs, double *S, double *SJ, double *SS, int *taustar, double *sigsquared, double *beta, int *nrows, double *coeffnew)
{

    // added interupt handler - D Grose 16-02-2017
  Progress p(0,false); // needs an instance for interrupt handler
    // end addition
  
  /*Stuff the function will provide
  double SJ[11]={23,4,2,1,4,3,5,3,2,5,12};
  double S[11]={2,41,2,1,2,4,5,9,6,8,12};
  double SS[11]={2,41,2,1,2,4,5,9,6,8,12};
  /*Actual stuff we need*/
  int i,j,sstar;
  double A,B,C,D,E,FF,seglen;
  /* coeffnew = (double *)calloc(5**nrows,sizeof(double)); /*take out this line to make it work */
  for(i = 0; i < *nrows; i++) {
    // added interupt handler - D Grose 16-02-2017
    if(i % 128 == 0 && Progress::check_abort()) { stop("User interrupt"); }
    // end addition
    *(coeffnew+5*i)=*taustar;
    *(coeffnew+5*i+1)=*(coeffs+5*i);
    sstar=*(coeffnew+5*i+1);
    seglen=*taustar-sstar;
    A=(seglen+1)*(2*seglen+1)/(12*seglen* *sigsquared);
    B=(seglen*seglen-1)/(6*seglen* *sigsquared);
    C=(-1)/(seglen* *sigsquared)*(*(SJ+*taustar)-*(SJ+sstar)-sstar*(*(S+*taustar)-*(S+sstar)));
    D=seglen/2*log(2*M_PI* *sigsquared)+1/(2* *sigsquared)*(*(SS+*taustar)-*(SS+sstar));
    E=(-1)*C-1/ *sigsquared*(*(S+*taustar)-*(S+sstar));
    FF=(seglen-1)*(2*seglen-1)/(12*seglen* *sigsquared);
    if((FF==0) && (*(coeffs+5*i+2)==0)){
      if(B==0){
	*(coeffnew+5*i+4)=*(coeffs+5*i+4)+D+ *beta;
	*(coeffnew+5*i+3)=C;
	*(coeffnew+5*i+2)=A;
      }
      else{
	*(coeffnew+5*i+4)=(-E-*(coeffs+5*i+3))/B+ *beta;
	*(coeffnew+5*i+3)=0;
	*(coeffnew+5*i+2)=0;
      }
    }
    else{
      *(coeffnew+5*i+4)=*(coeffs+5*i+4)+D-(*(coeffs+5*i+3)+E)*(*(coeffs+5*i+3)+E)/(4*(*(coeffs+5*i+2)+FF))+ *beta;
      *(coeffnew+5*i+3)=C-(*(coeffs+5*i+3)+E)*B/(2*(*(coeffs+5*i+2)+FF));
      *(coeffnew+5*i+2)=A-B*B/(4*(*(coeffs+5*i+2)+FF));
    }
  }  

}

