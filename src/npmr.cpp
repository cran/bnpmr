#include<iostream>
#include<fstream>
#include <stdio.h>
#include <math.h>
#include<time.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_statistics_int.h>
#include <gsl/gsl_sort.h>
#include "axl.h"
#include <fstream> /* for in/output*/
using namespace std;

/* Get transformed beta distribution parameters */
double getalpha(const double m, const double v){
  return m*(v+2);
}
/* Get transformed beta distribution parameters */
double getbeta(const double m, const double v){
  return  v+2-m*(v+2);
}

/* Build design matrix for 'transformed' input variable */
void buildXmat(const double x[], const double jl[], 
               const double jv[], const double jh[], gsl_matrix *X,
	       int *p, int *N, int *dimension){
  int i,j;
  double tdose;
  if(*p < 3){
    i = 0;
    while(1){
      if(i == *N-1){
	break;
      }
      tdose = 0;
      for(j = 0; j < *dimension; j++){
        tdose += jh[j]*ptsp(x[i], jl[j], jv[j]);
      }
      gsl_matrix_set(X, i, 0, 1);
      gsl_matrix_set(X, i, 1, tdose);
      i++;
      while(1){
        gsl_matrix_set(X, i, 0, 1);
        gsl_matrix_set(X, i, 1, tdose);
	if(i < *N - 1){
	  if(x[i]==x[i-1]){
	    i++;
	  } else {
	    break;
	  }
	} else {
	  break;
	}
      }
    }
  } else { /* the code below will segfault, so only p=2 is allowed! */
      for(i = 0; i < *N; i++){
        tdose = 0;
        for(j = 0; j < *dimension; j++){
          tdose += jh[j]*intptsp(x[i], jl[j], jv[j]);
        }
        gsl_matrix_set(X, i, 0, 1);
        gsl_matrix_set(X, i, 1, x[i]);   
        gsl_matrix_set(X, i++, 2, tdose);
        while(x[i]==x[i-1]){
          gsl_matrix_set(X, i, 0, 1);
          gsl_matrix_set(X, i, 1, x[i]);   
          gsl_matrix_set(X, i, 2, tdose);
          i++;
        }
      }
  }
}

/* Calculate marginal likelihood 'non-informative' version */
double getIntLikNI(gsl_vector *y, gsl_matrix *X, gsl_vector *tau,
                   gsl_vector *beta, gsl_vector *resids, gsl_matrix *X2,
                   double *as, double *a, double *d, int *N){
  double RSS=0, ds=0, IL=0, detXtX=0;

  /* use copy of X to not destroy X */
  gsl_matrix_memcpy (X2, X);
  /* Perform QR decomposition */
  gsl_linalg_QR_decomp (X2, tau);
  gsl_linalg_QR_lssolve(X2, tau, y, beta, resids);
  RSS = ssqElements(resids);
  *as = *a + RSS;
  ds = *d + *N;
  /* Calculate diagsum of R */
  detXtX = gsl_pow_2(proddiag(X2));
  IL = -0.5*log(detXtX) - ds/2*log(*as);
  /* remove created objects from memory */
  return IL;
}

/* Calculate marginal likelihood 'informative' version */
double getIntLikIN(gsl_vector *y, gsl_matrix *X, gsl_matrix *V, gsl_matrix *Vinv,
                 gsl_vector *m, gsl_vector *tau, gsl_vector *beta,
                 gsl_vector *resids, gsl_matrix *X2, gsl_matrix *V2, gsl_matrix *XtX,
		   double *as, double *a, double *d, int *N)  {
  /* To calculate as use p.310 of O'Hagan 2004 */
  double RSS=0, ds=0, IL=0, det=0, rslt=0;
  /* to not destroy X use copy of it */
  gsl_matrix_memcpy (X2, X);  

  /* Perform QR decomposition (to obtain betahat and RSS)*/
  gsl_linalg_QR_decomp (X2, tau);
  gsl_linalg_QR_lssolve(X2, tau, y, beta, resids);

  RSS = ssqElements(resids);
  *as = *a + RSS;
  ds = *d + *N;

  /* invert V+X'X^-1 */
  gsl_matrix_memcpy (X2, X);  /* remove QR output from X2 */
  gsl_matrix_memcpy (V2, V);  /* use copy of V */
  gsl_blas_dgemm (CblasTrans, CblasNoTrans, 1, X2, X2, 0, XtX); /* result in XtX */
  invert(XtX);
  gsl_matrix_add(V2, XtX); /* result in V2 */
  invert(V2);
  gsl_vector_memcpy (tau, m); /* copy m in tau (tau not needed at this moment) */
  gsl_vector_sub(tau, beta); /* result in tau */ 
  gsl_blas_dsymv(CblasUpper, 1, V2, tau, 0, beta); /* result in beta */
  gsl_blas_ddot(beta, tau, &rslt);
  *as = *as + rslt;
 
  /* Vinv + X'X  */
  gsl_matrix_memcpy (V2, Vinv);  /* use copy of Vinv */ 
  gsl_blas_dgemm (CblasTrans, CblasNoTrans, 1, X2, X2, 1, V2);
  det = 1/determinant(V2); /* no need to invert Vinv + X'X */

  IL = 0.5*log(det) - ds/2*log(*as); 
  return IL;
}

/* wrapper for non-inf. and inf. version */
double getIntLik(gsl_vector *y, gsl_matrix *X, gsl_matrix *V, gsl_matrix *Vinv,
                 gsl_vector *m, gsl_vector *tau, gsl_vector *beta,
                 gsl_vector *resids, gsl_matrix *X2, gsl_matrix *V2, gsl_matrix *XtX,
                 double *as, double *a, double *d, int *N){
  double IL=0;
  if(gsl_matrix_get(V, 0, 0) == -1){
    IL = getIntLikNI(y, X, tau, beta, resids, X2, as, a, d, N); /* with improper prior */
  } else {
    IL = getIntLikIN(y, X, V, Vinv, m, tau, beta, resids, X2, V2, XtX, as,
                     a, d, N); /* with proper prior */ 
  }
  /*return IL;*/
  return IL;
}

/* UPDATE step */
void UPDATE(double jl[], double jv[], double jh[], double jlProp[], double jvProp[],
            double jhProp[], double alphaVec[], double hVarVec[], double hVarVecRev[],
            double mu[], double muProp[], double x[], gsl_vector *y, gsl_matrix *X, 
            gsl_matrix *Xprop, gsl_matrix *V, gsl_matrix *Vinv, gsl_vector *m,
            gsl_rng * rngPntr, gsl_vector *tau, gsl_vector *beta,
            gsl_vector *resids, gsl_matrix *X2, gsl_matrix *V2, 
            gsl_matrix *XtX, double *ILold, double *as,
	    int *dimension, double *hVar, int *p, 
	    double *vL, double *vU, double *varPr, double *locPr, double *la, double *lb,
	    double *a, double *d, int *N){
  double likRat=0, priorRat=0, propRat=0, Rat=0, ILnew=0;
  double asTemp = *as;
  int i;

  /* update jump heights with Dirichlet distribution */            
  for(i=0; i<*dimension; i++){
    hVarVec[i] = jh[i]**hVar + 1; /* adding 1 to avoid too small values*/
  }
  gsl_ran_dirichlet(rngPntr, *dimension, hVarVec, jhProp);
  for(i=0; i<*dimension; i++){
    hVarVecRev[i] = jhProp[i]**hVar + 1;
  }
  buildXmat(x, jl, jv, jhProp, Xprop, p, N, dimension);
  /* likelihood ratio*/
  ILnew = getIntLik(y, Xprop, V, Vinv, m, tau, beta, resids, X2, V2, XtX, as, a, d, N);
  likRat = ILnew - *ILold;
  /* prior ratio*/
  priorRat = lddirichlet (*dimension, alphaVec, jhProp) - lddirichlet (*dimension, alphaVec, jh);
  /* proposal ratio */
  propRat = gsl_ran_dirichlet_lnpdf (*dimension, hVarVecRev, jh) - gsl_ran_dirichlet_lnpdf (*dimension, hVarVec, jhProp);
  Rat = likRat + priorRat + propRat;
  if(-gsl_ran_exponential (rngPntr, 1) < Rat){
    for(i=0;i<*dimension;i++){
      jh[i] = jhProp[i];
    }
    gsl_matrix_memcpy (X, Xprop);
    *ILold = ILnew;
    asTemp = *as;
  }

  /* update jump variances with beta distribution on [vL,vU] */ 
  double alp=0, bet=0;
  propRat = 0;priorRat = 0;
  /* proposals and proposal ratio */
  for(i = 0; i < *dimension; i++){
    mu[i] = (jv[i]-*vL)/(*vU-*vL);
    alp = getalpha(mu[i], *varPr)+1;
    bet = getbeta(mu[i], *varPr)+1;
    muProp[i] = gsl_ran_beta(rngPntr, alp, bet);
    jvProp[i] = muProp[i]*(*vU-*vL)+*vL;
    propRat -= nldbeta (muProp[i], alp, bet);
    alp = getalpha(muProp[i], *varPr)+1;
    bet = getbeta(muProp[i], *varPr)+1;
    propRat += nldbeta (mu[i], alp, bet);
  }
  buildXmat(x, jl, jvProp, jh, Xprop, p, N, dimension);
  /* likelihood ratio */
  ILnew = getIntLik(y, Xprop, V, Vinv, m, tau, beta, resids, X2, V2, XtX, as, a, d, N);
  likRat = ILnew - *ILold;
  /* prior ratio is 1 -> log(1) = 0 */
  Rat = likRat + priorRat + propRat;
  if(-gsl_ran_exponential (rngPntr, 1) < Rat){
    for(i=0;i<*dimension;i++){
      jv[i] = jvProp[i];
    }
    gsl_matrix_memcpy (X, Xprop);
    *ILold = ILnew;
    asTemp = *as;
  }

  /* update jump locations with beta distribution */
  double mn=0, mnprop=0;
  propRat = 0;priorRat = 0;
  /* proposals; proposal and prior ratio */
  for(i = 0; i < *dimension; i++){
    alp = getalpha(jl[i], *locPr)+1;
    bet = getbeta(jl[i], *locPr)+1;
    jlProp[i] = gsl_ran_beta(rngPntr, alp, bet);
    propRat -= nldbeta (jlProp[i], alp, bet);
    alp = getalpha(jlProp[i], *locPr)+1;
    bet = getbeta(jlProp[i], *locPr)+1;
    propRat += nldbeta(jl[i], alp, bet);
    priorRat += ldbeta (jlProp[i], *la, *lb) - ldbeta (jl[i], *la, *lb);
  }
  buildXmat(x, jlProp, jv, jh, Xprop, p, N, dimension);
  /* likelihood ratio */
  ILnew = getIntLik(y, Xprop, V, Vinv, m, tau, beta, resids, X2, V2, XtX, as, a, d, N);
  likRat = ILnew - *ILold;
  /* overall ratio */
  Rat = likRat + priorRat + propRat;
  if(-gsl_ran_exponential (rngPntr, 1) < Rat){
    for(i=0;i<*dimension;i++){
      jl[i] = jlProp[i];
    }
    gsl_matrix_memcpy (X, Xprop);
    *ILold = ILnew;
    asTemp = *as;
  }
  *as = asTemp;
}

/* ADD step */
void ADD(double jl[], double jv[], double jh[], double jlProp[], double jvProp[],
         double jhProp[], double x[],
         gsl_vector *y, gsl_matrix *X, gsl_matrix *Xprop, gsl_matrix *V, gsl_matrix *Vinv,
         gsl_vector *m, double pUp, double pDown, gsl_rng * rngPntr, gsl_vector *tau,
         gsl_vector *beta, gsl_vector *resids, gsl_matrix *X2, gsl_matrix *V2, 
         gsl_matrix *XtX, double *ILold, double *as,
	 int *dimension, double *vL, double *vU, double *alpha, int *p, int *N,
	 double *a, double *d, double *la, double *lb, double *lambda){
  double likRat=0, priorRat=0, propRat=0, Rat=0, jacobi=0, cs=0, ILnew=0;
  double asTemp = *as;
  int J,i,r;
  J = *dimension;

  /* sample proposals */  
  r = gsl_rng_uniform_int(rngPntr, J+1); /* rnbr from 0,...,J */
  jlProp[r] = gsl_ran_flat(rngPntr, 0, 1);
  jvProp[r] = gsl_ran_flat(rngPntr, *vL, *vU);
  jhProp[r] = gsl_ran_beta(rngPntr, *alpha, *alpha*J); /* = Beta(alpha, (J+1)alpha - alpha)/frueher: Beta(1, (J+1)) in */
  cs += jhProp[r];                                       /* REMOVE: Beta(1, J)*/

  for(i = 0; i < r; i++){
    jlProp[i] = jl[i];
    jvProp[i] = jv[i];
    jhProp[i] = jh[i]*(1-jhProp[r]);
    cs += jhProp[i];
  }
  for(i = r; i < J; i++){
    jlProp[i+1] = jl[i];
    jvProp[i+1] = jv[i];
    jhProp[i+1] = jh[i]*(1-jhProp[r]);
    cs += jhProp[i+1];
  }
  *dimension += 1;
  buildXmat(x, jlProp, jvProp, jhProp, Xprop, p, N, dimension);
  /* likelihood ratio */
  ILnew = getIntLik(y, Xprop, V, Vinv, m, tau, beta, resids, X2, V2, XtX, as, a, d, N);
  likRat = ILnew - *ILold;
  /* prior ratio */
  priorRat = -log(*vU - *vL) + nldbeta (jlProp[r], *la, *lb)+ log( (double) J) + log(*lambda) - log( (double) J+1);
  if(*alpha != 1){ /* not uniform distribution on simplex */
    priorRat = priorRat - log( (double) J) + gsl_sf_lngamma ((J+1)**alpha) + (*alpha-1)*log(jhProp[r]) + J*(*alpha-1)*log(1-jhProp[r]) - gsl_sf_lngamma (J**alpha) - gsl_sf_lngamma (*alpha);
  }
  /* proposal ratio and jacobian */  
  propRat = 0 + nldbeta (jhProp[r], *alpha, *alpha*J) - log(*vU - *vL); /* need normalized version of beta density here */
  jacobi = (J-1)*log(1-jhProp[r]);
  propRat = log(pDown) + jacobi - log(pUp) - propRat;
  /* overall ratio */
  Rat = likRat + priorRat + propRat;
  if(-gsl_ran_exponential (rngPntr, 1.0) < Rat){
    for(i=r;i<=J;i++){
      jl[i] = jlProp[i];
      jv[i] = jvProp[i];      
    }
    for(i=0;i<=J;i++){
      jh[i] = jhProp[i]/cs; /* cs (cumulated sum) to ensure sum = 1 */
    }   
    gsl_matrix_memcpy (X, Xprop);
    *ILold = ILnew;
  } else {
    *dimension -= 1;
    *as = asTemp;
  }
}


void REMOVE(double jl[], double jv[], double jh[], double jlProp[], double jvProp[],
	    double jhProp[], double x[], 
	    gsl_vector *y, gsl_matrix *X, gsl_matrix *Xprop, gsl_matrix *V, gsl_matrix *Vinv,
	    gsl_vector *m, double pUp, double pDown, gsl_rng * rngPntr, gsl_vector *tau,
	    gsl_vector *beta, gsl_vector *resids, gsl_matrix *X2, gsl_matrix *V2, 
	    gsl_matrix *XtX, double *ILold, double *as,
	    int *dimension, int *p, int *N, double *vU, double *vL, double *la, double *lb,
	    double *lambda, double *alpha, double *a, double *d){
  double likRat=0, priorRat=0, propRat=0, Rat=0, cs=0, ILnew=0;
  double asTemp = *as;
  int J,i,r;
  J = *dimension; /* number of non-zero entries */
  double jacobi=0;
  
  /* which removed? */  
  r = gsl_rng_uniform_int (rngPntr, J);
  for(i = 0; i < r; i++){
    jlProp[i] = jl[i];
    jvProp[i] = jv[i];
    jhProp[i] = jh[i]/(1-jh[r]);    
    cs += jhProp[i];
  }
  for(i = r; i < J-1; i++){
    jlProp[i] = jl[i+1];
    jvProp[i] = jv[i+1];
    jhProp[i] = jh[i+1]/(1-jh[r]);
    cs += jhProp[i];
  }
  *dimension -= 1;
  buildXmat(x, jlProp, jvProp, jhProp, Xprop, p, N, dimension);
  /* likelihood ratio */
  ILnew = getIntLik(y, Xprop, V, Vinv, m, tau, beta, resids, X2, V2, XtX, as, a, d, N);
  likRat = ILnew - *ILold;
  /* prior ratio */
  priorRat = log(*vU - *vL) - nldbeta (jl[r], *la, *lb) - log( (double) J-1) - log(*lambda) + log((double) J);
  if(*alpha != 1){ /* not uniform distribution on simplex */
    priorRat = priorRat + log( (double) J-1) + gsl_sf_lngamma(*alpha) + gsl_sf_lngamma((J-1)**alpha) - gsl_sf_lngamma(J**alpha) - (*alpha-1)*log(jh[r]) - (J-1)*(*alpha-1)*log(1-jh[r]);
  }
  /* proposal ratio and jacobian */ 
  propRat = 0 + nldbeta (jh[r], *alpha, *alpha*J-*alpha) - log(*vU - *vL); /* need normalized version of beta density here */
  jacobi = (2-J)*log(1-jh[r]);
  propRat = log(pUp) + jacobi + propRat - log(pDown);
  /* overall ratio */
  Rat = likRat + priorRat + propRat;
  if(-gsl_ran_exponential (rngPntr, 1.0) < Rat){
    for(i=r;i<*dimension;i++){
      jl[i] = jlProp[i];
      jv[i] = jvProp[i];
    }
    for(i=0;i<*dimension;i++){
      jh[i] = jhProp[i]/cs;
    }
    jl[*dimension] = 0;
    jv[*dimension] = 0;
    jh[*dimension] = 0;
    gsl_matrix_memcpy (X, Xprop);
    *ILold = ILnew;
  } else {
    *dimension += 1;
    *as = asTemp;
  }
}



extern "C" { 
void MH(double jl[], double jv[], double jh[], int *sizetrans, int *dimtrans, 
	int dimCount[], double x[], double ytrans[], int *Ntrans,
	double pMoves[], int *oR, double prior[], double Vtrans[], double Vinvtrans[], double mtrans[],
	double prop[], int *niter,  int *thin, int *burnIn, int *seed, double jlOut[], double jvOut[], double jhOut[],
	double betaOut[], double *s2Out){
  int dimension; /* current dimension of jl,jh,jv */
  int size; /* total number of entries of jl,jv,jh */
  int N; /* sample size */
  int p=2; /* umbrella or monotone */
  /* parameters of prior distributions */
  double vL, vU, la, lb, alpha, lambda, a, d;
  /* parameters of proposal distributions */
  double hVar, varPr, locPr;

  /* set global variables */
  size = *sizetrans; /* Länge des arrays für Parameter */
  dimension = *dimtrans; /* enthält aktuelle Dimension des Parameters */
  N = *Ntrans; /* Beobs */ 
  p = 2 + *oR; /* Anzahl Parameter */
  vL = prior[0]; /* Grenze für Präz. der Kerne */
  vU = prior[1];
  la = prior[2]; /* beta parameter Sprungorte */ 
  lb = prior[3];
  alpha = prior[4]; /* Dirichlet Vtlg. */
  lambda = prior[5]; /* Poisson Vtlg. */
  a = prior[6]; /* NIG */
  d = prior[7];
  /* scale parameters of proposal distributions */
  hVar = prop[0]; /* Dirichlet Update */
  varPr = prop[1]; /* Präz. Update */
  locPr = prop[2]; /* Sprungort Update */

  int i,j,count=0,cumDimCount=0;
  double u=0, pMov[3]={0}, ILold = 0, as = 0, s2=0;
  double *jhProp, *jvProp, *jlProp, *mu, *muProp, *alphaVec;
  double *hVarVec, *hVarVecRev;
  double *res, *aux;
  jhProp = new double [size];
  jvProp = new double [size];
  jlProp = new double [size];
  mu = new double [size];
  muProp = new double [size];
  alphaVec = new double [size];
  hVarVec = new double [size];
  hVarVecRev = new double [size];
  res = new double [p];
  aux = new double [p];
  
  gsl_rng *rngPntr;
  gsl_matrix *X = gsl_matrix_alloc (N, p);
  gsl_matrix *Xprop = gsl_matrix_alloc (N, p);
  gsl_vector *y = gsl_vector_alloc(N);
  gsl_matrix *V = gsl_matrix_alloc(p,p);
  gsl_matrix *Vinv = gsl_matrix_alloc(p,p);
  gsl_vector *m = gsl_vector_alloc(p);
  
  /* Variablen für Berechn. der integ. Likelihood */
  gsl_vector *tau = gsl_vector_alloc(p);
  gsl_vector *beta = gsl_vector_alloc(p);
  gsl_vector *resids = gsl_vector_alloc(N);
  gsl_matrix *X2 = gsl_matrix_alloc(N,p);  
  gsl_matrix *V2 = gsl_matrix_alloc(p,p);
  gsl_matrix *XtX = gsl_matrix_alloc(p,p);

  /* Create V,Vinv Matrix and m vector and yvec */
  array2gslmatrix(Vtrans, V, p);
  array2gslmatrix(Vinvtrans, Vinv, p);
  array2gslvector(mtrans, m, p);
  array2gslvector(ytrans, y, N);

  for(i=0; i<size; i++){
    alphaVec[i] = alpha;
  }
  
  /* initialization of rng */ 
  rngPntr=gsl_rng_alloc(gsl_rng_default); 
  gsl_rng_set(rngPntr, *seed);
  
  /* first iteration of design matrix and L'hood */
  buildXmat(x, jl, jv, jh, X, &p, &N, &dimension);
  ILold = getIntLik(y, X, V, Vinv, m, tau, beta, resids, X2, V2, XtX, &as, &a, &d, &N);

  for(i=1; i<=*niter; i++){
    u = gsl_ran_flat(rngPntr, 0, 1);
    if(jl[1]==0){ /* no remove step */
      pMov[0] = pMoves[0]/(1-pMoves[2]);
      pMov[1] = pMoves[1]/(1-pMoves[2]);
      pMov[2] = 0;
    } else {
      pMov[0] = pMoves[0];
      pMov[1] = pMoves[1];
      pMov[2] = pMoves[2];
    }
    
    if(u < pMov[0]){
      UPDATE(jl, jv, jh, jlProp, jvProp, jhProp, alphaVec, hVarVec, hVarVecRev,
	     mu, muProp, x, y, X, Xprop, V, Vinv, m, rngPntr, tau, beta, resids,
	     X2, V2, XtX, &ILold, &as, &dimension, &hVar, &p, &vL, &vU, &varPr, 
	     &locPr, &la, &lb, &a, &d, &N);
    } else if(u < pMov[0] + pMov[1]){
        if(dimension==1){
          pMov[2] = pMoves[2];
        }
        ADD(jl, jv, jh, jlProp, jvProp, jhProp, x, y, X, Xprop, V, Vinv, m,
	    pMov[1], pMov[2], rngPntr, tau, beta, resids, X2, V2, XtX, &ILold, &as,
	    &dimension, &vL, &vU, &alpha, &p, &N, &a, &d, &la, &lb, &lambda);
    } else {
        if(dimension==2){
          pMov[1] = pMoves[1]/(1-pMoves[2]);
        }
        REMOVE(jl, jv, jh, jlProp, jvProp, jhProp, x, y, X, Xprop, V, Vinv, 
	       m, pMov[1], pMov[2], rngPntr, tau, beta, resids, X2, V2, XtX, &ILold, &as,
	       &dimension, &p, &N, &vU, &vL, &la, &lb, &lambda, &alpha, &a, &d);
    }

    if(!(i % *thin) & i>*burnIn){ 
      /* store realization */
      dimCount[count] = dimension;
      for(j=0;j<dimension;j++){
        jlOut[cumDimCount+j] = jl[j];
        jvOut[cumDimCount+j] = jv[j]; 
        jhOut[cumDimCount+j] = jh[j];
      }
      cumDimCount += dimension;
      /* simulate beta,s2 cond. on the rest */
      /* calculate cov-matrix and post.mean */
      if(gsl_matrix_get(V,0,0) == -1){
         gsl_blas_dgemm (CblasTrans, CblasNoTrans, 1, X, X, 0, V2);       
         invert(V2);
         gsl_blas_dgemv (CblasTrans, 1, X, y, 0, tau);
         gsl_blas_dgemv (CblasNoTrans, 1, V2, tau, 0, beta);       
      } else {
        gsl_matrix_memcpy(V2, Vinv);
        gsl_blas_dgemm (CblasTrans, CblasNoTrans, 1, X, X, 1, V2);
        invert(V2);
        gsl_matrix_memcpy(XtX, Vinv);      
        gsl_blas_dgemv (CblasNoTrans, 1, XtX, m, 0, tau);
        gsl_blas_dgemv (CblasTrans, 1, X, y, 1, tau);
        gsl_blas_dgemv (CblasNoTrans, 1, V2, tau, 0, beta);
      }
      rNIG(as, d+N, beta, V2, rngPntr, res, aux, &s2);
      for(j=0;j<p;j++){
        betaOut[(count*p)+j] = res[j];
      }
      s2Out[count] = s2;
      count++;
    }
  }
  delete [] jhProp;
  delete [] jvProp;
  delete [] jlProp;
  delete [] mu;
  delete [] muProp;
  delete [] alphaVec;
  delete [] hVarVec;
  delete [] hVarVecRev;
  delete [] res;
  delete [] aux;
  gsl_rng_free(rngPntr);
  gsl_matrix_free(X);
  gsl_matrix_free(Xprop);
  gsl_matrix_free(V);
  gsl_matrix_free(Vinv);
  gsl_vector_free(y);
  gsl_vector_free(m);
  gsl_vector_free(beta);
  gsl_vector_free(tau);
  gsl_vector_free(resids);
  gsl_matrix_free(X2);
  gsl_matrix_free(V2);
  gsl_matrix_free(XtX);
}


}
