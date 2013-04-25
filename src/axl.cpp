#include<iostream>
//#include<conio.h>
#include<fstream>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_sf_gamma.h>
#include <fstream> /* for in/output*/
using namespace std;

/* Inverts symmetric 2x2 or 3x3 gsl_matrix */
void invert(gsl_matrix *X){
  double den;
  int i,j,dim;
  dim = X->size2;
  if(dim==2){
    double a=0,b=0,c=0,bcop=0;
    a = gsl_matrix_get(X, 0, 0);
    b = gsl_matrix_get(X, 0, 1);
    c = gsl_matrix_get(X, 1, 1);
    den = a*c - b*b;
    gsl_matrix_set(X, 0, 0, c/den);
    bcop = -b/den;
    gsl_matrix_set(X, 0, 1, bcop);
    gsl_matrix_set(X, 1, 0, bcop);
    gsl_matrix_set(X, 1, 1, a/den);
  } else {
    double a=0,b=0,c=0,d=0,e=0,f=0;
    double cop01 = 0, cop02 = 0, cop12 = 0;
    a = gsl_matrix_get(X, 0, 0);
    b = gsl_matrix_get(X, 0, 1);
    c = gsl_matrix_get(X, 0, 2);
    d = gsl_matrix_get(X, 1, 1);
    e = gsl_matrix_get(X, 1, 2);
    f = gsl_matrix_get(X, 2, 2);
    den = a*(d*f-e*e)+b*(c*e-b*f)+c*(b*e-c*d);
    gsl_matrix_set(X, 0, 0, (d*f-e*e)/den);
    cop01 = (c*e-b*f)/den;
    gsl_matrix_set(X, 0, 1, cop01);
    gsl_matrix_set(X, 1, 0, cop01);
    cop02 = (b*e-c*d)/den;
    gsl_matrix_set(X, 0, 2, cop02);    
    gsl_matrix_set(X, 2, 0, cop02);
    gsl_matrix_set(X, 1, 1, (a*f-c*c)/den);
    cop12 = (b*c-a*e)/den;
    gsl_matrix_set(X, 1, 2, cop12);
    gsl_matrix_set(X, 2, 1, cop12);
    gsl_matrix_set(X, 2, 2, (a*d-b*b)/den);    
  }  
}

/* calculate determinant of symmetric matrix */
double determinant(gsl_matrix *X){
  double den;
  int dim;
  dim = X->size2;
  if(dim==2){
    double a=0,b=0,c=0;
    a = gsl_matrix_get(X, 0, 0);
    b = gsl_matrix_get(X, 0, 1);
    c = gsl_matrix_get(X, 1, 1);
    return a*c - b*b;
  } else {
    double a=0,b=0,c=0,d=0,e=0,f=0;
    a = gsl_matrix_get(X, 0, 0);
    b = gsl_matrix_get(X, 0, 1);
    c = gsl_matrix_get(X, 0, 2);
    d = gsl_matrix_get(X, 1, 1);
    e = gsl_matrix_get(X, 1, 2);
    f = gsl_matrix_get(X, 2, 2);
    return a*(d*f-e*e)+b*(c*e-b*f)+c*(b*e-c*d);
  }  
}


/* Calculate the product of diagonal elements of a gsl matrix */
double proddiag(gsl_matrix *X){
  int dm,i;
  double out=1;
  dm = X->size2;
  for(i=0;i<dm;i++){
    out *= gsl_matrix_get(X, i, i);
  }
  return out;
}

/* Read an array into a square gsl matrix */
void array2gslmatrix(double arr[], gsl_matrix *M, int dim){
  int j,i;
  for(i=0;i<dim;i++){
    for(j=0;j<dim;j++){
      gsl_matrix_set(M, i, j, arr[i*dim+j]);
    }
  }
}

/* read an array into a gsl vector */
void array2gslvector(double arr[], gsl_vector *vec, int dim){
  int i;
  for(i=0;i<dim;i++){
    gsl_vector_set(vec, i, arr[i]);
  }
}

/* Sum of the squared elements of a gsl vector */
double ssqElements(gsl_vector *v){
  int i,n;
  double res;
  n = v->size;
  gsl_vector_mul (v,v);
  for(i = 0; i < n; i++){
    res += gsl_vector_get (v, i);
  }
  return res;
}

/* function to read numbers in a double array */
void read(const char *name, double vec[]){
  int i=0;
  double x2=0;
  ifstream inFile;
  
  inFile.open(name);
  if (!inFile) {
    //cout << "Unable to open file";
    //exit(1); // terminate with error
  }
  
  while (inFile >> x2) {
      vec[i] = x2;
      i++;
  }
  inFile.close();
}

/* function to read numbers in a vector */
int lg(const char *name){
  int i=0;
  double x2=0;
  ifstream inFile;
  
  inFile.open(name);
  if (!inFile) {
    //cout << "Unable to open file";
    //exit(1); // terminate with error
  }
  
  while (inFile >> x2) {
      i++;
  }
  inFile.close();
  return i;
}

/* non-normalized beta density on log scale */
double ldbeta (const double x, const double a, const double b){
  if (x < 0 || x > 1)
    {
      return 0 ;
    }
  else 
    {
      double p;
      p = log(x) * (a - 1)  + log1p(-x) * (b - 1);
      return p;
    }
}

/* normalized beta density on log scale */
double nldbeta (const double x, const double a, const double b){
  if (x < 0 || x > 1)
    {
      return 0 ;
    }
  else 
    {
      double p;

      double gab = gsl_sf_lngamma (a + b);
      double ga = gsl_sf_lngamma (a);
      double gb = gsl_sf_lngamma (b);

      p = gab - ga - gb + log(x) * (a - 1)  + log1p(-x) * (b - 1);

      return p;
    }
}

/* non-normalized Dirichlet density on log scale */
double lddirichlet(const size_t K, const double alpha[], const double theta[]){
  size_t i;
  double log_p = 0.0;
  double sum_alpha = 0.0;

  for (i = 0; i < K; i++)
    {
      log_p += (alpha[i] - 1.0) * log (theta[i]);
    }

  return log_p;
}

/* distribution fct. of two-sided power distribution */
double ptsp(const double x, const double m, const double n){
  if(x <= m){
    return m*pow(x/m, n);
  } else {
    return 1.0-(1.0-m)*pow((1.0-x)/(1.0-m),n);
  }
}

/* integrated dist. fct. of two-sided power distribution */
double intptsp(const double x, const double m, const double n){
  if(x <= m){
    return m*m/(n+1)*pow(x/m,n+1);
  } else {
    return x-m+(1-m)*(1-m)/(n+1)*pow((1-x)/(1-m), n+1)+(2*m-1)/(n+1);
  }
}

/* Performs Cholesky decomp. Returns lower diagonal matrix */
void choldc(gsl_matrix *a){
	int i,j,k,n;
  n = a->size2;
	double sum,p;

	for (i=0;i<n;i++) {
		for (j=i;j<n;j++) {
			for (sum=gsl_matrix_get(a,i,j),k=i-1;k>=0;k--) sum -= gsl_matrix_get(a,i,k)*gsl_matrix_get(a,j,k);
			if (i == j) {
				if (sum <= 0.0)/* warning message */;
				p=sqrt(sum);
				gsl_matrix_set(a,i,j,p);
			} else gsl_matrix_set(a,j,i,sum/p);
		}
	}
  for(i=0;i<n;i++) {
    for(j=i+1;j<n;j++){
      gsl_matrix_set(a,i,j,0.0);
    }
  }
}

/* generate multivariate normal variate */
void rMVN(double *res, double *aux, gsl_matrix *S, gsl_vector *mn, const double s2, const gsl_rng *rng_ptr){
  // res - result (one random variate)
  // aux - auxiliary pointer of length p
  // s2*S - cov matrix, m - mean
	int i,j,p;
	double sum;
	p = S->size2;
	choldc(S);
	for(i=0;i<p;i++){
	    aux[i] = gsl_ran_gaussian (rng_ptr,sqrt(s2));
  }
 	for (i=0;i<p;i++){
  	sum=0.0;
		for(j=0;j<p;j++) sum += aux[j]*gsl_matrix_get(S,i,j);
		res[i] = sum + gsl_vector_get(mn,i);
	}
}

/* simulate from normal inverse gamma distribution */
void rNIG(double a, double d, gsl_vector *mn, 
           gsl_matrix *V, const gsl_rng * rngPntr, 
           double *res, double *aux, double *s2){
  int i=0,dim=0;
  *s2 = 1/gsl_ran_gamma_mt (rngPntr, d/2, 2/a);
  rMVN(res, aux, V, mn, *s2, rngPntr);
}
