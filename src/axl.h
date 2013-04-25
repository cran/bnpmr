void invert(gsl_matrix *X);
double determinant(gsl_matrix *X);
double proddiag(gsl_matrix *X);
double ssqElements(gsl_vector *v);
void read(const char *name, double vec[]);
int lg(const char *name);
double ldbeta (const double x, const double a, const double b);
double nldbeta (const double x, const double a, const double b);
double lddirichlet(const size_t K, const double alpha[], const double theta[]);
double ptsp(const double x, const double m, const double n);
double intptsp(const double x, const double m, const double n);
void choldc(gsl_matrix *a);
void rMVN(double *res, double *aux, gsl_matrix *S, gsl_vector *m, const double s2, const gsl_rng *rng_ptr);
void rNIG(const double a, const double d, gsl_vector *m, gsl_matrix *V, const gsl_rng * rngPntr, 
           double *res, double *aux, double *s2);
void array2gslmatrix(double arr[], gsl_matrix *M, int dim);
void array2gslvector(double arr[], gsl_vector *vec, int dim);
