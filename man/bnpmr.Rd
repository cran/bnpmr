\name{bnpmr}
\alias{bnpmr}
\title{
  Monotonic regression
}
\description{
  Bayesian monotonic regression as described in Bornkamp and Ickstadt (2009).
}
\usage{
bnpmr(y, x, prior = NULL, start = NULL, niter = 10000, pMoves = NULL, thin = 1, burnIn = 0, prop = NULL, seed = 1, size = 50)
}
\arguments{
  \item{y}{
    vector of dependent values
  }
  \item{x}{
    vector of independent values (the code
    internally standardizes x to [0,1])
  }
  \item{prior}{
    A list specifying prior parameters\cr
    V, m, d, a - as defined in Biometrics paper p. 201  
    by default the noninformative choice of eqn (6)    
    is chosen.\cr
    vL, vU - lower and upper bound for uniform distribution
    of nu\cr
    la, lb - alpha, beta parameter of beta prior for m 
    (the TSP distribution mode)\cr
    alpha - prior parameter for Dirichlet distribution 
    (called gamma in the Biometrics paper)\cr
    lambda - prior parameter for truncated Poisson distr\cr
  }
  \item{start}{
    starting values for \cr
    nJ: number of components\cr
    jl: modes of TSP distributions (called m in the 
    Biometrics paper see eqn (5))\cr
    jv: called nu in the Biometrics paper (eqn (5))\cr
    jh: weights of the components (sum to 1)
    called w_i in the Biometrics paper\cr
  }
  \item{niter,burnIn,thin}{
    number of iterations, thinning
    and burn-in
  }
  \item{pMoves}{
    probabilities for the different move
    types
  }
  \item{prop}{
  }
  \item{seed}{
  }
  \item{size}{
    size of the vectors in C++ code (all vectors only have finite size
    determined by \code{size})
  }
}
\value{
  A list with entries (among others)\cr
  dimcount - posterior simulations of number of components\cr
  jl - posterior sims of m\cr
  jv - posterior sims of nu\cr
  jh - posterior sims of w\cr
  beta - posterior sims of beta vector\cr
  s2 - posterior sims of s**2\cr
}
\references{
Bornkamp, B. and Ickstadt, K. (2009). Bayesian Nonparametric Estimation of Continuous Monotone
Functions with Applications to DoseResponse Analysis. Biometrics, 65, 198-205
}
\author{
  Bjoern Bornkamp
}
\seealso{
   \code{\link{pred.bnpmr}}
}
\examples{
########################################################################
## example 1
## generate some example data
x <- seq(0,1,length=100) 
y <- 2+3*x/(0.05+x)+rnorm(100, 0, 1)
## run bnpmr function (with "default" parameters and priors)
res <- bnpmr(y, x)
sq <- seq(0,1,length=101)
aa <- pred.bnpmr(sq, res)
out005 <- apply(aa, 2, quantile, prob = 0.05)
out050 <- apply(aa, 2, median)
out095 <- apply(aa, 2, quantile, prob = 0.95)
## plot result
plot(x,y)
lines(sq, out005)
lines(sq, out050)
lines(sq, out095)
curve(2+3*x/(x+0.05), add=TRUE, col=2)

########################################################################
## example 2 with a sparse dose-design
## closer to what we actually see in pharmaceutical dose-finding trials
x <- rep(c(0,0.05,0.2,0.6,1), each = 10)
y <- 2+3*x^5/(0.05^5+x^5)+rnorm(length(x), 0, 1)

res <- bnpmr(y, x)
sq <- seq(0,1,length=101)
aa <- pred.bnpmr(sq, res)
out005 <- apply(aa, 2, quantile, prob = 0.05)
out050 <- apply(aa, 2, median)
out095 <- apply(aa, 2, quantile, prob = 0.95)

plot(x,y, ylim = c(0,8))
lines(sq, out005)
lines(sq, out050)
lines(sq, out095)
curve(2+3*x^5/(x^5+0.05^5), add=TRUE, col=2)

#### now reanalyse using different prior
## use prior that says placebo response = 0 with small uncertainty
## (just to check code)
V <- matrix(c(0.01,0,0,10), nrow=2)
prior <- list(alpha = 1, lambda = 0.5, m = c(0, 1), V=V, a=3.6, d=4, la=1,lb=2)
res2 <- bnpmr(y, x, prior = prior)
aa <- pred.bnpmr(sq, res2)
out005 <- apply(aa, 2, quantile, prob = 0.05)
out050 <- apply(aa, 2, median)
out095 <- apply(aa, 2, quantile, prob = 0.95)
lines(sq, out005, col = "green")
lines(sq, out050, col = "green")
lines(sq, out095, col = "green")

}
\keyword{ models }