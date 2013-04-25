\name{pred.bnpmr}
\alias{pred.bnpmr}
\title{
  Predict Bayesian monotonic regression object
}
\description{
  Predict Bayesian monotonic regression object
}
\usage{
pred.bnpmr(x, res)
}
\arguments{
  \item{x}{
    where to predict the function (should be a vector 
    with entries in [0,1])
}
\item{res}{
  the output of the bnpmr function
}
}
\value{
  A matrix containing the function simulations in the
rows.
}
\author{
  Bjoern Bornkamp
}

\seealso{
\code{\link{bnpmr}}
}
\keyword{ models }

