\name{ptsp}
\alias{ptsp}

\title{
  Distribution function of the two-sided power distribution
}
\description{
  Distribution function of the two-sided power distribution
}
\usage{
ptsp(x, m, n)
}
\arguments{
  \item{x}{
    Vector of numeric values in [0,1]
}
  \item{m,n}{
    Distribution parameters, m is the mode of the distribution
  }
}
\references{
  van Dorp, R. and Kotz, S. (2002) The standard two-sided power
  distribution and its properties. The American Statistician, 56, 90-99
}
\author{
  Bjoern Bornkamp
}

\examples{
curve(ptsp(x, 0.5, 1), 0, 1)
curve(ptsp(x, 0.5, 2), 0, 1, add = TRUE)
curve(ptsp(x, 0.5, 5), 0, 1, add = TRUE)
curve(ptsp(x, 0.5, 10), 0, 1, add = TRUE)
}
\keyword{ models}

