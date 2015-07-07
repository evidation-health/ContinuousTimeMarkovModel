library(smfsb)

#simulate 5 state MJP
Q <- matrix(c(-6, 2, 2, 1, 1,
              1, -4, 0, 1, 2,
              1, 0, -4, 2, 1,
              2, 1, 0, -3, 0,
              1, 1, 1, 1, -4),
            nrow=5, ncol=5, byrow=TRUE)

pi <- c(0.2, 0.2, 0.2, 0.2, 0.2)

set.seed(1991)
system.time(mjp <- rcfmc(1e8, Q, pi))

simulate_entire_mjp <- function(mjp, B, B0, Z, tauVec, Tn){
  
  time_steps <- sample(tauVec, size=Tn, replace=TRUE)
  times <- cumsum(time_steps)
  
  S = mjp(times)
  
  
}

tauVec <- c(0.01, 0.1, 1)
system.time(C <- sample_mjp(3.7*1e6, tauVec, mjp, 5))