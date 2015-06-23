library(smfsb)

#simulate 5 state MJP
Q <- matrix(c(-6, 2, 2, 1, 1,
              1, -4, 0, 1, 2,
              1, 0, -4, 2, 1,
              2, 1, 0, -3, 0,
              1, 1, 1, 1, -4),
            nrow=5, ncol=5, byrow=TRUE)

pi <- c(0.2, 0.2, 0.2, 0.2, 0.2)

system.time(mjp <- rcfmc(1e8, Q, pi))

#discreetly samples MJP up to time T with intervals
#between samples uniformly selected from vector tau.
sample_mjp <- function(T_stop, tauVec, mjp, nStates){
  C <- array(0, c(nStates, nStates, length(tauVec)))
  
  t <- 0
  while(t <= T_stop){
    tau <- sample(tau, size=1)
    i <- mjp(t)
    j <- mjp(t+tau)
    
    C[i,j,which(tauVec == tau)] <- C[i,j,which(tauVec == tau)] + 1
    t <- t + tau
  }
  
  return(C)
}

tauVec <- c(0.01, 0.1, 1)
system.time(sample_mjp(3.7*1e6, tauVec, mjp, 5))
