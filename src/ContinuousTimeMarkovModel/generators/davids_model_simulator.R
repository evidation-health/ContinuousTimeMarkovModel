library(smfsb)

#simulate 6 state MJP
Q <- matrix(c(-1, 1, 0, 0, 0, 0,
              0, -0.5, 0.5, 0, 0, 0,
              0, 0, -0.5, 0.5, 0, 0,
              0, 0, 0, -0.3, 0.3, 0,
              0, 0, 0, 0, -0.5, 0.5,
              0, 0, 0, 0, 0.001, -0.001),
            nrow=6, ncol=6, byrow=TRUE)

pi <- c(1, 0, 0, 0, 0, 0)

set.seed(1991)
mjp <- rcfmc(5, Q, pi)
plot(mjp)

simulate_entire_mjp <- function(mjp, B, B0, Z, L, tauVec, Tn, K=10, s=1991){
  
  set.seed(s)
  time_steps <- sample(tauVec, size=Tn, replace=TRUE)
  times <- cumsum(time_steps)
  
  S <- mjp(times)
  
  #draw X given S
  draw_Xk <- function(S, k){
    Xt <- vector(mode="integer", length=Tn)
    
    Xt[1] <- rbinom(1, 1, B0[k,S[1]])
    for(t in 2:Tn){
      #X can only change when S changes
      if(S[t] == S[t-1])
        Xt[t] <- Xt[t-1]
      else{
        Xt[t] <- rbinom(1, 1, B[k,S[t],Xt[t-1]+1])
      }
    }
    return(Xt)
  }
  X <- matrix(nrow=K, ncol=Tn)
  for(k in 1:K){
    X[k,]  <- draw_Xk(S, k)
  }
  
  #draw O given X
  draw_Ot <- function(Xt, t){
    p <- prod(1-)
  }
  
  
}

B <- array(0.0, c(10, 6, 2))
B[,,1] <- matrix(c(0.4, 0.6, 0.9, 0.9, 0.9, 1.0,
                   0.1, 0.2, 0.3, 0.3, 0.3, 0.4,
                   0.4, 0.6, 0.7, 0.8, 0.9, 1.0,
                   0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                   0.0, 0.0, 0.0, 0.1, 0.1, 0.1,
                   0.1, 0.2, 0.2, 0.2, 0.4, 0.5,
                   0.2, 0.4, 0.4, 0.7, 0.7, 0.7,
                   0.0, 0.0, 0.1, 0.1, 0.5, 0.6,
                   0.1, 0.2, 0.3, 0.3, 0.3, 0.4,
                   0.0, 0.1, 0.1, 0.1, 0.2, 0.2),
  nrow=10, ncol=6, byrow=TRUE)
B[,,2] <- 1

B0 <- B[,,1]

Z <- 

tauVec <- c(1, 2, 4)
system.time(C <- sample_mjp(3.7*1e6, tauVec, mjp, 5))