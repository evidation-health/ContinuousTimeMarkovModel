S <- c(0,  1,  1,  2,  2,  5,
       5,  5,  5,  5,  5,  5,  5,
       5,  5,  5,  5,  5,  5,  5)
S <- S+1

observed_jumps <- c(2, 4, 2, 2, 4, 2, 2,
                    2, 1, 2, 2, 2,
                    4, 1, 1, 1, 1, 4, 2)

computeC <- function(S, observed_jumps){
  C <- array(0, dim=c(6,6,3))
  
  step_sizes <- sort(unique(observed_jumps))
  
  for(i in 1:length(observed_jumps)){
    tau_index = which(step_sizes == observed_jumps[i])
    
    i_ind <- S[i]
    j_ind <- S[i+1]
    C[i_ind,j_ind,tau_index] <- C[i_ind,j_ind,tau_index] + 1
  }
  
  return(C)
}

C <- computeC(S, observed_jumps)

Q <- matrix(c(-0.631925, 0.631921,0.000001,0.000001,0.000001,0.000001,
              0.000001,-0.229489, 0.229485,0.000001,0.000001,0.000001,
              0.000001,0.000001, -0.450542, 0.450538,0.000001,0.000001,
              0.000001,0.000001,0.000001, -0.206046, 0.206042,0.000001,
              0.000001,0.000001,0.000001,0.000001,-0.609586,0.609582,
              0.000001,0.000001,0.000001,0.000001,0.00001, -1.4e-05),
            nrow=6, ncol=6, byrow=TRUE)

logp <- function(Q, S, observed_jumps){
  C <- computeC(S, observed_jumps)
  step_sizes <- sort(unique(observed_jumps))
  
  l = 0.0
  for(i in 1:length(step_sizes)){
    e <- eigen(Q)
    lambda <- e$values
    U <- e$vectors
    
    tau <- step_sizes[i]
    exp_tD <- diag(exp(tau*lambda))
    
    U_inv <- solve(U)
    P <- U %*% exp_tD %*% U_inv
    
    l = l + sum(C[,,i]*log(P))
  }
  
  return(l)
}

logp(Q, S, observed_jumps)