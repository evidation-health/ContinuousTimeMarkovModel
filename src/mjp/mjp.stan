functions {
  matrix get_Pt(matrix Q, real t){
    
  }
}
data {
  int<lower=1> M; // num states
  int<lower=1> r; // num jump lengths
  vector<lower=0>[r] tau; // jump lengths
  
  matrix[M,M] C[r]; 
}
parameters {
  matrix[M-1, M-1]<lower=0> Q_raw;
}
transformed parameters {
  matrix[M, M] Q;
  
  int index;
  for(i in 1:M){
    index <- 1
    for(j in 1:M){
      if(i == j)
        Q[i,j] <- -sum(Q_raw[i]);
      else{
        Q[i,j] <- Q_raw[i,index];
        index <- index + 1;
      }
    }
  }
}
model {
  for (k in 1:K) 
    theta[k] ~ dirichlet(alpha);
  for (k in 1:K)
    phi[k] ~ dirichlet(beta);
  for (t in 1:T)
    w[t] ~ categorical(phi[z[t]]);
  for (t in 2:T)
    z[t] ~ categorical(theta[z[t-1]]);

  { 
    // forward algorithm computes log p(u|...)
    real acc[K];
    real gamma[T_unsup,K];
    for (k in 1:K)
      gamma[1,k] <- log(phi[k,u[1]]);
    for (t in 2:T_unsup) {
      for (k in 1:K) {
        for (j in 1:K)
          acc[j] <- gamma[t-1,j] + log(theta[j,k]) + log(phi[k,u[t]]);
        gamma[t,k] <- log_sum_exp(acc);
      }
    }
    increment_log_prob(log_sum_exp(gamma[T_unsup]));
  }
}