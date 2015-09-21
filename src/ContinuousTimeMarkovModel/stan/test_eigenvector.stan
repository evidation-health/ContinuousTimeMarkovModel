data {
  int M;
  matrix[M,M] Q; 
}
parameters {
  real mu;
}
model {
  vector[M] val;
  val <- eigenvalues(Q);
  //vec <- eigenvectors(Q);

  //print("val: ", val, "\n vec:", vec);

  mu ~ normal(0, 1);
}