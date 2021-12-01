data {
  int K; // Number of features
  int N; // Number of observations
  int D; // Number of classes
  int Y[N, D]; // Counts for each class in each observation
  matrix[N, K] X; // Number of features
}

parameters {
  matrix[K, D-1] beta;
}

transformed parameters {
  matrix[D, N] Yhat = (X * append_col(beta, rep_vector(0, K)))'; 
  matrix[D,N] probs;
  
  for (n in 1:N){
    probs[:, n] = softmax(Yhat[:, n]);
  }
}

model {
  to_vector(beta) ~ normal(0, 5);

  for (n in 1:N)
    Y[n, :] ~ multinomial(probs[:, n]);
}
