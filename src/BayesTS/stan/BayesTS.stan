data {
  int T; // Length of time interval
  vector[T] t; // time vector
  // ADD_Y_INIT // Time Series // Need to update so it can have multple demes

  int K; // Number of features
  matrix[T, K] features; // Features 
}

parameters {
  vector[K] b;
  // ADD_OBS_PARMS
}

transformed parameters{
  // ADD_EY_TRANS
}

model {
  // ADD_MORE_PRIORS
  
  // ADD_OBS_PRIOR
  
  Y ~ // ADD_OBS_DIST
}

// Add generative quantity section for post preds?
//
// generated quanatities {
// vector[T] Y_pred;
// Y_pred = // ADD_GEN_RNG
//
// }
