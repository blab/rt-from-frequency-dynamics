functions {

// #include /functions/convolutions.stan

  real convolve_with_lifetime(vector X, vector Y_rev, int t, int l){
    int lx = max(1, t-l);
    int hx = min(l, t-1);
    return dot_product(X[lx:(t-1)], tail(Y_rev, hx));  
  }
//
    vector get_weekend_mat(int L,  real[] rho){
        matrix[L, 7] wk_mat = rep_matrix(0, L, 7);
        for (i in 1:L){
           wk_mat[i, (i % 7)+1] = 1;
        }
        return(wk_mat * to_vector(rho));
    }

// #include /functions/infections.stan

  vector get_infections(vector R, real I0, vector g_rev, int T, int l){
    vector[T] I;
    I[1] = I0;
    for (t in 2:T){
      I[t] = R[t] * convolve_with_lifetime(I, g_rev, t, l);
    }
    return(I);
  }

  vector[] get_infections_with_immunity(vector R, real I0, real phi0, real N, vector g_rev, int T, int l){
    vector[T] I;
    vector[T] phi;
    I[1] = I0;
    phi[1] = phi0;

    for (t in 2:T){
      I[t] =  (1 - phi[t-1]) * R[t] * convolve_with_lifetime(I, g_rev, t, l);
      phi[t] = phi[t-1] + (I[t] / N);
    }
    return({I, phi});
  }

// #include /functions/PCR_detectable.stan

  vector get_PCR_detectable(vector I, vector onset_rev, int T, int l){
    vector[T] I_prev;
    I_prev[1] = I[1] * onset_rev[l];
    for (t in 2:T){
      I_prev[t] = convolve_with_lifetime(I, onset_rev, t, l);
    }
    return(I_prev);
  }

// #include /functions/observe_neg_binom.stan

  vector get_cases_NB_rng(vector I, real alpha, real rho, int L){
    return(to_vector(neg_binomial_2_rng(rho * I, inv(alpha))));
  }

  vector get_cases_NB_wk_rng(vector I, real alpha, vector rho, int L){
    return(to_vector(neg_binomial_2_rng(rho .* I, inv(alpha))));
  }
}

data {
  int <lower=1> L;  // number of observed days 
  int seed_L;  // number of days to seed 
  int forecast_L;  // number of days to forecast 
  int cases[L]; // number of cases
  int<lower=1> lg; // max length of generation time
  real g[lg];
  int<lower=1> lo; // max length of onset distribution
  real onset[lo];
  int K; // Number of features
  matrix[(seed_L + L + forecast_L), K] features; // Features in linear regression on R
}

transformed data {
  // #include /data/trans_reverse_lifetimes.stan
  vector[lg] g_rev = to_vector(reverse(g));
  vector[lo] onset_rev = to_vector(reverse(onset));
  
  //#include /data/trans_obs_times.stan
  int L_ws = L + seed_L; // L with seed time
  int L_wf = L + forecast_L; // L with forecast included
  int L_wsf = L + seed_L + forecast_L; // Entire interval including seed and forecast
  real I0_max = 3./7. * sum(cases[1:7]); 
}

parameters {
  vector[K] b; // beta coeffiecients for features

// ADD_MORE_PARMS

//  #include /parms/parms_neg_binom.stan
  real<lower=0> zalpha;
  real<lower=0, upper=1> rho[7];
  real<lower=1> I0;
}

transformed parameters {

// ADD_MORE_TRANS_PARMS

  vector[L_wsf] R = exp(features * b);
  vector[L_wf] rho_vec = get_weekend_mat(L, rho);
  real<lower = 0> alpha = square(zalpha);

  vector[L_wsf] I = get_infections(R, I0, g_rev, L_wsf, lg);
  vector[L_wsf] I_prev = get_PCR_detectable(I, onset_rev, L_wsf, lo);
}

model {
  zalpha ~ normal(0.,1.);
  rho ~ beta(10,10);
  I0 ~ uniform(0.0, I0_max);

  // ADD_MORE_PRIORS
  
  cases ~ neg_binomial_2(rho_vec[1:L] .* I_prev[(seed_L+1):L_ws], inv(alpha));
}

generated quantities {
  //#include /generated_quants/quants_neg_binom.stan
  vector[L_wf] C;
  vector[L_wf] EC;

  C = get_cases_NB_wk_rng(tail(I_prev, L_wf), alpha, rho_vec, L_wf);
  EC = rho_vec .* tail(I_prev, L_wf);
}
