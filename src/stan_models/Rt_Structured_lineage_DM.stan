
functions {
  
  int get_L_nonzero(int[] Z, int L){
    int L_nonzero = 0;
    for (i in 1:L){
      if(Z[i] > 0){ 
        L_nonzero += 1;
      }
    }
    return L_nonzero;
  }

  int[] get_nonzero_idx(int[] Z, int L_nonzero, int L){
    int cur_idx = 1;
    int non_zero_idx[L_nonzero];

    for (i in 1:L){
      if(Z[i] > 0){
        non_zero_idx[cur_idx] = i;
        cur_idx += 1;
      }
    }
    return non_zero_idx;
  }

  matrix get_weekend_mat(int L){
        matrix[L, 7] wk_mat = rep_matrix(0, L, 7);
        for (i in 1:L){
           wk_mat[i, (i % 7)+1] = 1;
        }
        return(wk_mat);
    }
  
  vector get_infections(vector R, real I0, vector g_rev, int T, int l, int seed_L){
    vector[T] I;
    I[1:seed_L] = rep_vector(I0, seed_L);

    // Loop for dates without full backlog
    for (t in seed_L:(l-1)){
      I[t+1] = R[t-seed_L+1] * dot_product(I[1:t], tail(g_rev, t));
    }
    // Loop for days with full backlog
    for (t in l:(T-1)){
      I[t+1] = R[t-seed_L+1] * dot_product(I[(t-l+1):t], g_rev);
    }

    return(I);
  }

  vector get_PCR_detectable(vector I, vector onset_rev, int T, int l){
    vector[T] I_prev;
    I_prev[1] = 0.;
    
    // Loop for days without full backlog
    for (t in 1:(l-1)){
      I_prev[t+1] = dot_product(I[1:t], tail(onset_rev, t));
    }
    // Loop for days with full backlog
    for (t in l:(T-1)){
      I_prev[t+1] = dot_product(I[(t-l+1):t], onset_rev);
    }

    return(I_prev);
  }

  // General function for convolving lifetimes
  // Useful for adding series of delays
  vector convolve_with_lifetime(vector Y, vector G_rev){
    int T = rows(Y);
    int l = rows(G_rev);

    vector[T] Z;
    Z[1] = 0.;

    for (t in 1:(l-1)){
      Z[t+1] = dot_product(Y[1:t], tail(G_rev, t));
    }
    for (t in l:(T-1)){
      Z[t+1] = dot_product(Y[(t-l+1):t], G_rev);
    }

    return(Z);
  }

  vector get_cases_NB_rng(vector I, real inv_alpha, real rho, int L){
    return(to_vector(neg_binomial_2_rng(rho * I, inv_alpha)));
  }

  vector get_cases_NB_wk_rng(vector I, real inv_alpha, vector rho, int L){
    return(to_vector(neg_binomial_2_rng(rho .* I, inv_alpha)));
  }
  
  vector row_sum(matrix X){
    return(X * ones_vector(cols(X)));
  }

  matrix get_raw_freq(matrix I_prev, vector total_prev){
    return(diag_pre_multiply(inv(total_prev), I_prev));
  }

   matrix[] observe_frequencies_rng(matrix sim_freqs, int[] N_sequences, int[] idx_nonzero, real trans_xi, int L, int N_lineage){
    matrix[L, N_lineage] obs_counts = rep_matrix(0., L, N_lineage);
    matrix[L, N_lineage] obs_freqs = rep_matrix(0., L, N_lineage);
    for (t in idx_nonzero){
        obs_counts[t, :] = to_row_vector(multinomial_rng(dirichlet_rng( 0.001 + trans_xi * to_vector(sim_freqs[t, :])), N_sequences[t]));
        obs_freqs[t, :] = obs_counts[t, :] / N_sequences[t];
    }
    return({obs_counts, obs_freqs});
  }

/*      matrix[] observe_frequencies_rng(matrix sim_freqs, int[] N_sequences, int[] idx_nonzero, int L, int N_lineage){ */
/*     matrix[L, N_lineage] obs_counts = rep_matrix(0., L, N_lineage); */
/*     matrix[L, N_lineage] obs_freqs = rep_matrix(0., L, N_lineage); */
/*     for (t in idx_nonzero){ */
/*         obs_counts[t, :] = to_row_vector(multinomial_rng(to_vector(sim_freqs[t, :]), N_sequences[t])); */
/*         obs_freqs[t, :] = obs_counts[t, :] / N_sequences[t]; */
/*     } */
/*     return({obs_counts, obs_freqs}); */
/*   } */

  real dir_multinomial_lpmf(int [] x, vector alpha){
    int n = sum(x);
    real alpha_0 = sum(alpha);
    return lgamma(alpha_0) - lgamma(n + alpha_0) + sum(lgamma(to_vector(x) + alpha)) - sum(lgamma(alpha));
  }
}

data {
  int <lower=1> N_lineage; // number of lineages
  int <lower=1> L;  // number of days 
  int seed_L;  // number of days to seed 
  int forecast_L;  // number of days to forecast 
  int cases[L]; // number of cases
  int num_sequenced[L, N_lineage]; // number of occurances in each lineage in sample from that day
  int N_sequences[L]; // total count of sequences
  int<lower=1> l; // max length of generation time
  real g[l];
  real onset[l];
  int K; // Number of features
  matrix[L, K] features; // Features in linear regression on R
}

transformed data {
    vector[l] g_rev = to_vector(reverse(g));
    vector[l] onset_rev = to_vector(reverse(onset));

    int L_ws = L + seed_L; // L with seed time
    int L_wf = L + forecast_L; // L with forecast included
    int L_wsf = L + seed_L + forecast_L; // Entire interval including seed and forecast

    //vector[N_lineage] I0_max = (3./7. * sum(cases[1:7]) * to_vector(num_sequenced[1, :]) / N_sequences[1]) + 0.01 * cases[1];
    real I0_max = 10. / 7. * sum(cases[1:7]);

    // Find days with lineage data
    int L_nonzero = get_L_nonzero(N_sequences, L);
    int idx_nonzero[L_nonzero] = get_nonzero_idx(N_sequences, L_nonzero, L);
    
    // Weekday matrix for reporting rate
    matrix[L_wf, 7] wk_mat = get_weekend_mat(L_wf);
}

parameters {
  vector[K] b;  // beta coeffiecients for features
  vector[N_lineage-1] v; // Variant specific modifiers
  
// ADD_MORE_PARMS

  real<lower=0> zalpha;
  real<lower=0, upper=1> rho[7];
  real<lower=0, upper=1> zI0[N_lineage];
  /* real<lower=0, upper=I0_max> I0[N_lineage]; */
  real<lower=0, upper=0.05> xi; // Overdispersion relative to multinomial
}

transformed parameters {

// ADD_MORE_TRANS_PARMS

  /* vector[L_wsf] R_base; */ // Last column of R
  vector[N_lineage] growth_advantage =  append_row(exp(v), 1.);
  matrix[L_wf, N_lineage] R =  exp(features * b)  * growth_advantage'; // log(R) = X \beta + \delta_v
  vector[L_wf] rho_vec = wk_mat * to_vector(rho);
  real inv_alpha = inv_square(zalpha);
  real trans_xi = inv(xi) - 1; // (1-xi)/xi
  vector[N_lineage] I0 = I0_max * to_vector(zI0);

  matrix[L_wsf, N_lineage] I;
  matrix[L_wsf, N_lineage] I_prev;

  for (lineage in 1:N_lineage){
      I[:, lineage] = get_infections(R[:,lineage], I0[lineage], g_rev, L_wsf, l, seed_L);
      I_prev[:, lineage] = get_PCR_detectable(I[:, lineage], onset_rev, L_wsf, l);
  }

  vector[L_wsf] total_prev = row_sum(I_prev);
  matrix[L_wf, N_lineage] sim_freqs = get_raw_freq(I_prev[(seed_L+1):L_wsf, :], total_prev[(seed_L+1):L_wsf]);
}

model {
  zalpha ~ std_normal();
  rho ~ beta(10,10);
  zI0 ~ uniform(0.0, 1.0);
  xi ~ beta(1, 99);
  v ~ std_normal();

  // ADD_MORE_PRIORS

  // Get case likelihood
  cases ~ neg_binomial_2(rho_vec[1:L] .* total_prev[(seed_L+1):L_ws], inv_alpha);
   
  // Get lineage likelihood
  for (t in idx_nonzero){
    num_sequenced[t,:] ~ dir_multinomial(0.001 + trans_xi * to_vector(sim_freqs[t, :]));
    /* num_sequenced[t,:] ~ multinomial(to_vector(sim_freqs[t, :])); */
  }
}

generated quantities {
  vector[L_wf] R_average = rows_dot_product(R, sim_freqs); 

  // Lineage specific counts
  matrix[L_wf, N_lineage] scaled_prev;
  matrix[L_wf, N_lineage] lineage_prev;
  real mean_rho = mean(rho_vec);

  for (lineage in 1:N_lineage){
    lineage_prev[:, lineage] = I_prev[(seed_L+1):, lineage];
    scaled_prev[:, lineage] = mean_rho * I_prev[(seed_L+1):, lineage];
  }

  // Aggregate counts
  vector[L_wf] EC_smooth = mean_rho * total_prev[(seed_L+1):L_wsf];
  vector[L_wf] EC = rho_vec .* total_prev[(seed_L+1):L_wsf];
  //vector[L_wf] C_sample = to_vector(neg_binomial_2_rng(EC, inv_alpha));

  // Posterior predictive counts and frequencies.
  matrix[L, N_lineage] obs_counts;
  matrix[L, N_lineage] obs_freqs;
  {
  // Intermediate output. Do not save.
  matrix[L, N_lineage] obs_quants[2] = observe_frequencies_rng(sim_freqs[1:L,:], N_sequences, idx_nonzero, trans_xi, L, N_lineage);
  // Multinomial version
  /* matrix[L, N_lineage] obs_quants[2] = observe_frequencies_rng(sim_freqs[1:L,:], N_sequences, idx_nonzero, L, N_lineage); */

  // Save these.
  obs_counts = obs_quants[1];
  obs_freqs = obs_quants[2];
  }

}

