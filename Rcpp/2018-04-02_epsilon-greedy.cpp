#define ARMA_64BIT_WORD
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

/**
 * @title The epsilon-greedy algorithm for multi-armed bandits
 * @author Pavan Gurazada
 * @licence MIT 
 * @summary Implementation of the epsilon greedy algorithm for the multi-armed 
 * bandit problem.
 *
*/

struct EpsilonGreedy {
  double epsilon;
  arma::uvec counts;
  arma::vec values;
};

int index_max(arma::uvec& v) {
  return v.index_max();
}

int index_rand(arma::vec& v) {
  int s = arma::randi<int>(arma::distr_param(0, v.n_elem-1));
  return s;
}

int select_arm(EpsilonGreedy& algo) {
  if (R::runif(0, 1) > algo.epsilon) {
    return index_max(algo.values);
  } else {
    return index_rand(algo.values);
  }
}

void update(EpsilonGreedy& algo, int chosen_arm, double reward) {
  algo.counts[chosen_arm] += 1;
  
  int n = algo.counts[chosen_arm];
  double value = algo.values[chosen_arm];
  
  algo.values[chosen_arm] = ((n-1)/n) * value + (1/n) * reward;
}

struct BernoulliArm {
  double p;
};

int draw(BernoulliArm arm) {
  if (R::runif(0, 1) > arm.p) {
    return 0;
  } else {
    return 1;
  }
}

// [[Rcpp::export]]
DataFrame test_algorithm(double epsilon, std::vector<double>& means, int n_sims, int horizon) {
  
  std::vector<BernoulliArm> arms;
  
  for (auto& mu : means) {
    BernoulliArm b = {mu};
    arms.push_back(b);
  }
  
  std::vector<int> sim_num, time, chosen_arms;
  std::vector<double> rewards;
  
  for (int sim = 1; sim <= n_sims; ++sim) {
    
    arma::uvec counts(means.size(), arma::fill::zeros);
    arma::vec values(means.size(), arma::fill::zeros); 
    
    EpsilonGreedy algo = {epsilon, counts, values};
    
    for (int t = 1; t <= horizon; ++t) {
      int chosen_arm = select_arm(algo);
      double reward = draw(arms[chosen_arm]);
      update(algo, chosen_arm, reward);
      
      sim_num.push_back(sim);
      time.push_back(t);
      chosen_arms.push_back(chosen_arm);
      rewards.push_back(reward);
    }
  }
  
  DataFrame results = DataFrame::create(Named("sim_num") = sim_num,
                                        Named("time") = time,
                                        Named("chosen_arm") = chosen_arms,
                                        Named("reward") = rewards);
  
  return results;
}


/***R

library(tidyverse)
means <- c(0.1, 0.1, 0.1, 0.1, 0.9)

total_results <- data.frame(sim_num = integer(), time = integer(), chosen_arm = integer(),
                            reward = numeric(), epsilon = numeric())

for (epsilon in seq(0.1, 0.5, length.out = 5)) {

  cat("Starting with ", epsilon, " at: ", format(Sys.time(), "%H:%M"), "\n")

  results <- test_algorithm(epsilon, means, 5000, 250)
  results$epsilon <- epsilon

  total_results <- rbind(total_results, results)

}

avg_reward <- total_results %>% group_by(time, epsilon) %>%
                                summarize(avg_reward = mean(reward))

dev.new()

ggplot(avg_reward) +
  geom_line(aes(x = time, y = avg_reward,
                group = epsilon, color = epsilon), size = 1) +
  scale_color_gradient(low = "grey", high = "black") +
  labs(x = "Time",
       y = "Average reward",
       title = "Performance of the Epsilon-Greedy Algorithm",
       color = "epsilon\n")

#' Frequency of selecting the correct arm

freq_correct <- total_results %>% group_by(time, epsilon) %>%
                                  summarize_at(vars(chosen_arm), function(x) mean(x == 4))

ggplot(freq_correct) +
  geom_line(aes(x = time, y = chosen_arm,
                group = epsilon, color = epsilon), size = 1) +
                  scale_color_gradient(low = "grey", high = "black") +
                  labs(x = "Time",
                       y = "Probability of choosing the correct arm",
                       title = "Performance of the Epsilon-Greedy Algorithm",
                       color = "epsilon\n")
*/