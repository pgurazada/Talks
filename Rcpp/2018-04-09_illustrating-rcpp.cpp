#include <Rcpp.h>

using namespace Rcpp;

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins("cpp11")]]

// Example 1: Illustrating the data types that map R objects to Rcpp objects

// [[Rcpp::export]]
double calculate_sum(NumericVector x) {
  double total = 0;
  for (NumericVector::iterator it = x.begin(); it != x.end(); ++it) {
    total += *it;
  }
  return total;
}

// Example 2: Illustrating how STL and C++11 features can be used

// [[Rcpp::export]]
int simple_product(std::vector<int> vec) {
  auto prod = 1;
  for (auto &x : vec) {       
    prod *= x;              
  }
  return prod;
}

// [[Rcpp::export]]
double mse(NumericVector y_pred, NumericVector y_truth) {
  return mean(pow(y_pred - y_truth, 2));
}


/*** R

*/
