#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]

int rand(int a, int b) {
  //const unsigned int seed = 20130810;
  
  std::mt19937::result_type seed = 20130810;
  
  auto dice_rand = std::bind(std::uniform_int_distribution<int>(1,6),
                             std::mt19937(seed));
  
  return dice_rand();
}

/*** R

*/
