#ifndef FFTPP_FFTSIZE_HPP
#define FFTPP_FFTSIZE_HPP

/* Author: Cory Brinck
 * FFT++ is a templated, header-only, Fast Fourier Transform (FFT) library
 * library which supports FFTs of arbitrary sizes with O(N*log(N)) operations
 * using a simple interface. Transform sizes consisting of small primes (<= 7)
 * perform best, and a function (fftSize(N)) is provided to determine a
 * preferable size >= N if the transform size is flexible.
 * Dependent only on the C++ standard library.
 */

#include <cstdint>

namespace fftpp {

inline size_t nextPowerOf2(size_t n) {
  if (n < 1 || (n << 1) < n)
    return 0; // Integer overflow;

  size_t powerOf2 = 1;
  while (powerOf2 < n)
    powerOf2 = powerOf2 << 1;

  return powerOf2;
}

// Returns a size >= N for which the FFT can be computed highly efficiently
inline size_t fftSize(size_t N) {
  if (N <= 16)
    return N;

  // Get the next power of 2
  size_t M = nextPowerOf2(N);

  auto checkReduction = [&M, N](size_t numerator, size_t denominator) {
    if (denominator <= M && M%denominator == 0) {
      size_t P = (M * numerator) / denominator;
      if (P >= N) {
        M = P;
        return true;
      }
    }
    return false;
  };

  if(!checkReduction(9, 16))
   checkReduction(3, 4);
  checkReduction(5, 6);
  checkReduction(27, 32);
  checkReduction(7, 8);
  
  return M;
}
} // namespace fftpp

#endif
