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
  // Get the next power of 2
  size_t M = nextPowerOf2(N);

  if (M >= 16) {
    size_t M_9_16 = (M * 9) / 16;
    if (M_9_16 >= N)
      return M_9_16;
  }

  if (M >= 8) {
    size_t M_5_8 = (M * 5) / 8;
    if (M_5_8 >= N)
      return M_5_8;
  }

  if (M >= 4) {
    size_t M_3_4 = (M * 3) / 4;
    if (M_3_4 >= N)
      return M_3_4;
  }

  return M;
}
} // namespace fftpp

#endif
