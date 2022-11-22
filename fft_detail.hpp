#ifndef FFTPP_FFT_DETAIL_HPP
#define FFTPP_FFT_DETAIL_HPP

/* Author: Cory Brinck
 * FFT++ is a templated, header-only, Fast Fourier Transform (FFT) library
 * which supports efficient FFTs of arbitrary sizes. Transform sizes with
 *  small prime factors (<= 7) perform best, but sizes with large prime
 * factors are O(N*log(N)) efficient.
 *
 * A function (M = fftSize(N)) is provided to determine a preferable size >= N
 * when the choice of transform size is flexible.
 *
 * Dependent only on the C++ standard library using features of C++11.
 */

#include "fft_size.hpp"
#include <array>
#include <complex>
#include <map>
#include <memory>
#include <vector>

namespace fftpp {

namespace detail {
template <typename T> T twoPiOverN(size_t N) {
  return static_cast<T>(2 * 3.141592653589793238462643383279L / N);
}

template <typename T>
std::complex<T> getTwiddleFactor(size_t n, T negativeTwoPiOverN) {
  return std::polar<T>(1, static_cast<T>(n * negativeTwoPiOverN));
}

template <typename T>
std::shared_ptr<std::complex<T>[]> getNp1TwiddleFactors(size_t N) {
  std::shared_ptr<std::complex<T>[]> factors(new std::complex<T>[N + 1]);
  factors[0] = 1;
  factors[N] = 1;
  auto negativeTwoPiOverN = -twoPiOverN<T>(N);
  for (size_t n = 1, Nmid = (N + 2) / 2; n < Nmid; ++n) {
    factors[n] = getTwiddleFactor<T>(n, negativeTwoPiOverN);
    factors[N - n] = std::conj(factors[n]);
  }

  return factors;
}

// Base class for FFTs where the size is known at runtime
template <typename T>
class FFTBase {
public:
  virtual size_t size() const = 0;

  virtual void fftN(std::complex<T> *pY, size_t ystride,
                    const std::complex<T> *pX, size_t xstride,
                    const std::complex<T> *pTwiddles, int64_t twiddleStride,
                    std::complex<T> *pTmp) const = 0;
};

// Concept class for FFTs of a fixed compile-time size that use no temporary
// buffer
template <typename T, int N> class FixedSizeBufferlessFFT {
public:
  // static constexpr int size() { return N; }
  // static void fft(std::complex<T>* pY, size_t ystride, const std::complex<T>*
  // pX,
  //   size_t xstride, const std::complex<T>* pTwiddles, int64_t twiddleStride);
};

// FFT of size 2
template <typename T> class FixedSizeBufferlessFFT<T, 2> {
public:
  static constexpr int size() { return 2; }
  static void fft(std::complex<T> &y0, std::complex<T> &y1,
                  const std::complex<T> &x0, const std::complex<T> &x1) {
    y0 = x0 + x1;
    y1 = x0 - x1;
  }

  static void fft(std::complex<T> *pY, size_t ystride,
                  const std::complex<T> *pX, size_t xstride) {
    fft(pY[0], pY[ystride], pX[0], pX[xstride]);
  }

  static void fft(std::complex<T> *pY, size_t ystride,
                  const std::complex<T> *pX, size_t xstride,
                  const std::complex<T> *, int64_t) {
    fft(pY[0], pY[ystride], pX[0], pX[xstride]);
  }
};

// Subfunction to efficiently accumulate sub-portions of odd-sized FFTs
template <typename T>
inline void subFFTOddAccumulate(std::complex<T> &y1, std::complex<T> &y2,
                                const std::complex<T> &xSum,
                                const std::complex<T> &xDif,
                                const std::complex<T> &twiddle) {
  // y1 = x0 + std::complex<T>(x1.real*tr - x1.imag*ti, x1.real*ti + x1.imag*tr)
  // + std::complex<T>(x2.real*tr + x2.imag*ti, x2.imag*tr - x2.real*ti); y2 =
  // x0 + std::complex<T>(x2.real*tr - x2.imag*ti, x2.real*ti + x2.imag*tr) +
  // std::complex<T>(x1.real*tr + x1.imag*ti, x1.imag*tr - x1.real*ti);

  // y1 = x0 + std::complex<T>(x1.real*tr - x1.imag*ti + x2.real*tr +
  // x2.imag*ti, x1.real*ti + x1.imag*tr + x2.imag*tr - x2.real*ti); y2 = x0 +
  // std::complex<T>(x1.real*tr + x1.imag*ti + x2.real*tr - x2.imag*ti,
  // x2.real*ti + x2.imag*tr + x1.imag*tr - x1.real*ti);

  // y1 = x0 + std::complex<T>((x1.real + x2.real)*tr + (x2.imag - x1.imag)*ti,
  // (x1.real - x2.real)*ti + (x1.imag + x2.imag)*tr); y2 = x0 +
  // std::complex<T>((x1.real + x2.real)*tr + (x1.imag  - x2.imag)*ti, (x2.real
  // - x1.real)*ti + (x1.imag + x2.i)*tr);

  // y1 = x0 + std::complex<T>(sum.real*tr - dif.imag*ti, dif.real*ti +
  // sum.imag*tr); y2 = x0 + std::complex<T>(sum.real*tr + dif.imag*ti,
  // sum.imag*tr - dif.real*ti);

  // y1 = x0 + std::complex<T>(trSum.real - tiDif.imag, trSum.imag +
  // tiDif.real); y2 = x0 + std::complex<T>(trSum.real + tiDif.imag, trSum.imag
  // - tiDif.real);

  std::complex<T> trSum = xSum * twiddle.real();
  std::complex<T> tiDif = xDif * twiddle.imag();
  y1.real(y1.real() + trSum.real() - tiDif.imag());
  y1.imag(y1.imag() + trSum.imag() + tiDif.real());
  y2.real(y2.real() + trSum.real() + tiDif.imag());
  y2.imag(y2.imag() + trSum.imag() - tiDif.real());
}

// Subfunction to efficiently accumulate and assign sub-portions of odd-sized
// FFTs
template <typename T>
inline void
subFFTOddAssign(std::complex<T> &y1, std::complex<T> &y2,
                const std::complex<T> &x0, const std::complex<T> &xSum,
                const std::complex<T> &xDif, const std::complex<T> &twiddle) {
  // y1 = x0 + std::complex<T>(x1.real*tr - x1.imag*ti, x1.real*ti + x1.imag*tr)
  // + std::complex<T>(x2.real*tr + x2.imag*ti, x2.imag*tr - x2.real*ti); y2 =
  // x0 + std::complex<T>(x2.real*tr - x2.imag*ti, x2.real*ti + x2.imag*tr) +
  // std::complex<T>(x1.real*tr + x1.imag*ti, x1.imag*tr - x1.real*ti);

  // y1 = x0 + std::complex<T>(x1.real*tr - x1.imag*ti + x2.real*tr +
  // x2.imag*ti, x1.real*ti + x1.imag*tr + x2.imag*tr - x2.real*ti); y2 = x0 +
  // std::complex<T>(x1.real*tr + x1.imag*ti + x2.real*tr - x2.imag*ti,
  // x2.real*ti + x2.imag*tr + x1.imag*tr - x1.real*ti);

  // y1 = x0 + std::complex<T>((x1.real + x2.real)*tr + (x2.imag - x1.imag)*ti,
  // (x1.real - x2.real)*ti + (x1.imag + x2.imag)*tr); y2 = x0 +
  // std::complex<T>((x1.real + x2.real)*tr + (x1.imag  - x2.imag)*ti, (x2.real
  // - x1.real)*ti + (x1.imag + x2.i)*tr);

  // y1 = x0 + std::complex<T>(sum.real*tr - dif.imag*ti, dif.real*ti +
  // sum.imag*tr); y2 = x0 + std::complex<T>(sum.real*tr + dif.imag*ti,
  // sum.imag*tr - dif.real*ti);

  // y1 = x0 + std::complex<T>(trSum.real - tiDif.imag, trSum.imag +
  // tiDif.real); y2 = x0 + std::complex<T>(trSum.real + tiDif.imag, trSum.imag
  // - tiDif.real);

  std::complex<T> trSum = xSum * twiddle.real();
  std::complex<T> tiDif = xDif * twiddle.imag();
  std::complex<T> x0pTrSum = x0 + trSum;

  // std::complex<T> jtiDif(-tiDif.imag, tiDif.real);
  // y1 = x0pTrSum + jTiDif;
  // y2 = x0pTrSum - jTiDif;
  y1.real(x0pTrSum.real() - tiDif.imag());
  y1.imag(x0pTrSum.imag() + tiDif.real());
  y2.real(x0pTrSum.real() + tiDif.imag());
  y2.imag(x0pTrSum.imag() - tiDif.real());
}

// FFT of size 3
template <typename T> class FixedSizeBufferlessFFT<T, 3> {
public:
  static constexpr int size() { return 3; }
  static void fft(std::complex<T> &y0, std::complex<T> &y1, std::complex<T> &y2,
                  const std::complex<T> &x0, const std::complex<T> &x1,
                  const std::complex<T> &x2, const std::complex<T> &t1) {
    std::complex<T> xSum = x1 + x2;
    std::complex<T> xDif = x1 - x2;

    y0 = x0 + xSum;
    subFFTOddAssign(y1, y2, x0, xSum, xDif, t1);
  }

  static void fft(std::complex<T> *pY, size_t ystride,
                  const std::complex<T> *pX, size_t xstride,
                  const std::complex<T> *pTwiddles, int64_t twiddleStride) {
    fft(pY[0], pY[ystride], pY[2 * ystride], pX[0], pX[xstride],
        pX[2 * xstride], pTwiddles[twiddleStride]);
  }
};

// Efficient multiplication by sqrt(-1)
template <typename T> inline std::complex<T> timesJ(const std::complex<T> &x) {
  return std::complex<T>(-x.imag(), x.real());
}

// Efficient multiplication by -sqrt(-1)
template <typename T>
inline std::complex<T> timesJConj(const std::complex<T> &x) {
  return std::complex<T>(x.imag(), -x.real());
}

template <typename T> constexpr T sqrtOf2() {
  return static_cast<T>(
      1.4142135623730950488016887242096980785696718753769480731766797379907324784621);
}
template <typename T> constexpr T sqrtOf2Inv() {
  return static_cast<T>(1 / sqrtOf2<long double>());
}

// Efficient multiplication by e^(j*pi/4)
template <typename T>
inline std::complex<T> timesExpJPiFourths(const std::complex<T> &x) {
  return sqrtOf2Inv<T>() *
         std::complex<T>(x.real() - x.imag(), x.real() + x.imag());
}

// Efficient multiplication by e^(j*3*pi/4)
template <typename T>
inline std::complex<T> timesExpJ3PiFourths(const std::complex<T> &x) {
  return sqrtOf2Inv<T>() *
         std::complex<T>(-x.imag() - x.real(), x.real() - x.imag());
}

// Efficient multiplication by e^(j*5*pi/4)
template <typename T>
inline std::complex<T> timesExpJ5PiFourths(const std::complex<T> &x) {
  return -timesExpJPiFourths(x);
}

// Efficient multiplication by e^(j*7*pi/4)
template <typename T>
inline std::complex<T> timesExpJ7PiFourths(const std::complex<T> &x) {
  return -timesExpJ3PiFourths(x);
}

// FFT of length 4
template <typename T> class FixedSizeBufferlessFFT<T, 4> {
public:
  static constexpr int size() { return 4; }

  static void fft(std::complex<T> &y0, std::complex<T> &y1, std::complex<T> &y2,
                  std::complex<T> &y3, const std::complex<T> &x0,
                  const std::complex<T> &x1, const std::complex<T> &x2,
                  const std::complex<T> &x3, bool inv) {
    // y0 = x0 + x1 + x2 + x3;
    // y1 = x0 + j*x1 - x2 - jx3;
    // y2 = x0 - x1 + x2 -x3;
    // y3 = x0 -jx1 - x2 + jx3;

    std::complex<T> s0 = x0 + x2;
    std::complex<T> d0 = x0 - x2;
    std::complex<T> s1 = x1 + x3;
    std::complex<T> jd1 = timesJ(x1 - x3);
    std::complex<T> a = d0 + jd1;
    std::complex<T> b = d0 - jd1;

    y0 = s0 + s1;
    y2 = s0 - s1;

    if (inv) {
      y1 = a;
      y3 = b;
    } else {
      y1 = b;
      y3 = a;
    }
  }

  static void fft(std::complex<T> *pY, size_t ystride,
                  const std::complex<T> *pX, size_t xstride,
                  const std::complex<T> *pTwiddles, int64_t twiddleStride) {
    bool inv = (twiddleStride < 0);
    fft(pY[0], pY[ystride], pY[2 * ystride], pY[3 * ystride], pX[0],
        pX[xstride], pX[2 * xstride], pX[3 * xstride], inv);
  }
};

// FFT of length 8
template <typename T> class FixedSizeBufferlessFFT<T, 8> {
public:
  static constexpr int size() { return 8; }

  static void fft(std::complex<T> *pY, size_t ystride,
                  const std::complex<T> *pX, size_t xstride,
                  const std::complex<T> *pTwiddles, int64_t twiddleStride) {
    std::array<std::complex<T>, 8> tmp;
    for (int i = 0; i < 4; ++i)
      FixedSizeBufferlessFFT<T, 2>::fft(tmp.data() + i, 4, pX + i * xstride,
                                        4 * xstride);

    if (twiddleStride < 0) {
      tmp[5] = timesExpJPiFourths(tmp[5]);
      tmp[6] = timesJ(tmp[6]);
      tmp[7] = timesExpJ3PiFourths(tmp[7]);
    } else {
      tmp[5] = timesExpJ7PiFourths(tmp[5]);
      tmp[6] = timesJConj(tmp[6]);
      tmp[7] = timesExpJ5PiFourths(tmp[7]);
    }

    FixedSizeBufferlessFFT<T, 4>::fft(pY, 2 * ystride, tmp.data(), 1, pTwiddles,
                                      twiddleStride * 2);
    FixedSizeBufferlessFFT<T, 4>::fft(pY + ystride, 2 * ystride, tmp.data() + 4,
                                      1, pTwiddles, twiddleStride * 2);
  }
};

// FFT of length 5
template <typename T> class FixedSizeBufferlessFFT<T, 5> {
public:
  static constexpr int size() { return 5; }

  static void fft(std::complex<T> &y0, std::complex<T> &y1, std::complex<T> &y2,
                  std::complex<T> &y3, std::complex<T> &y4,
                  const std::complex<T> &x0, const std::complex<T> &x1,
                  const std::complex<T> &x2, const std::complex<T> &x3,
                  const std::complex<T> &x4, const std::complex<T> &t1,
                  const std::complex<T> &t2) {
    // y3 = x0 + x1 * t3 + x2 * t1 + x3 * t4 + x4 * t2;
    // y4 = x0 + x1 * t4 + x2 * t3 + x3 * t2 + x4 * t1;

    // y0 = x0 + x1 + x2 + x3 + x4;
    // y1 = x0 + x1 * t1 + x2 * t2 + x3 * t2c + x4 * t1c;
    // y2 = x0 + x1 * t2 + x2 * t1c + x3 * t1 + x4 * t2c;
    // y3 = x0 + x1 * t2c + x2 * t1 + x3 * t1c + x4 * t2;
    // y4 = x0 + x1 * t1c + x2 * t2c + x3 * t2 + x4 * t1;

    std::complex<T> s1 = x1 + x4;
    std::complex<T> d1 = x1 - x4;
    std::complex<T> s2 = x2 + x3;
    std::complex<T> d2 = x2 - x3;

    y0 = x0 + s1 + s2;

    subFFTOddAssign(y1, y4, x0, s1, d1, t1);
    subFFTOddAssign(y2, y3, x0, s1, d1, t2);
    subFFTOddAccumulate(y1, y4, s2, d2, t2);
    subFFTOddAccumulate(y3, y2, s2, d2, t1);
  }

  static void fft(std::complex<T> *pY, size_t ystride,
                  const std::complex<T> *pX, size_t xstride,
                  const std::complex<T> *pTwiddles, int64_t twiddleStride) {
    fft(pY[0], pY[ystride], pY[2 * ystride], pY[3 * ystride], pY[4 * ystride],
        pX[0], pX[xstride], pX[2 * xstride], pX[3 * xstride], pX[4 * xstride],
        pTwiddles[twiddleStride], pTwiddles[2 * twiddleStride]);
  }
};

// FFT of length 7
template <typename T> class FixedSizeBufferlessFFT<T, 7> {
public:
  static constexpr int size() { return 7; }

  static void fft(std::complex<T> &y0, std::complex<T> &y1, std::complex<T> &y2,
                  std::complex<T> &y3, std::complex<T> &y4, std::complex<T> &y5,
                  std::complex<T> &y6, const std::complex<T> &x0,
                  const std::complex<T> &x1, const std::complex<T> &x2,
                  const std::complex<T> &x3, const std::complex<T> &x4,
                  const std::complex<T> &x5, const std::complex<T> &x6,
                  const std::complex<T> &t1, const std::complex<T> &t2,
                  const std::complex<T> &t3) {
    // y3 = x0 + x1 * t3 + x2 * t1 + x3 * t4 + x4 * t2;
    // y4 = x0 + x1 * t4 + x2 * t3 + x3 * t2 + x4 * t1;

    // y0 = x0 + x1 + x2 + x3 + x4 + x5 + x6;
    // y1 = x0 + x1 * t1 + x2 * t2 + x3 * t3 + x4 * t4 + x5*t5 + x6*t6;
    // y2 = x0 + x1 * t2 + x2 * t4 + x3 * t6 + x4 * t1 + x5*t3 + x6*t5;
    // y3 = x0 + x1 * t3 + x2 * t6 + x3 * t2 + x4 * t5 + x5*t1 + x6*t4;
    // y4 = x0 + x1 * t4 + x2 * t1 + x5 * t2 + x4 * t2 + x5*t6 + x6*t3;
    // y5 = x0 + x1 * t5 + x2 * t3 + x3 * t1 + x4 * t6 + x5*t4 + x6*t2;
    // y6 = x0 + x1 * t6 + x2 * t5 + x3 * t4 + x4 * t3 + x5*t2 + x6*t1;

    std::complex<T> s1 = x1 + x6;
    std::complex<T> d1 = x1 - x6;
    std::complex<T> s2 = x2 + x5;
    std::complex<T> d2 = x2 - x5;
    std::complex<T> s3 = x3 + x4;
    std::complex<T> d3 = x3 - x4;

    // y0 = x0 + x1 + x2 + x3 + x4 + x5 + x6;
    // y1 = x0 + x1 * t1 + x2 * t2 + x3 * t3 + x4 * t3c + x5*t2c + x6*t1c;
    // y2 = x0 + x1 * t2 + x2 * t3c + x3 * t1c + x4 * t1 + x5*t3 + x6*t2c;
    // y3 = x0 + x1 * t3 + x2 * t1c + x3 * t2 + x4 * t2c + x5*t1 + x6*t3c;
    // y4 = x0 + x1 * t3c + x2 * t1 + x5 * t2 + x4 * t2 + x5*t1c + x6*t3;
    // y5 = x0 + x1 * t2c + x2 * t3 + x3 * t1 + x4 * t1c + x5*t3c + x6*t2;
    // y6 = x0 + x1 * t1c + x2 * t2c + x3 * t3c + x4 * t3 + x5*t2 + x6*t1;
    y0 = x0 + s1 + s2 + s3;
    subFFTOddAssign(y1, y6, x0, s1, d1, t1);
    subFFTOddAssign(y2, y5, x0, s1, d1, t2);
    subFFTOddAssign(y3, y4, x0, s1, d1, t3);
    subFFTOddAccumulate(y1, y6, s2, d2, t2);
    subFFTOddAccumulate(y5, y2, s2, d2, t3);
    subFFTOddAccumulate(y4, y3, s2, d2, t1);
    subFFTOddAccumulate(y1, y6, s3, d3, t3);
    subFFTOddAccumulate(y5, y2, s3, d3, t1);
    subFFTOddAccumulate(y3, y4, s3, d3, t2);
  }

  static void fft(std::complex<T> *pY, size_t ystride,
                  const std::complex<T> *pX, size_t xstride,
                  const std::complex<T> *pTwiddles, int64_t twiddleStride) {
    fft(pY[0], pY[ystride], pY[2 * ystride], pY[3 * ystride], pY[4 * ystride],
        pY[5 * ystride], pY[6 * ystride], pX[0], pX[xstride], pX[2 * xstride],
        pX[3 * xstride], pX[4 * xstride], pX[5 * xstride], pX[6 * xstride],
        pTwiddles[twiddleStride], pTwiddles[2 * twiddleStride],
        pTwiddles[3 * twiddleStride]);
  }
};

// Template for wrapping compile-time length FFTs to the FFTBase class
template <typename T, int N> class FixedSizeFFT : public FFTBase<T> {
public:
  virtual size_t size() const override final { return N; }

  virtual void fftN(std::complex<T> *pY, size_t ystride,
                    const std::complex<T> *pX, size_t xstride,
                    const std::complex<T> *pTwiddles, int64_t twiddleStride,
                    std::complex<T> *) const override final {
    FixedSizeBufferlessFFT<T, N>::fft(pY, ystride, pX, xstride, pTwiddles,
                                      twiddleStride);
  }
};

// Efficient (O(N^2/4)) small-prime FFT implementation for arbitrary small odd
// FFT sizes
template <typename T> class FFTSmallPrime : public FFTBase<T> {
public:
  FFTSmallPrime(int N) : m_N(N) {
    if (2 * (N / 2) == N)
      throw std::runtime_error("Odd FFT cannot be run on even size data");
  }

  virtual size_t size() const override final { return m_N; }
  virtual void fftN(std::complex<T> *pY, size_t ystride,
                    const std::complex<T> *pX, size_t xstride,
                    const std::complex<T> *pTwiddles, int64_t twiddleStride,
                    std::complex<T> *) const override final {
    size_t L = m_N / 2 + 1;

    for (size_t l = 0; l < L; ++l)
      pY[l * ystride] = *pX;
    for (size_t n = L; n < m_N; ++n)
      pY[n * ystride] = std::complex<T>(0, 0);

    for (size_t l = 1; l < L; ++l) {
      std::complex<T> sum = pX[l * xstride] + pX[(m_N - l) * xstride];
      std::complex<T> dif = pX[l * xstride] - pX[(m_N - l) * xstride];
      std::complex<T> jdif(-dif.imag(), dif.real());
      *pY += sum;

      for (size_t m = 1, n = l; m < L; n -= m_N) {
        auto dstFwd = pY + m * ystride;
        auto dstBwd = pY + (m_N - m) * ystride;

        int64_t factorStride = twiddleStride * l;
        auto pFactor = pTwiddles + twiddleStride * n;
        for (; n < L && m < L; n += l, ++m, pFactor += factorStride,
                               dstFwd += ystride, dstBwd -= ystride) {
          *dstFwd += pFactor->real() * sum;
          *dstBwd -= pFactor->imag() * jdif;
        }

        pFactor = pTwiddles + twiddleStride * (m_N - n);
        for (; n < m_N && m < L; n += l, ++m, pFactor -= factorStride,
                                 dstFwd += ystride, dstBwd -= ystride) {
          *dstFwd += pFactor->real() * sum;
          *dstBwd += pFactor->imag() * jdif;
        }
      }
    }

    for (size_t l = 1; l < L; ++l) {
      size_t idx1 = l * ystride, idx2 = (m_N - l) * ystride;
      std::complex<T> tmp = pY[idx1];
      pY[idx1] -= pY[idx2];
      pY[idx2] += tmp;
    }
  }

private:
  size_t m_N;
};

// Composite length FFT where the length = N1*N2 and N1 and N2 have
// FixedSizeBufferlessFFT implementations
template <typename T, int N1, int N2>
class FixedSizeCompositeFFT : public FFTBase<T> {
public:
  virtual size_t size() const override final { return N1 * N2; }

  static void fft(std::complex<T> *pY, size_t ystride,
                  const std::complex<T> *pX, size_t xstride,
                  const std::complex<T> *pTwiddles, int64_t twiddleStride,
                  std::complex<T> *pTmp) {
    // Col-wise FFT with transpose and twiddle multiply
    for (int n1 = 0; n1 < N1; ++n1, pX += xstride)
      FixedSizeBufferlessFFT<T, N2>::fft(pTmp + n1, N1, pX, xstride * N1,
                                         pTwiddles, twiddleStride * N1);

    for (int n1 = 1; n1 < N1; ++n1)
      for (int n2 = 1; n2 < N2; ++n2)
        (pTmp[n2 * N1 + n1] *= pTwiddles[n1 * n2 * twiddleStride]);

    for (int n2 = 0; n2 < N2; ++n2, pY += ystride, pTmp += N1)
      FixedSizeBufferlessFFT<T, N1>::fft(pY, ystride * N2, pTmp, 1, pTwiddles,
                                         twiddleStride * N2);
  }

  virtual void fftN(std::complex<T> *pY, size_t ystride,
                    const std::complex<T> *pX, size_t xstride,
                    const std::complex<T> *pTwiddles, int64_t twiddleStride,
                    std::complex<T> *pTmp) const override final {
    fft(pY, ystride, pX, xstride, pTwiddles, twiddleStride, pTmp);
  }
};

// Composite length FFT of 2 arbitrary lengths (N = N1 * N2)
template <typename T> class DynamicCompositeFFT : public FFTBase<T> {
public:
  DynamicCompositeFFT(const std::shared_ptr<FFTBase<T>> &pFFT1,
                      const std::shared_ptr<FFTBase<T>> &pFFT2)
      : m_pFFT1(pFFT1), m_pFFT2(pFFT2), m_N(pFFT1->size() * pFFT2->size()) {}

  virtual size_t size() const override final { return m_N; }

  virtual void fftN(std::complex<T> *pY, size_t ystride,
                    const std::complex<T> *pX, size_t xstride,
                    const std::complex<T> *pTwiddles, int64_t twiddleStride,
                    std::complex<T> *pTmp) const override final {
    size_t N1 = m_pFFT1->size();
    size_t N2 = m_pFFT2->size();

    // TODO, use Y as a temporary buffer to reduce memory usage or find a better
    // way to handle memory use
    std::vector<std::complex<T>> tmp2(std::max(N1, N2));

    // Col-wise FFT with transpose and twiddle multiply
    for (size_t n1 = 0; n1 < N1; ++n1, pX += xstride)
      m_pFFT2->fftN(pTmp + n1, N1, pX, xstride * N1, pTwiddles,
                    twiddleStride * N1, tmp2.data());

    for (size_t n1 = 1; n1 < N1; ++n1)
      for (size_t n2 = 1; n2 < N2; ++n2)
        pTmp[n2 * N1 + n1] *= pTwiddles[n1 * n2 * twiddleStride];

    for (size_t n2 = 0; n2 < N2; ++n2, pTmp += N1)
      m_pFFT1->fftN(pY + ystride * n2, ystride * N2, pTmp, 1, pTwiddles,
                    twiddleStride * N2, tmp2.data());
  }

private:
  // Shared pointers are used to easily manage pointer lifetimes while also
  // avoiding duplicate FFT objects if FFT1 and FFT2 are the same
  std::shared_ptr<FFTBase<T>> m_pFFT1;
  std::shared_ptr<FFTBase<T>> m_pFFT2;
  size_t m_N;
};

// Composite sized FFTs where N1 and N2 are co-prime. Currently this is a
// placeholder for a
// future implementation of the prime factor algorithm.
template <typename T, int N1, int N2>
class CoprimeFFT : public FixedSizeCompositeFFT<T, N1, N2> {};

// Explicit implementations of common composite sizes using
// compile-time sizes allowing a compiler to optimize these

template <typename T> class FixedSizeFFT<T, 28> : public CoprimeFFT<T, 7, 4> {};

template <typename T> class FixedSizeFFT<T, 24> : public CoprimeFFT<T, 8, 3> {};

template <typename T> class FixedSizeFFT<T, 14> : public CoprimeFFT<T, 7, 2> {};

template <typename T> class FixedSizeFFT<T, 15> : public CoprimeFFT<T, 3, 5> {};

template <typename T> class FixedSizeFFT<T, 20> : public CoprimeFFT<T, 4, 5> {};

template <typename T> class FixedSizeFFT<T, 10> : public CoprimeFFT<T, 2, 5> {};

template <typename T> class FixedSizeFFT<T, 35> : public CoprimeFFT<T, 7, 5> {};

template <typename T> class FixedSizeFFT<T, 21> : public CoprimeFFT<T, 7, 3> {};

template <typename T> class FixedSizeFFT<T, 12> : public CoprimeFFT<T, 3, 4> {};

template <typename T> class FixedSizeFFT<T, 6> : public CoprimeFFT<T, 2, 3> {};

template <typename T>
class FixedSizeFFT<T, 49> : public FixedSizeCompositeFFT<T, 7, 7> {};

template <typename T>
class FixedSizeFFT<T, 25> : public FixedSizeCompositeFFT<T, 5, 5> {};

template <typename T>
class FixedSizeFFT<T, 125> : public FixedSizeCompositeFFT<T, 25, 5> {};

template <typename T>
class FixedSizeFFT<T, 9> : public FixedSizeCompositeFFT<T, 3, 3> {};

template <typename T>
class FixedSizeFFT<T, 27> : public FixedSizeCompositeFFT<T, 9, 3> {};

template <typename T>
class FixedSizeFFT<T, 81> : public FixedSizeCompositeFFT<T, 9, 9> {};

template <typename T>
class FixedSizeFFT<T, 16> : public FixedSizeCompositeFFT<T, 8, 2> {};

template <typename T>
class FixedSizeFFT<T, 32> : public FixedSizeCompositeFFT<T, 8, 4> {};

template <typename T>
class FixedSizeFFT<T, 64> : public FixedSizeCompositeFFT<T, 8, 8> {};
} // namespace detail

// Forward declare template
template <typename T> class FFT;

enum class TransformDirection {
  Forward, // Uses e^(-j*2*pi*n/N)
  Backward // Uses e^(j*2*pi*n/N)
};

namespace detail {
// Implementation of Bluestein's algorithm for efficient FFT when length has
// large prime factors
template <typename T> class BluesteinFFT : public FFTBase<T> {
public:
  BluesteinFFT(size_t N) : bnc(N), Bm(fftSize(2 * N - 1), 0), fftB(Bm.size()) {
    size_t M = Bm.size();
    T scale = 1 / static_cast<T>(M);
    std::vector<std::complex<T>> bm(M, 0);
    bnc[0] = 1;
    bm[0] = scale;
    T piOverN = twoPiOverN<T>(N) / 2;
    for (size_t n = 1; n < N; ++n) {
      std::complex<T> bn = getTwiddleFactor<T>((n * n) % (2 * N), piOverN);
      bm[n] = bm[M - n] = bn * scale;
      bnc[n] = std::conj(bn);
    }

    fftB.transform(Bm.data(), 1, bm.data(), 1,
                   fftpp::TransformDirection::Forward);
  }

  virtual size_t size() const { return bnc.size(); }

  virtual void fftN(std::complex<T> *pY, size_t ystride,
                    const std::complex<T> *pX, size_t xstride,
                    const std::complex<T> *pTwiddles, int64_t twiddleStride,
                    std::complex<T> *) const override final {
    size_t N = size();
    size_t M = fftB.size();
    std::vector<std::complex<T>> am(M), Am(M), tmp(M);

    am[0] = pX[0];

    for (size_t n = 1; n < N; ++n)
      am[n] = pX[n * xstride] * bnc[n];

    for (size_t m = N; m < M; ++m)
      am[m] = 0;

    fftB.transform(Am.data(), 1, am.data(), 1, tmp.data(),
                   fftpp::TransformDirection::Forward);

    for (size_t m = 0; m < M; ++m)
      Am[m] *= Bm[m];

    fftB.transform(am.data(), 1, Am.data(), 1, tmp.data(),
                   fftpp::TransformDirection::Backward);

    pY[0] = am[0];
    if (twiddleStride > 0) {
      for (size_t n = 1; n < N; ++n)
        pY[n * ystride] = am[n] * bnc[n];
    } else {
      for (size_t n = 1; n < N; ++n)
        pY[(N - n) * ystride] = am[n] * bnc[n];
    }
  }

private:
  std::vector<std::complex<T>> bnc;
  std::vector<std::complex<T>> Bm;
  FFT<T> fftB;
};

// Add an FFT to the map for a particular factor
template <typename T, int Factor>
size_t
addFFTsForFactor(std::multimap<size_t, std::shared_ptr<FFTBase<T>>> &ffts,
                 size_t N) {
  size_t nextN = N / Factor;
  std::shared_ptr<FFTBase<T>> pFFT = nullptr;
  while (nextN * Factor == N) {
    if (!pFFT)
      pFFT = std::make_shared<FixedSizeFFT<T, Factor>>();
    ffts.emplace(Factor, pFFT);
    N = nextN;
    nextN /= Factor;
  }

  return N;
}

// Add an FFT to the map for a particular small prime factor
template <typename T>
size_t
addFFTsForSmallPrime(std::multimap<size_t, std::shared_ptr<FFTBase<T>>> &ffts,
                     size_t N, int factor) {
  size_t nextN = N / factor;
  std::shared_ptr<FFTBase<T>> pFFT = nullptr;
  while (nextN * factor == N) {
    if (!pFFT)
      pFFT = std::make_shared<FFTSmallPrime<T>>(factor);
    ffts.emplace(factor, pFFT);
    N = nextN;
    nextN /= factor;
  }

  return N;
}

// Add FFTs to the map for any remaining factors of N
template <typename T>
void addFFTsForRemainingFactors(
    std::multimap<size_t, std::shared_ptr<FFTBase<T>>> &ffts, size_t N) {
  if (N > 1)
    ffts.emplace(N, std::make_shared<BluesteinFFT<T>>(N));
}
} // namespace detail
} // namespace fftpp

#endif
