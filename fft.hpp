#ifndef FFTPP_FFT_HPP
#define FFTPP_FFT_HPP

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

#include "fft_detail.hpp"
#include "thread_scheduler.hpp"
#include <map>

namespace fftpp {

/* Object containing information needed to compute a FFT of a particular size
 * that can be precomputed prior to execution of one or many transforms. This
 * object is copyable with little copy overhead by using shared pointers
 * internally. It can be used to execute many FFTs simultaneously from multiple
 * threads safely. All memory allocation/destruction is handled internally on
 * construction/destruction of the class.
 */
template <typename T = double> class FFT {
public:
  explicit FFT(size_t N)
      : m_pFFT(), m_pTwiddles(detail::getNp1TwiddleFactors<T>(N)), m_N(N) {
    using namespace detail;

    if (N > 1) {
      size_t M = N;
      std::multimap<size_t, std::shared_ptr<FFTBase<T>>> ffts;

      M = addFFTsForFactor<T, 15>(ffts, M);
      M = addFFTsForFactor<T, 10>(ffts, M);
      M = addFFTsForFactor<T, 12>(ffts, M);
      M = addFFTsForFactor<T, 6>(ffts, M);
      M = addFFTsForFactor<T, 64>(ffts, M);
      M = addFFTsForFactor<T, 32>(ffts, M);
      M = addFFTsForFactor<T, 16>(ffts, M);
      M = addFFTsForFactor<T, 25>(ffts, M);
      M = addFFTsForFactor<T, 9>(ffts, M);
      M = addFFTsForFactor<T, 49>(ffts, M);

      M = addFFTsForFactor<T, 8>(ffts, M);
      M = addFFTsForFactor<T, 7>(ffts, M);
      M = addFFTsForFactor<T, 5>(ffts, M);
      M = addFFTsForFactor<T, 4>(ffts, M);
      M = addFFTsForFactor<T, 3>(ffts, M);
      M = addFFTsForFactor<T, 2>(ffts, M);

      // Other small primes
      M = addFFTsForSmallPrime<T>(ffts, M, 11);
      M = addFFTsForSmallPrime<T>(ffts, M, 13);
      M = addFFTsForSmallPrime<T>(ffts, M, 17);
      M = addFFTsForSmallPrime<T>(ffts, M, 19);
      M = addFFTsForSmallPrime<T>(ffts, M, 23);
      M = addFFTsForSmallPrime<T>(ffts, M, 29);

      // Large primes
      addFFTsForRemainingFactors<T>(ffts, M);

      while (ffts.size() > 1) {
        // Combine largest with the smallest iteratively for better cache
        // efficiency
        // TODO: Take advantage of co-primes using prime factor algorithm
        auto pComposite = std::make_shared<DynamicCompositeFFT<T>>(
            ffts.begin()->second, ffts.rbegin()->second);
        ffts.erase(ffts.begin());
        auto it = ffts.begin();
        std::advance(it, ffts.size() - 1);
        ffts.erase(it);
        ffts.emplace(pComposite->size(), pComposite);
      }

      m_pFFT = ffts.begin()->second;
    }
  }

  size_t size() const { return m_N; }

  void transform(std::complex<T> *pY, size_t ystride, const std::complex<T> *pX,
                 size_t xstride, std::complex<T> *pTmp,
                 TransformDirection direction) const {
    if (pY == pX) {
      std::vector<std::complex<T>> tmp(size());
      transform(tmp.data(), 1, pX, xstride, pTmp, direction);
      for (size_t i = 0; i < size(); ++i)
        pY[i * ystride] = tmp[i];
    } else {
      if (size() == 1)
        *pY = *pX;
      else if (size() > 1) {
        if (direction == TransformDirection::Forward)
          m_pFFT->fftN(pY, ystride, pX, xstride, m_pTwiddles.get(), 1, pTmp);
        else
          m_pFFT->fftN(pY, ystride, pX, xstride, m_pTwiddles.get() + size(), -1,
                       pTmp);
      }
    }
  }

  void transform(std::complex<T> *pY, size_t ystride, const std::complex<T> *pX,
                 size_t xstride, TransformDirection direction) const {
    std::vector<std::complex<T>> tmp(size());
    transform(pY, ystride, pX, xstride, tmp.data(), direction);
  }

  void transformMultiple(std::complex<T> *pY, size_t ySampleStride,
                         size_t yTransformStride, const std::complex<T> *pX,
                         size_t xSampleStride, size_t xTransformStride,
                         size_t numTransforms,
                         TransformDirection direction) const {
    std::vector<std::complex<T>> tmp(size());
    for (size_t i = 0; i < numTransforms;
         ++i, pY += yTransformStride, pX += xTransformStride)
      transform(pY, ySampleStride, pX, xSampleStride, tmp.data(), direction);
  }

private:
  std::shared_ptr<const detail::FFTBase<T>> m_pFFT;
  std::shared_ptr<const std::complex<T>[]> m_pTwiddles;
  size_t m_N;
};

namespace detail {
// Reuse a FFT if its size matches N, otherwise create a new FFT of size N
template <typename T> FFT<T> getFFT(size_t N, const FFT<T> *pFFT) {
  return (pFFT && pFFT->size() == N) ? *pFFT : FFT<T>(N);
}
} // namespace detail


 /*  Perform multiple Fast Fourier Transforms
 * T: The floating point type (float, double, long double)
 * RowMajor: Indicates the underlying storage of the 2D array. If the elements
 * are contiguous along the row dimension this must be set to true, otherwise
 * false.
 * pY: Pointer to start of the output buffer at the first transform
 * ySampleStride: Stride between elements of the output buffer
 * yTransformStride: Stride between each transform of the output buffer
 * pX: Pointer to the start of the input buffer at the first transform
 * xSampleStride: Stride between elements of the input buffer
 * xTransformStride: Stride between each transform of the input buffer
 * N: Size of the transform
 * numTransforms: Number of transforms to perform
 * direction: Direction of the fourier transform. See TransformDirection for
 * details.
 * pFFT: An optional pointer to FFT of length cols. If nullptr or not
 * the needed length a new FFT will be generated internally of the appropriate
 * length.
 * pScheduler: Pointer to an optional thread scheduler
 * returns: The FFT of length N that was used to perform the
 * transform
 */
template <typename T>
FFT<T> fftMultiple(std::complex<T> *pY, size_t ySampleStride,
                   size_t yTransformStride, const std::complex<T> *pX,
                   size_t xSampleStride, size_t xTransformStride, size_t N,
                   size_t numTransforms, TransformDirection direction,
                   const FFT<T> *pFFT = nullptr,
                   ThreadScheduler *pScheduler = nullptr) {
  FFT<T> fftObj = detail::getFFT(N, pFFT);

  if (pScheduler && pScheduler->executeAndWait && pScheduler->numThreads > 1 &&
      numTransforms > 1) {
    size_t transformsPerThread =
        (numTransforms + pScheduler->numThreads - 1) / pScheduler->numThreads;
    std::vector<std::function<void()>> tasks;
    for (size_t i = 0; i < numTransforms; i += transformsPerThread) {
      size_t transformsThisThread =
          std::min(transformsPerThread, numTransforms - i);
      tasks.emplace_back([=, &fftObj]() {
        fftObj.transformMultiple(pY + yTransformStride * i, ySampleStride,
                                 yTransformStride, pX + xTransformStride * i,
                                 xSampleStride, xTransformStride,
                                 transformsThisThread, direction);
      });
    }
    pScheduler->executeAndWait(tasks);
  } else {
    fftObj.transformMultiple(pY, ySampleStride, yTransformStride, pX,
                             xSampleStride, xTransformStride, numTransforms,
                             direction);
  }
  return fftObj;
}

/** Perform a 1D Discrete Fourier Transform (FFT)
 * T: The floating point type (float, double, long double)
 * pY: Pointer to start of an output buffer with N elements at the specified
 * stride. This can be the same as the input buffer.
 * ystride: Stride of the output buffer in elements
 * pX: Pointer to the start of an input buffer with N elements at the specified
 * stride
 * xstride: Stride of the input buffer in elements pTmp: Pointer to a
 * temporary buffer of length N
 * N: Number of  elements in the input and output buffers
 * direction: Direction of the fourier transform. See TransformDirection
 * for details.
 * pFFT: An optional pointer to a FFT of length
 */
template <typename T>
FFT<T> fft(std::complex<T> *pY, size_t ystride, const std::complex<T> *pX,
           size_t xstride, std::complex<T> *pTmp, size_t N,
           TransformDirection direction, const FFT<T> *pFFT = nullptr) {
  FFT<T> fftObj = detail::getFFT(N, pFFT);
  fftObj.transform(pY, ystride, pX, xstride, pTmp, direction);
  return fftObj;
}

/* Perform a 1D Discrete Fourier Transform (FFT)
 * T: The floating point type (float, double, long double)
 * x: Data to transform
 * pTmp: Pointer to a temporary buffer of length N cols: Number of
 * columns in the input and output buffers direction: Direction of the fourier
 * transform. See TransformDirection for details. pFFT: An optional pointer to a
 * FFT of length
 */
template <typename T>
std::vector<std::complex<T>> fft(const std::vector<std::complex<T>> &x,
                                 TransformDirection direction,
                                 const FFT<T> *pFFT = nullptr) {
  std::vector<std::complex<T>> y(x.size());
  std::vector<std::complex<T>> tmp(x.size());
  fft(y.data(), 1, x.data(), 1, tmp.data(), x.size(), direction, pFFT);
  return y;
}

/* Perform a 1D Discrete Fourier Transform (FFT) on each row of a 2D array
 * T: The floating point type (float, double, long double)
 * RowMajor: Indicates the underlying storage of the 2D array. If the elements
 * are contiguous along the row dimension this must be set to true, otherwise
 * false.
 * pY: Pointer to start of a 2D output buffer
 * pX: Pointer to the start of  a 2D input buffer
 * rows: Number of rows in the input and output buffers
 * cols:  Number of columns in the input and output buffers
 * direction: Direction of the fourier transform. See TransformDirection for
 * details.
 * pFFT: An optional pointer to FFT of length cols. If nullptr or not
 * the needed length a new FFT will be generated internally of the appropriate
 * length.
 * pScheduler: Pointer to an optional thread scheduler
 * returns: The FFT of length cols that was used to perform the
 * transform
 */
template <typename T, bool RowMajor = true>
FFT<T> fftEachRow(std::complex<T> *pY, const std::complex<T> *pX, size_t rows,
                  size_t cols, TransformDirection direction,
                  const FFT<T> *pFFT = nullptr,
                  ThreadScheduler *pScheduler = nullptr) {
  size_t sampleStride = RowMajor ? 1 : rows;
  size_t transformStride = RowMajor ? cols : 1;

  FFT<T> fftObj =
      fftMultiple(pY, sampleStride, transformStride, pX, sampleStride,
                  transformStride, cols, rows, direction, &fftObj, pScheduler);

  return fftObj;
}

/** Perform a 1D Discrete Fourier Transform (FFT) on each column of a 2D array
 * T: The floating point type (float, double, long double)
 * RowMajor: Indicates the underlying storage of the 2D array. If the elements
 * are contiguous along the row dimension this must be set to true, otherwise
 * false.
 * pY: Pointer to start of a 2D output buffer
 * pX: Pointer to the start of  a 2D input buffer
 * rows: Number of rows in the input and output buffers
 * cols:  Number of columns in the input and output buffers
 * direction: Direction of the fourier transform. See TransformDirection for
 * details.
 * pFFT: An optional pointer to FFT of length cols. If nullptr or not
 * the needed length a new FFT will be generated internally of the appropriate
 * length.
 * pScheduler: Pointer to an optional thread scheduler
 * returns: The FFT of length rows that was used to perform the
 * transform
 */
template <typename T, bool RowMajor = true>
FFT<T> fftEachCol(std::complex<T> *pY, const std::complex<T> *pX, size_t rows,
                  size_t cols, TransformDirection direction,
                  const FFT<T> *pFFT = nullptr,
                  ThreadScheduler *pScheduler = nullptr) {
  return fftEachRow<T, !RowMajor>(pY, pX, cols, rows, direction, pFFT,
                                  pScheduler);
}

/* Perform a 1D Discrete Fourier Transform (FFT) on a 2D array
 * T: The floating point type (float, double, long double)
 * RowMajor: Indicates the underlying storage of the 2D array. If the elements
 * are contiguous along the row dimension this must be set to true, otherwise
 * false.
 * pY: Pointer to start of a 2D output buffer
 * pX: Pointer to the start of  a 2D input buffer
 * rows: Number of rows in the input and output buffers
 * cols:  Number of columns in the input and output buffers
 * direction: Direction of the fourier transform. See TransformDirection for
 * details.
 * pFFT: An optional pointer to FFT of length cols. If nullptr or not
 * the needed length a new FFT will be generated internally of the appropriate
 * length.
 * pScheduler: Pointer to an optional thread scheduler
 * returns: The FFT of length cols that was used to perform the
 * transform
 */
template <typename T, bool RowMajor = true>
std::pair<FFT<T>, FFT<T>>
fft2D(std::complex<T> *pY, const std::complex<T> *pX, size_t rows, size_t cols,
      TransformDirection direction,
      const std::pair<FFT<T>, FFT<T>> *pPlans = nullptr,
      ThreadScheduler *pScheduler = nullptr) {
  const FFT<T> *pPlanRowwise = nullptr;
  const FFT<T> *pPlanColwise = nullptr;

  if (pPlans) {
    pPlanColwise = &pPlans->first;
    pPlanRowwise = &pPlans->second;
  }

  // Take advantage of memory alignment alignment by doing the
  // in-place transform (which internally performs a copy) along the
  // strided dimension
  if (RowMajor) {
    FFT<T> rowWiseFFT = fftEachRow<T, RowMajor>(pY, pX, rows, cols, direction,
                                                pPlanRowwise, pScheduler);
    FFT<T> colWiseFFT = fftEachCol<T, RowMajor>(pY, pY, rows, cols, direction,
                                                pPlanColwise, pScheduler);
    return std::pair<FFT<T>, FFT<T>>(rowWiseFFT, colWiseFFT);
  } else {
    FFT<T> colWiseFFT = fftEachCol<T, RowMajor>(pY, pX, rows, cols, direction,
                                                pPlanColwise, pScheduler);
    FFT<T> rowWiseFFT = fftEachRow<T, RowMajor>(pY, pY, rows, cols, direction,
                                                pPlanRowwise, pScheduler);
    return std::pair<FFT<T>, FFT<T>>(colWiseFFT, rowWiseFFT);
  }
}
} // namespace fftpp

#endif
