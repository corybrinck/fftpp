#include "fft.hpp"
#include "fft_size.hpp"
#include <chrono>
#include <iostream>

namespace {

bool isFFTSize(size_t N) {
  std::vector<size_t> factors = {2, 3, 5, 7};
  for (size_t factor : factors) {
    while (N >= factor && N % factor == 0) {
      N /= factor;
    }
  }
  return N == 1;
}

bool testFFTSize() {
  using namespace fftpp;
  bool success = true;
  for (size_t i = 1; success && i <= 16; ++i)
    success = success && fftSize(i) == i;

  for (size_t i = 17; success && i <= 2000; ++i) {
    size_t j = fftSize(i);
    success = success && j >= i && j <= 2 * i && isFFTSize(j);
  }

  return success && fftSize(17) == 18 && fftSize(18) == 18 &&
         fftSize(19) == 20 && fftSize(20) == 20 && fftSize(27) == 27 &&
         fftSize(28) == 28 && fftSize(53) == 54 && fftSize(1023) == 1024;
}

template <typename T>
bool testFFTRand(size_t N, std::int64_t xstride, std::int64_t ystride) {
  T two_pi = static_cast<T>(2 * 3.141592653589793238462643383279);
  T maxError = 5 * 10 * N * std::numeric_limits<T>::epsilon();
  std::vector<std::complex<T>> x0(N * xstride, 0);
  for (size_t i = 0; i < N; ++i)
    x0[i * xstride] = std::polar<T>(i % 5 - .1f, T(std::rand()));

  std::int64_t i = N > 1 ? 1 : 0;
  T phaseStep = two_pi * i / N;

  std::vector<std::complex<T>> x = x0;
  std::vector<std::complex<T>> y(N * ystride);
  std::vector<std::complex<T>> tmp(N);

  auto fftObj = fftpp::fft(y.data(), ystride, x.data(), xstride, tmp.data(), N,
                           fftpp::TransformDirection::Forward);

  fftpp::fft(x.data(), xstride, y.data(), ystride, tmp.data(), N,
             fftpp::TransformDirection::Backward, &fftObj);

  T scalar = static_cast<T>(1) / N;
  T e = 0;
  for (size_t n = 0; n < N; ++n) {
    T error = std::abs(x0[n * xstride] - scalar * x[n * xstride]);
    e = std::max(e, error);
    if (maxError < error) {
      std::cout << "Random FFT failed for size " << N << " with error value "
                << error << " at index " << n << std::endl;
      return false;
    }
  }

  return true;
}

template <typename T>
bool testFFT(size_t N, std::int64_t xstride, std::int64_t ystride) {
  T two_pi = static_cast<T>(2 * 3.141592653589793238462643383279);
  T maxError = 10 * N * std::numeric_limits<T>::epsilon();
  std::vector<std::complex<T>> x0(N * xstride, 0);
  std::int64_t i = N > 1 ? 1 : 0;
  T phaseStep = two_pi * i / N;
  T phaseOffset = two_pi / 7;

  x0.at(i * xstride) = std::polar<T>(1, phaseOffset);
  std::vector<std::complex<T>> x = x0;
  std::vector<std::complex<T>> y(N * ystride);
  std::vector<std::complex<T>> tmp(N);

  auto fftObj = fftpp::fft(y.data(), ystride, x.data(), xstride, tmp.data(), N,
                           fftpp::TransformDirection::Forward);
  for (size_t n = 0; n < N; ++n) {
    T error = std::abs(y[n * ystride] -
                       std::polar<T>(1, phaseOffset - phaseStep * n));
    if (maxError < error) {
      std::cout << "FFT failed for size " << N << " with error value " << error
                << " > " << maxError << " at index " << n << std::endl;
      return false;
    }
  }

  fftpp::fft(x.data(), xstride, y.data(), ystride, tmp.data(), N,
             fftpp::TransformDirection::Backward, &fftObj);

  T scalar = static_cast<T>(1) / N;
  T e = 0;
  for (size_t n = 0; n < N; ++n) {
    T error = std::abs(x0[n * xstride] - scalar * x[n * xstride]);
    e = std::max(e, error);
    if (maxError < error) {
      std::cout << "IFFT failed for size " << N << " with error value " << error
                << " at index " << n << std::endl;
      return false;
    }
  }

  return true;
}

template <typename T>
bool testFFT2D(size_t M, size_t N, fftpp::ThreadScheduler *pScheduler) {
  T two_pi = static_cast<T>(2 * 3.141592653589793238462643383279);
  double maxError = 10 * (N + M) * std::numeric_limits<T>::epsilon();
  std::vector<std::complex<T>> x0(N * M, 0);
  std::int64_t i = M > 1 ? 1 : 0;
  std::int64_t j = N > 1 ? 1 : 0;
  T phaseStepM = two_pi * i / M;
  T phaseStepN = two_pi * j / N;
  T phaseOffset = 0 * two_pi / 5;

  x0.at(i * N + j) = std::polar<T>(1, phaseOffset);
  std::vector<std::complex<T>> x = x0;
  std::vector<std::complex<T>> y(M * N);

  auto fftObjects =
      fftpp::fft2D<T>(y.data(), x.data(), M, N,
                      fftpp::TransformDirection::Backward, nullptr, pScheduler);
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      auto value = y[N * m + n];
      T error = std::abs(value - std::polar<T>(1, phaseOffset + phaseStepN * n +
                                                      phaseStepM * m));
      if (maxError < error) {
        std::cout << "2D FFT failed for size " << M << "x" << N
                  << " with error value " << error << " > " << maxError
                  << " at index (" << m << "," << n << ") = " << value
                  << std::endl;
        return false;
      }
    }
  }

  fftpp::fft2D<T, false>(x.data(), y.data(), N, M,
                         fftpp::TransformDirection::Forward, &fftObjects,
                         pScheduler);
  T scale = static_cast<T>(1) / (M * N);
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      auto value = scale * x[N * m + n];
      T error = std::abs(value - x0[N * m + n]);
      if (maxError < error) {
        std::cout << "2D IFFT failed for size " << M << "x" << N
                  << " with error value " << error << " > " << maxError
                  << " at index (" << m << "," << n << ") = " << value
                  << std::endl;
        return false;
      }
    }
  }

  return true;
}

template <typename T> bool testFFTType() {
  auto t0 = std::chrono::high_resolution_clock::now();
  for (size_t n = 1; n < 10000; ++n) {
    if (!testFFT<T>(n, n % 4 + 1, n % 3 + 1)) {
      return false;
    }
    if (!testFFTRand<T>(n, (n + 1) % 4 + 1, (n + 2) % 3 + 1)) {
      return false;
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  size_t start2D = 1;
  size_t end2D = 2000;
  for (size_t n = start2D; n < end2D; ++n) {
    if (!testFFT2D<T>(n, n % 7 + 19, nullptr)) {
      return false;
    }
  }

  fftpp::ThreadScheduler scheduler(3);
  for (size_t n = start2D; n < end2D; ++n) {
    if (!testFFT2D<T>(n, n % 7 + 19, &scheduler)) {
      return false;
    }
  }

  auto t2 = std::chrono::high_resolution_clock::now();

  std::cout << "1D test time: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(t1 -
                                                                         t0)
                   .count()
            << std::endl;
  std::cout << "2D test time: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
                                                                         t1)
                   .count()
            << std::endl;
  return true;
}

template <typename T> void fftTimer(size_t N, size_t numIterations = 10000) {
  std::vector<std::complex<T>> x(N, 0);
  std::vector<std::complex<T>> y(N);
  std::vector<std::complex<T>> tmp(N);
  fftpp::FFT<T> dft(N);
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < numIterations; ++i) {
    dft.transform(y.data(), 1, x.data(), 1, tmp.data(),
                  fftpp::TransformDirection::Forward);
    dft.transform(x.data(), 1, y.data(), 1, tmp.data(),
                  fftpp::TransformDirection::Forward);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << numIterations << " iterations of size " << N << " runtime: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(t1 -
                                                                         t0)
                   .count()
            << std::endl;
}

template <typename T>
void fft2DTimer(size_t rows, size_t cols, size_t threads) {
  std::vector<std::complex<float>> x(rows * cols, 1);
  std::vector<std::complex<float>> y(rows * cols);
  fftpp::ThreadScheduler scheduler(threads);
  auto t0 = std::chrono::high_resolution_clock::now();
  fftpp::fft2D<T>(y.data(), x.data(), rows, cols,
                  fftpp::TransformDirection::Forward, nullptr, &scheduler);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "2D FFT (rows=" << rows << ",cols=" << cols << ") time with "
            << threads << " threads runtime : "
            << std::chrono::duration_cast<std::chrono::duration<double>>(t1 -
                                                                         t0)
                   .count()
            << std::endl;
}
} // namespace

#define RUN_TEST_FUNCTION(testFunctionName)                                    \
  std::cout << "Running " << #testFunctionName << std::endl;                   \
  totalTests++;                                                                \
  if (testFunctionName()) {                                                    \
    std::cout << "Passed " << #testFunctionName << std::endl;                  \
    succeededTests++;                                                          \
  } else {                                                                     \
    std::cout << "Failed " << #testFunctionName << std::endl;                  \
  }

int main() {
  size_t succeededTests = 0;
  size_t totalTests = 0;

  try {
    RUN_TEST_FUNCTION(testFFTSize)
    RUN_TEST_FUNCTION(testFFTType<float>)
    RUN_TEST_FUNCTION(testFFTType<double>)
    RUN_TEST_FUNCTION(testFFTType<long double>)
  } catch (std::exception &e) {
    std::cout << "Exception during FFT testing: " << e.what() << std::endl;
    return -2;
  }

  int returnValue = -1;
  if (succeededTests == totalTests) {
    std::cout << "Successfully tested FFT" << std::endl;
    returnValue = 0;
  } else {
    std::cout << "Failed FFT testing" << std::endl;
    returnValue = -1;
  }

  std::cout << "Timing several FFTs" << std::endl;
  for (int i = 0; i < 3; ++i) {

    fftTimer<float>(128 * 15);
    fftTimer<float>(128 * 16);
    fftTimer<float>(128 * 32);
  }

  for (int i = 0; i < 3; ++i) {
    size_t rows = 4456;
    size_t cols = 4500;
    fft2DTimer<float>(rows, cols, 1);
    fft2DTimer<float>(rows, cols, 4);
  }

  return returnValue;
}