#include "fft.hpp"
#include "fft_size.hpp"
#include <chrono>
#include <iostream>

namespace {
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
             fftpp::TransformDirection::Inverse, &fftObj);

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

  if (e > 1e-10 && std::is_same_v<T, double>)
    std::cout << N << ":" << e << std::endl;

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
             fftpp::TransformDirection::Inverse, &fftObj);

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
                      fftpp::TransformDirection::Inverse, nullptr, pScheduler);
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
  size_t num2D = 1000;
  for (size_t n = 1; n < num2D; ++n) {
    if (!testFFT2D<T>(n, n % 7 + 9, nullptr)) {
      return false;
    }
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  fftpp::ThreadScheduler scheduler(3);
  for (size_t n = 1; n < num2D; ++n) {
    if (!testFFT2D<T>(n, n % 7 + 9, &scheduler)) {
      return false;
    }
  }

  auto t3 = std::chrono::high_resolution_clock::now();

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
  std::cout << "Threaded 2D test time: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(t3 -
                                                                         t2)
                   .count()
            << std::endl;
  return true;
}
} // namespace

int main() {
  try {
    testFFTType<long double>();
    testFFTType<double>();
    testFFTType<float>();
  } catch (std::exception &e) {
    std::cout << "Exception during FFT testing: " << e.what() << std::endl;
    return -1;
  }
  std::cout << "Successfully tested FFT" << std::endl;
  return 0;
}