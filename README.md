# FFT++
Fast Discrete Fourier Transform (FFT, DFT) implementation in C++

### FFT++ is Fast
* At worst O(N*log(N)) operations for all sizes of N (Make sure you enable your compiler's optimizations)
* Even faster for sizes with small prime factors
* Multi-threaded interface provided

### FFT++ is Simple
* Header-only FFT implementation dependent only on C++11 standard library
* No global variables and const correctness makes multithreading safe and easy
* Smart memory management makes twiddle-factor/DFT generation safe and easy

### FFT++ is Free
* Boost license allows you to use it easily in any project
