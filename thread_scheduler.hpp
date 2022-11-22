#ifndef FFTPP_THREAD_SCHEDULER_HPP
#define FFTPP_THREAD_SCHEDULER_HPP

/* Author: Cory Brinck
 */

#include <functional>
#include <list>
#include <thread>
#include <vector>

namespace fftpp {

/** Object for indicating the number of threads to schedule and how/where they
 * should be executed. */
struct ThreadScheduler {
  using SchedulingFunction =
      std::function<void(const std::vector<std::function<void()>> &)>;

  /*Execute tasks as new threads and wait until all are complete*/
  inline static void
  executeNewThreadsAndWait(const std::vector<std::function<void()>> &tasks) {
    std::list<std::thread> threads;
    for (size_t i = 0; i < tasks.size(); ++i) {
      if (i == tasks.size() - 1)
        tasks[i](); // Use the calling thread
      else
        threads.emplace_back(tasks[i]);
    }

    for (auto &&thread : threads)
      thread.join();
  }

  explicit ThreadScheduler(
      size_t numThreads_,
      SchedulingFunction executeAndWait_ = &executeNewThreadsAndWait)
      : numThreads(numThreads_), executeAndWait(executeAndWait_) {}

  /* Number of simultaneous threads to schedule*/
  size_t numThreads;

  /* Function to execute
            threads.This function must block *until they are
                complete.*/
  SchedulingFunction executeAndWait;
};

} // namespace fftpp

#endif
