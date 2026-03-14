#ifndef DIGTOOLS_H
#define DIGTOOLS_H

#include <iostream>
#include <chrono>
#include <cstdio>
#include <string>
#include <utility>

template <typename T>
void printer(const char* name, const T& a) {
    std::cout << name << " : " << a << std::endl;
}

#define PRINT(var) printer(#var, var)




// Simple RAII timer: prints elapsed ms when it goes out of scope.
struct ScopedTimer {
  using clock = std::chrono::high_resolution_clock;

  const char* name;
  clock::time_point t0;

  explicit ScopedTimer(const char* label)
      : name(label), t0(clock::now()) {}

  ~ScopedTimer() {
    const auto t1 = clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr, "[TIMER] %-28s %10.3f ms\n", name, ms);
  }

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;
};

// Macro to auto-name the variable so you can write: TIMER("stage");
#define TIMER(label) ScopedTimer timer_##__LINE__(label)

// Optional: explicit start/stop timer you can query (not RAII)
struct ManualTimer {
  using clock = std::chrono::high_resolution_clock;
  clock::time_point t0;

  void start() { t0 = clock::now(); }

  double ms() const {
    const auto t1 = clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
};

#endif 