// imi's thread team. The workers stay up between steps; inside a step they meet only at sync(), a spinning
// barrier, so a step has exactly as many stops as the math needs. Between steps they spin a little, then sleep.
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define IMI_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define IMI_PAUSE() __asm__ __volatile__("yield")
#else
#define IMI_PAUSE() ((void) 0)
#endif

struct imi_team {
    int n = 1;

    void start(int nthreads) {
        n = nthreads < 1 ? 1 : nthreads;
        spin_sync = pauses_for(30);     // inside a step a teammate is usually microseconds away
        spin_idle = pauses_for(1000);   // between steps, a millisecond before a worker sleeps
        for (int t = 1; t < n; t++) th.emplace_back([this, t] { worker(t); });
    }
    // every member runs f(tid), the caller as tid 0; returns when all are done
    void run(const std::function<void(int)> & f) {
        job = &f;
        done.store(0, std::memory_order_relaxed);
        gen.fetch_add(1, std::memory_order_release);
        wake();
        f(0);
        wait_for([&] { return done.load(std::memory_order_acquire) == n - 1; });
    }
    // inside f: wait for the whole team
    void sync() {
        if (n == 1) return;
        const int p = phase.load(std::memory_order_acquire);
        if (arrived.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
            arrived.store(0, std::memory_order_relaxed);
            phase.store(p + 1, std::memory_order_release);
        } else {
            wait_for([&] { return phase.load(std::memory_order_acquire) != p; });
        }
    }
    ~imi_team() {
        quit.store(true);
        gen.fetch_add(1, std::memory_order_release);
        wake();
        for (auto & t : th) t.join();
    }

private:
    std::vector<std::thread> th;
    const std::function<void(int)> * job = nullptr;
    alignas(64) std::atomic<int> gen{0};
    alignas(64) std::atomic<int> done{0};
    alignas(64) std::atomic<int> arrived{0};
    alignas(64) std::atomic<int> phase{0};
    std::atomic<bool> quit{false};
    int spin_sync = 1000, spin_idle = 30000;

    // Waiting for a teammate: spin a short while (the answer is usually microseconds away), then give the processor
    // back on every look. A teammate can be pushed off its processor by anything else on the computer (the browser,
    // the virus scanner); spinning on would keep every processor busy while it waits to get back, and one busy program
    // made a step 5x slower that way (8 threads on an 8-thread computer). Giving way lets it back at once.
    template <class Done> void wait_for(Done done_now) const {
        for (int spins = 0; !done_now();) {
            if (spins < spin_sync) { IMI_PAUSE(); spins++; } else std::this_thread::yield();
        }
    }
    // How many pauses make the given microseconds on this processor: one pause is about 10 cycles on older Intel
    // chips (Haswell) and about 140 on Skylake and newer, so a count alone would wait 14x longer on a new
    // laptop than on an older one.
    static int pauses_for(double us) {
        const int probe = 20000;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < probe; i++) IMI_PAUSE();
        const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / probe;
        const double n = us * 1000.0 / (ns > 0.1 ? ns : 0.1);
        return n < 64 ? 64 : n > 4e6 ? 4000000 : (int) n;
    }

    // a sleeping worker wakes when gen moves on. C++20's atomic wait does it where the library has it (Mac, Linux);
    // GCC 10's library (Ubuntu 22.04's Windows cross-compiler) predates it, so there a lock and a signal do the same:
    // the worker checks gen under the lock before it sleeps, and the waker takes the lock after moving gen, so a wake
    // can never fall between the check and the sleep.
#if defined(__cpp_lib_atomic_wait)
    void wake() { gen.notify_all(); }
    void sleep_while(int seen) { gen.wait(seen, std::memory_order_acquire); }
#else
    std::mutex m;
    std::condition_variable cv;
    void wake() { { std::lock_guard<std::mutex> l(m); } cv.notify_all(); }
    void sleep_while(int seen) {
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&] { return gen.load(std::memory_order_acquire) != seen; });
    }
#endif

    void worker(int tid) {
        int seen = 0;
        for (;;) {
            int g, spins = 0;
            while ((g = gen.load(std::memory_order_acquire)) == seen) {
                if (++spins < spin_idle) IMI_PAUSE(); else sleep_while(seen);
            }
            seen = g;
            if (quit.load()) return;
            (*job)(tid);
            done.fetch_add(1, std::memory_order_acq_rel);
        }
    }
};
