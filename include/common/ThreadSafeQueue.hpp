#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T>
class ThreadSafeQueue {
public:
  explicit ThreadSafeQueue(size_t capacity) : cap_(capacity) {}

  // Push blocks if full (backpressure). Low-latency variant could "drop oldest".
  void push(T&& item) {
    std::unique_lock<std::mutex> lk(m_);
    cv_not_full_.wait(lk, [&]{ return q_.size() < cap_ || stopped_; });
    if (stopped_) return;
    q_.push(std::move(item));
    cv_not_empty_.notify_one();
  }

  // Pop blocks if empty
  bool pop(T& out) {
    std::unique_lock<std::mutex> lk(m_);
    cv_not_empty_.wait(lk, [&]{ return !q_.empty() || stopped_; });
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop();
    cv_not_full_.notify_one();
    return true;
  }

  void stop() {
    std::lock_guard<std::mutex> lk(m_);
    stopped_ = true;
    cv_not_empty_.notify_all();
    cv_not_full_.notify_all();
  }

private:
  size_t cap_;
  std::mutex m_;
  std::condition_variable cv_not_empty_;
  std::condition_variable cv_not_full_;
  std::queue<T> q_;
  bool stopped_ = false;
};
