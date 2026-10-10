#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_CPWAITSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_CPWAITSTATS_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace Libs::Graphics {

// KYTY_DIAG_WAITS=1: totals the time spent in marked scopes (waits and syncs, any thread) and
// prints "WAITDIAG name count total_ms max_ms" for each every 10 s.
class CpWaitStats {
public:
	static constexpr int kMax = 32;

	struct Slot {
		const char*           name = nullptr;
		std::atomic<uint64_t> count {0};
		std::atomic<uint64_t> ns {0};
		std::atomic<uint64_t> max_ns {0};
	};

	static bool Enabled() {
		static const bool enabled = std::getenv("KYTY_DIAG_WAITS") != nullptr;
		return enabled;
	}

	static Slot* Slots() {
		static Slot slots[kMax];
		return slots;
	}

	static Slot* Find(const char* name) {
		static std::mutex mutex;
		std::lock_guard   lock(mutex);
		auto*             slots = Slots();
		for (int i = 0; i < kMax; ++i) {
			if (slots[i].name == name) return &slots[i];
			if (slots[i].name == nullptr) {
				slots[i].name = name;
				return &slots[i];
			}
		}
		return &slots[kMax - 1];
	}

	static void Record(Slot* slot, uint64_t ns) {
		slot->count.fetch_add(1, std::memory_order_relaxed);
		slot->ns.fetch_add(ns, std::memory_order_relaxed);
		uint64_t prev = slot->max_ns.load(std::memory_order_relaxed);
		while (ns > prev && !slot->max_ns.compare_exchange_weak(prev, ns)) {
		}
		static std::atomic<int64_t> next_print {0};
		const int64_t               now = std::chrono::steady_clock::now().time_since_epoch().count();
		int64_t                     due = next_print.load(std::memory_order_relaxed);
		if (now >= due && next_print.compare_exchange_strong(due, now + 10'000'000'000ll)) {
			Print();
		}
	}

	static void Print() {
		auto* slots = Slots();
		for (int i = 0; i < kMax && slots[i].name != nullptr; ++i) {
			auto& slot = slots[i];
			std::printf("WAITDIAG %-28s n=%-8llu total_ms=%-9.1f max_ms=%.2f\n", slot.name,
			            static_cast<unsigned long long>(slot.count.load()),
			            static_cast<double>(slot.ns.load()) / 1e6,
			            static_cast<double>(slot.max_ns.load()) / 1e6);
			slot.count  = 0;
			slot.ns     = 0;
			slot.max_ns = 0;
		}
		std::fflush(stdout);
	}

	// RAII scope; costs one flag test when the diagnostic is off.
	class Scope {
	public:
		explicit Scope(const char* name) {
			if (Enabled()) {
				m_slot  = Find(name);
				m_start = std::chrono::steady_clock::now();
			}
		}
		~Scope() {
			if (m_slot != nullptr) {
				Record(m_slot, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
				                                         std::chrono::steady_clock::now() - m_start)
				                                         .count()));
			}
		}
		Scope(const Scope&)            = delete;
		Scope& operator=(const Scope&) = delete;

	private:
		Slot*                                 m_slot = nullptr;
		std::chrono::steady_clock::time_point m_start;
	};

};

} // namespace Libs::Graphics

#define KYTY_WAIT_CAT2(a, b) a##b
#define KYTY_WAIT_CAT(a, b) KYTY_WAIT_CAT2(a, b)
#define KYTY_WAIT_SCOPE(name) \
	::Libs::Graphics::CpWaitStats::Scope KYTY_WAIT_CAT(kyty_wait_scope_, __COUNTER__) {name}

#endif
