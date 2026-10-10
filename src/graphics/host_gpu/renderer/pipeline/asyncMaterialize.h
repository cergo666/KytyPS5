#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCMATERIALIZE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCMATERIALIZE_H_

#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/perfLog.h"
#include "kernel/memory.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#if defined(_MSC_VER) || defined(__clang__)
#include <immintrin.h>
#endif

namespace Libs::Graphics {

// One shader-stage resource materialization executed off the command-processor thread.
//
// MaterializeResources is a pure function of the shader's resource plan, the stage's user data
// and the guest memory it reads. The worker runs it ahead of the command processor and records
// every guest-memory range it read, so the command processor can later prove that no GPU write
// has touched those ranges and otherwise discard the result and recompute it synchronously.
struct MaterializeTask {
	struct Range {
		uint64_t address = 0;
		uint64_t size    = 0;
	};

	// Inputs, written by the command-processor thread before Submit().
	const ShaderRecompiler::IR::ResourcePlan* plan = nullptr;
	std::array<uint32_t, 40>                  user_data {};
	uint32_t                                  user_data_count = 0;
	uint64_t                                  shader_base     = 0;
	std::array<uint32_t, 3>                   workgroup_counts {};
	uint32_t                                  workgroup_count_size = 0;
	// BufferCache::GpuDirtyClearEpoch() sampled at submission.
	uint64_t dirty_epoch = 0;

	// Outputs, written by the worker before it publishes completion.
	ShaderRecompiler::IR::ResourceSnapshot       snapshot;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	std::vector<Range>                           reads;
	bool                                         ok = false;

	// 0 = idle, 1 = queued or running, 2 = complete.
	std::atomic<uint32_t> state {0};
};

// Runs materializations on dedicated worker threads. Work is split into lanes and every shader
// plan must always be submitted to the same lane: a plan's evaluation scratch is not shared-safe,
// so one thread per plan keeps it single-threaded. Vertex and pixel plans never coincide.
class AsyncMaterializer {
public:
	static constexpr size_t kLaneCount = 6;
	static constexpr size_t kTaskCount = 512;

	AsyncMaterializer() {
		m_tasks.reset(new MaterializeTask[kTaskCount]);
		m_free.reserve(kTaskCount);
		for (size_t i = 0; i < kTaskCount; i++) {
			m_tasks[i].reads.reserve(32);
			m_free.push_back(&m_tasks[i]);
		}
		for (auto& lane: m_lanes) {
			lane.thread = std::jthread([&lane](std::stop_token stop) { Run(lane, stop); });
		}
	}

	~AsyncMaterializer() {
		for (auto& lane: m_lanes) {
			lane.thread.request_stop();
			Wake(lane);
		}
		// std::jthread joins on destruction.
	}

	AsyncMaterializer(const AsyncMaterializer&)            = delete;
	AsyncMaterializer& operator=(const AsyncMaterializer&) = delete;

	// Command-processor thread only. Returns nullptr when the pool is exhausted.
	MaterializeTask* Acquire() {
		if (m_free.empty()) {
			return nullptr;
		}
		auto* task = m_free.back();
		m_free.pop_back();
		return task;
	}

	// Command-processor thread only. The task must no longer be referenced by anyone.
	void Release(MaterializeTask* task) {
		task->state.store(0, std::memory_order_relaxed);
		m_free.push_back(task);
	}

	// Command-processor thread only.
	void Submit(MaterializeTask* task, size_t lane_index) {
		auto& lane = m_lanes[lane_index % kLaneCount];
		task->ok   = false;
		task->state.store(1, std::memory_order_relaxed);
		const auto head = lane.head.load(std::memory_order_relaxed);
		// The pool is smaller than the ring, so a queued task always finds a free slot.
		lane.ring[head % kRingSize] = task;
		lane.head.store(head + 1, std::memory_order_seq_cst);
		lane.submitted++;
		if (lane.sleeping.load(std::memory_order_seq_cst)) {
			Wake(lane);
		}
	}

	static void Wait(const MaterializeTask* task) {
		uint32_t spins  = 0;
		bool     warned = false;
		auto     start  = std::chrono::steady_clock::now();
		while (task->state.load(std::memory_order_acquire) != 2) {
			Relax(++spins);
			if (!warned && (spins & 0xfffu) == 0 &&
			    std::chrono::steady_clock::now() - start > std::chrono::seconds(5)) {
				warned = true;
				std::fprintf(stderr, "[AsyncDraw] materialization worker stalled for 5s\n");
			}
		}
	}

	// Blocks until every submitted task has completed. Command-processor thread only.
	void WaitIdle() {
		for (auto& lane: m_lanes) {
			uint32_t spins = 0;
			while (lane.completed.load(std::memory_order_acquire) != lane.submitted) {
				Relax(++spins);
			}
		}
	}

private:
	static constexpr size_t kRingSize = 1024;
	static_assert(kRingSize > kTaskCount);

	struct Lane {
		std::array<MaterializeTask*, kRingSize> ring {};
		alignas(64) std::atomic<uint64_t> head {0};
		alignas(64) std::atomic<uint64_t> completed {0};
		uint64_t                          submitted = 0; // command-processor thread only
		alignas(64) std::atomic<bool> sleeping {false};
		std::mutex              sleep_mutex;
		std::condition_variable sleep_cv;
		std::jthread            thread;
	};

	static void Relax(uint32_t spins) {
		if (spins < 4000) {
#if defined(_MSC_VER) || defined(__clang__)
			_mm_pause();
#endif
		} else {
			std::this_thread::yield();
		}
	}

	static void Wake(Lane& lane) {
		std::lock_guard lock(lane.sleep_mutex);
		lane.sleep_cv.notify_one();
	}

	static bool ReadGuest(void* userdata, uint64_t address, std::span<uint32_t> values) {
		if (values.empty()) {
			return false;
		}
		// No GPU-dirty download here: that mutates renderer state. The command-processor
		// thread validates the recorded ranges against GPU-modified bytes before using the
		// result and recomputes synchronously (with downloads) if any were touched.
		if (!LibKernel::Memory::TryReadBacking(address, values.data(), values.size_bytes())) {
			return false;
		}
		static_cast<std::vector<MaterializeTask::Range>*>(userdata)->push_back(
		    {address, values.size_bytes()});
		return true;
	}

	// Ordinary (non-specialization) reads. The synchronous path dereferences the guest mapping
	// directly, but that mapping can be protected while the GPU owns the data; touching it from
	// this thread would fault into a handler that waits for the GPU thread, which in turn waits for
	// us. Read the backing store instead and let the command-processor thread's dirty check
	// reject anything the GPU may have written.
	static bool ReadGuestRaw(void* userdata, uint64_t address, std::span<uint32_t> values) {
		return ReadGuest(userdata, address, values);
	}

	static void Execute(MaterializeTask& task) {
		task.reads.clear();
		ShaderRecompiler::IR::SrtRuntime runtime;
		runtime.user_data                  = std::span<const uint32_t>(task.user_data.data(),
		                                                               task.user_data_count);
		runtime.shader_base                = task.shader_base;
		runtime.read_memory                = ReadGuestRaw;
		runtime.userdata                   = &task.reads;
		runtime.read_specialization_memory = ReadGuest;
		runtime.workgroup_counts           = std::span<const uint32_t>(
            task.workgroup_counts.data(), task.workgroup_count_size);
		task.ok = ShaderRecompiler::IR::MaterializeResources(*task.plan, runtime, task.snapshot,
		                                                     task.specialization);
	}

	static void Run(Lane& lane, std::stop_token stop) {
		Libs::Graphics::PerfLog::RegisterThisThread("lane");
		// Spin iterations before a lane sleeps (KYTY_ASYNC_SPIN). Longer spinning was measured to burn
		// CPU without making frames faster.
		static const uint32_t spin_limit = [] {
			const char* env = std::getenv("KYTY_ASYNC_SPIN");
			return env != nullptr ? static_cast<uint32_t>(std::atoi(env)) : 20000u;
		}();
		uint64_t tail  = 0;
		uint32_t spins = 0;
		while (!stop.stop_requested()) {
			if (tail == lane.head.load(std::memory_order_acquire)) {
				if (++spins < spin_limit) {
#if defined(_MSC_VER) || defined(__clang__)
					_mm_pause();
#endif
					continue;
				}
				// Idle for a while: sleep until the next submission.
				std::unique_lock lock(lane.sleep_mutex);
				lane.sleeping.store(true, std::memory_order_seq_cst);
				if (tail == lane.head.load(std::memory_order_seq_cst) && !stop.stop_requested()) {
					lane.sleep_cv.wait_for(lock, std::chrono::milliseconds(50));
				}
				lane.sleeping.store(false, std::memory_order_seq_cst);
				spins = 0;
				continue;
			}
			spins      = 0;
			auto* task = lane.ring[tail % kRingSize];
			Execute(*task);
			task->state.store(2, std::memory_order_release);
			tail++;
			lane.completed.store(tail, std::memory_order_release);
		}
	}

	std::unique_ptr<MaterializeTask[]> m_tasks;
	std::vector<MaterializeTask*>      m_free; // command-processor thread only
	std::array<Lane, kLaneCount>       m_lanes;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_ASYNCMATERIALIZE_H_
