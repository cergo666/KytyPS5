#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/cpWaitStats.h"
#include "graphics/host_gpu/vulkanRecorder.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "libs/errno.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <mutex>
#if defined(_WIN32)
#include <windows.h>
#endif

namespace Libs::Graphics {

#if defined(_WIN32)
// KYTY_DIAG_FAULT=1: groups guest read faults by call stack and prints the costliest ones.
// Addresses are exe-relative; resolve them with llvm-symbolizer against kyty_emulator.pdb.
static void DiagFaultRead(uint64_t ns, uint64_t addr) {
	struct Entry {
		uint64_t count = 0, ns = 0, first_addr = 0;
		void*    frames[14] {};
		DWORD    thread = 0;
	};
	static std::mutex                 mutex;
	static std::map<uint64_t, Entry>  entries;
	static auto                       last = std::chrono::steady_clock::now();
	void*                             frames[14] {};
	const auto count = RtlCaptureStackBackTrace(2, 14, frames, nullptr);
	uint64_t   hash  = 1469598103934665603ull;
	for (unsigned i = 0; i < count; ++i) {
		hash = (hash ^ reinterpret_cast<uint64_t>(frames[i])) * 1099511628211ull;
	}
	std::lock_guard lock(mutex);
	auto& e = entries[hash];
	if (e.count == 0) {
		std::copy(frames, frames + 14, e.frames);
		e.first_addr = addr;
		e.thread     = GetCurrentThreadId();
	}
	e.count++;
	e.ns += ns;
	const auto now = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(30)) {
		return;
	}
	last = now;
	std::vector<std::pair<uint64_t, const Entry*>> sorted;
	for (const auto& [h, v]: entries) sorted.emplace_back(v.ns, &v);
	std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.first > b.first; });
	const auto base = reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
	std::printf("FAULTDIAG ---- top read-fault stacks\n");
	for (size_t i = 0; i < sorted.size() && i < 8; ++i) {
		const auto& v = *sorted[i].second;
		std::printf("FAULTDIAG n=%llu total_ms=%.1f thread=%lu addr=0x%llx:", static_cast<unsigned long long>(v.count),
		            static_cast<double>(v.ns) / 1e6, static_cast<unsigned long>(v.thread),
		            static_cast<unsigned long long>(v.first_addr));
		for (auto* f: v.frames) {
			if (f != nullptr) std::printf(" 0x%llx", static_cast<unsigned long long>(reinterpret_cast<uint64_t>(f) - base));
		}
		std::printf("\n");
	}
	std::fflush(stdout);
}
#endif

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache),
      m_geometry_motion(graphics, m_command_scheduler) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		// The guest GPU thread is gone; let its queued driver work finish before touching the
		// command buffer from this thread.
		VkRecorder::Drain();
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	if (access == PageFaultAccess::Write) {
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		KYTY_PROFILER_BLOCK("Ctx::FaultRead");
#if defined(_WIN32)
		static const bool diag = std::getenv("KYTY_DIAG_FAULT") != nullptr;
		if (diag) {
			const auto t0 = std::chrono::steady_clock::now();
			m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
			DiagFaultRead(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                  std::chrono::steady_clock::now() - t0).count()), fault_vaddr);
			return true;
		}
#endif
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	// Shader resource evaluation asks about many small reads inside the same mapping.
	struct Cached {
		const RenderContext* owner      = nullptr;
		uint64_t             begin      = 0;
		uint64_t             end        = 0;
		uint64_t             generation = 0;
	};
	thread_local Cached cached;
	if (cached.owner == this && vaddr >= cached.begin && vaddr < cached.end &&
	    size <= cached.end - vaddr &&
	    cached.generation == m_mapped_generation.load(std::memory_order_acquire)) {
		return true;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	uint64_t         begin = 0;
	uint64_t         end   = 0;
	if (!m_mapped_ranges.FindContaining(vaddr, size, begin, end)) {
		return false;
	}
	cached = {this, begin, end, m_mapped_generation.load(std::memory_order_relaxed)};
	return true;
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	m_mapped_generation.fetch_add(1, std::memory_order_acq_rel);
	// Buffers left from an earlier mapping may hold CPU writes made while unmapped.
	m_buffer_cache.AddBdaPending(vaddr, size);
}

/// Retire GPU ownership and overlapping guest writebacks before removing a mapping.
/// Cache checks run on the GPU command lane, or directly during teardown. Deferred
/// completion callbacks cannot call this method because they would wait on themselves.
void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		KYTY_WAIT_SCOPE("unmap.total");
		// Check cache ownership on the GPU thread. Guest-memory callbacks can still
		// access a range with no cached data, so they must finish before it is unmapped.
		if (m_command_scheduler.Active() &&
		    (m_buffer_cache.IsRegionRegistered(vaddr, size) ||
		     m_texture_cache.IsRegionRegistered(vaddr, size) ||
		     m_command_scheduler.HasPendingPriorityOperations({vaddr, size}))) {
			KYTY_WAIT_SCOPE("unmap.finish");
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
		m_mapped_generation.fetch_add(1, std::memory_order_acq_rel);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda(bool has_address_writes) {
	KYTY_WAIT_SCOPE("prepare.bda");
	KYTY_PROFILER_BLOCK("Draw::PrepareBda");
	if (has_address_writes) {
		m_buffer_cache.NoteUnknownWrite();
	}
	if (!m_bda_logged) {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		m_bda_logged = true;
	}
	{
		KYTY_PROFILER_BLOCK("Draw::PrepareBda(sync)");
		std::shared_lock lock(m_mapped_ranges_mutex);
		m_buffer_cache.SynchronizeBda(m_mapped_ranges);
	}
	m_fault_process_pending = true;
}

void RenderContext::RunGarbageCollector() {
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_texture_cache.ProcessDownloadImages();
	{
		KYTY_WAIT_SCOPE("gc.textures");
		m_texture_cache.RunGarbageCollector();
	}
	{
		KYTY_WAIT_SCOPE("gc.buffers");
		m_buffer_cache.RunGarbageCollector();
	}
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
