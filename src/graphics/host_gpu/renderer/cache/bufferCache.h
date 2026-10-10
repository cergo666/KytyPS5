#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <chrono>
#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// Address span covered by GPU-modified bytes; false when there are none.
	[[nodiscard]] bool GpuDirtyBounds(uint64_t& low, uint64_t& high) const {
		return m_gpu_modified_ranges.Bounds(low, high);
	}
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// A compute dispatch that fills [vaddr, vaddr + size) with one 32-bit value was just recorded.
	// Until anything else writes there, the bytes are known without reading them back.
	void RecordKnownFill(uint64_t vaddr, uint64_t size, uint32_t value);
	// When the GPU-modified bytes of the range are entirely covered by a known fill, publishes them
	// to guest memory as a download would, but without waiting for the GPU. GPU thread only.
	[[nodiscard]] bool TryResolveKnownFill(uint64_t vaddr, uint64_t size);
	void               ProcessFaultBuffer();
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	// Advances when a cached buffer is added or removed.
	// Advances whenever GPU-modified bytes are downloaded to guest memory (and so stop being
	// dirty), letting asynchronous guest-memory readers detect that data they read may be stale.
	[[nodiscard]] uint64_t GpuDirtyClearEpoch() const {
		return m_gpu_dirty_clear_epoch.load(std::memory_order_acquire);
	}
	// Whether GPU-modified bytes overlapping [begin, end) were published after `epoch`. GPU
	// thread only. Conservatively true once the epoch has left the bounded log.
	[[nodiscard]] bool ClearedSince(uint64_t epoch, uint64_t begin, uint64_t end) const {
		const auto current = m_gpu_dirty_clear_epoch.load(std::memory_order_acquire);
		if (current == epoch) {
			return false;
		}
		if (current - epoch > m_clear_log.size()) {
			return true;
		}
		for (uint64_t e = epoch + 1; e <= current; e++) {
			const auto& entry = m_clear_log[e % m_clear_log.size()];
			if (entry.begin < end && begin < entry.end) {
				return true;
			}
		}
		return false;
	}
	[[nodiscard]] uint64_t BuffersGeneration() const {
		return m_buffers_generation.load(std::memory_order_acquire);
	}
	// GPU-modified ranges nothing has written since MarkUnwritten; a later GPU write drops
	// them (CPU writes clear the GPU-modified state instead).
	void MarkUnwritten(uint64_t vaddr, uint64_t size) {
		std::scoped_lock lock(m_unwritten_lock);
		m_unwritten.Add(vaddr, size);
	}
	[[nodiscard]] bool IsUnwritten(uint64_t vaddr, uint64_t size) const {
		std::scoped_lock lock(m_unwritten_lock);
		return m_unwritten.Contains(vaddr, size);
	}
	// BDA stores choose addresses in the shader; serialize readback behind this tick.
	void NoteUnknownWrite();
	// Marks a range for the next SynchronizeBda. Any thread.
	void AddBdaPending(uint64_t vaddr, uint64_t size);
	// Uploads CPU writes made since the last call to the buffers inside `mapped`.
	void               SynchronizeBda(const RangeSet& mapped);
	void               RunGarbageCollector();
	// Called after recording a draw or dispatch: the ranges bound for writing, and any address
	// writes (which can reach untracked ranges), belong to the current command buffer.
	void RecordPendingWrites(bool address_writes = false);

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void                     NoteCurrentWrite(uint64_t vaddr, uint64_t size);
	// The GPU-written run around a guest read, clipped to its buffer.
	[[nodiscard]] std::pair<uint64_t, uint64_t> FaultAround(const Buffer& buffer, uint64_t vaddr,
	                                                        uint64_t size) const;
	// True when the recording command buffer neither writes the pages of the range nor
	// defers a publish: a readback can then run ahead of it.
	[[nodiscard]] bool CanReadAhead(uint64_t vaddr, uint64_t size);
	void ReadAhead(Buffer& buffer, std::span<const vk::BufferCopy> copies, uint64_t total_size);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// GPU writes of the current command buffer, and bound writes not recorded yet.
	void               MarkUnsubmittedWrite(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasPendingWrite(uint64_t vaddr, uint64_t size) const;
	[[nodiscard]] bool HasUnsubmittedWrite(uint64_t vaddr, uint64_t size) const;
	[[nodiscard]] bool CanDownloadDetached(uint64_t                        buffer_address,
	                                       std::span<const vk::BufferCopy> copies);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	std::atomic_uint64_t                              m_buffers_generation {0};
	std::atomic_uint64_t                              m_gpu_dirty_clear_epoch {0};
	struct DirtyClear {
		uint64_t begin = 0, end = 0;
	};
	// Range published at each epoch, indexed by epoch modulo the size.
	std::array<DirtyClear, 1024> m_clear_log {};
	void                         LogDirtyClear(uint64_t begin, uint64_t end) {
		const auto epoch = m_gpu_dirty_clear_epoch.load(std::memory_order_relaxed) + 1;
		m_clear_log[epoch % m_clear_log.size()] = {begin, end};
		m_gpu_dirty_clear_epoch.store(epoch, std::memory_order_release);
	}
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	RangeSet                                          m_unsubmitted_writes;
	RangeSet                                          m_pending_writes;
	uint64_t                                          m_unsubmitted_tick   = 0;
	uint64_t                                          m_address_write_tick = 0;
	// KYTY_DIAG_READBACK=1: recent GPU writes with the tick that recorded them, to report
	// whether synchronous readbacks wait for work that is still running.
	struct WriteTick {
		uint64_t begin = 0, end = 0, tick = 0;
	};
	std::vector<WriteTick> m_write_ticks;
	[[nodiscard]] uint64_t LastWriteTick(uint64_t begin, uint64_t end) const;
	void ReportReadback(uint64_t writer_tick, bool writer_done, uint64_t bytes, bool detached,
	                    uint64_t elapsed_ns);
	struct KnownFill {
		uint64_t address = 0, size = 0;
		uint32_t value = 0;
	};
	void                   EraseKnownFills(uint64_t vaddr, uint64_t size);
	std::mutex             m_known_fill_mutex;
	std::vector<KnownFill> m_known_fills;
	std::atomic<uint32_t>  m_known_fill_count {0};
	RangeSet                                           m_unwritten;
	mutable std::mutex                                 m_unwritten_lock;
	RangeSet                                           m_current_writes; // in the recording tick
	uint64_t                                           m_current_writes_tick = 0;
	uint64_t                                           m_unknown_write_tick = 0;
	std::unique_ptr<Buffer>                            m_ahead_readback;
	// CPU-written ranges and new buffers since the last SynchronizeBda. Guest threads add
	// write faults, so it has its own lock.
	std::mutex                                        m_bda_pending_lock;
	RangeSet                                          m_bda_pending;
	MemoryTracker                                     m_memory_tracker;
	uint64_t                                          m_registration_epoch = 0;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	std::chrono::steady_clock::time_point             m_usage_refreshed {};
	std::chrono::steady_clock::time_point             m_gc_idle_until {};
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
