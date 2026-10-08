#include "graphics/host_gpu/vulkanRecorder.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string_view>
#include <thread>
#include <tuple>

#if defined(_MSC_VER) || defined(__clang__)
#include <immintrin.h>
#endif

namespace Libs::Graphics::VkRecorder {

namespace {

// Real driver entry points, captured from the dispatcher before it is redirected.
struct Real {
	PFN_vkCmdPipelineBarrier                  vkCmdPipelineBarrier                  = nullptr;
	PFN_vkCmdPipelineBarrier2                 vkCmdPipelineBarrier2                 = nullptr;
	PFN_vkCmdBindPipeline                     vkCmdBindPipeline                     = nullptr;
	PFN_vkCmdPushDescriptorSetKHR             vkCmdPushDescriptorSetKHR             = nullptr;
	PFN_vkCmdBeginRendering                   vkCmdBeginRendering                   = nullptr;
	PFN_vkCmdCopyImage                        vkCmdCopyImage                        = nullptr;
	PFN_vkCmdDispatch                         vkCmdDispatch                         = nullptr;
	PFN_vkCmdEndRendering                     vkCmdEndRendering                     = nullptr;
	PFN_vkCmdPushConstants                    vkCmdPushConstants                    = nullptr;
	PFN_vkCmdCopyBuffer                       vkCmdCopyBuffer                       = nullptr;
	PFN_vkCmdDraw                             vkCmdDraw                             = nullptr;
	PFN_vkCmdClearColorImage                  vkCmdClearColorImage                  = nullptr;
	PFN_vkCmdCopyBufferToImage                vkCmdCopyBufferToImage                = nullptr;
	PFN_vkCmdCopyImageToBuffer                vkCmdCopyImageToBuffer                = nullptr;
	PFN_vkCmdDrawIndexed                      vkCmdDrawIndexed                      = nullptr;
	PFN_vkCmdFillBuffer                       vkCmdFillBuffer                       = nullptr;
	PFN_vkCmdSetScissor                       vkCmdSetScissor                       = nullptr;
	PFN_vkCmdSetViewport                      vkCmdSetViewport                      = nullptr;
	PFN_vkCmdBindDescriptorSets               vkCmdBindDescriptorSets               = nullptr;
	PFN_vkCmdBindIndexBuffer                  vkCmdBindIndexBuffer                  = nullptr;
	PFN_vkCmdBindVertexBuffers2               vkCmdBindVertexBuffers2               = nullptr;
	PFN_vkCmdBlitImage                        vkCmdBlitImage                        = nullptr;
	PFN_vkCmdClearDepthStencilImage           vkCmdClearDepthStencilImage           = nullptr;
	PFN_vkCmdDispatchIndirect                 vkCmdDispatchIndirect                 = nullptr;
	PFN_vkCmdDrawIndirect                     vkCmdDrawIndirect                     = nullptr;
	PFN_vkCmdDrawMeshTasksEXT                 vkCmdDrawMeshTasksEXT                 = nullptr;
	PFN_vkCmdResolveImage                     vkCmdResolveImage                     = nullptr;
	PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT vkCmdSetAttachmentFeedbackLoopEnableEXT = nullptr;
	PFN_vkCmdSetBlendConstants                vkCmdSetBlendConstants                = nullptr;
	PFN_vkCmdSetCheckpointNV                  vkCmdSetCheckpointNV                  = nullptr;
	PFN_vkCmdSetColorWriteEnableEXT           vkCmdSetColorWriteEnableEXT           = nullptr;
	PFN_vkCmdSetDepthBias                     vkCmdSetDepthBias                     = nullptr;
	PFN_vkCmdSetDepthBiasEnable               vkCmdSetDepthBiasEnable               = nullptr;
	PFN_vkCmdSetDepthBounds                   vkCmdSetDepthBounds                   = nullptr;
	PFN_vkCmdSetDepthBoundsTestEnable         vkCmdSetDepthBoundsTestEnable         = nullptr;
	PFN_vkCmdSetDepthCompareOp                vkCmdSetDepthCompareOp                = nullptr;
	PFN_vkCmdSetDepthTestEnable               vkCmdSetDepthTestEnable               = nullptr;
	PFN_vkCmdSetDepthWriteEnable              vkCmdSetDepthWriteEnable              = nullptr;
	PFN_vkCmdSetLineWidth                     vkCmdSetLineWidth                     = nullptr;
	PFN_vkCmdSetScissorWithCount              vkCmdSetScissorWithCount              = nullptr;
	PFN_vkCmdSetStencilCompareMask            vkCmdSetStencilCompareMask            = nullptr;
	PFN_vkCmdSetStencilOp                     vkCmdSetStencilOp                     = nullptr;
	PFN_vkCmdSetStencilReference              vkCmdSetStencilReference              = nullptr;
	PFN_vkCmdSetStencilTestEnable             vkCmdSetStencilTestEnable             = nullptr;
	PFN_vkCmdSetStencilWriteMask              vkCmdSetStencilWriteMask              = nullptr;
	PFN_vkCmdSetViewportWithCount             vkCmdSetViewportWithCount             = nullptr;
	PFN_vkBeginCommandBuffer                  vkBeginCommandBuffer                  = nullptr;
	PFN_vkEndCommandBuffer                    vkEndCommandBuffer                    = nullptr;
	PFN_vkQueueSubmit                         vkQueueSubmit                         = nullptr;
	PFN_vkQueuePresentKHR                     vkQueuePresentKHR                     = nullptr;
	PFN_vkQueueWaitIdle                       vkQueueWaitIdle                       = nullptr;
	PFN_vkDeviceWaitIdle                      vkDeviceWaitIdle                      = nullptr;
	PFN_vkAllocateCommandBuffers              vkAllocateCommandBuffers              = nullptr;
	PFN_vkFreeCommandBuffers                  vkFreeCommandBuffers                  = nullptr;
	PFN_vkDestroyCommandPool                  vkDestroyCommandPool                  = nullptr;
};

Real                   g_real {};
bool                   g_enabled = false;
thread_local bool      t_defer   = false;
// Serialises VkQueue access between the recorder thread and threads that submit directly.
std::mutex             g_queue_mutex;

// ------------------------------------------------------------------------------------------------
// Single-producer (guest GPU thread) / single-consumer (recorder thread) ring of variable-size
// records. A record is a 16-byte header (size, replay function) followed by its payload.

constexpr uint64_t kRingSize = 64ull * 1024 * 1024;

using ReplayFn = void (*)(std::byte* payload);

struct Header {
	uint64_t size;
	ReplayFn fn; // nullptr: padding up to the end of the ring
};
static_assert(sizeof(Header) == 16);

struct Ring {
	std::byte*                       data = nullptr;
	alignas(64) std::atomic<uint64_t> head {0}; // published by the producer
	alignas(64) std::atomic<uint64_t> tail {0}; // consumed (executed) by the recorder
	alignas(64) std::atomic<uint32_t> sleeping {0};
	uint64_t                          write = 0; // producer only
};

Ring* g_ring = nullptr;

constexpr size_t Align8(size_t v) {
	return (v + 7u) & ~size_t {7};
}

constexpr size_t Align16(size_t v) {
	return (v + 15u) & ~size_t {15};
}

void Pause(uint32_t spins) {
	if (spins < 2000) {
#if defined(_MSC_VER) || defined(__clang__)
		_mm_pause();
#endif
	} else {
		std::this_thread::yield();
	}
}

struct Counter {
	size_t size = 0;
	template <class T>
	void Put(const T& /*value*/) {
		size = Align8(size) + sizeof(T);
	}
	template <class T>
	void PutArray(const T* /*values*/, size_t count) {
		size = Align8(size) + sizeof(T) * count;
	}
};

struct Writer {
	std::byte* base = nullptr;
	size_t     off  = 0;
	template <class T>
	void Put(const T& value) {
		off = Align8(off);
		new (base + off) T(value);
		off += sizeof(T);
	}
	template <class T>
	void PutArray(const T* values, size_t count) {
		off = Align8(off);
		if (count != 0) {
			std::memcpy(base + off, values, sizeof(T) * count);
		}
		off += sizeof(T) * count;
	}
};

struct Reader {
	std::byte* base = nullptr;
	size_t     off  = 0;
	template <class T>
	T* Get() {
		off      = Align8(off);
		auto* at = reinterpret_cast<T*>(base + off);
		off += sizeof(T);
		return at;
	}
	template <class T>
	T* GetArray(size_t count) {
		off      = Align8(off);
		auto* at = count != 0 ? reinterpret_cast<T*>(base + off) : nullptr;
		off += sizeof(T) * count;
		return at;
	}
};

void WakeRecorder() {
	if (g_ring->sleeping.load(std::memory_order_seq_cst) != 0) {
		g_ring->head.notify_one();
	}
}

void WaitForSpace(size_t need) {
	auto&    ring  = *g_ring;
	uint32_t spins = 0;
	while (ring.write + need - ring.tail.load(std::memory_order_acquire) > kRingSize) {
		WakeRecorder();
		Pause(++spins);
	}
}

std::byte* Reserve(size_t need) {
	auto& ring = *g_ring;
	auto  off  = ring.write % kRingSize;
	if (off + need > kRingSize) {
		const auto pad = kRingSize - off;
		WaitForSpace(pad);
		auto* header = reinterpret_cast<Header*>(ring.data + off);
		header->size = pad;
		header->fn   = nullptr;
		ring.write += pad;
	}
	WaitForSpace(need);
	return ring.data + ring.write % kRingSize;
}

void Publish(size_t size) {
	auto& ring = *g_ring;
	ring.write += size;
	ring.head.store(ring.write, std::memory_order_seq_cst);
	WakeRecorder();
}

template <class F>
void Emit(ReplayFn fn, F&& serialize) {
	Counter counter;
	serialize(counter);
	const size_t need = Align16(sizeof(Header) + counter.size);
	EXIT_IF(need > kRingSize / 4);
	auto* at     = Reserve(need);
	auto* header = reinterpret_cast<Header*>(at);
	header->size = need;
	header->fn   = fn;
	Writer writer {at + sizeof(Header)};
	serialize(writer);
	Publish(need);
}

void RecorderThread() {
	auto&    ring  = *g_ring;
	uint64_t tail  = 0;
	uint32_t spins = 0;
	for (;;) {
		auto head = ring.head.load(std::memory_order_acquire);
		if (head == tail) {
			if (++spins < 4000) {
#if defined(_MSC_VER) || defined(__clang__)
				_mm_pause();
#endif
				continue;
			}
			ring.sleeping.store(1, std::memory_order_seq_cst);
			head = ring.head.load(std::memory_order_seq_cst);
			if (head == tail) {
				ring.head.wait(head, std::memory_order_acquire);
			}
			ring.sleeping.store(0, std::memory_order_seq_cst);
			spins = 0;
			continue;
		}
		spins = 0;
		while (tail != head) {
			auto* header = reinterpret_cast<Header*>(ring.data + tail % kRingSize);
			if (header->fn != nullptr) {
				header->fn(reinterpret_cast<std::byte*>(header) + sizeof(Header));
			}
			tail += header->size;
			ring.tail.store(tail, std::memory_order_release);
		}
	}
}

// A call that cannot be deferred (unknown pNext chains): let the recorder catch up, then call the
// driver directly so the order is preserved.
void BeforeDirectCall() {
	if (t_defer) {
		Drain();
	}
}

template <class T>
bool AnyNext(const T* items, uint32_t count) {
	for (uint32_t i = 0; i < count; i++) {
		if (items[i].pNext != nullptr) {
			return true;
		}
	}
	return false;
}

// ------------------------------------------------------------------------------------------------
// Commands whose arguments are plain values.

template <auto Member, class... Args>
struct ScalarCmd {
	using Pack = std::tuple<VkCommandBuffer, Args...>;
	static void Replay(std::byte* payload) {
		Reader reader {payload};
		std::apply(g_real.*Member, *reader.Get<Pack>());
	}
	static void VKAPI_CALL Wrap(VkCommandBuffer command, Args... args) {
		if (!t_defer) {
			(g_real.*Member)(command, args...);
			return;
		}
		Emit(&Replay, [&](auto& sink) { sink.Put(Pack {command, args...}); });
	}
};

// ------------------------------------------------------------------------------------------------
// Commands with arrays or nested structures.

struct BarrierHead {
	VkCommandBuffer      command;
	VkPipelineStageFlags src, dst;
	VkDependencyFlags    flags;
	uint32_t             memory, buffer, image;
};

void ReplayPipelineBarrier(std::byte* payload) {
	Reader r {payload};
	auto*  h = r.Get<BarrierHead>();
	auto*  m = r.GetArray<VkMemoryBarrier>(h->memory);
	auto*  b = r.GetArray<VkBufferMemoryBarrier>(h->buffer);
	auto*  i = r.GetArray<VkImageMemoryBarrier>(h->image);
	g_real.vkCmdPipelineBarrier(h->command, h->src, h->dst, h->flags, h->memory, m, h->buffer, b,
	                            h->image, i);
}

void VKAPI_CALL WrapPipelineBarrier(VkCommandBuffer command, VkPipelineStageFlags src,
                                    VkPipelineStageFlags dst, VkDependencyFlags flags, uint32_t nm,
                                    const VkMemoryBarrier* pm, uint32_t nb,
                                    const VkBufferMemoryBarrier* pb, uint32_t ni,
                                    const VkImageMemoryBarrier* pi) {
	if (!t_defer || AnyNext(pm, nm) || AnyNext(pb, nb) || AnyNext(pi, ni)) {
		BeforeDirectCall();
		g_real.vkCmdPipelineBarrier(command, src, dst, flags, nm, pm, nb, pb, ni, pi);
		return;
	}
	Emit(&ReplayPipelineBarrier, [&](auto& s) {
		s.Put(BarrierHead {command, src, dst, flags, nm, nb, ni});
		s.PutArray(pm, nm);
		s.PutArray(pb, nb);
		s.PutArray(pi, ni);
	});
}

struct DependencyHead {
	VkCommandBuffer  command;
	VkDependencyInfo info;
};

void ReplayPipelineBarrier2(std::byte* payload) {
	Reader r {payload};
	auto*  h                         = r.Get<DependencyHead>();
	h->info.pMemoryBarriers          = r.GetArray<VkMemoryBarrier2>(h->info.memoryBarrierCount);
	h->info.pBufferMemoryBarriers    = r.GetArray<VkBufferMemoryBarrier2>(h->info.bufferMemoryBarrierCount);
	h->info.pImageMemoryBarriers     = r.GetArray<VkImageMemoryBarrier2>(h->info.imageMemoryBarrierCount);
	g_real.vkCmdPipelineBarrier2(h->command, &h->info);
}

void VKAPI_CALL WrapPipelineBarrier2(VkCommandBuffer command, const VkDependencyInfo* info) {
	if (!t_defer || info->pNext != nullptr ||
	    AnyNext(info->pMemoryBarriers, info->memoryBarrierCount) ||
	    AnyNext(info->pBufferMemoryBarriers, info->bufferMemoryBarrierCount) ||
	    AnyNext(info->pImageMemoryBarriers, info->imageMemoryBarrierCount)) {
		BeforeDirectCall();
		g_real.vkCmdPipelineBarrier2(command, info);
		return;
	}
	Emit(&ReplayPipelineBarrier2, [&](auto& s) {
		s.Put(DependencyHead {command, *info});
		s.PutArray(info->pMemoryBarriers, info->memoryBarrierCount);
		s.PutArray(info->pBufferMemoryBarriers, info->bufferMemoryBarrierCount);
		s.PutArray(info->pImageMemoryBarriers, info->imageMemoryBarrierCount);
	});
}

struct PushDescriptorHead {
	VkCommandBuffer     command;
	VkPipelineBindPoint bind_point;
	VkPipelineLayout    layout;
	uint32_t            set;
	uint32_t            count;
};

void ReplayPushDescriptorSet(std::byte* payload) {
	Reader r {payload};
	auto*  h      = r.Get<PushDescriptorHead>();
	auto*  writes = r.GetArray<VkWriteDescriptorSet>(h->count);
	for (uint32_t i = 0; i < h->count; i++) {
		auto&      w = writes[i];
		const auto n = w.descriptorCount;
		const auto images  = w.pImageInfo != nullptr ? n : 0u;
		const auto buffers = w.pBufferInfo != nullptr ? n : 0u;
		const auto views   = w.pTexelBufferView != nullptr ? n : 0u;
		w.pImageInfo       = r.GetArray<VkDescriptorImageInfo>(images);
		w.pBufferInfo      = r.GetArray<VkDescriptorBufferInfo>(buffers);
		w.pTexelBufferView = r.GetArray<VkBufferView>(views);
	}
	g_real.vkCmdPushDescriptorSetKHR(h->command, h->bind_point, h->layout, h->set, h->count, writes);
}

void VKAPI_CALL WrapPushDescriptorSet(VkCommandBuffer command, VkPipelineBindPoint bind_point,
                                      VkPipelineLayout layout, uint32_t set, uint32_t count,
                                      const VkWriteDescriptorSet* writes) {
	if (!t_defer || AnyNext(writes, count)) {
		BeforeDirectCall();
		g_real.vkCmdPushDescriptorSetKHR(command, bind_point, layout, set, count, writes);
		return;
	}
	Emit(&ReplayPushDescriptorSet, [&](auto& s) {
		s.Put(PushDescriptorHead {command, bind_point, layout, set, count});
		s.PutArray(writes, count);
		for (uint32_t i = 0; i < count; i++) {
			const auto& w = writes[i];
			const auto  n = w.descriptorCount;
			s.PutArray(w.pImageInfo, w.pImageInfo != nullptr ? n : 0u);
			s.PutArray(w.pBufferInfo, w.pBufferInfo != nullptr ? n : 0u);
			s.PutArray(w.pTexelBufferView, w.pTexelBufferView != nullptr ? n : 0u);
		}
	});
}

struct RenderingHead {
	VkCommandBuffer command;
	VkRenderingInfo info;
	uint32_t        has_depth;
	uint32_t        has_stencil;
};

void ReplayBeginRendering(std::byte* payload) {
	Reader r {payload};
	auto*  h                = r.Get<RenderingHead>();
	h->info.pColorAttachments = r.GetArray<VkRenderingAttachmentInfo>(h->info.colorAttachmentCount);
	h->info.pDepthAttachment   = h->has_depth != 0 ? r.Get<VkRenderingAttachmentInfo>() : nullptr;
	h->info.pStencilAttachment = h->has_stencil != 0 ? r.Get<VkRenderingAttachmentInfo>() : nullptr;
	g_real.vkCmdBeginRendering(h->command, &h->info);
}

void VKAPI_CALL WrapBeginRendering(VkCommandBuffer command, const VkRenderingInfo* info) {
	const bool unknown =
	    info->pNext != nullptr || AnyNext(info->pColorAttachments, info->colorAttachmentCount) ||
	    (info->pDepthAttachment != nullptr && info->pDepthAttachment->pNext != nullptr) ||
	    (info->pStencilAttachment != nullptr && info->pStencilAttachment->pNext != nullptr);
	if (!t_defer || unknown) {
		BeforeDirectCall();
		g_real.vkCmdBeginRendering(command, info);
		return;
	}
	Emit(&ReplayBeginRendering, [&](auto& s) {
		s.Put(RenderingHead {command, *info, info->pDepthAttachment != nullptr ? 1u : 0u,
		                     info->pStencilAttachment != nullptr ? 1u : 0u});
		s.PutArray(info->pColorAttachments, info->colorAttachmentCount);
		if (info->pDepthAttachment != nullptr) {
			s.Put(*info->pDepthAttachment);
		}
		if (info->pStencilAttachment != nullptr) {
			s.Put(*info->pStencilAttachment);
		}
	});
}

struct CopyBufferHead {
	VkCommandBuffer command;
	VkBuffer        src, dst;
	uint32_t        count;
};
void ReplayCopyBuffer(std::byte* payload) {
	Reader r {payload};
	auto*  h = r.Get<CopyBufferHead>();
	g_real.vkCmdCopyBuffer(h->command, h->src, h->dst, h->count, r.GetArray<VkBufferCopy>(h->count));
}
void VKAPI_CALL WrapCopyBuffer(VkCommandBuffer command, VkBuffer src, VkBuffer dst, uint32_t count,
                               const VkBufferCopy* regions) {
	if (!t_defer) {
		g_real.vkCmdCopyBuffer(command, src, dst, count, regions);
		return;
	}
	Emit(&ReplayCopyBuffer, [&](auto& s) {
		s.Put(CopyBufferHead {command, src, dst, count});
		s.PutArray(regions, count);
	});
}

struct CopyImageHead {
	VkCommandBuffer command;
	VkImage         src;
	VkImageLayout   src_layout;
	VkImage         dst;
	VkImageLayout   dst_layout;
	uint32_t        count;
	VkFilter        filter; // blit only
};
void ReplayCopyImage(std::byte* payload) {
	Reader r {payload};
	auto*  h = r.Get<CopyImageHead>();
	g_real.vkCmdCopyImage(h->command, h->src, h->src_layout, h->dst, h->dst_layout, h->count,
	                      r.GetArray<VkImageCopy>(h->count));
}
void VKAPI_CALL WrapCopyImage(VkCommandBuffer command, VkImage src, VkImageLayout src_layout,
                              VkImage dst, VkImageLayout dst_layout, uint32_t count,
                              const VkImageCopy* regions) {
	if (!t_defer) {
		g_real.vkCmdCopyImage(command, src, src_layout, dst, dst_layout, count, regions);
		return;
	}
	Emit(&ReplayCopyImage, [&](auto& s) {
		s.Put(CopyImageHead {command, src, src_layout, dst, dst_layout, count, VK_FILTER_NEAREST});
		s.PutArray(regions, count);
	});
}
void ReplayBlitImage(std::byte* payload) {
	Reader r {payload};
	auto*  h = r.Get<CopyImageHead>();
	g_real.vkCmdBlitImage(h->command, h->src, h->src_layout, h->dst, h->dst_layout, h->count,
	                      r.GetArray<VkImageBlit>(h->count), h->filter);
}
void VKAPI_CALL WrapBlitImage(VkCommandBuffer command, VkImage src, VkImageLayout src_layout,
                              VkImage dst, VkImageLayout dst_layout, uint32_t count,
                              const VkImageBlit* regions, VkFilter filter) {
	if (!t_defer) {
		g_real.vkCmdBlitImage(command, src, src_layout, dst, dst_layout, count, regions, filter);
		return;
	}
	Emit(&ReplayBlitImage, [&](auto& s) {
		s.Put(CopyImageHead {command, src, src_layout, dst, dst_layout, count, filter});
		s.PutArray(regions, count);
	});
}
void ReplayResolveImage(std::byte* payload) {
	Reader r {payload};
	auto*  h = r.Get<CopyImageHead>();
	g_real.vkCmdResolveImage(h->command, h->src, h->src_layout, h->dst, h->dst_layout, h->count,
	                         r.GetArray<VkImageResolve>(h->count));
}
void VKAPI_CALL WrapResolveImage(VkCommandBuffer command, VkImage src, VkImageLayout src_layout,
                                 VkImage dst, VkImageLayout dst_layout, uint32_t count,
                                 const VkImageResolve* regions) {
	if (!t_defer) {
		g_real.vkCmdResolveImage(command, src, src_layout, dst, dst_layout, count, regions);
		return;
	}
	Emit(&ReplayResolveImage, [&](auto& s) {
		s.Put(CopyImageHead {command, src, src_layout, dst, dst_layout, count, VK_FILTER_NEAREST});
		s.PutArray(regions, count);
	});
}

struct BufferImageHead {
	VkCommandBuffer command;
	VkBuffer        buffer;
	VkImage         image;
	VkImageLayout   layout;
	uint32_t        count;
};
void ReplayCopyBufferToImage(std::byte* payload) {
	Reader r {payload};
	auto*  h = r.Get<BufferImageHead>();
	g_real.vkCmdCopyBufferToImage(h->command, h->buffer, h->image, h->layout, h->count,
	                              r.GetArray<VkBufferImageCopy>(h->count));
}
void VKAPI_CALL WrapCopyBufferToImage(VkCommandBuffer command, VkBuffer buffer, VkImage image,
                                      VkImageLayout layout, uint32_t count,
                                      const VkBufferImageCopy* regions) {
	if (!t_defer) {
		g_real.vkCmdCopyBufferToImage(command, buffer, image, layout, count, regions);
		return;
	}
	Emit(&ReplayCopyBufferToImage, [&](auto& s) {
		s.Put(BufferImageHead {command, buffer, image, layout, count});
		s.PutArray(regions, count);
	});
}
void ReplayCopyImageToBuffer(std::byte* payload) {
	Reader r {payload};
	auto*  h = r.Get<BufferImageHead>();
	g_real.vkCmdCopyImageToBuffer(h->command, h->image, h->layout, h->buffer, h->count,
	                              r.GetArray<VkBufferImageCopy>(h->count));
}
void VKAPI_CALL WrapCopyImageToBuffer(VkCommandBuffer command, VkImage image, VkImageLayout layout,
                                      VkBuffer buffer, uint32_t count,
                                      const VkBufferImageCopy* regions) {
	if (!t_defer) {
		g_real.vkCmdCopyImageToBuffer(command, image, layout, buffer, count, regions);
		return;
	}
	Emit(&ReplayCopyImageToBuffer, [&](auto& s) {
		s.Put(BufferImageHead {command, buffer, image, layout, count});
		s.PutArray(regions, count);
	});
}

struct PushConstantsHead {
	VkCommandBuffer    command;
	VkPipelineLayout   layout;
	VkShaderStageFlags stages;
	uint32_t           offset, size;
};
void ReplayPushConstants(std::byte* payload) {
	Reader r {payload};
	auto*  h = r.Get<PushConstantsHead>();
	g_real.vkCmdPushConstants(h->command, h->layout, h->stages, h->offset, h->size,
	                          r.GetArray<std::byte>(h->size));
}
void VKAPI_CALL WrapPushConstants(VkCommandBuffer command, VkPipelineLayout layout,
                                  VkShaderStageFlags stages, uint32_t offset, uint32_t size,
                                  const void* values) {
	if (!t_defer) {
		g_real.vkCmdPushConstants(command, layout, stages, offset, size, values);
		return;
	}
	Emit(&ReplayPushConstants, [&](auto& s) {
		s.Put(PushConstantsHead {command, layout, stages, offset, size});
		s.PutArray(static_cast<const std::byte*>(values), size);
	});
}

struct ClearColorHead {
	VkCommandBuffer   command;
	VkImage           image;
	VkImageLayout     layout;
	VkClearColorValue color;
	uint32_t          count;
};
void ReplayClearColorImage(std::byte* payload) {
	Reader r {payload};
	auto*  h = r.Get<ClearColorHead>();
	g_real.vkCmdClearColorImage(h->command, h->image, h->layout, &h->color, h->count,
	                            r.GetArray<VkImageSubresourceRange>(h->count));
}
void VKAPI_CALL WrapClearColorImage(VkCommandBuffer command, VkImage image, VkImageLayout layout,
                                    const VkClearColorValue* color, uint32_t count,
                                    const VkImageSubresourceRange* ranges) {
	if (!t_defer) {
		g_real.vkCmdClearColorImage(command, image, layout, color, count, ranges);
		return;
	}
	Emit(&ReplayClearColorImage, [&](auto& s) {
		s.Put(ClearColorHead {command, image, layout, *color, count});
		s.PutArray(ranges, count);
	});
}

struct ClearDepthHead {
	VkCommandBuffer          command;
	VkImage                  image;
	VkImageLayout            layout;
	VkClearDepthStencilValue value;
	uint32_t                 count;
};
void ReplayClearDepthStencilImage(std::byte* payload) {
	Reader r {payload};
	auto*  h = r.Get<ClearDepthHead>();
	g_real.vkCmdClearDepthStencilImage(h->command, h->image, h->layout, &h->value, h->count,
	                                   r.GetArray<VkImageSubresourceRange>(h->count));
}
void VKAPI_CALL WrapClearDepthStencilImage(VkCommandBuffer command, VkImage image,
                                           VkImageLayout layout,
                                           const VkClearDepthStencilValue* value, uint32_t count,
                                           const VkImageSubresourceRange* ranges) {
	if (!t_defer) {
		g_real.vkCmdClearDepthStencilImage(command, image, layout, value, count, ranges);
		return;
	}
	Emit(&ReplayClearDepthStencilImage, [&](auto& s) {
		s.Put(ClearDepthHead {command, image, layout, *value, count});
		s.PutArray(ranges, count);
	});
}

struct RectsHead {
	VkCommandBuffer command;
	uint32_t        first, count;
};
void ReplaySetScissor(std::byte* payload) {
	Reader r {payload};
	auto*  h = r.Get<RectsHead>();
	g_real.vkCmdSetScissor(h->command, h->first, h->count, r.GetArray<VkRect2D>(h->count));
}
void VKAPI_CALL WrapSetScissor(VkCommandBuffer command, uint32_t first, uint32_t count,
                               const VkRect2D* rects) {
	if (!t_defer) {
		g_real.vkCmdSetScissor(command, first, count, rects);
		return;
	}
	Emit(&ReplaySetScissor, [&](auto& s) {
		s.Put(RectsHead {command, first, count});
		s.PutArray(rects, count);
	});
}
void ReplaySetScissorWithCount(std::byte* payload) {
	Reader r {payload};
	auto*  h = r.Get<RectsHead>();
	g_real.vkCmdSetScissorWithCount(h->command, h->count, r.GetArray<VkRect2D>(h->count));
}
void VKAPI_CALL WrapSetScissorWithCount(VkCommandBuffer command, uint32_t count,
                                        const VkRect2D* rects) {
	if (!t_defer) {
		g_real.vkCmdSetScissorWithCount(command, count, rects);
		return;
	}
	Emit(&ReplaySetScissorWithCount, [&](auto& s) {
		s.Put(RectsHead {command, 0, count});
		s.PutArray(rects, count);
	});
}
void ReplaySetViewport(std::byte* payload) {
	Reader r {payload};
	auto*  h = r.Get<RectsHead>();
	g_real.vkCmdSetViewport(h->command, h->first, h->count, r.GetArray<VkViewport>(h->count));
}
void VKAPI_CALL WrapSetViewport(VkCommandBuffer command, uint32_t first, uint32_t count,
                                const VkViewport* viewports) {
	if (!t_defer) {
		g_real.vkCmdSetViewport(command, first, count, viewports);
		return;
	}
	Emit(&ReplaySetViewport, [&](auto& s) {
		s.Put(RectsHead {command, first, count});
		s.PutArray(viewports, count);
	});
}
void ReplaySetViewportWithCount(std::byte* payload) {
	Reader r {payload};
	auto*  h = r.Get<RectsHead>();
	g_real.vkCmdSetViewportWithCount(h->command, h->count, r.GetArray<VkViewport>(h->count));
}
void VKAPI_CALL WrapSetViewportWithCount(VkCommandBuffer command, uint32_t count,
                                         const VkViewport* viewports) {
	if (!t_defer) {
		g_real.vkCmdSetViewportWithCount(command, count, viewports);
		return;
	}
	Emit(&ReplaySetViewportWithCount, [&](auto& s) {
		s.Put(RectsHead {command, 0, count});
		s.PutArray(viewports, count);
	});
}
void ReplaySetColorWriteEnable(std::byte* payload) {
	Reader r {payload};
	auto*  h = r.Get<RectsHead>();
	g_real.vkCmdSetColorWriteEnableEXT(h->command, h->count, r.GetArray<VkBool32>(h->count));
}
void VKAPI_CALL WrapSetColorWriteEnable(VkCommandBuffer command, uint32_t count,
                                        const VkBool32* enables) {
	if (!t_defer) {
		g_real.vkCmdSetColorWriteEnableEXT(command, count, enables);
		return;
	}
	Emit(&ReplaySetColorWriteEnable, [&](auto& s) {
		s.Put(RectsHead {command, 0, count});
		s.PutArray(enables, count);
	});
}

struct BlendConstantsHead {
	VkCommandBuffer command;
	float           values[4];
};
void ReplaySetBlendConstants(std::byte* payload) {
	Reader r {payload};
	auto*  h = r.Get<BlendConstantsHead>();
	g_real.vkCmdSetBlendConstants(h->command, h->values);
}
void VKAPI_CALL WrapSetBlendConstants(VkCommandBuffer command, const float values[4]) {
	if (!t_defer) {
		g_real.vkCmdSetBlendConstants(command, values);
		return;
	}
	BlendConstantsHead head {command, {values[0], values[1], values[2], values[3]}};
	Emit(&ReplaySetBlendConstants, [&](auto& s) { s.Put(head); });
}

struct DescriptorSetsHead {
	VkCommandBuffer     command;
	VkPipelineBindPoint bind_point;
	VkPipelineLayout    layout;
	uint32_t            first, count, dynamic_count;
};
void ReplayBindDescriptorSets(std::byte* payload) {
	Reader r {payload};
	auto*  h       = r.Get<DescriptorSetsHead>();
	auto*  sets    = r.GetArray<VkDescriptorSet>(h->count);
	auto*  offsets = r.GetArray<uint32_t>(h->dynamic_count);
	g_real.vkCmdBindDescriptorSets(h->command, h->bind_point, h->layout, h->first, h->count, sets,
	                               h->dynamic_count, offsets);
}
void VKAPI_CALL WrapBindDescriptorSets(VkCommandBuffer command, VkPipelineBindPoint bind_point,
                                       VkPipelineLayout layout, uint32_t first, uint32_t count,
                                       const VkDescriptorSet* sets, uint32_t dynamic_count,
                                       const uint32_t* offsets) {
	if (!t_defer) {
		g_real.vkCmdBindDescriptorSets(command, bind_point, layout, first, count, sets,
		                               dynamic_count, offsets);
		return;
	}
	Emit(&ReplayBindDescriptorSets, [&](auto& s) {
		s.Put(DescriptorSetsHead {command, bind_point, layout, first, count, dynamic_count});
		s.PutArray(sets, count);
		s.PutArray(offsets, dynamic_count);
	});
}

struct VertexBuffersHead {
	VkCommandBuffer command;
	uint32_t        first, count;
	uint32_t        has_sizes, has_strides;
};
void ReplayBindVertexBuffers2(std::byte* payload) {
	Reader r {payload};
	auto*  h       = r.Get<VertexBuffersHead>();
	auto*  buffers = r.GetArray<VkBuffer>(h->count);
	auto*  offsets = r.GetArray<VkDeviceSize>(h->count);
	auto*  sizes   = r.GetArray<VkDeviceSize>(h->has_sizes != 0 ? h->count : 0u);
	auto*  strides = r.GetArray<VkDeviceSize>(h->has_strides != 0 ? h->count : 0u);
	g_real.vkCmdBindVertexBuffers2(h->command, h->first, h->count, buffers, offsets, sizes, strides);
}
void VKAPI_CALL WrapBindVertexBuffers2(VkCommandBuffer command, uint32_t first, uint32_t count,
                                       const VkBuffer* buffers, const VkDeviceSize* offsets,
                                       const VkDeviceSize* sizes, const VkDeviceSize* strides) {
	if (!t_defer) {
		g_real.vkCmdBindVertexBuffers2(command, first, count, buffers, offsets, sizes, strides);
		return;
	}
	Emit(&ReplayBindVertexBuffers2, [&](auto& s) {
		s.Put(VertexBuffersHead {command, first, count, sizes != nullptr ? 1u : 0u,
		                         strides != nullptr ? 1u : 0u});
		s.PutArray(buffers, count);
		s.PutArray(offsets, count);
		s.PutArray(sizes, sizes != nullptr ? count : 0u);
		s.PutArray(strides, strides != nullptr ? count : 0u);
	});
}

// ------------------------------------------------------------------------------------------------
// Command buffer lifecycle and submission. Results are checked on the recorder thread.

void Fail(const char* what, VkResult result) {
	std::printf("[VkRecorder] %s failed: %s (%d)\n", what,
	            vk::to_string(static_cast<vk::Result>(result)).c_str(), static_cast<int>(result));
	std::fflush(stdout);
	EXIT("%s failed on the Vulkan recorder thread\n", what);
}

struct BeginHead {
	VkCommandBuffer           command;
	VkCommandBufferUsageFlags flags;
};
void ReplayBegin(std::byte* payload) {
	Reader                   r {payload};
	auto*                    h = r.Get<BeginHead>();
	VkCommandBufferBeginInfo info {};
	info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	info.flags = h->flags;
	if (const auto result = g_real.vkBeginCommandBuffer(h->command, &info); result != VK_SUCCESS) {
		Fail("vkBeginCommandBuffer", result);
	}
}
VkResult VKAPI_CALL WrapBegin(VkCommandBuffer command, const VkCommandBufferBeginInfo* info) {
	if (!t_defer || info->pNext != nullptr || info->pInheritanceInfo != nullptr) {
		BeforeDirectCall();
		return g_real.vkBeginCommandBuffer(command, info);
	}
	Emit(&ReplayBegin, [&](auto& s) { s.Put(BeginHead {command, info->flags}); });
	return VK_SUCCESS;
}

void ReplayEnd(std::byte* payload) {
	Reader r {payload};
	auto*  command = r.Get<VkCommandBuffer>();
	if (const auto result = g_real.vkEndCommandBuffer(*command); result != VK_SUCCESS) {
		Fail("vkEndCommandBuffer", result);
	}
}
VkResult VKAPI_CALL WrapEnd(VkCommandBuffer command) {
	if (!t_defer) {
		return g_real.vkEndCommandBuffer(command);
	}
	Emit(&ReplayEnd, [&](auto& s) { s.Put(command); });
	return VK_SUCCESS;
}

struct SubmitHead {
	VkQueue  queue;
	VkFence  fence;
	uint32_t count;
};
struct SubmitEntry {
	VkSubmitInfo                  info;
	uint32_t                      has_timeline;
	VkTimelineSemaphoreSubmitInfo timeline;
};

const VkTimelineSemaphoreSubmitInfo* TimelineOf(const VkSubmitInfo& info, bool& supported) {
	supported = true;
	if (info.pNext == nullptr) {
		return nullptr;
	}
	const auto* next = static_cast<const VkBaseInStructure*>(info.pNext);
	if (next->sType != VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO || next->pNext != nullptr) {
		supported = false;
		return nullptr;
	}
	return reinterpret_cast<const VkTimelineSemaphoreSubmitInfo*>(next);
}

void ReplaySubmit(std::byte* payload) {
	Reader r {payload};
	auto*  h = r.Get<SubmitHead>();
	auto*  entries = r.GetArray<SubmitEntry>(h->count);
	std::array<VkSubmitInfo, 8> infos {};
	EXIT_IF(h->count > infos.size());
	for (uint32_t i = 0; i < h->count; i++) {
		auto& e                      = entries[i];
		auto& info                   = e.info;
		info.pWaitSemaphores         = r.GetArray<VkSemaphore>(info.waitSemaphoreCount);
		info.pWaitDstStageMask       = r.GetArray<VkPipelineStageFlags>(info.waitSemaphoreCount);
		info.pCommandBuffers         = r.GetArray<VkCommandBuffer>(info.commandBufferCount);
		info.pSignalSemaphores       = r.GetArray<VkSemaphore>(info.signalSemaphoreCount);
		info.pNext                   = nullptr;
		if (e.has_timeline != 0) {
			e.timeline.pWaitSemaphoreValues = r.GetArray<uint64_t>(e.timeline.waitSemaphoreValueCount);
			e.timeline.pSignalSemaphoreValues =
			    r.GetArray<uint64_t>(e.timeline.signalSemaphoreValueCount);
			e.timeline.pNext = nullptr;
			info.pNext       = &e.timeline;
		}
		infos[i] = info;
	}
	std::lock_guard lock(g_queue_mutex);
	if (const auto result = g_real.vkQueueSubmit(h->queue, h->count, infos.data(), h->fence);
	    result != VK_SUCCESS) {
		Fail("vkQueueSubmit", result);
	}
}

VkResult VKAPI_CALL WrapQueueSubmit(VkQueue queue, uint32_t count, const VkSubmitInfo* submits,
                                    VkFence fence) {
	bool supported = count <= 8;
	for (uint32_t i = 0; i < count && supported; i++) {
		(void)TimelineOf(submits[i], supported);
	}
	if (!t_defer || !supported) {
		// Work this thread submits may wait for the guest GPU thread's queued submissions.
		Drain();
		std::lock_guard lock(g_queue_mutex);
		return g_real.vkQueueSubmit(queue, count, submits, fence);
	}
	Emit(&ReplaySubmit, [&](auto& s) {
		s.Put(SubmitHead {queue, fence, count});
		for (uint32_t i = 0; i < count; i++) {
			bool        unused   = true;
			const auto* timeline = TimelineOf(submits[i], unused);
			SubmitEntry entry {submits[i], timeline != nullptr ? 1u : 0u, {}};
			if (timeline != nullptr) {
				entry.timeline = *timeline;
			}
			s.Put(entry);
		}
		for (uint32_t i = 0; i < count; i++) {
			const auto& info = submits[i];
			s.PutArray(info.pWaitSemaphores, info.waitSemaphoreCount);
			s.PutArray(info.pWaitDstStageMask, info.waitSemaphoreCount);
			s.PutArray(info.pCommandBuffers, info.commandBufferCount);
			s.PutArray(info.pSignalSemaphores, info.signalSemaphoreCount);
			bool        unused   = true;
			const auto* timeline = TimelineOf(info, unused);
			if (timeline != nullptr) {
				s.PutArray(timeline->pWaitSemaphoreValues, timeline->waitSemaphoreValueCount);
				s.PutArray(timeline->pSignalSemaphoreValues, timeline->signalSemaphoreValueCount);
			}
		}
	});
	return VK_SUCCESS;
}

VkResult VKAPI_CALL WrapQueuePresent(VkQueue queue, const VkPresentInfoKHR* info) {
	Drain();
	std::lock_guard lock(g_queue_mutex);
	return g_real.vkQueuePresentKHR(queue, info);
}

VkResult VKAPI_CALL WrapQueueWaitIdle(VkQueue queue) {
	Drain();
	std::lock_guard lock(g_queue_mutex);
	return g_real.vkQueueWaitIdle(queue);
}

VkResult VKAPI_CALL WrapDeviceWaitIdle(VkDevice device) {
	Drain();
	std::lock_guard lock(g_queue_mutex);
	return g_real.vkDeviceWaitIdle(device);
}

// Pool operations need the pool to be idle on the recorder thread.
VkResult VKAPI_CALL WrapAllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* info,
                                               VkCommandBuffer* buffers) {
	Drain();
	return g_real.vkAllocateCommandBuffers(device, info, buffers);
}

void VKAPI_CALL WrapFreeCommandBuffers(VkDevice device, VkCommandPool pool, uint32_t count,
                                       const VkCommandBuffer* buffers) {
	Drain();
	g_real.vkFreeCommandBuffers(device, pool, count, buffers);
}

void VKAPI_CALL WrapDestroyCommandPool(VkDevice device, VkCommandPool pool,
                                       const VkAllocationCallbacks* allocator) {
	Drain();
	g_real.vkDestroyCommandPool(device, pool, allocator);
}

} // namespace

bool Enabled() {
	return g_enabled;
}

void EnableForThisThread() {
	t_defer = g_enabled;
}

void Drain() {
	if (!g_enabled) {
		return;
	}
	auto&          ring   = *g_ring;
	const uint64_t target = ring.head.load(std::memory_order_acquire);
	uint32_t       spins  = 0;
	while (ring.tail.load(std::memory_order_acquire) < target) {
		WakeRecorder();
		Pause(++spins);
	}
}

void Install(GraphicContext& graphics) {
	(void)graphics;
	if (const char* env = std::getenv("KYTY_VK_THREAD"); env != nullptr && env[0] == '0') {
		return;
	}
	auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	// Every function the guest GPU thread records with must be deferred, or the recorder could run
	// behind a direct call into the same command buffer. Refuse to install if any is missing.
	bool complete = true;
#define KYTY_HOOK(NAME, ...)                                                                     \
	g_real.NAME = d.NAME;                                                                          \
	complete &= g_real.NAME != nullptr || std::string_view(#NAME).ends_with("NV") ||               \
	            std::string_view(#NAME).ends_with("EXT")
	KYTY_HOOK(vkCmdPipelineBarrier, WrapPipelineBarrier);
	KYTY_HOOK(vkCmdPipelineBarrier2, WrapPipelineBarrier2);
	KYTY_HOOK(vkCmdBindPipeline, 0);
	KYTY_HOOK(vkCmdPushDescriptorSetKHR, WrapPushDescriptorSet);
	KYTY_HOOK(vkCmdBeginRendering, WrapBeginRendering);
	KYTY_HOOK(vkCmdCopyImage, WrapCopyImage);
	KYTY_HOOK(vkCmdDispatch, 0);
	KYTY_HOOK(vkCmdEndRendering, 0);
	KYTY_HOOK(vkCmdPushConstants, WrapPushConstants);
	KYTY_HOOK(vkCmdCopyBuffer, WrapCopyBuffer);
	KYTY_HOOK(vkCmdDraw, 0);
	KYTY_HOOK(vkCmdClearColorImage, WrapClearColorImage);
	KYTY_HOOK(vkCmdCopyBufferToImage, WrapCopyBufferToImage);
	KYTY_HOOK(vkCmdCopyImageToBuffer, WrapCopyImageToBuffer);
	KYTY_HOOK(vkCmdDrawIndexed, 0);
	KYTY_HOOK(vkCmdFillBuffer, 0);
	KYTY_HOOK(vkCmdSetScissor, WrapSetScissor);
	KYTY_HOOK(vkCmdSetViewport, WrapSetViewport);
	KYTY_HOOK(vkCmdBindDescriptorSets, WrapBindDescriptorSets);
	KYTY_HOOK(vkCmdBindIndexBuffer, 0);
	KYTY_HOOK(vkCmdBindVertexBuffers2, WrapBindVertexBuffers2);
	KYTY_HOOK(vkCmdBlitImage, WrapBlitImage);
	KYTY_HOOK(vkCmdClearDepthStencilImage, WrapClearDepthStencilImage);
	KYTY_HOOK(vkCmdDispatchIndirect, 0);
	KYTY_HOOK(vkCmdDrawIndirect, 0);
	KYTY_HOOK(vkCmdDrawMeshTasksEXT, 0);
	KYTY_HOOK(vkCmdResolveImage, WrapResolveImage);
	KYTY_HOOK(vkCmdSetAttachmentFeedbackLoopEnableEXT, 0);
	KYTY_HOOK(vkCmdSetBlendConstants, WrapSetBlendConstants);
	KYTY_HOOK(vkCmdSetCheckpointNV, 0);
	KYTY_HOOK(vkCmdSetColorWriteEnableEXT, WrapSetColorWriteEnable);
	KYTY_HOOK(vkCmdSetDepthBias, 0);
	KYTY_HOOK(vkCmdSetDepthBiasEnable, 0);
	KYTY_HOOK(vkCmdSetDepthBounds, 0);
	KYTY_HOOK(vkCmdSetDepthBoundsTestEnable, 0);
	KYTY_HOOK(vkCmdSetDepthCompareOp, 0);
	KYTY_HOOK(vkCmdSetDepthTestEnable, 0);
	KYTY_HOOK(vkCmdSetDepthWriteEnable, 0);
	KYTY_HOOK(vkCmdSetLineWidth, 0);
	KYTY_HOOK(vkCmdSetScissorWithCount, WrapSetScissorWithCount);
	KYTY_HOOK(vkCmdSetStencilCompareMask, 0);
	KYTY_HOOK(vkCmdSetStencilOp, 0);
	KYTY_HOOK(vkCmdSetStencilReference, 0);
	KYTY_HOOK(vkCmdSetStencilTestEnable, 0);
	KYTY_HOOK(vkCmdSetStencilWriteMask, 0);
	KYTY_HOOK(vkCmdSetViewportWithCount, WrapSetViewportWithCount);
	KYTY_HOOK(vkBeginCommandBuffer, WrapBegin);
	KYTY_HOOK(vkEndCommandBuffer, WrapEnd);
	KYTY_HOOK(vkQueueSubmit, WrapQueueSubmit);
	KYTY_HOOK(vkQueuePresentKHR, WrapQueuePresent);
	KYTY_HOOK(vkQueueWaitIdle, WrapQueueWaitIdle);
	KYTY_HOOK(vkDeviceWaitIdle, WrapDeviceWaitIdle);
	KYTY_HOOK(vkAllocateCommandBuffers, WrapAllocateCommandBuffers);
	KYTY_HOOK(vkFreeCommandBuffers, WrapFreeCommandBuffers);
	KYTY_HOOK(vkDestroyCommandPool, WrapDestroyCommandPool);
#undef KYTY_HOOK
	if (!complete) {
		std::printf("Vulkan recorder thread: disabled (driver entry point missing)\n");
		return;
	}

	g_ring       = new Ring;
	g_ring->data = static_cast<std::byte*>(::operator new(kRingSize, std::align_val_t {64}));
	std::thread(RecorderThread).detach();

	// Redirect the dispatcher only after the recorder is ready.
	const auto redirect = [](auto& slot, auto wrapper) {
		if (slot != nullptr) {
			slot = wrapper;
		}
	};
#define KYTY_SCALAR(NAME, ...) redirect(d.NAME, &ScalarCmd<&Real::NAME __VA_OPT__(, ) __VA_ARGS__>::Wrap)
	KYTY_SCALAR(vkCmdBindPipeline, VkPipelineBindPoint, VkPipeline);
	KYTY_SCALAR(vkCmdDispatch, uint32_t, uint32_t, uint32_t);
	KYTY_SCALAR(vkCmdEndRendering);
	KYTY_SCALAR(vkCmdDraw, uint32_t, uint32_t, uint32_t, uint32_t);
	KYTY_SCALAR(vkCmdDrawIndexed, uint32_t, uint32_t, uint32_t, int32_t, uint32_t);
	KYTY_SCALAR(vkCmdFillBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, uint32_t);
	KYTY_SCALAR(vkCmdBindIndexBuffer, VkBuffer, VkDeviceSize, VkIndexType);
	KYTY_SCALAR(vkCmdDispatchIndirect, VkBuffer, VkDeviceSize);
	KYTY_SCALAR(vkCmdDrawIndirect, VkBuffer, VkDeviceSize, uint32_t, uint32_t);
	KYTY_SCALAR(vkCmdDrawMeshTasksEXT, uint32_t, uint32_t, uint32_t);
	KYTY_SCALAR(vkCmdSetAttachmentFeedbackLoopEnableEXT, VkImageAspectFlags);
	KYTY_SCALAR(vkCmdSetCheckpointNV, const void*);
	KYTY_SCALAR(vkCmdSetDepthBias, float, float, float);
	KYTY_SCALAR(vkCmdSetDepthBiasEnable, VkBool32);
	KYTY_SCALAR(vkCmdSetDepthBounds, float, float);
	KYTY_SCALAR(vkCmdSetDepthBoundsTestEnable, VkBool32);
	KYTY_SCALAR(vkCmdSetDepthCompareOp, VkCompareOp);
	KYTY_SCALAR(vkCmdSetDepthTestEnable, VkBool32);
	KYTY_SCALAR(vkCmdSetDepthWriteEnable, VkBool32);
	KYTY_SCALAR(vkCmdSetLineWidth, float);
	KYTY_SCALAR(vkCmdSetStencilCompareMask, VkStencilFaceFlags, uint32_t);
	KYTY_SCALAR(vkCmdSetStencilOp, VkStencilFaceFlags, VkStencilOp, VkStencilOp, VkStencilOp,
	            VkCompareOp);
	KYTY_SCALAR(vkCmdSetStencilReference, VkStencilFaceFlags, uint32_t);
	KYTY_SCALAR(vkCmdSetStencilTestEnable, VkBool32);
	KYTY_SCALAR(vkCmdSetStencilWriteMask, VkStencilFaceFlags, uint32_t);
#undef KYTY_SCALAR
	redirect(d.vkCmdPipelineBarrier, &WrapPipelineBarrier);
	redirect(d.vkCmdPipelineBarrier2, &WrapPipelineBarrier2);
	redirect(d.vkCmdPushDescriptorSetKHR, &WrapPushDescriptorSet);
	redirect(d.vkCmdBeginRendering, &WrapBeginRendering);
	redirect(d.vkCmdCopyImage, &WrapCopyImage);
	redirect(d.vkCmdPushConstants, &WrapPushConstants);
	redirect(d.vkCmdCopyBuffer, &WrapCopyBuffer);
	redirect(d.vkCmdClearColorImage, &WrapClearColorImage);
	redirect(d.vkCmdCopyBufferToImage, &WrapCopyBufferToImage);
	redirect(d.vkCmdCopyImageToBuffer, &WrapCopyImageToBuffer);
	redirect(d.vkCmdSetScissor, &WrapSetScissor);
	redirect(d.vkCmdSetViewport, &WrapSetViewport);
	redirect(d.vkCmdBindDescriptorSets, &WrapBindDescriptorSets);
	redirect(d.vkCmdBindVertexBuffers2, &WrapBindVertexBuffers2);
	redirect(d.vkCmdBlitImage, &WrapBlitImage);
	redirect(d.vkCmdClearDepthStencilImage, &WrapClearDepthStencilImage);
	redirect(d.vkCmdResolveImage, &WrapResolveImage);
	redirect(d.vkCmdSetBlendConstants, &WrapSetBlendConstants);
	redirect(d.vkCmdSetColorWriteEnableEXT, &WrapSetColorWriteEnable);
	redirect(d.vkCmdSetScissorWithCount, &WrapSetScissorWithCount);
	redirect(d.vkCmdSetViewportWithCount, &WrapSetViewportWithCount);
	redirect(d.vkBeginCommandBuffer, &WrapBegin);
	redirect(d.vkEndCommandBuffer, &WrapEnd);
	redirect(d.vkQueueSubmit, &WrapQueueSubmit);
	redirect(d.vkQueuePresentKHR, &WrapQueuePresent);
	redirect(d.vkQueueWaitIdle, &WrapQueueWaitIdle);
	redirect(d.vkDeviceWaitIdle, &WrapDeviceWaitIdle);
	redirect(d.vkAllocateCommandBuffers, &WrapAllocateCommandBuffers);
	redirect(d.vkFreeCommandBuffers, &WrapFreeCommandBuffers);
	redirect(d.vkDestroyCommandPool, &WrapDestroyCommandPool);
	g_enabled = true;
	std::printf("Vulkan recorder thread: enabled\n");
}

} // namespace Libs::Graphics::VkRecorder
