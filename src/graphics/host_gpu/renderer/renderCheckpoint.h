#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCHECKPOINT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCHECKPOINT_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>
#include <cstdio>

namespace Libs::Graphics::Checkpoint {

// KYTY_GPU_CHECKPOINTS=1 (NVIDIA only, VK_NV_device_diagnostic_checkpoints): every draw and dispatch
// leaves a numbered marker in the command buffer, and a ring remembers what each number was. After a
// device loss the driver reports the last markers the GPU reached, which names the faulting shader.
struct Record {
	uint64_t id          = 0;
	uint32_t kind        = 0; // 0 draw, 1 indirect draw, 2 dispatch
	uint64_t submit_id   = 0;
	uint64_t hash_a      = 0; // VS or CS shader hash
	uint64_t hash_b      = 0; // PS shader hash
	uint32_t count_a = 0, count_b = 0, count_c = 0;
	uint64_t address     = 0; // indirect arguments, if any
};

inline bool      g_enabled = false;
inline vk::Queue g_queue {};
inline std::array<Record, 65536> g_ring;
inline uint64_t                   g_counter = 0;

inline void Mark(vk::CommandBuffer command, uint32_t kind, uint64_t submit_id, uint64_t hash_a,
                 uint64_t hash_b, uint32_t a, uint32_t b, uint32_t c, uint64_t address = 0) {
	if (!g_enabled) {
		return;
	}
	const uint64_t id = ++g_counter;
	g_ring[id % g_ring.size()] = {id, kind, submit_id, hash_a, hash_b, a, b, c, address};
	command.setCheckpointNV(reinterpret_cast<const void*>(static_cast<uintptr_t>(id)));
}

inline void Dump(vk::Queue queue = g_queue) {
	if (!g_enabled || !queue) {
		return;
	}
	std::printf("[GpuFault] last GPU checkpoints (newest ring id %llu):\n",
	            static_cast<unsigned long long>(g_counter));
	auto data = queue.getCheckpointDataNV();
	for (const auto& item: data) {
		const auto id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(item.pCheckpointMarker));
		const auto& r = g_ring[id % g_ring.size()];
		if (r.id != id) {
			std::printf("[GpuFault]   stage=%s marker=%llu (record overwritten)\n",
			            vk::to_string(item.stage).c_str(), static_cast<unsigned long long>(id));
			continue;
		}
		std::printf("[GpuFault]   stage=%s marker=%llu kind=%u submit=%llu hashA=0x%llx hashB=0x%llx "
		            "counts=%u,%u,%u args=0x%llx\n",
		            vk::to_string(item.stage).c_str(), static_cast<unsigned long long>(id), r.kind,
		            static_cast<unsigned long long>(r.submit_id),
		            static_cast<unsigned long long>(r.hash_a), static_cast<unsigned long long>(r.hash_b),
		            r.count_a, r.count_b, r.count_c, static_cast<unsigned long long>(r.address));
	}
	std::fflush(stdout);
}

} // namespace Libs::Graphics::Checkpoint

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCHECKPOINT_H_
