#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDIAG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDIAG_H_

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::Diag {

// KYTY_DIAG_META=1: remember which compute shaders wrote which guest ranges so a metadata
// readback can say who produced the data it is waiting for. Command-processor thread only.
inline bool MetaEnabled() {
	static const bool enabled = [] {
		const char* env = std::getenv("KYTY_DIAG_META");
		return env != nullptr && env[0] == '1';
	}();
	return enabled;
}

struct WriteRecord {
	uint64_t address = 0, size = 0, shader_hash = 0;
	uint32_t groups[3] {};
	bool     has_xor = false;
};

inline std::array<WriteRecord, 256> g_writes;
inline size_t                       g_write_index = 0;
inline uint32_t                     g_reports     = 0;

inline void RecordWrite(uint64_t address, uint64_t size, uint64_t shader_hash, const uint32_t* groups,
                        bool has_xor) {
	auto& r       = g_writes[g_write_index++ % g_writes.size()];
	r.address     = address;
	r.size        = size;
	r.shader_hash = shader_hash;
	r.has_xor     = has_xor;
	for (int i = 0; i < 3; i++) r.groups[i] = groups[i];
}

inline void ReportMetaReadback(uint64_t address, uint64_t size) {
	if (g_reports++ >= 60) return;
	std::printf("[Meta] readback range=0x%llx+0x%llx writers:", static_cast<unsigned long long>(address),
	            static_cast<unsigned long long>(size));
	int found = 0;
	for (size_t i = 0; i < g_writes.size() && found < 4; i++) {
		const auto& r = g_writes[(g_write_index + g_writes.size() - 1 - i) % g_writes.size()];
		if (r.size == 0 || r.address >= address + size || r.address + r.size <= address) continue;
		std::printf(" [shader=0x%llx xor=%d groups=%ux%ux%u buf=0x%llx+0x%llx]",
		            static_cast<unsigned long long>(r.shader_hash), r.has_xor ? 1 : 0, r.groups[0],
		            r.groups[1], r.groups[2], static_cast<unsigned long long>(r.address),
		            static_cast<unsigned long long>(r.size));
		found++;
	}
	std::printf("%s\n", found == 0 ? " (none recorded)" : "");
}

} // namespace Libs::Graphics::Diag

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDIAG_H_
