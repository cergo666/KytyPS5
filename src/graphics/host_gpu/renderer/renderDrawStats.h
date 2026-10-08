#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAWSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAWSTATS_H_

#include "graphics/perfLog.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <vector>

namespace Libs::Graphics::DrawStats {

// KYTY_DIAG_DRAWS=1: every 10 seconds prints what the average frame is made of: render passes (by color and
// depth target), shader pairs and compute shaders, with draws per frame. Command-processor thread only.
inline bool Enabled() {
	static const bool enabled = [] {
		const char* env = std::getenv("KYTY_DIAG_DRAWS");
		return env != nullptr && env[0] == '1';
	}();
	return enabled;
}

struct Counter {
	uint64_t draws = 0, prims = 0, instances = 0;
};

using PassKey   = std::tuple<uint64_t, uint64_t, uint32_t, uint32_t, uint32_t>; // color, depth, w, h, colors
using ShaderKey = std::tuple<uint64_t, uint64_t>;                               // vs, ps

inline std::map<PassKey, Counter>                       g_passes;
inline std::map<ShaderKey, Counter>                     g_shaders;
inline std::map<uint64_t, Counter>                      g_compute; // cs hash
inline auto                                             g_window_start  = std::chrono::steady_clock::now();
inline uint64_t                                         g_window_frames = 0;

inline void MaybeReport() {
	const auto now    = std::chrono::steady_clock::now();
	const auto window = std::chrono::duration<double>(now - g_window_start).count();
	if (window < 10.0) {
		return;
	}
	const auto   frames_total = PerfLog::FrameCount();
	const double frames       = std::max<double>(1.0, static_cast<double>(frames_total - g_window_frames));
	uint64_t     draws = 0, prims = 0;
	for (const auto& [key, c]: g_passes) {
		draws += c.draws;
		prims += c.prims;
	}
	std::printf("[Draws] window %.1fs, %.0f frames: %.0f draws/frame, %.0fk prims/frame\n", window, frames,
	            static_cast<double>(draws) / frames, static_cast<double>(prims) / frames / 1000.0);
	std::vector<std::pair<PassKey, Counter>> passes(g_passes.begin(), g_passes.end());
	std::sort(passes.begin(), passes.end(), [](auto& a, auto& b) { return a.second.draws > b.second.draws; });
	for (size_t i = 0; i < passes.size() && i < 14; i++) {
		const auto& [k, c] = passes[i];
		std::printf("[Draws]  pass color=0x%llx depth=0x%llx %ux%u colors=%u : %.1f draws/f %.1fk prims/f\n",
		            static_cast<unsigned long long>(std::get<0>(k)),
		            static_cast<unsigned long long>(std::get<1>(k)), std::get<2>(k), std::get<3>(k),
		            std::get<4>(k), static_cast<double>(c.draws) / frames,
		            static_cast<double>(c.prims) / frames / 1000.0);
	}
	std::vector<std::pair<ShaderKey, Counter>> shaders(g_shaders.begin(), g_shaders.end());
	std::sort(shaders.begin(), shaders.end(), [](auto& a, auto& b) { return a.second.draws > b.second.draws; });
	for (size_t i = 0; i < shaders.size() && i < 24; i++) {
		const auto& [k, c] = shaders[i];
		std::printf("[Draws]  vs=0x%llx ps=0x%llx : %.1f draws/f %.1fk prims/f %.1f inst/draw\n",
		            static_cast<unsigned long long>(std::get<0>(k)),
		            static_cast<unsigned long long>(std::get<1>(k)), static_cast<double>(c.draws) / frames,
		            static_cast<double>(c.prims) / frames / 1000.0,
		            c.draws ? static_cast<double>(c.instances) / static_cast<double>(c.draws) : 0.0);
	}
	std::vector<std::pair<uint64_t, Counter>> compute(g_compute.begin(), g_compute.end());
	std::sort(compute.begin(), compute.end(), [](auto& a, auto& b) { return a.second.draws > b.second.draws; });
	for (size_t i = 0; i < compute.size() && i < 16; i++) {
		const auto& [k, c] = compute[i];
		std::printf("[Draws]  cs=0x%llx : %.1f dispatches/f, avg %.0f groups\n",
		            static_cast<unsigned long long>(k), static_cast<double>(c.draws) / frames,
		            c.draws ? static_cast<double>(c.prims) / static_cast<double>(c.draws) : 0.0);
	}
	std::fflush(stdout);
	g_passes.clear();
	g_shaders.clear();
	g_compute.clear();
	g_window_start  = now;
	g_window_frames = frames_total;
}

inline void RecordDraw(uint64_t vs, uint64_t ps, uint64_t color, uint64_t depth, uint32_t w, uint32_t h,
                       uint32_t colors, uint32_t index_count, uint32_t instances) {
	const uint64_t prims = static_cast<uint64_t>(index_count) * instances / 3;
	auto&          p     = g_passes[{color, depth, w, h, colors}];
	p.draws++;
	p.prims += prims;
	auto& s = g_shaders[{vs, ps}];
	s.draws++;
	s.prims += prims;
	s.instances += instances;
	MaybeReport();
}

inline void RecordDispatch(uint64_t cs, uint64_t groups) {
	auto& c = g_compute[cs];
	c.draws++;
	c.prims += groups;
}

} // namespace Libs::Graphics::DrawStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAWSTATS_H_
