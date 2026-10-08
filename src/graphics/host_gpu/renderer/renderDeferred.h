#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDEFERRED_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDEFERRED_H_

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"

#include <array>
#include <memory>

namespace Libs::Graphics {

// An indexed draw captured by the command processor together with the register state it must
// execute against. Its shader resources are materialized on a worker while the command
// processor keeps decoding; FlushDeferredDraws() later executes the draw in stream order.
struct DeferredDraw {
	enum class Kind : uint8_t { Draw, DrawAuto, Dispatch };
	Kind                                 kind      = Kind::Draw;
	uint64_t                             submit_id = 0;
	DrawIndexArgs                        args;
	DrawAutoArgs                         auto_args;
	HW::Context                          ctx;
	HW::UserConfig                       ucfg;
	HW::Shader                           sh;
	bool                                 ps_active = false;
	PipelineCache::GraphicsPrepared      prepared;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo                 ps_info;
	PipelineCache::AsyncGraphics         async;
	// Dispatch-only state (kind == Dispatch).
	std::array<uint32_t, 3>              groups {};
	uint32_t                             mode = 0;
	ShaderComputeInputInfo               compute_info;
	ShaderParams                         compute_params;
	PipelineCache::AsyncGraphics::Stage  compute_async;
};

struct DeferredDrawQueue {
	static constexpr size_t kCapacity = 128;

	std::array<std::unique_ptr<DeferredDraw>, kCapacity> slots;
	size_t                                               head     = 0;
	size_t                                               count    = 0;
	bool                                                 flushing = false;
	// KYTY_ASYNC_VERIFY=1 recomputes every deferred draw's programs synchronously and compares.
	bool     verify = false;
	// KYTY_ASYNC_DEBUG bit 0: ignore worker results; bit 1: flush after every deferred draw.
	uint32_t debug_flags = 0;
	uint64_t total_deferred = 0, total_async_hits = 0, total_fallbacks = 0, total_verified = 0,
	         total_mismatches = 0;
};


} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDEFERRED_H_
