#include "graphics/host_gpu/renderer/renderDraw.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/rasterScale.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderCheckpoint.h"
#include "graphics/host_gpu/renderer/renderDeferred.h"
#include "graphics/host_gpu/renderer/renderDrawStats.h"
#include "graphics/perfLog.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include "graphics/host_gpu/cpWaitStats.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t index_offset,
	                                           const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.fetch_embedded) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = *vs_input_info.stage.program;
	const auto& resources = *vs_input_info.stage.resources;
	if (index_offset == 0 &&
	    program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			vertex_offset = static_cast<int32_t>(resources.user_data[index]);
		}
	}
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			instance_offset = resources.user_data[index];
		}
	}

	return {vertex_offset, instance_offset};
}

static std::atomic<uint32_t> g_draw_state_log_count   = 0;
static std::atomic<uint32_t> g_draw_input_log_count   = 0;
static std::atomic<uint32_t> g_mrt_state_log_count    = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		default:
			// A fixed-point guest bias cannot be represented exactly by a floating-point host
			// attachment without VK_EXT_depth_bias_control.
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

static const char* RenderColorTypeName(const RenderColorInfo& color) {
	return color.image_id ? "RenderTexture" : "NoColorOutput";
}

static void LogFramebufferSkip(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF(
	    "DrawFramebufferSkip[%u]: %s color=%s color_addr=0x%010" PRIx64 " color_size=0x%016" PRIx64
	    " color_image=%s depth_format=%s depth_image=%s depth_vaddr_num=%d target_mask=0x%08" PRIx32
	    " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	    log_id, draw_name, RenderColorTypeName(color), color.desc.info.data.address,
	    color.desc.info.data.size, color.image_id ? "yes" : "no",
	    vk::to_string(depth.desc.view_info.format).c_str(), depth.image_id ? "yes" : "no",
	    static_cast<int>(!depth.desc.info.data.Empty()) +
	        static_cast<int>(depth.desc.info.HasStencil()),
	    ctx.GetRenderTargetMask(), static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags);
}

static void LogMrtState(const char* draw_name, const CommandBuffer& buffer,
                        const ShaderPixelInputInfo& ps_input_info) {
	const auto& ctx            = buffer.GetRegisters();
	const auto& sh_regs        = ctx.GetShaderRegisters();
	const auto  rt_mask        = ctx.GetRenderTargetMask();
	const auto  cb_shader_mask = sh_regs.m_cbShaderMask;
	const auto& bc0            = ctx.GetBlendControl(0);

	auto log_id = g_mrt_state_log_count.fetch_add(1);
	if (log_id >= 32) {
		return;
	}

	LOGF("MrtState[%u]: %s rt_mask=0x%08" PRIx32 " cb_shader_mask=0x%08" PRIx32
	     " blend0=%s src=%u dst=%u alpha_src=%u alpha_dst=%u sep_alpha=%s\n",
	     log_id, draw_name, rt_mask, cb_shader_mask, bc0.enable ? "true" : "false",
	     bc0.color_srcblend, bc0.color_destblend, bc0.alpha_srcblend, bc0.alpha_destblend,
	     bc0.separate_alpha_blend ? "true" : "false");

	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt  = ctx.GetRenderTarget(i);
		const auto& bc  = ctx.GetBlendControl(i);
		const auto  ctm = (rt_mask >> (i * 4u)) & 0x0fu;
		const auto  csm = (cb_shader_mask >> (i * 4u)) & 0x0fu;

		if (rt.base.addr == 0 && ps_input_info.target_output_mode[i] == 0 && ctm == 0 && csm == 0 &&
		    !bc.enable) {
			continue;
		}

		LOGF("MrtState[%u]: slot=%u addr=0x%010" PRIx64
		     " target_mask=0x%x shader_mask=0x%x out_mode=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32
		     " width=%u height=%u tile=%u"
		     " blend=%s src=%u dst=%u alpha_src=%u alpha_dst=%u\n",
		     log_id, i, rt.base.addr, ctm, csm, ps_input_info.target_output_mode[i],
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
		     rt.attrib2.height + 1, static_cast<uint32_t>(rt.attrib3.tile_mode),
		     bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.alpha_srcblend,
		     bc.alpha_destblend);
	}
}

static void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                               uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!color.image_id) {
		return;
	}

	auto log_id = g_draw_state_log_count.fetch_add(1);
	if (log_id >= 192) {
		return;
	}

	const auto& cc             = ctx.GetColorControl();
	const auto& bc             = ctx.GetBlendControl(color.target_slot);
	const auto& dc             = ctx.GetDepthControl();
	const auto& vp             = ctx.GetScreenViewport();
	const auto& vp0            = vp.viewports[0];
	const auto& ps_resources   = ps_input_info.stage.program->info;
	const auto  sampled_images = std::count_if(
	    ps_resources.images.begin(), ps_resources.images.end(), [](const auto& image) {
		    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
	    });

	const auto extent = color.Extent();
	const auto sc     = calc_final_scissor(vp, ctx.GetScanModeControl(), extent, 0);

	LOGF(
	    "DrawTargetState[%u]: frame=%d %s target=%s addr=0x%010" PRIx64
	    " extent=%ux%u prim=%u index_count=%u flags=0x%08" PRIx32 " color_mask=0x%08" PRIx32
	    " cc_mode=%u cc_op=0x%02x"
	    " blend=%s src=%u dst=%u comb=%u ps_tex=%d sampled=%d storage=%d ps_kill=%s target_mode0=%u"
	    " depth_test=%s depth_write=%s depth_func=%u depth_clear=%s viewport=(%.1f,%.1f %.1fx%.1f) "
	    "scissor=(%d,%d)-(%d,%d)\n",
	    log_id, buffer.GetContext().GetGpu().GetFrameNum(), draw_name, RenderColorTypeName(color),
	    color.desc.info.data.address, extent.width, extent.height,
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags, ctx.GetRenderTargetMask(),
	    cc.mode, cc.op,
	    bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
	    static_cast<int>(ps_resources.images.size()), static_cast<int>(sampled_images),
	    static_cast<int>(ps_resources.images.size() - sampled_images),
	    ps_input_info.ps_pixel_kill_enable ? "true" : "false", ps_input_info.target_output_mode[0],
	    dc.z_enable ? "true" : "false", dc.z_write_enable ? "true" : "false", dc.zfunc,
	    depth.depth_clear_enable ? "true" : "false", vp0.xoffset - vp0.xscale,
	    vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, sc.left, sc.top, sc.right,
	    sc.bottom);

	LogMrtState(draw_name, buffer, ps_input_info);
}

static void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
                              const ShaderVertexInputInfo& vs_input_info,
                              uint32_t index_type_and_size, uint32_t index_count,
                              const void* index_addr) {
	auto log_id = g_draw_input_log_count.fetch_add(1);
	if (log_id >= 64) {
		return;
	}

	LOGF("DrawInputState[%u]: frame=%d target=%s addr=0x%010" PRIx64
	     " index_type=%u index_count=%u index_addr=0x%016" PRIx64
	     " vs_resources=%d vs_buffers=%d\n",
	     log_id, buffer.GetContext().GetGpu().GetFrameNum(), RenderColorTypeName(color),
	     color.desc.info.data.address, index_type_and_size, index_count,
	     reinterpret_cast<uint64_t>(index_addr), vs_input_info.resources_num,
	     vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index);

		const auto* bytes = reinterpret_cast<const uint8_t*>(b.addr);
		if (bytes != nullptr && b.stride != 0) {
			const uint32_t records = std::min<uint32_t>(b.num_records, 4u);
			for (uint32_t rec = 0; rec < records; rec++) {
				const auto* rec_bytes = bytes + static_cast<uint64_t>(rec) * b.stride;
				const auto  dword_num = std::min<uint32_t>(b.stride / 4u, 12u);
				uint32_t    raw[12]   = {};
				float       flt[12]   = {};
				for (uint32_t i = 0; i < dword_num; i++) {
					std::memcpy(&raw[i], rec_bytes + i * 4u, sizeof(raw[i]));
					std::memcpy(&flt[i], rec_bytes + i * 4u, sizeof(flt[i]));
				}
				LOGF("DrawInputState[%u]: vb[%d].rec[%u] stride=%u dwords=%u raw=%08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " f=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
				     log_id, bi, rec, b.stride, dword_num, raw[0], raw[1], raw[2], raw[3], raw[4],
				     raw[5], raw[6], raw[7], raw[8], flt[0], flt[1], flt[2], flt[3], flt[4], flt[5],
				     flt[6], flt[7], flt[8]);

				for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
					const auto& r  = vs_input_info.resources[ai];
					const auto& rd = vs_input_info.resources_dst[ai];
					if (rd.buffer_index != bi) {
						continue;
					}
					const auto offset = static_cast<uint32_t>(r.Base48() - b.addr);
					if (offset + 4u <= b.stride &&
					    r.Format() == Prospero::BufferFormat::k8_8_8_8UNorm) {
						uint32_t packed = 0;
						std::memcpy(&packed, rec_bytes + offset, sizeof(packed));
						const auto r8 = (packed >> 0u) & 0xffu;
						const auto g8 = (packed >> 8u) & 0xffu;
						const auto b8 = (packed >> 16u) & 0xffu;
						const auto a8 = (packed >> 24u) & 0xffu;
						LOGF("DrawInputState[%u]: vb[%d].rec[%u].attr[%d] dst=v%d fmt=56 "
						     "rgba8=%02" PRIx32 "%02" PRIx32 "%02" PRIx32 "%02" PRIx32
						     " rgba=(%.3f,%.3f,%.3f,%.3f)\n",
						     log_id, bi, rec, ai, rd.register_start, r8, g8, b8, a8,
						     static_cast<double>(r8) / 255.0, static_cast<double>(g8) / 255.0,
						     static_cast<double>(b8) / 255.0, static_cast<double>(a8) / 255.0);
					}
				}
			}
		}

		for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
			const auto& r  = vs_input_info.resources[ai];
			const auto& rd = vs_input_info.resources_dst[ai];
			if (rd.buffer_index != bi) {
				continue;
			}
			LOGF("DrawInputState[%u]: attr[%d] offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, static_cast<uint32_t>(r.Base48() - b.addr), rd.register_start,
			     rd.registers_num, rd.fetch_index, r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
		}
	}
}

static void UpdateGraphicsDynamicState(const CommandBuffer& buffer, DynamicState& dynamic_state,
                                       const ShaderVertexInputInfo& vs_input_info,
                                       const RenderDepthInfo& depth, const RenderState& rendering) {
	KYTY_PROFILER_FUNCTION();

	const auto& ctx = buffer.GetRegisters();
	const auto&        vp  = ctx.GetScreenViewport();
	const auto&        effective = buffer.EffectiveRenderState();
	const float        scale_x   = float(effective.width) / rendering.width;
	const float        scale_y   = float(effective.height) / rendering.height;
	const vk::Extent2D framebuffer_extent {rendering.width, rendering.height};
	const auto& outputs = vs_input_info.stage.program->info.outputs;
	const bool  indexed_viewports =
	    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
		    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	    });
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	std::array<vk::Viewport, viewport_slots> viewports {};
	std::array<vk::Rect2D, viewport_slots>   scissors {};
	const uint32_t viewport_count = indexed_viewports ? viewport_slots : 1;
	for (uint32_t i = 0; i < viewport_count; i++) {
		const auto& guest    = vp.viewports[i];
		auto&       viewport = viewports[i];
		if (ctx.GetClipControl().clip_disable) {
			const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
			viewport.width  = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
			viewport.height = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
		} else {
			viewport.x      = guest.xoffset - guest.xscale;
			viewport.y      = guest.yoffset - guest.yscale;
			viewport.width  = guest.xscale * 2.0f;
			viewport.height = guest.yscale * 2.0f;
		}
		viewport.minDepth =
		    guest.zoffset - (ctx.GetClipControl().dx_clip_space ? 0.0f : guest.zscale);
		viewport.maxDepth = guest.zscale + guest.zoffset;

		const auto final_scissor =
		    calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent, i);
		auto& scissor  = scissors[i];
		scissor.offset = {final_scissor.left, final_scissor.top};
		scissor.extent = {static_cast<uint32_t>(final_scissor.right - final_scissor.left),
		                  static_cast<uint32_t>(final_scissor.bottom - final_scissor.top)};
		if (viewport.width == 0.0f) {
			// Keep empty slots at their guest index; Vulkan requires a positive viewport width.
			viewport.width = 1.0f;
			scissor.extent = {0, 0};
		}
		viewport.x *= scale_x;
		viewport.y *= scale_y;
		viewport.width *= scale_x;
		viewport.height *= scale_y;
		scissor =
		    ScaleRasterScissor(scissor, scale_x, scale_y, {effective.width, effective.height});
	}
	dynamic_state.SetViewports(std::span {viewports}.first(viewport_count));
	dynamic_state.SetScissors(std::span {scissors}.first(viewport_count));

	float line_width = ctx.GetLineWidth();
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	dynamic_state.SetLineWidth(line_width);
	const auto&      blend = ctx.GetBlendColor();
	const std::array blend_constants {blend.red, blend.green, blend.blue, blend.alpha};
	dynamic_state.SetBlendConstants(blend_constants);
	dynamic_state.SetDepthTestEnabled(depth.depth_test_enable);
	dynamic_state.SetDepthWriteEnabled(depth.depth_write_enable);
	dynamic_state.SetDepthCompareOp(depth.depth_compare_op);
#if !defined(__APPLE__)
	dynamic_state.SetDepthBoundsTestEnabled(depth.depth_bounds_test_enable);
	if (depth.depth_bounds_test_enable) {
		dynamic_state.SetDepthBounds(ctx.GetDepthBoundsMin(), ctx.GetDepthBoundsMax());
	}
#endif

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	dynamic_state.SetDepthBiasEnabled(depth_bias_enable);
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor = ConvertPolygonOffsetConstantFactor(
		    guest_constant_factor, poly_offset, depth.desc.view_info.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		dynamic_state.SetDepthBias(constant_factor, poly_offset.clamp, slope_factor);
	}

	dynamic_state.SetStencilTestEnabled(depth.stencil_test_enable);
	if (depth.stencil_test_enable) {
		const auto& front = depth.stencil_front;
		const auto& back  = depth.stencil_back;
		dynamic_state.SetStencilOps({front.failOp, front.passOp, front.depthFailOp, front.compareOp},
		                            {back.failOp, back.passOp, back.depthFailOp, back.compareOp});
		dynamic_state.SetStencilCompareMasks(front.compareMask, back.compareMask);
		dynamic_state.SetStencilWriteMasks(front.writeMask, back.writeMask);
		dynamic_state.SetStencilReferences(front.reference, back.reference);
	}

#if defined(__APPLE__)
	// MoltenVK has no VK_EXT_color_write_enable; the pipeline is created without the
	// eColorWriteEnableEXT dynamic state and relies on the static colorWriteMask instead.
#else
	vk::Bool32 enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t slot = 0; slot < rendering.num_color_attachments; slot++) {
		enable[slot] = rendering.color_attachments[slot].image_view != nullptr;
	}
	dynamic_state.SetColorWriteEnables(std::span {enable}.first(rendering.num_color_attachments));
#endif
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

static bool PixelShaderHasDepthOrCoverageSideEffects(const HW::ShaderRegisters& sh_regs) {
	const auto& db = sh_regs.db_shader_control;
	return db.shader_kill_enable || db.shader_z_export_enable || db.shader_mask_export_enable ||
	       db.shader_dual_export_enable || db.shader_execute_on_noop;
}

struct DrawRenderState {
	RenderDepthInfo       depth_info;
	RenderColorInfo       color_info[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	uint32_t              color_count                              = 0;
	bool                  ps_active                                = true;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo  ps_input_info;
	PipelineCache::GraphicsPrograms programs;
};

struct DrawCallInfo {
	CommandBufferDebugOp debug_op       = CommandBufferDebugOp::DrawIndex;
	uint32_t             index_count    = 0;
	uint32_t             instance_count = 0;
	uint32_t             first_instance = 0;

	[[nodiscard]] bool IsIndexed() const { return debug_op == CommandBufferDebugOp::DrawIndex; }
	[[nodiscard]] const char* Name() const { return IsIndexed() ? "DrawIndex" : "DrawIndexAuto"; }
};

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages) {
	KYTY_PROFILER_BLOCK("Draw::AcquireRenderTargets");
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	feedback_aspects = {};
	auto&       cache = m_context.GetTextureCache();
	RenderState state {};
	std::tie(state.raster_scale_x, state.raster_scale_y) = m_context.GetRasterScale();
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto owner = cache.m_slot_images.try_get(target.image_id);
		if (owner == nullptr || (!owner->registered && !owner->info.data.Empty()) ||
		    owner->binding.needs_rebind) {
			EXIT("color target changed after render-state discovery\n");
		}
		const auto image_view = cache.FindRenderTarget(target.image_id, target.desc);
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		const auto& view   = target.desc.view_info;
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		const auto extent       = target.Extent();
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view   = image_view;
		attachment.image_layout = layout;
		attachment.image            = &image;
		attachment.mip_level        = view.base_level;
		attachment.base_layer       = view.base_layer;
	}
	if (depth.image_id) {
		const auto owner = cache.m_slot_images.try_get(depth.image_id);
		if (owner == nullptr || !owner->registered || owner->binding.needs_rebind) {
			EXIT("depth target changed after render-state discovery\n");
		}
		const auto  image_view = cache.FindDepthTarget(depth.image_id, depth.desc);
		const auto& metadata   = depth.desc.info.metadata;
		if (metadata.kind == ImageMetadataKind::Htile && depth.depth_clear_enable &&
		    !cache.ClearMeta(metadata.range.address)) {
			EXIT("failed to acquire HTile metadata for a depth clear\n");
		}
		const bool meta_clear =
		    metadata.kind == ImageMetadataKind::Htile &&
		    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer);
		depth.depth_load_clear_enable = depth.depth_clear_enable || meta_clear;
		if (meta_clear &&
		    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
			EXIT("failed to consume HTile clear state\n");
		}
		auto& image = cache.GetImage(depth.image_id);
		EXIT_IF(image_view == nullptr || image.backing.samples != depth.desc.info.samples);
		const auto draw_writes = depth.AttachmentWriteAspects();
		vk::ImageAspectFlags sampled_aspects;
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture) continue;
				const auto native =
				    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == image.views.end());
				sampled_aspects |= native->info.aspect;
				feedback_aspects |= DepthFeedbackAspects(draw_writes, depth.desc.view_info,
				                                         native->info);
			}
		}
		if (feedback_aspects && !m_context.GetGraphics().attachment_feedback_loop_enabled) {
			EXIT("depth attachment feedback loop is not supported by the host\n");
		}
		auto layout = depth_attachment_layout(depth);
		if (sampled_aspects & ~DepthReadableAspects(layout)) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		const auto access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		const auto& view                = depth.desc.view_info;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		state.width               = std::min(state.width, depth.desc.info.extent.width);
		state.height              = std::min(state.height, depth.desc.info.extent.height);
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.image          = &image;
		attachment.mip_level      = view.base_level;
		attachment.base_layer     = view.base_layer;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear    = depth.depth_load_clear_enable;
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear  = depth.stencil_clear_enable;
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	return state;
}

static uint32_t DrawColorOutputMask(const HW::Context& ctx) {
	const auto& sh_regs     = ctx.GetShaderRegisters();
	const auto  write_mask  = ctx.GetRenderTargetMask() & sh_regs.m_cbShaderMask;
	uint32_t    output_mask = 0;
	for (uint32_t index = 0; index < RENDER_COLOR_ATTACHMENTS_MAX; index++) {
		const auto slot = ShaderPixelExportTarget(sh_regs.m_cbShaderMask, index);
		if (slot >= RENDER_COLOR_ATTACHMENTS_MAX) {
			break;
		}
		if (sh_regs.target_output_mode[index] != 0 &&
		    render_target_mask_slot(write_mask, slot) != 0) {
			output_mask |= 1u << slot;
		}
	}
	return output_mask;
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

static bool ConsumeMetadataColorOperation(const CommandBuffer& buffer) {
	KYTY_PROFILER_BLOCK("Draw::ConsumeMetadataColorOp");
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	// These special modes run color-buffer metadata or decompression operations. The shader is a
	// vehicle for that operation, and its exported color must not be applied as a normal draw.
	// Kyty stores expanded Vulkan images rather than compressed guest surfaces, so no equivalent
	// hardware pass is emitted. Tracked DCC clear state is materialized on attachment bind;
	// future CMask/FMask support can consume its state through the same TextureCache path.
	return mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear) ||
	       mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

struct DrawEmitInfo {
	int32_t  vertex_offset = 0;
	uint32_t first_vertex  = 0;
	uint32_t first_instance = 0;
};

struct DrawIndexBufferSource {
	uint64_t      address   = 0;
	const void*   host_data = nullptr;
	uint64_t      size      = 0;
	vk::IndexType type      = vk::IndexType::eUint16;
	uint32_t      guest_element_size = 0;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

static uint64_t VertexBufferDescriptorSize(int binding, const ShaderVertexInputInfo& info) {
	const auto& buffer = info.buffers[binding];
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < info.resources_num; i++) {
		if (info.resources_dst[i].buffer_index != binding) {
			continue;
		}
		const auto& resource = info.resources[i];
		// RDNA2 OOB_SELECT=2 only checks NumRecords != 0. A constant attribute still
		// fetches its entire format; NumRecords is not a byte count in this mode.
		const uint64_t extent = resource.OutOfBounds() == 2
		                            ? resource.Base48() - buffer.addr +
		                                  ShaderRecompiler::Format::GetFormatInfo(resource.Format()).byte_size
		                            : buffer.num_records;
		size = std::max(size, extent);
	}
	return size;
}

struct VertexBufferRange {
	uint64_t                     base_address  = 0;
	uint64_t                     requested_end = 0;
	uint64_t                     acquired_end  = 0;
	std::pair<Buffer*, uint64_t> binding;

	[[nodiscard]] uint64_t RequestedSize() const { return requested_end - base_address; }
};

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	std::array<vk::Buffer, MaxBuffers>     buffers {};
	std::array<vk::DeviceSize, MaxBuffers> offsets {};
	std::array<vk::DeviceSize, MaxBuffers> sizes {};
	uint32_t                               count = 0;
};

static PreparedVertexBuffers AcquireVertexBuffers(CommandBuffer&               buffer,
                                                  const ShaderVertexInputInfo& vs_input_info) {
	KYTY_PROFILER_BLOCK("Draw::AcquireVertexBuffers");
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);

	// Collect the non-empty guest vertex ranges.
	std::array<uint64_t, ShaderVertexInputInfo::RES_MAX>          sizes {};
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges {};
	uint32_t                                                      range_count = 0;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(i, vs_input_info);
		sizes[i]           = size;
		if (size == 0) {
			continue;
		}
		if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vertex.addr, size);
		}
		ranges[range_count++] = {vertex.addr, vertex.addr + size};
	}

	std::sort(ranges.begin(), ranges.begin() + range_count,
	          [](const VertexBufferRange& left, const VertexBufferRange& right) {
		          return left.base_address < right.base_address;
	          });

	// Merge overlapping or touching ranges before acquiring host buffers.
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> merged_ranges {};
	uint32_t                                                      merged_count = 0;
	for (uint32_t i = 0; i < range_count; i++) {
		const auto& range = ranges[i];
		if (merged_count != 0 &&
		    merged_ranges[merged_count - 1].requested_end >= range.base_address) {
			merged_ranges[merged_count - 1].requested_end =
			    std::max(merged_ranges[merged_count - 1].requested_end, range.requested_end);
			continue;
		}
		merged_ranges[merged_count++] = {range.base_address, range.requested_end};
	}

	auto& cache = buffer.GetContext().GetBufferCache();
	for (uint32_t i = 0; i < merged_count; i++) {
		auto& range = merged_ranges[i];
		// PPSA20298
		const auto size =
		    Libs::LibKernel::Memory::ClampRangeSize(range.base_address, range.RequestedSize());
		range.acquired_end = range.base_address + size;
		range.binding      = cache.ObtainBuffer(range.base_address, size, false);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	PreparedVertexBuffers prepared;
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		const auto range = std::find_if(merged_ranges.begin(), merged_ranges.begin() + merged_count,
		                                [&](const VertexBufferRange& value) {
			                                return vertex.addr >= value.base_address &&
			                                       vertex.addr < value.acquired_end;
		                                });
		if (range == merged_ranges.begin() + merged_count) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vertex.addr);
		}

		prepared.buffers[i] = range->binding.first->Handle();
		prepared.offsets[i] = range->binding.second + vertex.addr - range->base_address;
		prepared.sizes[i]   = std::min(size, range->acquired_end - vertex.addr);
	}

	return prepared;
}

static void SetDrawDebugPhase(CommandBuffer& buffer, uint64_t submit_id, const DrawCallInfo& draw,
                              uint32_t phase) {
	buffer.SetDebugInfo(static_cast<uint32_t>(draw.debug_op), submit_id, phase, draw.index_count, 0,
	                    draw.instance_count, draw.first_instance);
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kPolygon:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) return false;
			[[fallthrough]];
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("Skipping draw with unknown primitive type: %u\n",
				            static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			return false;
		}
	}

	return true;
}

static bool ResolvePrimitiveRestart(const CommandBuffer& buffer,
                                    const DrawIndexBufferSource& source) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	EXIT_NOT_IMPLEMENTED((control & ~0x3u) != 0);
	if ((control & 0x1u) == 0) {
		return false;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return false;
	}

	const auto element_size = source.guest_element_size;
	const auto index_mask   = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index  = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return false;
	}
	const auto restart_index = reset_index & index_mask;
	if (restart_index == index_mask) {
		// Use native restart; the 8-bit path widens its marker to 0xffff.
		return true;
	}

	// A game can set a custom reset value without using it in the index buffer.
	// Keep restart off in that case; fail if we actually find the value.
	// Scan before preparing draw resources: readback can restart the command buffer.
	EXIT_NOT_IMPLEMENTED(source.address == 0);
	const auto* indices = reinterpret_cast<const uint8_t*>(source.address);
	for (uint64_t offset = 0; offset < source.size; offset += element_size) {
		uint32_t index = 0;
		std::memcpy(&index, indices + offset, element_size);
		EXIT_NOT_IMPLEMENTED(index == restart_index);
	}
	return false;
}

static std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
ComputeTargetExportMapping(const HW::Context& ctx, uint32_t color_output_mask) {
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> mapping {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
			mapping[slot] = TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
			                                             rt.info.channel_order)
			                    .export_mapping;
		}
	}
	return mapping;
}

static void RefreshShaders(CommandBuffer& buffer, const DrawCallInfo& draw,
                           uint32_t color_output_mask, DrawRenderState& state,
                           bool allow_geometry_motion = true) {
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	const auto& vertex_shader_info = sh_ctx.GetVs();
	const auto& pixel_shader_info  = sh_ctx.GetPs();
	const auto& shader_regs        = ctx.GetShaderRegisters();

	state.programs      = {};
	state.ps_input_info = {};
	const auto target_export_mapping = ComputeTargetExportMapping(ctx, color_output_mask);
	auto& pipeline_cache = buffer.GetContext().GetPipelineCache();
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "GetGraphicsPrograms");
	}
	state.programs = pipeline_cache.GetGraphicsPrograms(
	    vertex_shader_info, pixel_shader_info, shader_regs, ctx, buffer.GetUserConfig(),
	    target_export_mapping, state.ps_active, state.vertex_info, state.ps_input_info,
	    allow_geometry_motion && buffer.GetContext().TemporalInputsNeeded());
}

static bool SnapshotsEqual(const ShaderRecompiler::IR::ResourceSnapshot* a,
                           const ShaderRecompiler::IR::ResourceSnapshot* b) {
	if (a == nullptr || b == nullptr) {
		return a == b;
	}
	return a->buffers == b->buffers && a->images == b->images && a->samplers == b->samplers &&
	       a->flattened_srt == b->flattened_srt && a->user_data == b->user_data &&
	       a->specialization_reads == b->specialization_reads && a->uniform_fill == b->uniform_fill;
}

// Command-processor thread. Replaces RefreshShaders for a deferred draw.
static void ApplyDeferredShaders(CommandBuffer& buffer, const DrawCallInfo& draw,
                                 DeferredDraw& deferred, DrawRenderState& state,
                                 DeferredDrawQueue* queue) {
	auto& pipeline      = buffer.GetContext().GetPipelineCache();
	state.vertex_info   = deferred.vertex_info;
	state.ps_input_info = deferred.ps_info;
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "GetGraphicsProgramsAsync");
	}
	state.programs = pipeline.ResolveGraphicsProgramsAsync(deferred.prepared, deferred.async,
	                                                       state.vertex_info, state.ps_input_info);
	if (queue == nullptr) {
		return;
	}
	const bool used_async = (deferred.async.vertex.valid && deferred.async.vertex.task != nullptr);
	queue->total_async_hits += used_async ? 1 : 0;
	queue->total_fallbacks += used_async ? 0 : 1;
	if (!queue->verify) {
		return;
	}
	auto vertex_info = deferred.vertex_info;
	auto ps_info     = deferred.ps_info;
	pipeline.WaitAsyncIdle();
	const auto reference =
	    pipeline.ResolveGraphicsPrograms(deferred.prepared, vertex_info, ps_info);
	const bool same = reference.vertex[0].id == state.programs.vertex[0].id &&
	                  reference.pixel.id == state.programs.pixel.id &&
	                  SnapshotsEqual(vertex_info[0].stage.resources,
	                                 state.vertex_info[0].stage.resources) &&
	                  (!deferred.ps_active ||
	                   SnapshotsEqual(ps_info.stage.resources, state.ps_input_info.stage.resources));
	queue->total_verified++;
	if (!same) {
		queue->total_mismatches++;
		if (queue->total_mismatches <= 16) {
			std::printf("[AsyncDraw] MISMATCH #%llu (async_used=%d vs=%llu/%llu ps=%llu/%llu)\n",
			            static_cast<unsigned long long>(queue->total_mismatches), used_async ? 1 : 0,
			            static_cast<unsigned long long>(state.programs.vertex[0].id),
			            static_cast<unsigned long long>(reference.vertex[0].id),
			            static_cast<unsigned long long>(state.programs.pixel.id),
			            static_cast<unsigned long long>(reference.pixel.id));
		}
	}
	if (queue->total_verified % 200000 == 0) {
		std::printf("[AsyncDraw] verified=%llu mismatches=%llu async_hits=%llu fallbacks=%llu\n",
		            static_cast<unsigned long long>(queue->total_verified),
		            static_cast<unsigned long long>(queue->total_mismatches),
		            static_cast<unsigned long long>(queue->total_async_hits),
		            static_cast<unsigned long long>(queue->total_fallbacks));
	}
}

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            uint32_t            render_target_slice_offset,
	                                        DrawRenderState& state,
	                                        DeferredDraw* deferred) {
	KYTY_PROFILER_BLOCK("Draw::PrepareDrawRenderState");
	const auto& shader_regs       = buffer.GetRegisters().GetShaderRegisters();
	const auto  color_output_mask = DrawColorOutputMask(buffer.GetRegisters());
	state.ps_active = buffer.GetShaders().GetPs().ps_regs.data_addr != 0 &&
	                  (color_output_mask != 0 ||
	                   PixelShaderHasDepthOrCoverageSideEffects(shader_regs));
	if (deferred != nullptr) {
		ApplyDeferredShaders(buffer, draw, *deferred, state, m_deferred.get());
	} else {
		RefreshShaders(buffer, draw, color_output_mask, state);
	}
	uint32_t mrt_mask = 0;
	if (state.ps_active) {
		for (const auto& output: state.ps_input_info.stage.program->info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt) {
				mrt_mask |= 1u << output.location;
			}
		}
	}
	mrt_mask &= color_output_mask;
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderColorTarget");
	}
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if ((mrt_mask & (1u << slot)) != 0) {
			ResolveRenderColorTarget(buffer, state.color_info[state.color_count],
			                         render_target_slice_offset, slot);
			if (state.color_info[state.color_count].image_id) {
				state.color_count++;
			}
		}
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderDepthTarget");
	}
	ResolveRenderDepthTarget(buffer, state.depth_info);
	if (state.ps_input_info.geometry_motion_dword != UINT32_MAX) {
		const auto& color  = state.color_info[0];
		const auto& view   = color.desc.view_info;
		const auto  extent = color.Extent();
		if (state.color_count != 1 || color.target_slot != 0 || view.base_level != 0 ||
		    view.base_layer != 0 || view.layer_count != 1 ||
		    !m_context.GetGeometryMotion().SupportsSurface(
		        m_context.GetTextureCache().GetImage(color.image_id), extent) ||
		    (state.depth_info.image_id &&
		     (state.depth_info.desc.info.extent.width < extent.width ||
		      state.depth_info.desc.info.extent.height < extent.height))) {
			RefreshShaders(buffer, draw, color_output_mask, state, false);
		}
	}

	if (state.color_count == 0 && !state.depth_info.image_id && !state.ps_active) {
		LogFramebufferSkip(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   draw.index_count, 0);
		return false;
	}

	return true;
}

static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	KYTY_PROFILER_BLOCK("Draw::PrepareIndexBuffer");
	PreparedIndexBuffer prepared;
	if (source.size == 0) {
		return prepared;
	}
	prepared.type = source.type;
	if (source.host_data != nullptr) {
		auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = stream.Copy(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
	}
	return prepared;
}

static void CommitVertexBuffers(vk::CommandBuffer            vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	KYTY_PROFILER_BLOCK("Draw::CommitVertexBuffers");
	for (uint32_t i = 0; i < prepared.count; i++) {
		EXIT_IF(prepared.buffers[i] == nullptr);
	}
	if (prepared.count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache.
		vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
		                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
	}
}

static void CommitIndexBuffer(vk::CommandBuffer vk_buffer, const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
	                             const DrawRenderState& state, uint32_t index_type_and_size,
                                 const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.IsIndexed() && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   state.ps_input_info, draw.index_count, 0);
	}
	LogDrawInputState(buffer, state.color_info[0], state.vertex_info[0], index_type_and_size,
	                  draw.index_count, index_addr);
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, vk::CommandBuffer vk_buffer,
                               const DrawCallInfo& draw, const DrawEmitInfo& emit) {
	KYTY_PROFILER_BLOCK("Draw::EmitDrawPrimitives");
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kPolygon:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch:
			if (draw.IsIndexed()) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count, emit.first_vertex,
				               emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.IsIndexed()) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count, i + emit.first_vertex,
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
}

void RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw_in, DrawRenderState& state,
                                         vk::PrimitiveTopology topology, const DrawEmitInfo& emit_in,
                                         const DrawIndexBufferSource& index_source,
	                                     bool primitive_restart_enable, uint64_t gpu_args_addr) {
	KYTY_PROFILER_BLOCK("Draw::ExecutePreparedDraw");
	KYTY_WAIT_SCOPE("x.total");
	const char* draw_category = "cat.color.small";
	if (CpWaitStats::Enabled()) {
		const auto& depth_extent = state.depth_info.desc.info.extent;
		if (state.color_count == 0) {
			draw_category = depth_extent.width == depth_extent.height && depth_extent.width >= 1023
			                    ? "cat.depth.square"
			                    : "cat.depth.other";
		} else {
			draw_category = state.color_info[0].Extent().width >= 1500 ? "cat.color.big"
			                                                              : "cat.color.small";
		}
	}
	KYTY_WAIT_SCOPE(draw_category);
	auto& ucfg = buffer.GetUserConfig();
	const auto vertex_stages =
	    std::span {state.vertex_info.data(), state.programs.VertexStageCount()};
	const bool mesh_active = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;

	// A GPU-resident argument buffer is drawn natively. Mesh shaders need the counts on the host,
	// so they read the arguments back here (waiting for the GPU) and draw as before.
	DrawCallInfo draw = draw_in;
	DrawEmitInfo emit = emit_in;
	if (gpu_args_addr != 0 &&
	    (mesh_active || ucfg.GetPrimType() == Prospero::PrimitiveType::kQuadListLegacy)) {
		m_context.GetBufferCache().ReadMemory(gpu_args_addr, 16);
		uint32_t raw[4] {};
		if (!LibKernel::Memory::TryReadBacking(gpu_args_addr, raw, sizeof(raw))) {
			EXIT("indirect draw arguments are unreadable\n");
		}
		draw.index_count    = raw[0];
		draw.instance_count = raw[1];
		const bool native   = !state.vertex_info[0].fetch_embedded;
		emit.first_vertex   = native ? 0 : raw[2];
		emit.first_instance = native ? 0 : raw[3];
		gpu_args_addr       = 0;
		if (draw.index_count == 0 || draw.instance_count == 0) {
			return;
		}
	}
	uint32_t   mesh_groups = 0;
	if (mesh_active) {
		const auto& mesh = state.vertex_info[0].mesh;
		EXIT_NOT_IMPLEMENTED(mesh.fast_launch && (draw.IsIndexed() || primitive_restart_enable));
		static std::atomic_bool restart_warned = false;
		if (primitive_restart_enable && !restart_warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("Warning: primitive restart is not implemented for mesh shaders; "
			            "continuing draw (primitive=%u indexed=%u)\n",
			            static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed());
		}
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed(), primitive_restart_enable);
		}
		const auto primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (primitives == 0 || draw.instance_count == 0) {
			return;
		}
		mesh_groups        = (primitives - 1u) / mesh.primitives_per_group + 1u;
		const auto& limits = m_context.GetGraphics().mesh_shader_properties;
		if (mesh_groups > limits.maxMeshWorkGroupCount[0] ||
		    draw.instance_count > limits.maxMeshWorkGroupCount[1] ||
		    static_cast<uint64_t>(mesh_groups) * draw.instance_count >
		        limits.maxMeshWorkGroupTotalCount) {
			EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", mesh_groups,
			     draw.instance_count);
		}
	}

	if (mesh_active && draw.IsIndexed()) {
		// Register the original guest indices for shader reads; PrepareGraphicsBindings
		// synchronizes registered BDA ranges before any draw commands are committed.
		(void)m_context.GetBufferCache().FindBuffer(
		    index_source.address, static_cast<uint64_t>(draw.index_count) *
		                              index_source.guest_element_size);
	}
	LogDrawPhase(draw.Name(), "PrepareBindings");
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	{
		KYTY_WAIT_SCOPE("x.preparebindings");
		for (uint32_t i = 0; i < vertex_stages.size(); i++) {
			PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i]);
			descriptor_stages[stage_count++] = &bindings.vertex[i];
		}
		if (state.ps_active) {
			if (!bindings.pixel) bindings.pixel.emplace();
			PrepareBindings(state.ps_input_info.stage, *bindings.pixel);
			descriptor_stages[stage_count++] = &*bindings.pixel;
		}
	}
	const auto stages = std::span {descriptor_stages.data(), stage_count};
	{
		KYTY_WAIT_SCOPE("x.preparegraphicsbindings");
		PrepareGraphicsBindings(stages, std::span {state.color_info, state.color_count});
	}
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	if (!mesh_active) {
		KYTY_WAIT_SCOPE("x.vertexindex");
		LogDrawPhase(draw.Name(), "PrepareVertexBuffers");
		vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0]);
		index_binding   = PrepareIndexBuffer(buffer, index_source);
	}
	if (DrawStats::Enabled()) {
		const auto extent = state.color_count != 0 ? state.color_info[0].Extent() : vk::Extent2D {0, 0};
		DrawStats::RecordDraw(
		    state.vertex_info[0].stage.program->shader_hash,
		    state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0,
		    state.color_count != 0 ? state.color_info[0].desc.info.data.address : 0,
		    state.depth_info.image_id ? state.depth_info.desc.info.data.address : 0, extent.width,
		    extent.height, state.color_count, draw.index_count, draw.instance_count);
	}
	vk::Buffer args_buffer;
	uint64_t   args_offset = 0;
	if (gpu_args_addr != 0) {
		const auto [obtained, offset] =
		    m_context.GetBufferCache().ObtainBuffer(gpu_args_addr, 16, false);
		EXIT_IF(obtained == nullptr || (offset & 3u) != 0);
		args_buffer = obtained->Handle();
		args_offset = offset;
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "CreatePipeline");
	}
	KYTY_WAIT_SCOPE("x.pipeline+targets");
	auto& pipeline = m_context.GetPipelineCache().GetGraphicsPipeline(
	    std::span {state.color_info, state.color_count}, state.depth_info, vertex_stages, buffer,
	    state.ps_active ? &state.ps_input_info : nullptr, topology, primitive_restart_enable,
	    state.programs);
	{
		// KYTY_DIAG_RUNS=1: how often consecutive draws repeat pipeline, targets and bindings.
		static const bool diag = std::getenv("KYTY_DIAG_RUNS") != nullptr;
		if (diag) {
			struct Prev {
				const void* pipeline = nullptr;
				uint64_t targets = 0, images = 0, buffers = 0;
				uint64_t run = 0;
			};
			static Prev prev;
			static uint64_t n = 0, same_pipe = 0, same_targets = 0, same_both = 0, same_images = 0,
			                same_buffers = 0, same_all = 0, runs = 0, run_total = 0, push_only = 0;
			static auto last = std::chrono::steady_clock::now();
			uint64_t targets = 1469598103934665603ull, images = targets, buffers = targets;
			const auto mix = [](uint64_t& h, uint64_t v) { h = (h ^ v) * 1099511628211ull; };
			for (uint32_t i = 0; i < state.color_count; ++i) {
				mix(targets, state.color_info[i].image_id.index);
			}
			mix(targets, state.depth_info.image_id.index);
			for (const auto* stage: stages) {
				for (const auto& t: stage->images) {
					mix(images, t.image_id.index);
					mix(images, reinterpret_cast<uintptr_t>(static_cast<VkImageView>(t.image_view)));
				}
				for (const auto& b: stage->buffer_sources) {
					mix(buffers, b.id.index);
					mix(buffers, b.address);
					mix(buffers, b.size);
				}
			}
			const void* pipe = &pipeline;
			++n;
			const bool sp = pipe == prev.pipeline, st = targets == prev.targets;
			const bool si = images == prev.images, sb = buffers == prev.buffers;
			same_pipe += sp;
			same_targets += st;
			same_both += sp && st;
			same_images += si;
			same_buffers += sb;
			same_all += sp && st && si && sb;
			push_only += sp && st && si && !sb;
			if (sp) { ++prev.run; } else { ++runs; run_total += prev.run + 1; prev.run = 0; }
			prev.pipeline = pipe; prev.targets = targets; prev.images = images; prev.buffers = buffers;
			const auto now = std::chrono::steady_clock::now();
			if (now - last > std::chrono::seconds(10)) {
				last = now;
				const double d = static_cast<double>(n);
				std::printf("RUNDIAG draws=%llu same_pipeline=%.1f%% same_targets=%.1f%% pipe+targets=%.1f%% "
				            "same_images=%.1f%% same_buffers=%.1f%% all_same=%.1f%% images_same_buffers_diff=%.1f%% "
				            "avg_pipeline_run=%.2f\n",
				            static_cast<unsigned long long>(n), 100 * same_pipe / d, 100 * same_targets / d,
				            100 * same_both / d, 100 * same_images / d, 100 * same_buffers / d,
				            100 * same_all / d, 100 * push_only / d,
				            runs ? static_cast<double>(run_total) / runs : 0.0);
				std::fflush(stdout);
				n = same_pipe = same_targets = same_both = same_images = same_buffers = same_all = 0;
				runs = run_total = push_only = 0;
			}
		}
	}
	vk::ImageAspectFlags feedback_aspects;
	auto rendering = AcquireRenderTargets(buffer, state.color_info, state.color_count,
	                                      state.depth_info, feedback_aspects, stages);
	// Only a pass that survives target acquisition and is then split by motion
	// capture resumes with LOAD. A split during acquisition restarts with the
	// guest's own clears (DEPTH_CLEAR_ENABLE), as it did before capture existed.
	const auto               interrupted = buffer.ActiveRenderState();
	std::array<uint32_t, 14> motion_push {};
	if (state.ps_input_info.geometry_motion_dword != UINT32_MAX) {
		const auto&           vs       = state.vertex_info[0];
		const auto&           viewport = buffer.GetRegisters().GetScreenViewport().viewports[0];
		std::vector<uint64_t> key {vs.stage.program->shader_hash,
		                           state.ps_input_info.stage.program->shader_hash,
		                           uint64_t(topology),
		                           draw.index_count,
		                           uint32_t(emit.vertex_offset),
		                           emit.first_vertex,
		                           emit.first_instance,
		                           draw.instance_count,
		                           index_source.address,
		                           index_source.guest_element_size,
		                           rendering.width,
		                           rendering.height,
		                           std::bit_cast<uint32_t>(viewport.xscale),
		                           std::bit_cast<uint32_t>(viewport.yscale),
		                           std::bit_cast<uint32_t>(viewport.xoffset),
		                           std::bit_cast<uint32_t>(viewport.yoffset)};
		const uint64_t        last_vertex = uint64_t(emit.first_vertex) + draw.index_count;
		uint32_t              capacity =
            draw.IsIndexed() || last_vertex > UINT32_MAX ? 0 : uint32_t(last_vertex);
		for (int i = 0; i < vs.buffers_num; ++i) {
			const auto& vb = vs.buffers[i];
			key.insert(key.end(), {vb.addr, vb.stride, vb.num_records, vb.fetch_index});
			if (vb.fetch_index == 0 && vb.stride != 0)
				capacity = std::max(capacity, vb.num_records);
		}
		key.push_back(capacity);
		auto& motion         = m_context.GetGeometryMotion();
		bool  scene_geometry = true;
		for (const auto* stage: stages) {
			for (const auto& texture: stage->images) {
				if (!texture.image_id) continue;
				const auto& sampled = m_context.GetTextureCache().GetImage(texture.image_id);
				// A post-processing quad describes its own geometry, not the
				// scene sampled underneath it. Do not advertise that as scene motion.
				if (sampled.usage.render_target || sampled.usage.storage) scene_geometry = false;
			}
		}
		if (scene_geometry)
			motion_push =
			    motion.PrepareDraw(buffer, key, capacity, emit.first_instance, draw.instance_count);
		if (state.depth_info.depth_compare_op == vk::CompareOp::eGreater ||
		    state.depth_info.depth_compare_op == vk::CompareOp::eGreaterOrEqual)
			motion_push[7] |= 2;
		EXIT_IF(!motion.Attach(buffer, rendering));
		const float normalized_viewport[] {(viewport.xoffset - viewport.xscale) / rendering.width,
		                                   (viewport.yoffset - viewport.yscale) / rendering.height,
		                                   viewport.xscale * 2 / rendering.width,
		                                   viewport.yscale * 2 / rendering.height};
		std::memcpy(motion_push.data() + 8, normalized_viewport, sizeof(normalized_viewport));
	}

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory.
	auto vk_buffer = buffer.Handle();
	SetDrawDebugPhase(buffer, submit_id, draw, draw.IsIndexed() ? 0x100u : 0x200u);
	if (!mesh_active) {
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	if (state.ps_active && !draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x300u);
	}
	CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
	if (mesh_active) {
		const uint32_t draw_data[] {
		    draw.index_count,
		    draw.IsIndexed() ? static_cast<uint32_t>(emit.vertex_offset) : emit.first_vertex,
		    emit.first_instance, index_source.guest_element_size,
		    static_cast<uint32_t>(index_source.address),
		    static_cast<uint32_t>(index_source.address >> 32u)};
		static_assert(std::size(draw_data) == ShaderRecompiler::IR::PushData::MeshDrawDwordCount);
		vk_buffer.pushConstants(pipeline.pipeline_layout,
		                        vk::ShaderStageFlagBits::eMeshEXT |
		                            vk::ShaderStageFlagBits::eFragment,
		                        0, sizeof(draw_data), draw_data);
	} else {
		CommitIndexBuffer(vk_buffer, index_binding);
	}

	LogDrawPhase(draw.Name(), "BeginRendering");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x400u);
	}
	if (gpu_args_addr != 0) {
		// The arguments may have been written by earlier shaders or copies on the GPU.
		m_context.GetCommandScheduler().EndRendering();
		vk::MemoryBarrier barrier {};
		barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics |
		                              vk::PipelineStageFlagBits::eComputeShader |
		                              vk::PipelineStageFlagBits::eTransfer,
		                          vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &barrier, 0, nullptr,
		                          0, nullptr);
	}
	const auto guest_state = [](RenderState state) {
		if (state.geometry_motion_attachment) {
			state.color_attachments[7]       = {};
			state.num_color_attachments      = 1;
			state.geometry_motion_attachment = false;
		}
		return state;
	};
	// Capture barriers may split an otherwise unchanged guest pass. Resume its
	// attachments with LOAD; clearing depth again would change guest visibility.
	buffer.BeginRendering(rendering,
	                      interrupted && guest_state(*interrupted) == guest_state(rendering));
	auto& dynamic_state = m_context.GetCommandScheduler().GetDynamicState();
	UpdateGraphicsDynamicState(buffer, dynamic_state, vertex_stages.back(), state.depth_info,
	                           rendering);
	dynamic_state.SetAttachmentFeedbackLoopAspects(feedback_aspects);
	dynamic_state.Commit(m_context.GetGraphics(), vk_buffer);
	if (state.ps_input_info.geometry_motion_dword != UINT32_MAX) {
		m_context.GetGeometryMotion().BeginPass(buffer);
		const auto& effective = buffer.EffectiveRenderState();
		const float reciprocal_extent[] {1.f / effective.width, 1.f / effective.height};
		std::memcpy(motion_push.data() + 12, reciprocal_extent, sizeof(reciprocal_extent));
		vk_buffer.pushConstants(pipeline.pipeline_layout,
		                        vk::ShaderStageFlagBits::eVertex |
		                            vk::ShaderStageFlagBits::eFragment,
		                        0, sizeof(motion_push), motion_push.data());
	}
	if (state.ps_active && state.ps_input_info.raster_scale_dword != UINT32_MAX) {
		const auto&          effective = buffer.EffectiveRenderState();
		const float          inverse_scale[] {float(rendering.width) / effective.width,
		                                      float(rendering.height) / effective.height};
		vk::ShaderStageFlags push_stages = vk::ShaderStageFlagBits::eFragment;
		for (const auto& stage: vertex_stages)
			push_stages |= NativeShaderStage(stage.logical_stage);
		vk_buffer.pushConstants(pipeline.pipeline_layout, push_stages,
		                        state.ps_input_info.raster_scale_dword * sizeof(uint32_t),
		                        sizeof(inverse_scale), inverse_scale);
	}
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x500u);
	}
	if (mesh_active) {
		vk_buffer.drawMeshTasksEXT(mesh_groups, draw.instance_count, 1);
	} else if (gpu_args_addr != 0) {
		Checkpoint::Mark(vk_buffer, 1, submit_id, state.vertex_info[0].stage.program->shader_hash,
		                 state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0, 0, 0,
		                 0, gpu_args_addr);
		vk_buffer.drawIndirect(args_buffer, args_offset, 1, 16);
	} else {
		Checkpoint::Mark(vk_buffer, 0, submit_id, state.vertex_info[0].stage.program->shader_hash,
		                 state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0,
		                 draw.index_count, draw.instance_count, emit.first_instance);
		EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
	}
	m_context.GetBufferCache().RecordPendingWrites(
	    std::any_of(stages.begin(), stages.end(), [](const PreparedBindings* stage) {
		    return stage->runtime->program->has_address_writes;
	    }));

	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x600u);
	}
	vk::PipelineStageFlags shader_write_stages = {};
	for (const auto& stage: vertex_stages) {
		if (HasShaderBufferWrites(stage.stage)) {
			shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
		}
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (shader_write_stages) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, shader_write_stages);
	}
	LogDrawPhase(draw.Name(), "DrawComplete");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x700u);
	}
}

void RenderExecutor::DrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                               const DrawIndexArgs& args) {
	DrawIndexImpl(submit_id, buffer, args, nullptr);
}

void RenderExecutor::DrawIndexImpl(uint64_t submit_id, CommandBuffer& buffer,
                                   const DrawIndexArgs& args, DeferredDraw* deferred) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    args.index_count, 0, 1, args.instance_count,
	                    reinterpret_cast<uint64_t>(args.index_addr));

	Common::LockGuard lock(m_context.GetMutex());
	if (args.index_count == 0 || args.instance_count == 0) {
		return;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndex():Shader:\n");
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t base_vertex         = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.index_type_and_size, args.index_count,
		     reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
		     static_cast<uint32_t>(args.base_vertex), args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		return;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16:
			index_source.type               = vk::IndexType::eUint16;
			index_source.guest_element_size = 2;
			break;
		case Prospero::IndexType::kIndex32:
			index_source.type               = vk::IndexType::eUint32;
			index_source.guest_element_size = 4;
			break;
		case Prospero::IndexType::kIndex8:
			index_source.guest_element_size = 1;
			break;
		default: EXIT("unknown index_type_and_size: %u\n", args.index_type_and_size);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	const bool primitive_restart = ResolvePrimitiveRestart(buffer, index_source);

	std::vector<uint16_t> expanded_indices;
	if (index_source.guest_element_size == 1) {
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			expanded_indices[i] = primitive_restart && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = expanded_indices.data();
		index_source.size      = expanded_indices.size() * sizeof(uint16_t);
	}

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndex, args.index_count,
	                        args.instance_count, args.first_instance};
	DrawRenderState state {};
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state,
	                            deferred)) {
		ResetBindings();
		return;
	}

	LogDrawStateIfNeeded(buffer, draw, state, args.index_type_and_size,
	                     args.index_addr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);

	DrawEmitInfo emit {};
	// Native fetches consume the patched SGPRs; rewritten fetches need Vulkan offsets.
	const bool native_indirect = indirect && !state.vertex_info[0].fetch_embedded;
	emit.vertex_offset  = native_indirect ? 0 : vertex_offset + args.base_vertex;
	emit.first_instance = native_indirect ? 0 : instance_offset;

	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
	                    primitive_restart);
	ResetBindings();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args) {
	DrawAutoImpl(submit_id, buffer, args, nullptr);
}

void RenderExecutor::DrawAutoImpl(uint64_t submit_id, CommandBuffer& buffer,
                                  const DrawAutoArgs& args, DeferredDraw* deferred) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    args.vertex_count, 0, args.first_vertex, args.instance_count,
	                    args.first_instance);

	Common::LockGuard lock(m_context.GetMutex());
	if (args.gpu_args_addr == 0 && (args.vertex_count == 0 || args.instance_count == 0)) {
		return;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndexAuto():Shader:\n");
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t vertex_count        = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndexAuto,
	                         args.vertex_count, args.instance_count, args.first_instance};

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		ResetBindings();
		return;
	}
	DrawRenderState state {};
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state, deferred)) {
		ResetBindings();
		return;
	}

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.program->param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return;
	}

	LogDrawStateIfNeeded(buffer, draw, state, 0, nullptr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);
	DrawEmitInfo emit {};
	const bool native_indirect = indirect && !state.vertex_info[0].fetch_embedded;
	emit.first_vertex = native_indirect ? 0 :
	    static_cast<uint32_t>(vertex_offset + static_cast<int32_t>(args.first_vertex));
	emit.first_instance = native_indirect ? 0 : instance_offset;

	DrawIndexBufferSource index_source {};
	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source, false,
	                    args.gpu_args_addr);
	ResetBindings();
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer, uint32_t render_target_slice_offset) {
	KYTY_PROFILER_BLOCK("Draw::ResolveColorTargets");
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(buffer, src, render_target_slice_offset, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, render_target_slice_offset, 1, true, true);
	if (!src.image_id || !dst.image_id) {
		return false;
	}
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	destination.Resolve(source, {src.guest_mip_level, 1, src.guest_array_layer, 1},
	                    {dst.guest_mip_level, 1, dst.guest_array_layer, 1});
	return true;
}

RenderExecutor::RenderExecutor(RenderContext& context): m_context(context) {}

RenderExecutor::~RenderExecutor() = default;

bool RenderExecutor::HasDeferredDraws() const noexcept {
	return m_deferred != nullptr && m_deferred->count != 0;
}

// Personal-build quality knobs for the shadow-map passes (depth-only, square target, no pixel shader),
// which make up about half of a fight frame's draws:
//   KYTY_SHADOW_OFF=1         never draw them
//   KYTY_SHADOW_EVERY=N       draw them only on every Nth frame (the map keeps its previous contents)
//   KYTY_SHADOW_MIN_PRIMS=T   skip draws with fewer than T primitives
// The first two are decided once per run of consecutive shadow draws so a map is never half updated.
static bool ShadowFilterSkips(const HW::Context& ctx, const HW::Shader& sh, uint32_t count,
                              uint32_t instances) {
	static const struct Config {
		bool     off       = false;
		uint32_t every     = 1;
		uint32_t min_prims = 0;
		bool     any       = false;
	} config = [] {
		Config c;
		const auto get = [](const char* name) -> long {
			const char* env = std::getenv(name);
			return env != nullptr ? std::atol(env) : 0;
		};
		c.off       = get("KYTY_SHADOW_OFF") == 1;
		c.every     = static_cast<uint32_t>(std::max(1L, get("KYTY_SHADOW_EVERY")));
		c.min_prims = static_cast<uint32_t>(std::max(0L, get("KYTY_SHADOW_MIN_PRIMS")));
		c.any       = c.off || c.every > 1 || c.min_prims > 0;
		return c;
	}();
	if (!config.any) {
		return false;
	}
	static bool     in_run      = false;
	static bool     run_skipped = false;
	static uint64_t skipped = 0, kept = 0;

	const auto& depth = ctx.GetDepthRenderTarget();
	const bool  ps_present = sh.GetPs().ps_regs.data_addr != 0;
	const bool  depth_only =
	    DrawColorOutputMask(ctx) == 0 && depth.z_write_base_addr != 0 && depth.size.valid &&
	    !(ps_present && PixelShaderHasDepthOrCoverageSideEffects(ctx.GetShaderRegisters()));
	const bool square_map = depth.size.x_max == depth.size.y_max && depth.size.x_max >= 1023;
	if (!depth_only || !square_map) {
		in_run = false;
		return false;
	}
	if (!in_run) {
		in_run      = true;
		run_skipped = config.off || (config.every > 1 &&
		                             PerfLog::FrameCount() % config.every != 0);
	}
	bool skip = run_skipped;
	if (!skip && config.min_prims != 0 &&
	    static_cast<uint64_t>(count) * instances / 3 < config.min_prims) {
		skip = true;
	}
	(skip ? skipped : kept)++;
	if (((skipped + kept) & 0x3ffffu) == 0) {
		std::printf("[Shadow] kept=%llu skipped=%llu\n", static_cast<unsigned long long>(kept),
		            static_cast<unsigned long long>(skipped));
	}
	return skip;
}

bool RenderExecutor::TryDeferDrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                                       const DrawIndexArgs& args) {
	return TryDeferGraphics(submit_id, buffer, &args, nullptr);
}

bool RenderExecutor::TryDeferDrawAuto(uint64_t submit_id, CommandBuffer& buffer,
                                      const DrawAutoArgs& args) {
	static const bool disabled = [] {
		const char* env = std::getenv("KYTY_NO_DEFER_AUTO");
		return env != nullptr && env[0] == '1';
	}();
	if (disabled || args.gpu_args_addr != 0) {
		return false;
	}
	return TryDeferGraphics(submit_id, buffer, nullptr, &args);
}

bool RenderExecutor::TryDeferGraphics(uint64_t submit_id, CommandBuffer& buffer,
                                      const DrawIndexArgs* indexed,
                                      const DrawAutoArgs* automatic) {
	KYTY_PROFILER_BLOCK("Draw::Defer");
	auto& pipeline = m_context.GetPipelineCache();
	if (!pipeline.AsyncEnabled()) {
		return false;
	}
	if (m_deferred == nullptr) {
		m_deferred = std::make_unique<DeferredDrawQueue>();
		const char* verify = std::getenv("KYTY_ASYNC_VERIFY");
		m_deferred->verify = verify != nullptr && verify[0] == '1';
		if (const char* debug = std::getenv("KYTY_ASYNC_DEBUG"); debug != nullptr) {
			m_deferred->debug_flags = static_cast<uint32_t>(std::atoi(debug));
		}
	}
	auto& queue = *m_deferred;
	const bool        is_indexed = indexed != nullptr;
	const uint32_t    count      = is_indexed ? indexed->index_count : automatic->vertex_count;
	const uint32_t    instances  = is_indexed ? indexed->instance_count : automatic->instance_count;
	const auto        source     = is_indexed ? indexed->offset_source : automatic->offset_source;
	if (queue.flushing || buffer.IsInvalid() || count == 0 || instances == 0 ||
	    source != DrawOffsetSource::DrawState || graphics_debug_dump_enabled()) {
		return false;
	}
	// Deferred draws do not capture geometry motion for temporal upscaling / frame generation
	// (presentation then estimates it with optical flow). KYTY_FG_MOTION=1 draws synchronously
	// while those inputs are wanted, for better generated frames at a CPU cost.
	static const bool fg_motion = [] {
		const char* env = std::getenv("KYTY_FG_MOTION");
		return env != nullptr && env[0] == '1';
	}();
	if (fg_motion && buffer.GetContext().TemporalInputsNeeded()) {
		return false;
	}

	KYTY_PROFILER_BLOCK("Defer::Checks");
	KYTY_WAIT_SCOPE("defer.total");
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	const auto& sh   = buffer.GetShaders();
	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	// Tessellation and merged (NGG) vertex stages keep using the synchronous path.
	if (ucfg.GetPrimType() == Prospero::PrimitiveType::kPatch ||
	    (ctx.GetShaderStages() & 0x20u) != 0 || !DrawHasValidVertexShader(sh) ||
	    !GetDrawTopology(ucfg, topology)) {
		return false;
	}

	bool shadow_skip = false;
	{
		KYTY_WAIT_SCOPE("defer.shadow");
		shadow_skip = ShadowFilterSkips(ctx, sh, is_indexed ? indexed->index_count : automatic->vertex_count,
		                                instances);
	}
	if (shadow_skip) {
		return true; // consumed without drawing
	}
	if (queue.count == DeferredDrawQueue::kCapacity) {
		KYTY_WAIT_SCOPE("flush.cap");
		FlushDeferredDraws();
	}
	auto& slot = queue.slots[(queue.head + queue.count) % DeferredDrawQueue::kCapacity];
	if (slot == nullptr) {
		slot = std::make_unique<DeferredDraw>();
	}
	DeferredDraw& draw = *slot;
	draw.kind          = is_indexed ? DeferredDraw::Kind::Draw : DeferredDraw::Kind::DrawAuto;
	draw.submit_id     = submit_id;
	if (is_indexed) {
		draw.args = *indexed;
	} else {
		draw.auto_args = *automatic;
	}
	{
		KYTY_PROFILER_BLOCK("Defer::Snapshot");
		KYTY_WAIT_SCOPE("defer.snapshot");
		draw.ctx = ctx;
		draw.ucfg = ucfg;
		draw.sh = sh;
	}

	const auto& regs             = draw.ctx;
	uint32_t    color_output_mask = 0;
	{
		KYTY_WAIT_SCOPE("defer.psactive");
		color_output_mask = DrawColorOutputMask(regs);
		draw.ps_active = draw.sh.GetPs().ps_regs.data_addr != 0 &&
		                 (color_output_mask != 0 ||
		                  PixelShaderHasDepthOrCoverageSideEffects(regs.GetShaderRegisters()));
	}
	{
		KYTY_PROFILER_BLOCK("Defer::Reset");
		draw.vertex_info = {};
		draw.ps_info     = {};
	}
	const auto mapping = [&] {
		KYTY_WAIT_SCOPE("defer.mapping");
		return ComputeTargetExportMapping(regs, color_output_mask);
	}();
	{
		KYTY_PROFILER_BLOCK("Defer::Prepare");
		KYTY_WAIT_SCOPE("defer.prepare");
		draw.prepared = pipeline.PrepareGraphicsPrograms(
		    draw.sh.GetVs(), draw.sh.GetPs(), regs.GetShaderRegisters(), regs, draw.ucfg, mapping,
		    draw.ps_active, draw.vertex_info, draw.ps_info);
	}
	{
		KYTY_PROFILER_BLOCK("Defer::Submit");
		KYTY_WAIT_SCOPE("defer.submit");
		pipeline.SubmitGraphicsAsync(draw.prepared, draw.vertex_info, draw.ps_info,
		                             m_context.GetBufferCache().GpuDirtyClearEpoch(), draw.async);
	}
	queue.count++;
	queue.total_deferred++;
	if ((queue.debug_flags & 2u) != 0) {
		FlushDeferredDraws();
	}
	return true;
}

bool RenderExecutor::TryDeferDispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                            uint32_t thread_group_x, uint32_t thread_group_y,
                                            uint32_t thread_group_z, uint32_t mode) {
	KYTY_PROFILER_BLOCK("Dispatch::Defer");
	// Deferring compute dispatches measured no faster than running them in order (the cost is in
	// the in-order binding work, not in resource materialization), so it is opt-in.
	static const bool compute_enabled = [] {
		const char* env = std::getenv("KYTY_ASYNC_COMPUTE");
		return env != nullptr && env[0] == '1';
	}();
	auto& pipeline = m_context.GetPipelineCache();
	if (!compute_enabled || !pipeline.AsyncEnabled()) {
		return false;
	}
	if (m_deferred == nullptr) {
		m_deferred = std::make_unique<DeferredDrawQueue>();
		const char* verify = std::getenv("KYTY_ASYNC_VERIFY");
		m_deferred->verify = verify != nullptr && verify[0] == '1';
		if (const char* debug = std::getenv("KYTY_ASYNC_DEBUG"); debug != nullptr) {
			m_deferred->debug_flags = static_cast<uint32_t>(std::atoi(debug));
		}
	}
	auto& queue = *m_deferred;
	if (queue.flushing || buffer.IsInvalid() || thread_group_x == 0 || thread_group_y == 0 ||
	    thread_group_z == 0 || graphics_debug_dump_enabled() ||
	    buffer.GetShaders().GetCs().cs_regs.data_addr == 0) {
		return false;
	}

	if (queue.count == DeferredDrawQueue::kCapacity) {
		FlushDeferredDraws();
	}
	auto& slot = queue.slots[(queue.head + queue.count) % DeferredDrawQueue::kCapacity];
	if (slot == nullptr) {
		slot = std::make_unique<DeferredDraw>();
	}
	DeferredDraw& op = *slot;
	op.kind          = DeferredDraw::Kind::Dispatch;
	op.submit_id     = submit_id;
	op.groups        = {thread_group_x, thread_group_y, thread_group_z};
	op.mode          = mode;
	op.ctx           = buffer.GetRegisters();
	op.ucfg          = buffer.GetUserConfig();
	op.sh            = buffer.GetShaders();

	constexpr uint32_t kUseThreadDimensions = 1u << 5u;
	auto&              info                 = op.compute_info;
	info                                    = {};
	const bool use_thread_dimensions        = (mode & kUseThreadDimensions) != 0;
	info.dispatch_thread_dimensions         = use_thread_dimensions;
	info.workgroup_counts[0]                = thread_group_x;
	info.workgroup_counts[1]                = thread_group_y;
	info.workgroup_counts[2]                = thread_group_z;
	if (use_thread_dimensions) {
		const auto&    cs          = op.sh.GetCs().cs_regs;
		const uint32_t group_sizes[] = {cs.num_thread_x, cs.num_thread_y, cs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size    = std::max(group_sizes[axis], 1u);
			const auto threads = info.workgroup_counts[axis];
			info.workgroup_counts[axis] = threads / size + (threads % size != 0u);
		}
	}
	op.compute_params = pipeline.PrepareComputeProgram(op.sh.GetCs(),
	                                                   op.ctx.GetShaderRegisters(), info);
	op.compute_async  = pipeline.SubmitComputeAsync(
        op.compute_params, info, m_context.GetBufferCache().GpuDirtyClearEpoch());
	queue.count++;
	queue.total_deferred++;
	return true;
}

// Whether a worker result can still be trusted: it must have succeeded, and none of the guest
// memory it read may have been written by the GPU (or downloaded) since it was submitted.
static bool AsyncTaskStillValid(BufferCache& cache, const MaterializeTask& task) {
	KYTY_PROFILER_BLOCK("Draw::ValidateAsync");
	if (!task.ok) {
		return false;
	}
	// Bytes published since the task was submitted invalidate it only where it read them.
	if (cache.GpuDirtyClearEpoch() != task.dirty_epoch) {
		for (const auto& read: task.reads) {
			if (cache.ClearedSince(task.dirty_epoch, read.address, read.address + read.size)) {
				return false;
			}
		}
	}
	uint64_t low = 0, high = 0;
	if (!cache.GpuDirtyBounds(low, high)) {
		return true; // nothing is GPU-modified, so no read can have raced a GPU write
	}
	for (const auto& read: task.reads) {
		// Most reads (constant tables) lie far from GPU-written buffers.
		if (read.address >= high || read.address + read.size <= low) {
			continue;
		}
		if (cache.HasGpuDirtyBytes(read.address, read.size)) {
			return false;
		}
	}
	return true;
}

bool RenderExecutor::TryDeferBarrier() {
	static const bool enabled = [] {
		const char* env = std::getenv("KYTY_DEFER_BARRIER"); // opt-in: caused artifacts and lower FPS in UFC 5
		return env != nullptr && env[0] == '1';
	}();
	if (!enabled || m_deferred == nullptr || m_deferred->count == 0 || m_deferred->flushing) {
		return false;
	}
	auto&      queue = *m_deferred;
	const auto last  = (queue.head + queue.count - 1) % DeferredDrawQueue::kCapacity;
	if (queue.slots[last]->kind == DeferredDraw::Kind::Barrier) {
		return true; // nothing ran since the previous barrier
	}
	if (queue.count == DeferredDrawQueue::kCapacity) {
		FlushDeferredDraws();
		return false;
	}
	auto& slot = queue.slots[(queue.head + queue.count) % DeferredDrawQueue::kCapacity];
	if (slot == nullptr) {
		slot = std::make_unique<DeferredDraw>();
	}
	slot->kind = DeferredDraw::Kind::Barrier;
	queue.count++;
	return true;
}

void RenderExecutor::RecordGlobalBarrier() {
	Common::LockGuard lock(m_context.GetMutex());

	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;

	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	auto& scheduler               = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	scheduler.Current().Handle().pipelineBarrier2(dependency);
}

void RenderExecutor::FlushDeferredDraws() {
	if (m_deferred == nullptr || m_deferred->count == 0 || m_deferred->flushing) {
		return;
	}
	KYTY_PROFILER_BLOCK("Draw::FlushDeferred");
	KYTY_WAIT_SCOPE("flush.all");
	auto& queue    = *m_deferred;
	queue.flushing = true;
	auto& pipeline = m_context.GetPipelineCache();
	while (queue.count != 0) {
		DeferredDraw& draw = *queue.slots[queue.head];
		if (draw.kind == DeferredDraw::Kind::Barrier) {
			RecordGlobalBarrier();
			queue.head = (queue.head + 1) % DeferredDrawQueue::kCapacity;
			queue.count--;
			continue;
		}
		const bool    is_dispatch = draw.kind == DeferredDraw::Kind::Dispatch;
		std::array<PipelineCache::AsyncGraphics::Stage*, 2> stages {
		    is_dispatch ? &draw.compute_async : &draw.async.vertex,
		    is_dispatch ? nullptr : &draw.async.pixel};
		for (auto* stage: stages) {
			if (stage != nullptr && stage->task != nullptr) {
				{
					KYTY_PROFILER_BLOCK("Flush::WaitWorker");
					KYTY_WAIT_SCOPE("flush.waitworker");
					AsyncMaterializer::Wait(stage->task);
				}
				stage->valid = (queue.debug_flags & 1u) == 0 &&
				               AsyncTaskStillValid(m_context.GetBufferCache(), *stage->task);
			}
		}
		auto& buffer      = m_context.GetCommandScheduler().Current();
		const auto saved  = buffer.SwapState(draw.ctx, draw.ucfg, draw.sh);
		if (is_dispatch) {
			DispatchDirectImpl(draw.submit_id, buffer, draw.groups[0], draw.groups[1],
			                   draw.groups[2], draw.mode, false, &draw);
		} else {
			if (draw.kind == DeferredDraw::Kind::DrawAuto) {
				DrawAutoImpl(draw.submit_id, buffer, draw.auto_args, &draw);
			} else {
				DrawIndexImpl(draw.submit_id, buffer, draw.args, &draw);
			}
		}
		buffer.RestoreState(saved);
		if (is_dispatch) {
			pipeline.ReleaseAsyncStage(draw.compute_async);
		} else {
			pipeline.ReleaseGraphicsAsync(draw.async);
		}
		queue.head = (queue.head + 1) % DeferredDrawQueue::kCapacity;
		queue.count--;
		m_ops_since_submit++;
	}
	queue.flushing = false;

	// The guest submits one large command buffer at a time, so without this the GPU would sit idle
	// until the command processor has recorded the whole thing. Hand the GPU work in slices.
	static const uint32_t submit_interval = [] {
		const char* env = std::getenv("KYTY_SUBMIT_INTERVAL");
		return env != nullptr ? static_cast<uint32_t>(std::atoi(env)) : 0u;
	}();
	if (submit_interval != 0 && m_ops_since_submit >= submit_interval) {
		m_ops_since_submit = 0;
		auto& scheduler    = m_context.GetCommandScheduler();
		if (scheduler.Active() && !scheduler.Current().IsInvalid()) {
			KYTY_PROFILER_BLOCK("Draw::SliceSubmit");
			scheduler.Flush();
		}
	}
}

} // namespace Libs::Graphics
