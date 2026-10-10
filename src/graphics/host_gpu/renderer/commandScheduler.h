#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <optional>
#include <condition_variable>
#include <memory>
#include <deque>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler {
public:
	CommandScheduler(RenderContext& context, GraphicContext& graphics, bool presentation = false);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering();
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           FlushAndWait();
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Records a small command buffer that runs behind all submitted work and waits for it. The
	// current command buffer stays open and the tick timeline does not advance. Its pool is not
	// shared: call it only from the thread that records the current command buffer.
	// With `writer_tick` and a readback queue, the copy runs there instead and waits only for
	// that tick of the master timeline (0: nothing to wait for).
	template <typename Record>
	void SubmitDetached(Record&& record, std::optional<uint64_t> writer_tick = std::nullopt) {
		// On the readback queue the copy is independent of the deferred Vulkan stream: record and
		// submit it directly, without waiting for the recorder to drain.
		const bool direct   = writer_tick.has_value() && HasReadbackQueue();
		const bool deferral = direct ? SuspendRecorderDeferral() : false;
		record(BeginDetached());
		EndDetached(writer_tick);
		if (direct) {
			ResumeRecorderDeferral(deferral);
		}
	}
	static bool SuspendRecorderDeferral();
	static void ResumeRecorderDeferral(bool previous);
	[[nodiscard]] bool HasReadbackQueue() const noexcept;
	// Work that runs after everything already submitted but ahead of the recording command
	// buffer. BeginAhead returns a recording buffer; SubmitAheadAndWait runs it and blocks.
	[[nodiscard]] vk::CommandBuffer BeginAhead();
	void                            SubmitAheadAndWait();
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	// Waits only for operations up to `tick` that may write one of `ranges` (operations without a
	// declared range count as writing everything); later or unrelated publications may follow.
	void                      WaitPriorityOperations(uint64_t tick, std::span<const GuestRange> ranges);
	// Guest-memory completions use the priority queue; normal callbacks maintain GPU resources.
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	// An empty range is host-only; an omitted range conservatively aliases all guest memory.
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation,
	                                                 std::optional<GuestRange>      range = std::nullopt);
	[[nodiscard]] bool        HasPendingPriorityOperations();
	// True while an operation deferred at `tick` or later has not completed.
	[[nodiscard]] bool        HasPriorityOperationsSince(uint64_t tick);
	[[nodiscard]] bool        HasPendingPriorityOperations(GuestRange range);
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] DynamicState&    GetDynamicState() noexcept { return m_dynamic_state; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }
	// Tick of the newest DeferPriorityOperation (publishes that land after GPU completion).
	[[nodiscard]] uint64_t LastPriorityTick() const noexcept { return m_last_priority_tick; }

private:
	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
		std::optional<GuestRange>    range;
	};

	void              BeginNext();
	vk::CommandBuffer BeginDetached();
	void              EndDetached(std::optional<uint64_t> writer_tick);
	void              PriorityOperationsThread(std::stop_token stop);
	void QueueOperation(Common::UniqueFunction<void>&& operation, bool priority,
	                    std::optional<GuestRange> range = std::nullopt);
	void RunOperation(Common::UniqueFunction<void>&& operation);

	MasterSemaphore              m_master;
	vk::CommandPool              m_detached_pool    = nullptr;
	vk::CommandBuffer            m_detached_command = nullptr;
	vk::Fence                    m_detached_fence   = nullptr;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	vk::Queue                        m_queue;
	Common::Mutex&                   m_queue_mutex;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	DynamicState                 m_dynamic_state;
	std::deque<PendingOperation> m_pending_operations;
	uint32_t                     m_refresh_skip = 0;
	std::deque<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	std::optional<GuestRange>    m_priority_active_range;
	OperationState               m_operation_state      = OperationState::Open;
	uint64_t                         m_last_priority_tick   = 0;
	vk::CommandPool                  m_ahead_pool           = nullptr;
	vk::CommandBuffer                m_ahead_buffer         = nullptr;
	std::unique_ptr<MasterSemaphore> m_ahead_semaphore;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
