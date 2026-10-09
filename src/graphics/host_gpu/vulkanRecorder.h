#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_VULKANRECORDER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_VULKANRECORDER_H_

namespace Libs::Graphics {
struct GraphicContext;
} // namespace Libs::Graphics

namespace Libs::Graphics::VkRecorder {

// Moves the guest GPU thread's Vulkan driver work (command recording, command buffer begin/end and
// queue submission) to a dedicated thread. The Vulkan entry points are replaced in the default
// dispatcher; on a thread that called EnableForThisThread() each call is copied, arguments and all,
// into a ring and executed later on the recorder thread in the same order. Other threads call the
// driver directly. KYTY_VK_THREAD=0 turns it off.

// Call once, right after the default dispatcher has been initialised for the device.
void Install(GraphicContext& graphics);
// Marks the calling thread as one whose recording is deferred.
void EnableForThisThread();
// Waits until everything queued so far has reached the driver. Any thread; a no-op when idle.
void Drain();
// Turns deferral off (false) or back on for the calling thread and returns the previous state.
// For work recorded and submitted independently of the deferred stream (readback copies on
// their own queue), which must not wait for the ring to drain.
bool SetDeferral(bool enabled);
[[nodiscard]] bool Enabled();

} // namespace Libs::Graphics::VkRecorder

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_VULKANRECORDER_H_
