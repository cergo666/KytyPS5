#ifndef EMULATOR_SRC_GRAPHICS_PERFLOG_H_
#define EMULATOR_SRC_GRAPHICS_PERFLOG_H_

namespace Libs::Graphics::PerfLog {

// Writes perf_log.csv (working directory) every two seconds: presented FPS, system and process
// CPU use, the command-processor thread's CPU use, GPU use and VRAM (NVML, NVIDIA only) and RAM.
// On by default; KYTY_PERF_LOG=0 turns it off.
void Start();
// Counts one presented frame.
void CountFrame();
// Number of frames counted so far.
unsigned long long FrameCount();
// Adaptive quality: 1 = shadow passes drawn every frame, 2 = every second frame. Switches to 2 when
// the measured FPS stays below KYTY_ADAPTIVE_LOW (default 22) and back to 1 once it reaches
// KYTY_ADAPTIVE_HIGH (default 33); KYTY_ADAPTIVE_SHADOWS=0 keeps it at 1.
unsigned RecommendedShadowEvery();
// Feeds one FPS sample (every two seconds) to the adaptive controller.
void AdaptiveQuality(double fps);
// Registers the calling thread under a label ("cp") so its CPU use is reported separately.
void RegisterThisThread(const char* label);

} // namespace Libs::Graphics::PerfLog

#endif // EMULATOR_SRC_GRAPHICS_PERFLOG_H_
