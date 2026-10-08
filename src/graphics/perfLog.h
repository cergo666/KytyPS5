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
// Registers the calling thread under a label ("cp") so its CPU use is reported separately.
void RegisterThisThread(const char* label);

} // namespace Libs::Graphics::PerfLog

#endif // EMULATOR_SRC_GRAPHICS_PERFLOG_H_
