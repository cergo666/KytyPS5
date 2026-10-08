#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_CPSAMPLER_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_CPSAMPLER_H_

namespace Libs::Graphics {

// Diagnostic sampling profiler for the calling thread (KYTY_CP_SAMPLE=1, Windows only). Every
// millisecond it records the thread's call stack and, every few seconds, writes the hottest
// functions and source lines to cp_samples_<n>.txt next to the working directory.
void CpSamplerStartIfEnabled();

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_CPSAMPLER_H_
