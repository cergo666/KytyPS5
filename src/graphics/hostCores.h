#ifndef EMULATOR_SRC_GRAPHICS_HOSTCORES_H_
#define EMULATOR_SRC_GRAPHICS_HOSTCORES_H_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace Libs::Graphics::HostCores {

// KYTY_PIN_CORES=1 (default on for hosts with SMT; KYTY_PIN_CORES=0 turns it off): the command
// processor and the Vulkan recorder thread each get a physical core of their own. Their SMT siblings
// are removed from the process affinity, so no spinning worker or game thread can share a core with
// them and slow them down.
#ifdef _WIN32
namespace Detail {

struct Plan {
	DWORD_PTR cp_cpu          = 0; // single logical CPU
	DWORD_PTR recorder_cpu    = 0;
	DWORD_PTR excluded        = 0; // siblings removed from the process mask
	bool      valid           = false;
};

inline const Plan& GetPlan() {
	static const Plan plan = [] {
		Plan result;
		const char* env = std::getenv("KYTY_PIN_CORES");
		if (env != nullptr && env[0] == '0') {
			return result;
		}
		DWORD length = 0;
		GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
		std::vector<uint8_t> storage(length);
		if (length == 0 ||
		    !GetLogicalProcessorInformationEx(
		        RelationProcessorCore,
		        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(storage.data()), &length)) {
			return result;
		}
		DWORD_PTR process_mask = 0, system_mask = 0;
		if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
			return result;
		}
		std::vector<DWORD_PTR> cores; // group-0 masks of cores with at least two logical CPUs
		for (DWORD offset = 0; offset < length;) {
			auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(storage.data() + offset);
			if (info->Relationship == RelationProcessorCore && info->Processor.GroupCount >= 1 &&
			    info->Processor.GroupMask[0].Group == 0) {
				const DWORD_PTR mask = info->Processor.GroupMask[0].Mask & process_mask;
				if (__builtin_popcountll(static_cast<unsigned long long>(mask)) >= 2) {
					cores.push_back(mask);
				}
			}
			offset += info->Size;
		}
		// Skip the first core (interrupts and OS work land there); need two cores for ourselves
		// and plenty left for the rest.
		if (cores.size() < 6) {
			return result;
		}
		const DWORD_PTR cp_core       = cores[1];
		const DWORD_PTR recorder_core = cores[2];
		result.cp_cpu       = cp_core & (~cp_core + 1);
		result.recorder_cpu = recorder_core & (~recorder_core + 1);
		result.excluded     = (cp_core & ~result.cp_cpu) | (recorder_core & ~result.recorder_cpu);
		result.valid        = true;
		return result;
	}();
	return plan;
}

inline void ApplyProcessMask() {
	static const bool once = [] {
		const auto& plan = GetPlan();
		if (plan.valid) {
			DWORD_PTR process_mask = 0, system_mask = 0;
			if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
				SetProcessAffinityMask(GetCurrentProcess(), process_mask & ~plan.excluded);
			}
		}
		return true;
	}();
	(void)once;
}

} // namespace Detail
#endif

// Call from the thread itself, early. `which`: 0 = command processor, 1 = recorder.
inline void PinCurrentThread(int which) {
#ifdef _WIN32
	const auto& plan = Detail::GetPlan();
	if (!plan.valid) {
		return;
	}
	Detail::ApplyProcessMask();
	SetThreadAffinityMask(GetCurrentThread(), which == 0 ? plan.cp_cpu : plan.recorder_cpu);
#else
	(void)which;
#endif
}

} // namespace Libs::Graphics::HostCores

#endif
