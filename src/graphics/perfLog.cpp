#include "graphics/perfLog.h"

#include <atomic>
#include <cstdlib>

namespace Libs::Graphics::PerfLog {

namespace {
std::atomic<unsigned long long> g_frames {0};
std::atomic<unsigned>           g_shadow_every {1};
}

unsigned RecommendedShadowEvery() {
	return g_shadow_every.load(std::memory_order_relaxed);
}

void AdaptiveQuality(double fps) {
	static const bool enabled = [] {
		const char* env = std::getenv("KYTY_ADAPTIVE_SHADOWS");
		return env != nullptr && env[0] == '1'; // off by default: skipping shadow updates flickers
	}();
	if (!enabled || fps < 5.0) { // loading and menus' transitions say nothing about the scene
		return;
	}
	static const double low = [] {
		const char* env = std::getenv("KYTY_ADAPTIVE_LOW");
		return env != nullptr ? std::atof(env) : 22.0;
	}();
	static const double high = [] {
		const char* env = std::getenv("KYTY_ADAPTIVE_HIGH");
		return env != nullptr ? std::atof(env) : 33.0;
	}();
	static unsigned level = 1, low_count = 0, high_count = 0;
	if (fps < low) {
		low_count++;
		high_count = 0;
	} else if (fps >= high) {
		high_count++;
		low_count = 0;
	} else {
		low_count = high_count = 0;
	}
	if (level == 1 && low_count >= 2) {
		level = 2;
	} else if (level == 2 && high_count >= 2) {
		level = 1;
	}
	g_shadow_every.store(level, std::memory_order_relaxed);
}

void CountFrame() {
	g_frames.fetch_add(1, std::memory_order_relaxed);
}

unsigned long long FrameCount() {
	return g_frames.load(std::memory_order_relaxed);
}

} // namespace Libs::Graphics::PerfLog

#ifdef _WIN32

#include <windows.h>

#include <psapi.h>
#include <tlhelp32.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "psapi.lib")

namespace Libs::Graphics::PerfLog {

namespace {

struct TrackedThread {
	std::string label;
	HANDLE      handle = nullptr;
	double      last_cpu_seconds = 0;
};

std::mutex                 g_threads_mutex;
std::vector<TrackedThread> g_threads;

const char* ThreadLabel(DWORD id) {
	std::lock_guard lock(g_threads_mutex);
	for (const auto& thread: g_threads) {
		if (GetThreadId(thread.handle) == id) return thread.label.c_str();
	}
	return nullptr;
}

double FileTimeSeconds(const FILETIME& ft) {
	ULARGE_INTEGER v;
	v.LowPart  = ft.dwLowDateTime;
	v.HighPart = ft.dwHighDateTime;
	return static_cast<double>(v.QuadPart) / 1e7;
}

double ThreadCpuSeconds(HANDLE thread) {
	FILETIME create, exit_time, kernel, user;
	if (!GetThreadTimes(thread, &create, &exit_time, &kernel, &user)) {
		return 0;
	}
	return FileTimeSeconds(kernel) + FileTimeSeconds(user);
}

// Per-thread CPU census (perf_threads.csv: t_s,tid,name,pct), threads above 2% of a core.
const char* ThreadLabel(DWORD id);

void WriteThreadCensus(FILE* out, double t_s, double elapsed,
                       std::vector<std::pair<DWORD, double>>& previous) {
	const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snapshot == INVALID_HANDLE_VALUE) {
		return;
	}
	THREADENTRY32 entry {};
	entry.dwSize = sizeof(entry);
	const DWORD pid = GetCurrentProcessId();
	std::vector<std::pair<DWORD, double>> now;
	if (Thread32First(snapshot, &entry)) {
		do {
			if (entry.th32OwnerProcessID != pid) {
				continue;
			}
			HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
			if (thread == nullptr) {
				continue;
			}
			FILETIME create, exit_time, kernel, user;
			if (GetThreadTimes(thread, &create, &exit_time, &kernel, &user)) {
				const double cpu = FileTimeSeconds(kernel) + FileTimeSeconds(user);
				now.emplace_back(entry.th32ThreadID, cpu);
				double before = 0;
				for (const auto& [id, value]: previous) {
					if (id == entry.th32ThreadID) {
						before = value;
						break;
					}
				}
				const double pct = 100.0 * (cpu - before) / elapsed;
				if (pct >= 2.0) {
					char       name[128] = "";
					PWSTR      description = nullptr;
					if (SUCCEEDED(GetThreadDescription(thread, &description)) && description != nullptr) {
						WideCharToMultiByte(CP_UTF8, 0, description, -1, name, sizeof(name) - 1, nullptr, nullptr);
						LocalFree(description);
					}
					if (name[0] == 0) {
						if (const char* label = ThreadLabel(entry.th32ThreadID); label != nullptr) {
							std::snprintf(name, sizeof(name), "%s", label);
						}
					}
					std::fprintf(out, "%.0f,%lu,%s,%.0f\n", t_s, static_cast<unsigned long>(entry.th32ThreadID),
					             name, pct);
				}
			}
			CloseHandle(thread);
		} while (Thread32Next(snapshot, &entry));
	}
	CloseHandle(snapshot);
	previous = std::move(now);
	std::fflush(out);
}


// Minimal NVML binding (nvml.dll ships with the NVIDIA driver); every call may be unavailable.
struct NvmlUtilization {
	unsigned int gpu;
	unsigned int memory;
};
struct NvmlMemory {
	unsigned long long total, free, used;
};

struct Nvml {
	HMODULE module = nullptr;
	void*   device = nullptr;
	int (*get_utilization)(void*, NvmlUtilization*) = nullptr;
	int (*get_memory)(void*, NvmlMemory*)           = nullptr;

	void Init() {
		module = LoadLibraryA("nvml.dll");
		if (module == nullptr) {
			return;
		}
		auto init   = reinterpret_cast<int (*)()>(GetProcAddress(module, "nvmlInit_v2"));
		auto handle = reinterpret_cast<int (*)(unsigned int, void**)>(
		    GetProcAddress(module, "nvmlDeviceGetHandleByIndex_v2"));
		get_utilization = reinterpret_cast<int (*)(void*, NvmlUtilization*)>(
		    GetProcAddress(module, "nvmlDeviceGetUtilizationRates"));
		get_memory = reinterpret_cast<int (*)(void*, NvmlMemory*)>(
		    GetProcAddress(module, "nvmlDeviceGetMemoryInfo"));
		if (init == nullptr || handle == nullptr || init() != 0 || handle(0, &device) != 0) {
			get_utilization = nullptr;
			get_memory      = nullptr;
		}
	}
};

void Run() {
	Nvml nvml;
	nvml.Init();
	FILE* out = std::fopen("perf_log.csv", "w");
	if (out == nullptr) {
		return;
	}
	FILE* census = std::fopen("perf_threads.csv", "w");
	if (census != nullptr) {
		std::fprintf(census, "t_s,tid,name,pct\n");
	}
	std::vector<std::pair<DWORD, double>> census_previous;
	std::fprintf(out, "t_s,fps,cpu_total_pct,proc_cpu_pct,cp_thread_pct,gpu_pct,vram_used_mb,ram_mb\n");
	std::fflush(out);

	const int cpus = static_cast<int>(std::thread::hardware_concurrency());
	FILETIME  idle0, kernel0, user0;
	GetSystemTimes(&idle0, &kernel0, &user0);
	FILETIME create, exit_time, pk0, pu0;
	GetProcessTimes(GetCurrentProcess(), &create, &exit_time, &pk0, &pu0);
	auto        last      = std::chrono::steady_clock::now();
	const auto  start     = last;
	unsigned long long last_frames = g_frames.load(std::memory_order_relaxed);
	double      last_idle   = FileTimeSeconds(idle0);
	double      last_system = FileTimeSeconds(kernel0) + FileTimeSeconds(user0);
	double      last_proc   = FileTimeSeconds(pk0) + FileTimeSeconds(pu0);

	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(2));
		const auto   now     = std::chrono::steady_clock::now();
		const double elapsed = std::chrono::duration<double>(now - last).count();
		last                 = now;

		const auto frames = g_frames.load(std::memory_order_relaxed);
		const double fps  = static_cast<double>(frames - last_frames) / elapsed;
		last_frames       = frames;
		AdaptiveQuality(fps);

		FILETIME idle, kernel, user;
		GetSystemTimes(&idle, &kernel, &user);
		const double idle_s   = FileTimeSeconds(idle);
		const double system_s = FileTimeSeconds(kernel) + FileTimeSeconds(user);
		const double total_delta = system_s - last_system;
		const double cpu_total   = total_delta > 0 ? 100.0 * (1.0 - (idle_s - last_idle) / total_delta) : 0;
		last_idle   = idle_s;
		last_system = system_s;

		FILETIME pk, pu;
		GetProcessTimes(GetCurrentProcess(), &create, &exit_time, &pk, &pu);
		const double proc_s   = FileTimeSeconds(pk) + FileTimeSeconds(pu);
		const double proc_cpu = 100.0 * (proc_s - last_proc) / (elapsed * (cpus > 0 ? cpus : 1));
		last_proc             = proc_s;

		double cp_thread = 0;
		{
			std::lock_guard lock(g_threads_mutex);
			for (auto& thread: g_threads) {
				const double cpu = ThreadCpuSeconds(thread.handle);
				if (thread.label == "cp") {
					cp_thread = 100.0 * (cpu - thread.last_cpu_seconds) / elapsed;
				}
				thread.last_cpu_seconds = cpu;
			}
		}

		NvmlUtilization util {};
		NvmlMemory      mem {};
		const bool      have_util = nvml.get_utilization != nullptr && nvml.get_utilization(nvml.device, &util) == 0;
		const bool      have_mem  = nvml.get_memory != nullptr && nvml.get_memory(nvml.device, &mem) == 0;

		PROCESS_MEMORY_COUNTERS counters {};
		GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters));

		std::fprintf(out, "%.0f,%.1f,%.0f,%.0f,%.0f,%d,%d,%d\n",
		             std::chrono::duration<double>(now - start).count(), fps, cpu_total, proc_cpu,
		             cp_thread, have_util ? static_cast<int>(util.gpu) : -1,
		             have_mem ? static_cast<int>(mem.used / (1024 * 1024)) : -1,
		             static_cast<int>(counters.WorkingSetSize / (1024 * 1024)));
		std::fflush(out);
		if (census != nullptr) {
			WriteThreadCensus(census, std::chrono::duration<double>(now - start).count(), elapsed,
			                  census_previous);
		}
	}
}

} // namespace

void RegisterThisThread(const char* label) {
	HANDLE self = nullptr;
	if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &self,
	                     THREAD_QUERY_INFORMATION, FALSE, 0)) {
		return;
	}
	std::lock_guard lock(g_threads_mutex);
	g_threads.push_back({label, self, ThreadCpuSeconds(self)});
}

void Start() {
	const char* env = std::getenv("KYTY_PERF_LOG");
	if (env != nullptr && env[0] == '0') {
		return;
	}
	static std::once_flag once;
	std::call_once(once, [] { std::thread(Run).detach(); });
}

} // namespace Libs::Graphics::PerfLog

#else

namespace Libs::Graphics::PerfLog {
void Start() {}
void RegisterThisThread(const char*) {}
} // namespace Libs::Graphics::PerfLog

#endif
