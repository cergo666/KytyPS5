#include "graphics/guest_gpu/cpSampler.h"

#ifdef _WIN32

#include <windows.h>

#include <dbghelp.h>
#include <timeapi.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#pragma comment(lib, "dbghelp.lib")
#pragma comment(lib, "winmm.lib")

namespace Libs::Graphics {

namespace {

constexpr size_t kMaxDepth = 40;

struct Sample {
	std::array<uint64_t, kMaxDepth> pc;
	uint32_t                        depth = 0;
};

struct SymbolInfo {
	std::string function;
	std::string line;
};

class Sampler {
public:
	explicit Sampler(HANDLE thread, double window_seconds, const char* prefix)
	    : m_thread(thread), m_window_seconds(window_seconds), m_prefix(prefix) {
		m_samples.reserve(40000);
	}

	void Run() {
		timeBeginPeriod(1);
		SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES |
		              SYMOPT_INCLUDE_32BIT_MODULES);
		SymInitialize(GetCurrentProcess(), nullptr, TRUE);
		LARGE_INTEGER freq, window_start, now;
		QueryPerformanceFrequency(&freq);
		QueryPerformanceCounter(&window_start);
		for (;;) {
			Sleep(1);
			Capture();
			QueryPerformanceCounter(&now);
			const double elapsed =
			    static_cast<double>(now.QuadPart - window_start.QuadPart) / freq.QuadPart;
			if (elapsed >= m_window_seconds) {
				Report();
				m_samples.clear();
				QueryPerformanceCounter(&window_start);
			}
		}
	}

private:
	void Capture() {
		if (SuspendThread(m_thread) == static_cast<DWORD>(-1)) {
			return;
		}
		CONTEXT context {};
		context.ContextFlags = CONTEXT_FULL;
		Sample sample;
		if (GetThreadContext(m_thread, &context)) {
			// Unwind with the RtlVirtualUnwind tables only: no heap use while the thread is stopped.
			while (sample.depth < kMaxDepth && context.Rip != 0) {
				sample.pc[sample.depth++] = context.Rip;
				DWORD64 image_base = 0;
				auto* entry = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
				if (entry == nullptr) {
					break;
				}
				void*   handler_data = nullptr;
				DWORD64 frame        = 0;
				RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, entry, &context,
				                 &handler_data, &frame, nullptr);
			}
		}
		ResumeThread(m_thread);
		if (sample.depth != 0) {
			m_samples.push_back(sample);
		}
	}

	const SymbolInfo& Resolve(uint64_t pc) {
		auto it = m_cache.find(pc);
		if (it != m_cache.end()) {
			return it->second;
		}
		SymbolInfo info;
		alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 512];
		auto* symbol         = reinterpret_cast<SYMBOL_INFO*>(buffer);
		symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
		symbol->MaxNameLen   = 511;
		DWORD64 displacement = 0;
		if (SymFromAddr(GetCurrentProcess(), pc, &displacement, symbol)) {
			info.function = symbol->Name;
		} else {
			char text[32];
			std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(pc));
			info.function = text;
		}
		IMAGEHLP_LINE64 line {};
		line.SizeOfStruct = sizeof(line);
		DWORD line_disp   = 0;
		if (SymGetLineFromAddr64(GetCurrentProcess(), pc, &line_disp, &line)) {
			const char* file = line.FileName;
			if (const char* slash = std::strrchr(file, '\\')) {
				file = slash + 1;
			}
			info.line = std::string(file) + ":" + std::to_string(line.LineNumber);
		} else {
			info.line = info.function;
		}
		return m_cache.emplace(pc, std::move(info)).first->second;
	}

	static void Print(FILE* out, const char* title, std::vector<std::pair<std::string, uint32_t>>& rows,
	                  size_t total, size_t limit) {
		std::sort(rows.begin(), rows.end(),
		          [](const auto& a, const auto& b) { return a.second > b.second; });
		std::fprintf(out, "\n== %s ==\n", title);
		for (size_t i = 0; i < rows.size() && i < limit; ++i) {
			std::fprintf(out, "%6.2f%%  %7u  %s\n", 100.0 * rows[i].second / total, rows[i].second,
			             rows[i].first.c_str());
		}
	}

	// Base name of the module containing pc ("kyty_emulator.exe", "nvoglv64.dll", ...).
	const std::string& ModuleOf(uint64_t pc) {
		const auto base = SymGetModuleBase64(GetCurrentProcess(), pc);
		auto       it   = m_modules.find(base);
		if (it != m_modules.end()) {
			return it->second;
		}
		char path[MAX_PATH] = "?";
		if (base == 0 || GetModuleFileNameA(reinterpret_cast<HMODULE>(base), path, MAX_PATH) == 0) {
			std::snprintf(path, sizeof(path), "?");
		}
		const char* file = std::strrchr(path, '\\');
		return m_modules.emplace(base, file != nullptr ? file + 1 : path).first->second;
	}

	void Report() {
		if (m_samples.empty()) {
			return;
		}
		std::unordered_map<std::string, uint32_t> leaf_function, leaf_line, inclusive;
		// Time outside the emulator, keyed by the module the emulator called into and the emulator
		// function that made the call (for example "nvoglv64.dll <- CommitBindings").
		std::unordered_map<std::string, uint32_t> external_by_caller, leaf_module;
		static const std::string self_module = [] {
			char path[MAX_PATH] = "?";
			GetModuleFileNameA(nullptr, path, MAX_PATH);
			const char* file = std::strrchr(path, '\\');
			return std::string(file != nullptr ? file + 1 : path);
		}();
		for (const auto& sample: m_samples) {
			leaf_module[ModuleOf(sample.pc[0])]++;
			if (ModuleOf(sample.pc[0]) != self_module) {
				for (uint32_t i = 1; i < sample.depth; ++i) {
					if (ModuleOf(sample.pc[i]) == self_module) {
						external_by_caller[ModuleOf(sample.pc[i - 1]) + " <- " +
						                   Resolve(sample.pc[i]).function]++;
						break;
					}
				}
			}
			leaf_function[Resolve(sample.pc[0]).function]++;
			leaf_line[Resolve(sample.pc[0]).line]++;
			std::unordered_set<std::string> seen;
			for (uint32_t i = 0; i < sample.depth; ++i) {
				if (seen.insert(Resolve(sample.pc[i]).function).second) {
					inclusive[Resolve(sample.pc[i]).function]++;
				}
			}
		}
		char name[64];
		std::snprintf(name, sizeof(name), "%s_%u.txt", m_prefix.c_str(), m_report_index++);
		FILE* out = std::fopen(name, "w");
		if (out == nullptr) {
			return;
		}
		std::fprintf(out, "samples: %zu\n", m_samples.size());
		const auto dump = [&](const char* title, auto& map, size_t limit) {
			std::vector<std::pair<std::string, uint32_t>> rows(map.begin(), map.end());
			Print(out, title, rows, m_samples.size(), limit);
		};
		dump("time by leaf module", leaf_module, 20);
		dump("time outside the emulator: entered module <- emulator caller", external_by_caller, 60);
		dump("inclusive (function appears anywhere in the stack)", inclusive, 80);
		dump("self time by function", leaf_function, 60);
		dump("self time by source line", leaf_line, 80);
		std::fclose(out);
	}

	HANDLE                                m_thread;
	double                                m_window_seconds;
	std::string                           m_prefix;
	std::vector<Sample>                   m_samples;
	std::unordered_map<uint64_t, SymbolInfo> m_cache;
	std::unordered_map<uint64_t, std::string> m_modules;
	uint32_t                              m_report_index = 0;
};

} // namespace

void CpSamplerStartIfEnabled() {
	CpSamplerStartIfEnabled("KYTY_CP_SAMPLE", "cp_samples");
}

void CpSamplerStartIfEnabled(const char* env_name, const char* file_prefix) {
	const char* enabled = std::getenv(env_name);
	if (enabled == nullptr || enabled[0] != '1') {
		return;
	}
	HANDLE self = nullptr;
	if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &self,
	                     THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE,
	                     0)) {
		return;
	}
	double window = 15.0;
	if (const char* env = std::getenv("KYTY_CP_SAMPLE_WINDOW"); env != nullptr) {
		window = (std::max)(2.0, std::atof(env));
	}
	auto* sampler = new Sampler(self, window, file_prefix); // lives until the process exits
	std::thread([sampler] { sampler->Run(); }).detach();
}

} // namespace Libs::Graphics

#else

namespace Libs::Graphics {
void CpSamplerStartIfEnabled() {}
void CpSamplerStartIfEnabled(const char*, const char*) {}
} // namespace Libs::Graphics

#endif
