#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_

#include "common/assert.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>

namespace Libs::Graphics {

class RangeSet final {
public:
	// Optional coarse filter for sets that are queried far more often than they change: one bit
	// per 64 KiB block says whether any range touches it, so most negative Intersects() queries
	// avoid walking the tree. Addresses at or above kFilterLimit always take the exact path.
	void EnableFilter() {
		if (m_filter.empty()) {
			m_filter.assign(kFilterWords, 0);
			for (const auto& [begin, last]: m_ranges) {
				FilterSet(begin, last);
			}
		}
	}

	void Add(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		FilterSet(address, end);
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second >= address) {
			it = std::prev(it);
		}
		uint64_t begin = address;
		uint64_t last  = end;
		while (it != m_ranges.end() && it->first <= last) {
			begin = std::min(begin, it->first);
			last  = std::max(last, it->second);
			it    = m_ranges.erase(it);
		}
		m_ranges.emplace(begin, last);
	}

	void Subtract(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		SubtractExact(address, end);
		FilterAfterSubtract(address, end);
	}

private:
	void SubtractExact(uint64_t address, uint64_t end) {
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			it = std::prev(it);
		}
		while (it != m_ranges.end() && it->first < end) {
			const auto begin = it->first;
			const auto last  = it->second;
			it               = m_ranges.erase(it);
			if (begin < address) {
				m_ranges.emplace(begin, address);
			}
			if (last > end) {
				m_ranges.emplace(end, last);
				break;
			}
		}
	}

public:
	void Clear() {
		m_ranges.clear();
		std::fill(m_filter.begin(), m_filter.end(), 0);
	}

	template <typename Func>
	void ForEach(Func&& func) const {
		for (const auto& [begin, end]: m_ranges) {
			func(begin, end);
		}
	}

	// Lowest start and highest end of all ranges; false when the set is empty.
	[[nodiscard]] bool Bounds(uint64_t& low, uint64_t& high) const {
		if (m_ranges.empty()) {
			return false;
		}
		low  = m_ranges.begin()->first;
		high = m_ranges.rbegin()->second;
		return true;
	}

	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		if (!FilterMayIntersect(address, end)) {
			return false;
		}
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			return true;
		}
		return it != m_ranges.end() && it->first < end;
	}

	// The range holding all of [address, address + size), as [begin, last).
	[[nodiscard]] bool FindContaining(uint64_t address, uint64_t size, uint64_t& begin,
	                                  uint64_t& last) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
		if (it == m_ranges.begin()) {
			return false;
		}
		--it;
		if (it->first > address || it->second < end) {
			return false;
		}
		begin = it->first;
		last  = it->second;
		return true;
	}

	[[nodiscard]] bool Contains(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
		if (it == m_ranges.begin()) {
			return false;
		}
		--it;
		return it->first <= address && it->second >= end;
	}

	template <typename Func>
	void ForEachInRange(uint64_t address, uint64_t size, Func&& func) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
		if (it != m_ranges.begin()) {
			--it;
		}
		for (; it != m_ranges.end() && it->first < end; ++it) {
			const auto begin = std::max(address, it->first);
			const auto last  = std::min(end, it->second);
			if (begin < last) {
				func(begin, last);
			}
		}
	}

	[[nodiscard]] bool Empty() const { return m_ranges.empty(); }

private:
	static constexpr uint32_t kBlockShift  = 16;
	static constexpr uint64_t kFilterLimit = 1ull << 42;
	static constexpr size_t   kFilterWords = static_cast<size_t>(kFilterLimit >> kBlockShift) / 64;

	[[nodiscard]] bool FilterBit(uint64_t block) const {
		return (m_filter[block >> 6] >> (block & 63)) & 1u;
	}
	void FilterFill(uint64_t first, uint64_t last, bool value) { // blocks [first, last]
		for (uint64_t block = first; block <= last;) {
			if ((block & 63) == 0 && last - block >= 63) {
				m_filter[block >> 6] = value ? ~0ull : 0ull;
				block += 64;
			} else {
				const uint64_t bit = 1ull << (block & 63);
				if (value) m_filter[block >> 6] |= bit; else m_filter[block >> 6] &= ~bit;
				++block;
			}
		}
	}
	void FilterSet(uint64_t address, uint64_t end) {
		if (m_filter.empty() || address >= kFilterLimit) {
			return;
		}
		FilterFill(address >> kBlockShift, (std::min(end, kFilterLimit) - 1) >> kBlockShift, true);
	}
	// Whether the exact set intersects one whole block, ignoring the filter.
	[[nodiscard]] bool BlockIntersects(uint64_t block) const {
		const uint64_t begin = block << kBlockShift;
		const uint64_t end   = begin + (1ull << kBlockShift);
		auto           it    = m_ranges.lower_bound(begin);
		if (it != m_ranges.begin() && std::prev(it)->second > begin) {
			return true;
		}
		return it != m_ranges.end() && it->first < end;
	}
	void FilterAfterSubtract(uint64_t address, uint64_t end) {
		if (m_filter.empty() || address >= kFilterLimit) {
			return;
		}
		end = std::min(end, kFilterLimit);
		const uint64_t first = address >> kBlockShift;
		const uint64_t last  = (end - 1) >> kBlockShift;
		for (uint64_t block = first; block <= last; ++block) {
			const bool whole = (block << kBlockShift) >= address &&
			                   ((block + 1) << kBlockShift) <= end;
			if (whole) {
				// Interior blocks lie entirely inside the removed range.
				const uint64_t run_end = std::min<uint64_t>(last, (end >> kBlockShift) - 1);
				FilterFill(block, run_end, false);
				block = run_end;
			} else {
				const bool on = BlockIntersects(block);
				FilterFill(block, block, on);
			}
		}
	}
	[[nodiscard]] bool FilterMayIntersect(uint64_t address, uint64_t end) const {
		if (m_filter.empty() || end > kFilterLimit) {
			return true;
		}
		const uint64_t first = address >> kBlockShift;
		const uint64_t last  = (end - 1) >> kBlockShift;
		if (last - first > 63) {
			return true;
		}
		for (uint64_t block = first; block <= last; ++block) {
			if (FilterBit(block)) {
				return true;
			}
		}
		return false;
	}

	static uint64_t End(uint64_t address, uint64_t size) {
		if (size == 0 || size > UINT64_MAX - address) {
			EXIT("invalid range-set address or size\n");
		}
		return address + size;
	}

	std::map<uint64_t, uint64_t> m_ranges;
	std::vector<uint64_t>        m_filter;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
