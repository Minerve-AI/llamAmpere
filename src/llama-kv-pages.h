#pragma once

#include <cstdint>
#include <vector>
#include <cassert>
#include <cstring>

//
// Paged KV Cache - Page Table Management
//
// The KV cache is organized into fixed-size pages (default: 64 tokens/page).
// Each sequence has a page table mapping logical token positions to physical pages.
// This eliminates fragmentation and allows efficient concurrent sequence processing.
//

struct llama_kv_page_table {
    static constexpr uint32_t DEFAULT_PAGE_SIZE = 64;

    uint32_t page_size;      // tokens per page
    uint32_t n_pages;        // total physical pages in the pool
    uint32_t n_seq;          // max number of sequences

    // page_table[seq_id * n_pages + page_idx] = physical_page_idx + 1 (0 = unallocated)
    // Using +1 to distinguish "unallocated" (0) from "page 0" (1)
    std::vector<uint32_t> page_table;

    // Stack of free physical page indices
    std::vector<uint32_t> free_pages;

    // Number of tokens allocated per sequence
    std::vector<uint32_t> seq_len;

    // GPU-side page table (device memory)
    uint32_t * d_page_table = nullptr;

    llama_kv_page_table() : page_size(DEFAULT_PAGE_SIZE), n_pages(0), n_seq(0) {}

    llama_kv_page_table(uint32_t n_pages_, uint32_t n_seq_, uint32_t page_size_ = DEFAULT_PAGE_SIZE)
        : page_size(page_size_), n_pages(n_pages_), n_seq(n_seq_) {
        init();
    }

    ~llama_kv_page_table() {
        // GPU memory freed by caller (cudaFree)
    }

    void init() {
        page_table.resize(n_seq * n_pages, 0);
        free_pages.reserve(n_pages);
        for (uint32_t i = n_pages; i > 0; --i) {
            free_pages.push_back(i - 1);
        }
        seq_len.resize(n_seq, 0);
    }

    // Allocate a new page for a sequence
    // Returns physical page index, or UINT32_MAX if pool is exhausted
    uint32_t alloc_page(uint32_t seq_id) {
        assert(seq_id < n_seq);
        if (free_pages.empty()) {
            return UINT32_MAX;
        }
        uint32_t phys_page = free_pages.back();
        free_pages.pop_back();

        uint32_t page_idx = seq_len[seq_id] / page_size;
        if (page_idx >= n_pages) {
            // Out of pages for this sequence
            free_pages.push_back(phys_page);
            return UINT32_MAX;
        }

        page_table[seq_id * n_pages + page_idx] = phys_page + 1; // +1 encoding
        return phys_page;
    }

    // Extend a sequence by n_tokens, allocating pages as needed
    // Returns true if all pages were allocated successfully
    bool extend_seq(uint32_t seq_id, uint32_t n_new_tokens) {
        assert(seq_id < n_seq);
        uint32_t old_len = seq_len[seq_id];
        uint32_t new_len = old_len + n_new_tokens;

        uint32_t old_n_pages = old_len / page_size;
        uint32_t new_n_pages = (new_len + page_size - 1) / page_size;

        for (uint32_t p = old_n_pages; p < new_n_pages; ++p) {
            uint32_t phys = alloc_page(seq_id);
            if (phys == UINT32_MAX) {
                // Rollback: we can't partially extend
                return false;
            }
        }

        seq_len[seq_id] = new_len;
        return true;
    }

    // Free all pages for a sequence
    void free_seq(uint32_t seq_id) {
        assert(seq_id < n_seq);
        uint32_t n_used = (seq_len[seq_id] + page_size - 1) / page_size;
        for (uint32_t i = 0; i < n_used; ++i) {
            uint32_t val = page_table[seq_id * n_pages + i];
            if (val != 0) {
                free_pages.push_back(val - 1);
                page_table[seq_id * n_pages + i] = 0;
            }
        }
        seq_len[seq_id] = 0;
    }

    // Shrink a sequence, freeing trailing pages
    void shrink_seq(uint32_t seq_id, uint32_t new_len) {
        assert(seq_id < n_seq);
        assert(new_len <= seq_len[seq_id]);

        uint32_t old_n_pages = (seq_len[seq_id] + page_size - 1) / page_size;
        uint32_t new_n_pages = (new_len + page_size - 1) / page_size;

        for (uint32_t p = new_n_pages; p < old_n_pages; ++p) {
            uint32_t val = page_table[seq_id * n_pages + p];
            if (val != 0) {
                free_pages.push_back(val - 1);
                page_table[seq_id * n_pages + p] = 0;
            }
        }
        seq_len[seq_id] = new_len;
    }

    // Get the physical token index for a logical (seq_id, token_pos)
    // Returns: phys_page * page_size + offset_in_page
    inline uint32_t get_phys_token_idx(uint32_t seq_id, uint32_t token_pos) const {
        assert(seq_id < n_seq);
        assert(token_pos < seq_len[seq_id]);
        uint32_t page_idx = token_pos / page_size;
        uint32_t offset   = token_pos % page_size;
        uint32_t val      = page_table[seq_id * n_pages + page_idx];
        assert(val != 0);
        return (val - 1) * page_size + offset;
    }

    // Get the physical token index (device-safe, no asserts)
    inline uint32_t get_phys_token_idx_fast(uint32_t seq_id, uint32_t token_pos) const {
        uint32_t page_idx = token_pos / page_size;
        uint32_t offset   = token_pos % page_size;
        uint32_t val      = page_table[seq_id * n_pages + page_idx];
        return (val - 1) * page_size + offset;
    }

    // Number of free pages
    inline uint32_t n_free() const { return free_pages.size(); }

    // Number of used pages
    inline uint32_t n_used() const { return n_pages - free_pages.size(); }

    // Total capacity in tokens
    inline uint32_t capacity_tokens() const { return n_pages * page_size; }

    // Get raw data pointer (for GPU upload)
    inline const uint32_t * data() const { return page_table.data(); }

    // Size in bytes
    inline size_t size_bytes() const { return page_table.size() * sizeof(uint32_t); }

    // Upload page table to GPU (caller manages the device memory)
    // d_page_table must be pre-allocated with cudaMalloc
    void upload_to_gpu() {
        if (d_page_table) {
            // cudaMemcpyAsync(d_page_table, page_table.data(), size_bytes(), cudaMemcpyHostToDevice, stream)
            // Actual copy done by caller with proper stream
            std::memcpy(d_page_table, page_table.data(), size_bytes());
        }
    }

    // Stats
    void print_stats() const {
        printf("[paged-kv] pages: %u total, %u used, %u free | page_size: %u | capacity: %u tokens\n",
               n_pages, n_used(), n_free(), page_size, capacity_tokens());
        for (uint32_t s = 0; s < n_seq; ++s) {
            if (seq_len[s] > 0) {
                printf("[paged-kv]   seq %u: %u tokens (%u pages)\n",
                       s, seq_len[s], (seq_len[s] + page_size - 1) / page_size);
            }
        }
    }
};

//
// GPU page table wrapper
//
struct llama_kv_page_table_gpu {
    uint32_t * d_data = nullptr;   // device memory
    uint32_t   size   = 0;         // size in bytes

    bool alloc(uint32_t n_elements) {
        size = n_elements * sizeof(uint32_t);
        // cudaMalloc(d_data, size) - done by caller
        return d_data != nullptr;
    }

    void free() {
        // cudaFree(d_data) - done by caller
        d_data = nullptr;
        size = 0;
    }

    ~llama_kv_page_table_gpu() { free(); }
};
