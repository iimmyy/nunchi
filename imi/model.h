// imi.cpp: one GGUF file read into memory, tensors looked up by name.
#pragma once
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include <string>

// the CPU's fast-kernel buffer type (CPU_REPACK), nullptr if this ggml has none
ggml_backend_buffer_type_t imi_repack_buft();

struct imi_file {
    gguf_context * meta = nullptr;
    ggml_context * ctx  = nullptr; // tensor headers; data lives in `data`
    void *         data = nullptr; // every tensor's bytes, one block (2 MB huge pages where the OS allows)
    size_t         size = 0;

    ggml_backend_buffer_t repacked = nullptr; // 4-bit tables in the CPU's fast-kernel layout (CPU_REPACK)

    // repack: put 4-bit matrix tables in the layout ggml's fastest CPU kernels read (as llama.cpp does at load)
    bool load(const std::string & path, bool repack = false);
    ggml_tensor * get(const std::string & name, bool required = true) const;
    int64_t       key_i(const std::string & name, int64_t def) const;
    float         key_f(const std::string & name, float def) const;
    ~imi_file();
};
