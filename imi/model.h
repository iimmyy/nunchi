// imi.cpp: one GGUF file loaded into memory, tensors looked up by name.
#pragma once
#include "ggml.h"
#include "gguf.h"
#include <string>

struct imi_file {
    gguf_context * meta = nullptr;
    ggml_context * ctx  = nullptr; // owns every tensor's data

    bool load(const std::string & path);
    ggml_tensor * get(const std::string & name, bool required = true) const;
    ~imi_file();
};
