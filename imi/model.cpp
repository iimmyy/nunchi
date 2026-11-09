#include "model.h"
#include <cstdio>
#include <cstdlib>

bool imi_file::load(const std::string & path) {
    gguf_init_params p = { /*no_alloc*/ false, /*ctx*/ &ctx };
    meta = gguf_init_from_file(path.c_str(), p);
    if (!meta) {
        fprintf(stderr, "imi: cannot read %s\n", path.c_str());
        return false;
    }
    return true;
}

ggml_tensor * imi_file::get(const std::string & name, bool required) const {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t && required) {
        fprintf(stderr, "imi: missing tensor %s\n", name.c_str());
        exit(1);
    }
    return t;
}

imi_file::~imi_file() {
    if (ctx) ggml_free(ctx);
    if (meta) gguf_free(meta);
}
