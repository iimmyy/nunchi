#include "model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#ifdef __linux__
#include <sys/mman.h>
#endif
#ifdef _WIN32
#include <malloc.h>
// Windows: 2 MB-aligned memory and 64-bit file offsets have their own names
static int posix_memalign(void ** p, size_t a, size_t n) { *p = _aligned_malloc(n, a); return *p ? 0 : 1; }
static void aligned_free(void * p) { _aligned_free(p); }
#define fseek64 _fseeki64
#else
static void aligned_free(void * p) { free(p); }
#define fseek64 fseeko
#endif

ggml_backend_buffer_type_t imi_repack_buft() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) return nullptr;
    auto fn = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    if (!fn) return nullptr;
    for (ggml_backend_buffer_type_t * b = fn(dev); b && *b; b++) {
        if (strcmp(ggml_backend_buft_name(*b), "CPU_REPACK") == 0) return *b;
    }
    return nullptr;
}

// a 2-D 4-bit matrix table that the AVX2 / NEON 8-row interleaved kernels can take
static bool repackable(const ggml_tensor * t) {
    return t->type == GGML_TYPE_Q4_0 && ggml_n_dims(t) == 2 && t->ne[1] % 8 == 0 && t->ne[0] % 32 == 0
        && strstr(t->name, "token_embd") == nullptr;
}

static void advise_huge(void * p, size_t n) {
#ifdef MADV_HUGEPAGE
    madvise(p, n, MADV_HUGEPAGE); // Immy 2026-11-08: tiny on the floor box (+0.5%), free, so in
#else
    (void) p; (void) n;
#endif
}

bool imi_file::load(const std::string & path, bool repack) {
    gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &ctx };
    meta = gguf_init_from_file(path.c_str(), p);
    if (!meta) {
        fprintf(stderr, "imi: cannot read %s\n", path.c_str());
        return false;
    }
    const int64_t nt = gguf_get_n_tensors(meta);
    ggml_backend_buffer_type_t rb = repack ? imi_repack_buft() : nullptr;
    std::vector<ggml_tensor *> plain, fast;
    size_t plain_size = 0, fast_size = 0;
    for (int64_t i = 0; i < nt; i++) {
        ggml_tensor * t = ggml_get_tensor(ctx, gguf_get_tensor_name(meta, i));
        if (rb && repackable(t)) { fast.push_back(t); fast_size += GGML_PAD(ggml_backend_buft_get_alloc_size(rb, t), 64); }
        else                     { plain.push_back(t); plain_size += GGML_PAD(ggml_nbytes(t), 64); }
    }
    const size_t huge = 2u << 20;
    const size_t alloc = (plain_size + huge - 1) / huge * huge + huge;
    if (posix_memalign(&data, huge, alloc) != 0) {
        fprintf(stderr, "imi: out of memory for %s (%zu MB)\n", path.c_str(), alloc >> 20);
        return false;
    }
    advise_huge(data, alloc);
    size = plain_size;
    ggml_tallocr ta{};
    if (!fast.empty()) {
        repacked = ggml_backend_buft_alloc_buffer(rb, fast_size + 4096);
        advise_huge(ggml_backend_buffer_get_base(repacked), ggml_backend_buffer_get_size(repacked));
        ggml_backend_buffer_set_usage(repacked, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ta = ggml_tallocr_new(repacked);
    }

    FILE * f = ggml_fopen(path.c_str(), "rb");   // UTF-8 names, also on Windows (like gguf)
    if (!f) return false;
    const size_t off0 = gguf_get_data_offset(meta);
    auto read_at = [&](const ggml_tensor * t, void * dst) {
        const int64_t i = gguf_find_tensor(meta, t->name);
        return fseek64(f, (int64_t) (off0 + gguf_get_tensor_offset(meta, i)), SEEK_SET) == 0 && fread(dst, 1, ggml_nbytes(t), f) == ggml_nbytes(t);
    };
    size_t o = 0;
    for (ggml_tensor * t : plain) {
        t->data = (char *) data + o;
        o += GGML_PAD(ggml_nbytes(t), 64);
        if (!read_at(t, t->data)) { fprintf(stderr, "imi: short read on %s\n", t->name); fclose(f); return false; }
    }
    std::vector<uint8_t> tmp;
    for (ggml_tensor * t : fast) {
        tmp.resize(ggml_nbytes(t));
        if (!read_at(t, tmp.data())) { fprintf(stderr, "imi: short read on %s\n", t->name); fclose(f); return false; }
        ggml_tallocr_alloc(&ta, t);
        ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size()); // the repack happens here
    }
    fclose(f);
    if (!fast.empty()) {
        fprintf(stderr, "imi: %zu tables in the fast-kernel layout (%zu MB), %zu as stored (%zu MB)\n",
                fast.size(), fast_size >> 20, plain.size(), plain_size >> 20);
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

int64_t imi_file::key_i(const std::string & name, int64_t def) const {
    const int64_t k = gguf_find_key(meta, name.c_str());
    if (k < 0) return def;
    switch (gguf_get_kv_type(meta, k)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(meta, k);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(meta, k);
        case GGUF_TYPE_UINT64: return (int64_t) gguf_get_val_u64(meta, k);
        default: return def;
    }
}

float imi_file::key_f(const std::string & name, float def) const {
    const int64_t k = gguf_find_key(meta, name.c_str());
    return k < 0 ? def : gguf_get_val_f32(meta, k);
}

// a tensor's bytes are in the file's block (not kept yet, not in the repacked buffer)
static bool in_block(const ggml_tensor * t, const void * data, size_t size) {
    return data && t->data && (const char *) t->data >= (const char *) data && (const char *) t->data < (const char *) data + size;
}

ggml_tensor * imi_file::keep(const std::string & name, bool required) {
    ggml_tensor * t = get(name, required);
    if (!t || !in_block(t, data, size)) return t;   // missing (and not required), or already its own
    void * p = nullptr;
    if (posix_memalign(&p, 64, GGML_PAD(ggml_nbytes(t), 64)) != 0) {
        fprintf(stderr, "imi: out of memory keeping %s\n", t->name);
        exit(1);
    }
    memcpy(p, t->data, ggml_nbytes(t));
    t->data = p;
    kept.push_back(p);
    return t;
}

void imi_file::drop_data() {
    if (!data) return;
    size_t mine = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (in_block(t, data, size)) t->data = nullptr;
        else if (t->data) mine += ggml_nbytes(t);
    }
    fprintf(stderr, "imi: %zu MB of the file let go, %zu MB kept\n", size >> 20, mine >> 20);
    aligned_free(data);
    data = nullptr;
    size = 0;
}

imi_file::~imi_file() {
    if (ctx) ggml_free(ctx);
    if (meta) gguf_free(meta);
    if (repacked) ggml_backend_buffer_free(repacked);
    aligned_free(data);
    for (void * p : kept) aligned_free(p);
}
