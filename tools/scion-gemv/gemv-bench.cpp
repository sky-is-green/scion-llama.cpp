// scion-gemv-bench: cold-cache batch-1 GEMV timing for one weight shape.
//
// Why this exists: test-backend-ops' MUL_MAT shapes fit in the 96 MB Infinity
// Cache, so they measure cache bandwidth (1.8 TB/s) instead of the DRAM stream
// a real model sees. This allocates `-r` distinct copies of the matrix and
// rotates through them, so every call reads cold weights, and reports GB/s of
// weight bytes read. It is the ceiling calibration for the decode path.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

static ggml_type type_by_name(const char * s) {
    if (!strcmp(s, "q4_0"))  return GGML_TYPE_Q4_0;
    if (!strcmp(s, "q4_1"))  return GGML_TYPE_Q4_1;
    if (!strcmp(s, "q5_0"))  return GGML_TYPE_Q5_0;
    if (!strcmp(s, "q5_1"))  return GGML_TYPE_Q5_1;
    if (!strcmp(s, "q8_0"))  return GGML_TYPE_Q8_0;
    if (!strcmp(s, "q2_K"))  return GGML_TYPE_Q2_K;
    if (!strcmp(s, "q3_K"))  return GGML_TYPE_Q3_K;
    if (!strcmp(s, "q4_K"))  return GGML_TYPE_Q4_K;
    if (!strcmp(s, "q5_K"))  return GGML_TYPE_Q5_K;
    if (!strcmp(s, "q6_K"))  return GGML_TYPE_Q6_K;
    if (!strcmp(s, "f16"))   return GGML_TYPE_F16;
    if (!strcmp(s, "bf16"))  return GGML_TYPE_BF16;
    if (!strcmp(s, "q2_0_g128"))  return GGML_TYPE_Q2_0_G128;
    if (!strcmp(s, "tq1_0_g128")) return GGML_TYPE_TQ1_0_G128;
    return GGML_TYPE_COUNT;
}

int main(int argc, char ** argv) {
    const char * dev_name = "ROCm0";
    const char * tname    = "q6_K";
    int64_t k = 2560;   // n_in
    int64_t m = 6144;   // n_out
    int     rot = 8;    // distinct matrices, to defeat the last-level cache
    int     iters = 12;

    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char * what) {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", what); exit(1); }
            return argv[++i];
        };
        if      (!strcmp(argv[i], "-t")) tname = need("-t");
        else if (!strcmp(argv[i], "-k")) k = atoll(need("-k"));
        else if (!strcmp(argv[i], "-m")) m = atoll(need("-m"));
        else if (!strcmp(argv[i], "-r")) rot = atoi(need("-r"));
        else if (!strcmp(argv[i], "-i")) iters = atoi(need("-i"));
        else if (!strcmp(argv[i], "-d")) dev_name = need("-d");
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }

    const ggml_type type = type_by_name(tname);
    if (type == GGML_TYPE_COUNT) { fprintf(stderr, "unknown type %s\n", tname); return 1; }

    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name);
    if (!dev) { fprintf(stderr, "no device %s\n", dev_name); return 1; }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) { fprintf(stderr, "cannot init %s\n", dev_name); return 1; }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    ggml_init_params ip = { /*.mem_size =*/ (size_t) 512*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx = ggml_init(ip);

    std::vector<ggml_tensor *> w(rot);
    for (int i = 0; i < rot; ++i) {
        w[i] = ggml_new_tensor_2d(ctx, type, k, m);
    }
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
    // The op nodes must exist before the allocation below: their outputs need device buffers too.
    std::vector<ggml_tensor *> out(rot);
    for (int i = 0; i < rot; ++i) {
        out[i] = ggml_mul_mat(ctx, w[i], x);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "allocation failed\n"); return 1; }

    // fill on the host and upload: quantized rows go through the backend, never through ->data
    const size_t row_bytes = ggml_row_size(type, k);
    std::vector<uint8_t> stage((size_t) m * row_bytes);
    std::vector<float> src(k);
    for (int64_t r = 0; r < m; ++r) {
        for (int64_t i = 0; i < k; ++i) {
            src[i] = (float) (((i * 2654435761u + r * 40503u) % 2000) - 1000) / 1000.0f;
        }
        ggml_quantize_chunk(type, src.data(), stage.data() + (size_t) r * row_bytes, 0, 1, k, nullptr);
    }
    for (int i = 0; i < rot; ++i) {
        ggml_backend_tensor_set(w[i], stage.data(), 0, stage.size());
    }
    std::vector<float> xv(k);
    for (int64_t i = 0; i < k; ++i) xv[i] = (float) sin(0.01 * i);
    ggml_backend_tensor_set(x, xv.data(), 0, k * sizeof(float));

    std::vector<ggml_cgraph *> gf(rot);
    for (int i = 0; i < rot; ++i) {
        gf[i] = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf[i], out[i]);
    }

    const double bytes_per_call = (double) k * m * (double) ggml_type_size(type) / (double) ggml_blck_size(type);
    std::vector<double> us;
    for (int pass = 0; pass < iters; ++pass) {
        for (int i = 0; i < rot; ++i) {
            const int64_t t0 = ggml_time_us();
            ggml_backend_graph_compute(backend, gf[i]);
            ggml_backend_synchronize(backend);
            const int64_t t1 = ggml_time_us();
            us.push_back((double) (t1 - t0));
        }
    }
    std::sort(us.begin(), us.end());
    const double med = us[us.size()/2];
    const double best = us[0];
    printf("%-10s k=%-6lld m=%-6lld  rot=%-2d  median %8.1f us  min %8.1f us  %7.1f GB/s (median)  %7.1f GB/s (min)\n",
           ggml_type_name(type), (long long) k, (long long) m, rot, med, best,
           bytes_per_call / (med * 1e-6) / 1e9, bytes_per_call / (best * 1e-6) / 1e9);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
