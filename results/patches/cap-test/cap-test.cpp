// llama-cap-test: how many users fit in memory at a fixed context length per user, for the full model or one
// pipeline stage (layer range, gemma4 + pp-layer-range patch).
//
//   llama-cap-test -m model.gguf [-a A -b B] --ctx 1024,4096,8192 --users 1,2,4,8,16,32 [--reserve-mib 1024] [-t 8]
//
// The model (or layers [A, B)) is loaded once. For every context length L and user count U it creates a context
// with U sequences of L cells each (same settings as llama-pp-stage: unified KV, n_ubatch 512; n_batch 512),
// sliding-window layers capped at 1024 cells per sequence (swa_full = false, the llama-server default),
// clears the KV cache (memset, so every page is resident), decodes one token per sequence, prints the process RSS
// and the system MemAvailable, and frees the context. llama.cpp allocates the whole KV cache when the context is
// created, so this is the memory a server with U users of up to L tokens needs.
// Before each step the next MemAvailable is predicted from the last two points; a step that would leave less
// than --reserve-mib is skipped, and the largest U that keeps the reserve is then computed and verified.

#include "llama.h"

#include <malloc.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static long status_kib(const char * file, const char * key) {
    std::ifstream f(file);
    std::string line;
    const size_t k = strlen(key);
    while (std::getline(f, line)) {
        if (line.compare(0, k, key) == 0) return std::atol(line.c_str() + k);
    }
    return -1;
}

// With --mmap the model weights live in the page cache, which MemAvailable counts as reclaimable. Using them
// as free memory would let the KV cache push the weights out (they would then be re-read from disk), so in
// mmap mode the resident file-backed pages of this process (RssFile) are subtracted from MemAvailable.
static bool g_sub_rssfile = false;
static long rss_mib()     { return status_kib("/proc/self/status", "VmRSS:") / 1024; }
static long rssfile_mib() { return status_kib("/proc/self/status", "RssFile:") / 1024; }
static long avail_mib()   {
    return status_kib("/proc/meminfo", "MemAvailable:") / 1024 - (g_sub_rssfile ? rssfile_mib() : 0);
}

static std::vector<int> parse_list(const std::string & s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::atoi(t.c_str()));
    return v;
}

// KV cache element type by name (K and V use the same type; quantized V needs flash attention, which is on by default)
static ggml_type kv_type_from_name(const std::string & s) {
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    fprintf(stderr, "unknown --kv-type %s (f16, q8_0, q4_0)\n", s.c_str());
    exit(1);
}

struct opts_t {
    std::string model, kv_type = "f16";
    bool kv_unified = true, mmap = false, avail_raw = false;   // avail_raw: with --mmap, use MemAvailable as is
    int a = 0, b = -1, n_threads = 8;
    long reserve_mib = 1024;
    std::vector<int> ctxs = {1024, 4096, 8192}, users = {1, 2, 4, 8, 16, 32};
};

struct point_t { int users; long rss, avail; bool ok; };

// one measurement: context with U sequences of L cells; returns false if the context or the decode failed
static bool measure(llama_model * m, const opts_t & o, bool first, bool last, int L, int U, point_t & p) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) ((long) U * L);
    cp.n_batch         = 512;
    cp.n_ubatch        = 512;
    cp.n_seq_max       = (uint32_t) U;
    cp.n_threads       = o.n_threads;
    cp.n_threads_batch = o.n_threads;
    cp.embeddings      = !last;
    cp.pooling_type    = LLAMA_POOLING_TYPE_NONE;
    cp.kv_unified      = o.kv_unified;
    cp.type_k          = kv_type_from_name(o.kv_type);
    cp.type_v          = cp.type_k;
    // sliding-window layers keep only n_swa (1024) cells per sequence, as llama-server does by default
    // (llama_context_default_params() sets swa_full = true, which sizes them like the full-attention layers)
    cp.swa_full        = false;
    p = { U, -1, -1, false };
    const double t0 = now_s();
    llama_context * c = llama_init_from_model(m, cp);
    if (!c) return false;
    llama_memory_clear(llama_get_memory(c), true);   // writes the whole KV buffer

    const int n_embd = llama_model_n_embd(m);
    llama_batch bt = llama_batch_init(U, first ? 0 : n_embd, 1);
    bt.n_tokens = U;
    for (int i = 0; i < U; ++i) {
        if (first) bt.token[i] = 2;                  // <bos>
        bt.pos[i] = 0; bt.n_seq_id[i] = 1; bt.seq_id[i][0] = i; bt.logits[i] = 1;
    }
    if (!first) memset(bt.embd, 0, sizeof(float) * (size_t) U * n_embd);
    const bool ok = llama_decode(c, bt) == 0;
    p.rss   = rss_mib();
    p.avail = avail_mib();
    p.ok    = ok;
    printf("[cap] L %6d  users %3d  n_ctx %8ld  rss %6ld MiB  rssfile %5ld MiB  avail %6ld MiB  %s  (%.1f s)\n",
           L, U, (long) U * L, p.rss, rssfile_mib(), p.avail, ok ? "ok" : "DECODE FAILED", now_s() - t0);
    fflush(stdout);
    llama_batch_free(bt);
    llama_free(c);
    malloc_trim(0);
    return ok;
}

int main(int argc, char ** argv) {
    opts_t o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return argv[++i]; };
        if      (a == "-m")            o.model = next();
        else if (a == "-a")            o.a = std::atoi(next().c_str());
        else if (a == "-b")            o.b = std::atoi(next().c_str());
        else if (a == "-t")            o.n_threads = std::atoi(next().c_str());
        else if (a == "--ctx")         o.ctxs = parse_list(next());
        else if (a == "--users")       o.users = parse_list(next());
        else if (a == "--reserve-mib") o.reserve_mib = std::atol(next().c_str());
        else if (a == "--kv-type")     o.kv_type = next();
        else if (a == "--no-kvu")      o.kv_unified = false;
        else if (a == "--mmap")        o.mmap = true;
        else if (a == "--avail-raw")   o.avail_raw = true;
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    if (o.model.empty()) { fprintf(stderr, "usage: %s -m model.gguf [-a A -b B] [--ctx L1,L2] [--users U1,U2] [--reserve-mib R] [-t N]\n", argv[0]); return 1; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.load_mode   = o.mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_DIRECT_IO;
    mp.layer_start = o.a;
    mp.layer_end   = o.b;
    g_sub_rssfile  = o.mmap && !o.avail_raw;
    const long avail0 = avail_mib();
    const double t0 = now_s();
    llama_model * m = llama_model_load_from_file(o.model.c_str(), mp);
    if (!m) { fprintf(stderr, "failed to load model\n"); return 1; }
    const int  n_layer = llama_model_n_layer(m);
    const bool first   = o.a == 0;
    const bool last    = o.b < 0 || o.b >= n_layer;
    malloc_trim(0);
    printf("[load] layers [%d, %d) of %d  %.1f s  rss %ld MiB  rssfile %ld MiB  avail before %ld / after %ld MiB  reserve %ld MiB%s\n",
           o.a, last ? n_layer : o.b, n_layer, now_s() - t0, rss_mib(), rssfile_mib(), avail0, avail_mib(), o.reserve_mib,
           g_sub_rssfile ? "  (avail = MemAvailable - RssFile)" : "");
    fflush(stdout);

    // the model alone (no context) is the first reference point, so even the first step of each L is predicted
    const point_t base = { 0, rss_mib(), avail_mib(), true };
    for (const int L : o.ctxs) {
        std::vector<point_t> pts = { base };
        bool stopped = false;
        for (const int U : o.users) {
            if (pts.size() >= 2) {
                const point_t & p1 = pts[pts.size() - 2];
                const point_t & p2 = pts.back();
                const double per_user = (double) (p1.avail - p2.avail) / (p2.users - p1.users);
                const double pred = p2.avail - per_user * (U - p2.users);
                if (pred < o.reserve_mib) {
                    printf("[cap] L %6d  users %3d  skipped: predicted avail %.0f MiB < reserve\n", L, U, pred);
                    stopped = true;
                    break;
                }
            }
            point_t p;
            if (!measure(m, o, first, last, L, U, p)) { stopped = true; break; }
            if (p.avail < o.reserve_mib) { printf("[cap] L %6d  users %3d  below reserve\n", L, U); stopped = true; break; }
            pts.push_back(p);
        }
        int best = pts.back().users;
        if (stopped && pts.size() >= 2) {
            // largest U that keeps the reserve, from the last two points; verify it
            const point_t & p1 = pts[pts.size() - 2];
            const point_t & p2 = pts.back();
            const double per_user = (double) (p1.avail - p2.avail) / (p2.users - p1.users);
            int cand = per_user > 0 ? p2.users + (int) ((p2.avail - o.reserve_mib) / per_user) : p2.users;
            while (cand > best) {
                point_t p;
                if (measure(m, o, first, last, L, cand, p) && p.avail >= o.reserve_mib) { best = cand; break; }
                --cand;
            }
            printf("[cap] L %6d  per user %.0f MiB\n", L, per_user);
        }
        printf("[result] L %6d  max users %3d%s\n", L, best, stopped ? "" : " (or more: all listed user counts fit)");
        fflush(stdout);
    }
    llama_model_free(m);
    llama_backend_free();
    return 0;
}
