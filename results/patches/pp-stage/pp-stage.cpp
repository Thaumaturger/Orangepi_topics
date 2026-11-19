// llama-pp-stage: run a gemma4 model as two pipeline stages in two processes (or on two boards) over TCP.
//
//   last stage (e.g. node2):  llama-pp-stage --role last  -m model.gguf -s S -N n_layer --listen 0.0.0.0:9000 [--n-seq 32] [-t 8]
//   first stage (e.g. node1): llama-pp-stage --role first -m model.gguf -s S --next 10.0.0.2:9000 --ref ref.bin
//                                            [--configs 1x1,2x2,8x2] [-n 32] [--n-seq 32] [-t 8] [--no-verify]
//                                            [--ctx-seq 256] [--prefix N] [--distinct]
//   middle stage (3+ stages):  llama-pp-stage --role mid -m model.gguf -s A -e B --listen host:port --next host:port
//                              [--stage-id 1] [--n-seq 32] [--ctx-seq 256] [-t 8]
//                              runs layers [A, B), forwards the hidden state to --next and passes the replies back;
//                              its compute time goes back in pp_rsp.t_mid_ms[stage-id - 1], so the first stage can
//                              print every stage's busy fraction (stage-id 1..3 in chain order)
// all roles also take [--kv-type f16|q8_0|q4_0] [--no-kvu] [--mmap]
// --kv-type : KV cache element type (default f16)
// --no-kvu  : one KV stream per sequence instead of one unified KV buffer
// --mmap    : load with mmap instead of direct I/O; tensors that are not repacked (the token embedding table)
//             then stay in the page cache, so only the rows that are used take memory
// --prefix N: every sequence starts with the same N-token prefix (filled once on sequence 0, then copied with
//             llama_memory_seq_cp on both stages), so generation runs with an N-token context per sequence
// --distinct: with --prefix, every sequence gets its own N-token prefix computed through both stages (its own
//             KV cells, like separate conversations). Sequences are filled incrementally: with configs in
//             ascending R, a config only fills the sequences the previous one did not have, and the previous
//             config's prompt and generation are removed with llama_memory_seq_rm(-1, N, -1)
//
// The first stage runs layers [0, S) and sends the hidden state (fp32, n_tokens x n_embd) plus per-row
// (pos, seq_id, output) to the last stage, which runs layers [S, N) + lm head and returns either logits rows
// (verification) or argmax tokens of the rows marked as output.
//
// verify pass (one sequence): replay the tokens of a reference produced by llama-stage-test --mode ref and
//                             compare every returned logits row with the single-model reference
// config RxM                : R sequences (same prompt), split into M micro-batches of R/M sequences;
//                             micro-batch i+1 runs on the first stage while micro-batch i runs on the last stage

#include "llama.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <deque>
#include <mutex>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static const uint32_t PP_MAGIC = 0x32535050; // "PPS2"

enum pp_msg_type : uint32_t {
    PP_BATCH    = 1, // payload: n_tokens x n_embd floats, then n_tokens x pp_row
    PP_RESET    = 2, // clear the KV cache (all sequences)
    PP_QUIT     = 3, // close this connection
    PP_SHUTDOWN = 4, // close and exit the server
    PP_SEQCP    = 5, // copy sequence 0 to sequences 1..n_tokens-1 (shared prefix), no payload
    PP_SEQRM    = 6, // remove every sequence's cells at positions >= n_tokens, no payload (n_rows -1 on failure)
};

#pragma pack(push, 1)
struct pp_req {
    uint32_t magic;
    uint32_t type;
    int32_t  n_tokens;
    int32_t  n_embd;
    int32_t  want_logits; // 1: return logits of the output rows, 0: return argmax token of the output rows
};
struct pp_row {
    int32_t pos;
    int32_t seq;
    int32_t out;
};
struct pp_rsp {
    uint32_t magic;
    int32_t  n_rows;      // number of output rows
    int32_t  n_vocab;     // 0 when tokens are returned
    float    t_compute_ms;
    float    t_mid_ms[3]; // compute time of middle stages 1..3 for this request (0 when absent)
};
#pragma pack(pop)

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static long rss_mib() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) return std::atol(line.c_str() + 6) / 1024;
    }
    return -1;
}

static void die(const char * msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static bool send_all(int fd, const void * buf, size_t n) {
    const char * p = (const char *) buf;
    while (n > 0) {
        const ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
        if (k <= 0) return false;
        p += k; n -= (size_t) k;
    }
    return true;
}

static bool recv_all(int fd, void * buf, size_t n) {
    char * p = (char *) buf;
    while (n > 0) {
        const ssize_t k = recv(fd, p, n, 0);
        if (k <= 0) return false;
        p += k; n -= (size_t) k;
    }
    return true;
}

static void set_nodelay(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

static void split_hostport(const std::string & s, std::string & host, std::string & port) {
    const size_t c = s.rfind(':');
    if (c == std::string::npos) die("expected host:port");
    host = s.substr(0, c);
    port = s.substr(c + 1);
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
    std::string role, model, listen = "0.0.0.0:9000", next, ref, configs = "1x1";
    int split = -1, end = -1, stage_id = 1, n_layer = -1, n_gen = 32, n_threads = 8, n_seq = 32, n_ctx_seq = 256;
    int prefix = 0; // > 0: every sequence starts with the same prefix of this many tokens (filled once, then copied)
    bool distinct = false; // with prefix: each sequence gets its own prefix (own KV cells) instead of a copy
    std::string kv_type = "f16";
    bool kv_unified = true;
    bool mmap = false;
    bool verify = true;
};

static llama_model * load_stage(const opts_t & o, int a, int b, const char * name) {
    llama_model_params mp = llama_model_default_params();
    mp.load_mode   = o.mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_DIRECT_IO;
    mp.layer_start = a;
    mp.layer_end   = b;
    const double t0 = now_s();
    llama_model * m = llama_model_load_from_file(o.model.c_str(), mp);
    if (!m) die("failed to load model");
    printf("[load] %s layers [%d, %d)  %.1f s  rss %ld MiB\n", name, a, b, now_s() - t0, rss_mib());
    fflush(stdout);
    return m;
}

static llama_context * make_ctx(const opts_t & o, llama_model * m, bool embeddings) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) (o.n_seq * o.n_ctx_seq);
    cp.n_batch         = std::min<uint32_t>(cp.n_ctx, 8192);
    cp.n_ubatch        = 512;
    cp.n_seq_max       = (uint32_t) o.n_seq;
    cp.n_threads       = o.n_threads;
    cp.n_threads_batch = o.n_threads;
    cp.embeddings      = embeddings;
    cp.pooling_type    = LLAMA_POOLING_TYPE_NONE;
    // unified KV: lets llama.cpp put all sequences of a micro-batch in one ubatch (split_simple).
    // without it, embeddings=true (first stage) makes the KV cache split the batch one sequence per
    // ubatch (split_seq), so a batch of K sequences costs K single-token passes (see 6.8_pipeline_sweep.md).
    // --no-kvu (one KV stream per sequence) needs llama-context-embd-all.patch, which keeps split_seq for
    // non-causal attention and pooling only
    cp.kv_unified      = o.kv_unified;
    // sliding-window layers keep n_swa cells per sequence, as llama-server does (the library default is full size)
    cp.swa_full        = false;
    cp.type_k          = kv_type_from_name(o.kv_type);
    cp.type_v          = cp.type_k;
    llama_context * c = llama_init_from_model(m, cp);
    if (!c) die("failed to create context");
    printf("[ctx] n_ctx %u, n_batch %u, n_seq_max %u, kv_unified %d, swa_full %d, kv %s, load %s, rss %ld MiB\n",
           cp.n_ctx, cp.n_batch, cp.n_seq_max, (int) cp.kv_unified, (int) cp.swa_full, o.kv_type.c_str(),
           o.mmap ? "mmap" : "dio", rss_mib());
    fflush(stdout);
    return c;
}

static int argmax(const float * v, int n) {
    return (int) (std::max_element(v, v + n) - v);
}

// ------------------------------------------------------------------ last stage (server)

static int run_last(const opts_t & o) {
    if (o.split <= 0 || o.n_layer <= o.split) die("last stage needs -s S and -N n_layer with S < N");
    llama_model * m = load_stage(o, o.split, o.n_layer, "last");
    llama_context * c = make_ctx(o, m, false);
    const int n_embd  = llama_model_n_embd(m);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m));
    const int n_max   = std::min(o.n_seq * o.n_ctx_seq, 8192); // largest batch accepted (= n_batch)

    std::string host, port;
    split_hostport(o.listen, host, port);
    const int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t) std::atoi(port.c_str()));
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) die("bad listen address");
    if (bind(ls, (sockaddr *) &addr, sizeof(addr)) != 0) die("bind failed");
    if (listen(ls, 1) != 0) die("listen failed");
    printf("[last] listening on %s (n_embd %d, n_vocab %d)\n", o.listen.c_str(), n_embd, n_vocab);
    fflush(stdout);

    llama_batch b = llama_batch_init(n_max, n_embd, 1);
    std::vector<float>   embd;
    std::vector<pp_row>  rows;
    std::vector<float>   out_f;
    std::vector<int32_t> out_t;

    bool running = true;
    while (running) {
        const int fd = accept(ls, nullptr, nullptr);
        if (fd < 0) continue;
        set_nodelay(fd);
        printf("[last] client connected\n");
        fflush(stdout);
        pp_req rq;
        while (recv_all(fd, &rq, sizeof(rq))) {
            if (rq.magic != PP_MAGIC) { fprintf(stderr, "bad magic\n"); break; }
            if (rq.type == PP_RESET) {
                llama_memory_clear(llama_get_memory(c), true);
                pp_rsp rs = { PP_MAGIC, 0, 0, 0.0f, { 0.0f, 0.0f, 0.0f } };
                send_all(fd, &rs, sizeof(rs));
                continue;
            }
            if (rq.type == PP_SEQCP) {
                for (int s = 1; s < rq.n_tokens; ++s) llama_memory_seq_cp(llama_get_memory(c), 0, s, -1, -1);
                pp_rsp rs = { PP_MAGIC, 0, 0, 0.0f, { 0.0f, 0.0f, 0.0f } };
                send_all(fd, &rs, sizeof(rs));
                continue;
            }
            if (rq.type == PP_SEQRM) {
                const bool ok = llama_memory_seq_rm(llama_get_memory(c), -1, rq.n_tokens, -1);
                pp_rsp rs = { PP_MAGIC, ok ? 0 : -1, 0, 0.0f, { 0.0f, 0.0f, 0.0f } };
                send_all(fd, &rs, sizeof(rs));
                continue;
            }
            if (rq.type == PP_QUIT) break;
            if (rq.type == PP_SHUTDOWN) { running = false; break; }
            if (rq.type != PP_BATCH || rq.n_embd != n_embd || rq.n_tokens <= 0 || rq.n_tokens > n_max) {
                fprintf(stderr, "bad request\n");
                break;
            }
            const int n = rq.n_tokens;
            embd.resize((size_t) n * n_embd);
            rows.resize(n);
            if (!recv_all(fd, embd.data(), embd.size() * sizeof(float))) break;
            if (!recv_all(fd, rows.data(), rows.size() * sizeof(pp_row))) break;

            const double t0 = now_s();
            b.n_tokens = n;
            memcpy(b.embd, embd.data(), embd.size() * sizeof(float));
            int n_out = 0;
            for (int i = 0; i < n; ++i) {
                b.pos[i] = rows[i].pos; b.n_seq_id[i] = 1; b.seq_id[i][0] = rows[i].seq; b.logits[i] = rows[i].out ? 1 : 0;
                n_out += rows[i].out ? 1 : 0;
            }
            if (llama_decode(c, b) != 0) { fprintf(stderr, "decode failed\n"); break; }

            pp_rsp rs = { PP_MAGIC, n_out, rq.want_logits ? n_vocab : 0, 0.0f, { 0.0f, 0.0f, 0.0f } };
            if (rq.want_logits) {
                out_f.resize((size_t) n_out * n_vocab);
                for (int i = 0, j = 0; i < n; ++i) {
                    if (rows[i].out) memcpy(&out_f[(size_t) j++ * n_vocab], llama_get_logits_ith(c, i), sizeof(float) * n_vocab);
                }
            } else {
                out_t.resize(n_out);
                for (int i = 0, j = 0; i < n; ++i) {
                    if (rows[i].out) out_t[j++] = argmax(llama_get_logits_ith(c, i), n_vocab);
                }
            }
            rs.t_compute_ms = (float) ((now_s() - t0) * 1000.0);
            bool ok = send_all(fd, &rs, sizeof(rs));
            ok = ok && (rq.want_logits ? send_all(fd, out_f.data(), out_f.size() * sizeof(float))
                                       : send_all(fd, out_t.data(), out_t.size() * sizeof(int32_t)));
            if (!ok) break;
        }
        close(fd);
        printf("[last] client disconnected\n");
        fflush(stdout);
    }
    close(ls);
    llama_batch_free(b);
    llama_free(c);
    llama_model_free(m);
    return 0;
}

// ------------------------------------------------------------------ first stage (client)

struct ref_t {
    int32_t n_vocab = 0, n_layer = 0, n_embd = 0;
    std::vector<int32_t> prompt, gen;
    std::vector<float>   L, G;
};

static ref_t load_ref(const std::string & path) {
    ref_t r;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) die("cannot read ref file");
    int32_t hdr[5];
    if (fread(hdr, sizeof(hdr), 1, f) != 1) die("bad ref file");
    r.n_vocab = hdr[0]; r.n_layer = hdr[3]; r.n_embd = hdr[4];
    r.prompt.resize(hdr[1]); r.gen.resize(hdr[2]);
    r.L.resize((size_t) hdr[1] * hdr[0]); r.G.resize((size_t) hdr[2] * hdr[0]);
    bool ok = fread(r.prompt.data(), 4, r.prompt.size(), f) == r.prompt.size() &&
              fread(r.gen.data(),    4, r.gen.size(),    f) == r.gen.size() &&
              fread(r.L.data(),      4, r.L.size(),      f) == r.L.size() &&
              fread(r.G.data(),      4, r.G.size(),      f) == r.G.size();
    fclose(f);
    if (!ok) die("truncated ref file");
    return r;
}

static int connect_to(const std::string & hp) {
    std::string host, port;
    split_hostport(hp, host, port);
    addrinfo hints = {}, * res = nullptr;
    hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) die("cannot resolve next stage");
    int fd = -1;
    for (int attempt = 0; attempt < 900 && fd < 0; ++attempt) { // wait up to ~15 min for the last stage to load
        fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (connect(fd, res->ai_addr, res->ai_addrlen) != 0) { close(fd); fd = -1; sleep(1); }
    }
    freeaddrinfo(res);
    if (fd < 0) die("cannot connect to next stage");
    set_nodelay(fd);
    return fd;
}

struct first_t {
    const opts_t & o;
    llama_context * c = nullptr;
    int fd = -1, n_embd = 0, nv = 0;
    llama_batch b;
    std::vector<float>   hidden;
    std::vector<pp_row>  rows;
    std::vector<float>   logits;
    std::vector<int32_t> toks;
    double busy_a = 0.0, busy_b = 0.0;
    double busy_m[3] = { 0.0, 0.0, 0.0 }; // middle stages, in chain order
    void clear_busy() { busy_a = busy_b = 0.0; busy_m[0] = busy_m[1] = busy_m[2] = 0.0; }
    int filled = 0; // --distinct: sequences 0..filled-1 hold their own prefix on both stages

    explicit first_t(const opts_t & o) : o(o) {}

    // stage A on a batch of tokens, then ship the hidden state (does not wait for the reply)
    void send_batch(const std::vector<llama_token> & tk, const std::vector<pp_row> & meta, bool want_logits) {
        const int n = (int) tk.size();
        const double t0 = now_s();
        b.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            b.token[i] = tk[i]; b.pos[i] = meta[i].pos; b.n_seq_id[i] = 1; b.seq_id[i][0] = meta[i].seq; b.logits[i] = 1;
        }
        if (llama_decode(c, b) != 0) die("decode (first) failed");
        hidden.resize((size_t) n * n_embd);
        for (int i = 0; i < n; ++i) memcpy(&hidden[(size_t) i * n_embd], llama_get_embeddings_ith(c, i), sizeof(float) * n_embd);
        busy_a += now_s() - t0;
        pp_req rq = { PP_MAGIC, PP_BATCH, n, n_embd, want_logits ? 1 : 0 };
        if (!send_all(fd, &rq, sizeof(rq)) || !send_all(fd, hidden.data(), sizeof(float) * hidden.size()) ||
            !send_all(fd, meta.data(), sizeof(pp_row) * meta.size())) die("send failed");
    }

    // wait for the reply of the oldest batch in flight
    int recv_reply(bool want_logits) {
        pp_rsp rs;
        if (!recv_all(fd, &rs, sizeof(rs)) || rs.magic != PP_MAGIC) die("bad response");
        bool ok;
        if (want_logits) { logits.resize((size_t) rs.n_rows * nv); ok = recv_all(fd, logits.data(), sizeof(float) * logits.size()); }
        else             { toks.resize(rs.n_rows);                 ok = recv_all(fd, toks.data(),   sizeof(int32_t) * toks.size()); }
        if (!ok) die("recv failed");
        busy_b += rs.t_compute_ms / 1000.0;
        for (int k = 0; k < 3; ++k) busy_m[k] += rs.t_mid_ms[k] / 1000.0;
        return rs.n_rows;
    }

    void reset() {
        llama_memory_clear(llama_get_memory(c), true);
        pp_req rq = { PP_MAGIC, PP_RESET, 0, n_embd, 0 };
        pp_rsp rs;
        if (!send_all(fd, &rq, sizeof(rq)) || !recv_all(fd, &rs, sizeof(rs))) die("reset failed");
        clear_busy();
    }

    // both stages: copy sequence 0 to sequences 1..R-1
    void seq_cp(int R) {
        for (int s = 1; s < R; ++s) llama_memory_seq_cp(llama_get_memory(c), 0, s, -1, -1);
        pp_req rq = { PP_MAGIC, PP_SEQCP, R, n_embd, 0 };
        pp_rsp rs;
        if (!send_all(fd, &rq, sizeof(rq)) || !recv_all(fd, &rs, sizeof(rs))) die("seq_cp failed");
    }

    // both stages: remove every sequence's cells at positions >= p0
    void seq_rm(int p0) {
        if (!llama_memory_seq_rm(llama_get_memory(c), -1, p0, -1)) die("seq_rm (first) failed");
        pp_req rq = { PP_MAGIC, PP_SEQRM, p0, n_embd, 0 };
        pp_rsp rs;
        if (!send_all(fd, &rq, sizeof(rq)) || !recv_all(fd, &rs, sizeof(rs)) || rs.n_rows != 0) die("seq_rm (last) failed");
    }
};

static void verify_pass(first_t & F, const ref_t & r) {
    F.reset();
    const int n = (int) r.prompt.size();
    int n_rows = 0, n_top1 = 0;
    double worst = 0.0;
    auto check = [&](const float * ref_row, const float * got_row) {
        double mx = 0.0;
        for (int i = 0; i < F.nv; ++i) mx = std::max(mx, std::fabs((double) ref_row[i] - (double) got_row[i]));
        worst = std::max(worst, mx);
        n_rows++;
        n_top1 += argmax(ref_row, F.nv) == argmax(got_row, F.nv);
    };
    std::vector<llama_token> tk(r.prompt.begin(), r.prompt.end());
    std::vector<pp_row> meta(n);
    for (int i = 0; i < n; ++i) meta[i] = { i, 0, 1 };
    F.send_batch(tk, meta, true);
    F.recv_reply(true);
    for (int i = 0; i < n; ++i) check(&r.L[(size_t) i * F.nv], &F.logits[(size_t) i * F.nv]);
    for (size_t i = 0; i < r.gen.size(); ++i) {
        F.send_batch({ r.gen[i] }, { { n + (int) i, 0, 1 } }, true);
        F.recv_reply(true);
        check(&r.G[i * F.nv], F.logits.data());
    }
    printf("[verify] rows %d  top1 agree %d/%d  worst max_abs %.3e  -> %s\n", n_rows, n_top1, n_rows, worst,
            (n_top1 == n_rows && worst < 1e-3) ? "PASS" : "FAIL");
    fflush(stdout);
}

// R sequences with the same prompt, M micro-batches; returns aggregate generation throughput
static void config_pass(first_t & F, const ref_t & r, int R, int M) {
    if (R % M != 0 || R > F.o.n_seq) { printf("[config %dx%d] skipped (R must be a multiple of M and <= n_seq)\n", R, M); return; }
    const int K = R / M; // sequences per micro-batch
    const int n = (int) r.prompt.size();
    const int G = F.o.n_gen;
    const int P = F.o.prefix; // shared prefix length; the prompt and generation follow it
    if (P + n + G > F.o.n_ctx_seq) { printf("[config %dx%d] skipped (prefix + prompt + gen > ctx-seq)\n", R, M); return; }
    const bool distinct = F.o.distinct && P > 0;
    if (!distinct || F.filled == 0 || R < F.filled) { F.reset(); F.filled = 0; }
    else F.seq_rm(P); // keep each sequence's own prefix, drop the previous config's prompt and generation

    double t_prefix = 0.0;
    int n_new = 0;
    if (distinct) {
        // own prefix per sequence for sequences filled..R-1, chunks of 512; the last stage handles batches in
        // the order they arrive, so two chunks are kept in flight and both boards work at the same time
        const double tq0 = now_s();
        int inflight = 0;
        for (int s = F.filled; s < R; ++s) {
            for (int p0 = 0; p0 < P; p0 += 512) {
                const int len = std::min(512, P - p0);
                std::vector<llama_token> tk(len);
                std::vector<pp_row> meta(len);
                for (int i = 0; i < len; ++i) { tk[i] = r.prompt[(p0 + i + 7 * s) % n]; meta[i] = { p0 + i, s, 0 }; }
                F.send_batch(tk, meta, false);
                if (++inflight == 2) { F.recv_reply(false); --inflight; }
            }
        }
        while (inflight-- > 0) F.recv_reply(false);
        n_new = R - F.filled;
        F.filled = R;
        t_prefix = now_s() - tq0;
    }

    // shared prefix: fill sequence 0 in chunks of 512 through both stages, then copy it to every sequence
    if (P > 0 && !distinct) {
        const double tq0 = now_s();
        for (int p0 = 0; p0 < P; p0 += 512) {
            const int len = std::min(512, P - p0);
            std::vector<llama_token> tk(len);
            std::vector<pp_row> meta(len);
            for (int i = 0; i < len; ++i) { tk[i] = r.prompt[(p0 + i) % n]; meta[i] = { p0 + i, 0, 0 }; }
            F.send_batch(tk, meta, false);
            F.recv_reply(false);
        }
        F.seq_cp(R);
        t_prefix = now_s() - tq0;
    }

    // prefill, one micro-batch at a time (not timed as generation); only the last prompt row of each sequence is output
    std::vector<std::vector<llama_token>> next(M, std::vector<llama_token>(K));
    const double tp0 = now_s();
    for (int mb = 0; mb < M; ++mb) {
        std::vector<llama_token> tk;
        std::vector<pp_row> meta;
        for (int k = 0; k < K; ++k) {
            const int seq = mb * K + k;
            for (int i = 0; i < n; ++i) { tk.push_back(r.prompt[i]); meta.push_back({ P + i, seq, i == n - 1 ? 1 : 0 }); }
        }
        F.send_batch(tk, meta, false);
        F.recv_reply(false);
        for (int k = 0; k < K; ++k) next[mb][k] = F.toks[k];
    }
    const double t_prefill = now_s() - tp0;

    // generation: keep every micro-batch in flight; the reply of micro-batch mb (round t) arrives while
    // the first stage is computing micro-batch mb+1, so the two boards work at the same time
    std::vector<llama_token> seq0;
    F.clear_busy();
    const double t0 = now_s();
    for (int t = 0; t < G; ++t) {
        for (int mb = 0; mb < M; ++mb) {
            if (t > 0) {
                // reply of this micro-batch from the previous round
                F.recv_reply(false);
                for (int k = 0; k < K; ++k) next[mb][k] = F.toks[k];
            }
            if (mb == 0) seq0.push_back(next[0][0]);
            std::vector<pp_row> meta(K);
            for (int k = 0; k < K; ++k) meta[k] = { P + n + t, mb * K + k, 1 };
            F.send_batch(next[mb], meta, false);
        }
    }
    for (int mb = 0; mb < M; ++mb) F.recv_reply(false); // drain the last round
    const double t_gen = now_s() - t0;

    int agree = 0;
    // with a prefix the context differs from the reference, so the comparison does not apply
    const int n_cmp = P > 0 ? 0 : std::min<int>((int) seq0.size(), (int) r.gen.size());
    for (int i = 0; i < n_cmp; ++i) agree += seq0[i] == r.gen[i];
    const double tps = (double) R * G / t_gen;
    std::string mids;
    for (int k = 0; k < 3; ++k) {
        if (F.busy_m[k] > 0.0) { char buf[32]; snprintf(buf, sizeof(buf), "  M%d %.0f%%", k + 1, 100.0 * F.busy_m[k] / t_gen); mids += buf; }
    }
    printf("[config %2dx%d] R %2d  M %d  per-mb %2d  gen %d tok/seq  time %.2f s  -> %.2f t/s total, %.3f t/s per seq  "
           "| busy A %.0f%%%s  B %.0f%%  | prefix %d tok %s (%d new seq) %.1f s | prefill %.1f s | seq0 vs ref %d/%d\n",
            R, M, R, M, K, G, t_gen, tps, tps / R, 100.0 * F.busy_a / t_gen, mids.c_str(), 100.0 * F.busy_b / t_gen, P,
            distinct ? "own" : "shared", distinct ? n_new : (P > 0 ? 1 : 0), t_prefix, t_prefill, agree, n_cmp);
    fflush(stdout);
}

// ------------------------------------------------------------------ middle stage (server and client)

// payload size of a reply: argmax tokens or logits rows of the output rows; control replies have none
static size_t rsp_payload_bytes(const pp_rsp & rs) {
    if (rs.n_rows <= 0) return 0;
    return (size_t) rs.n_rows * (rs.n_vocab > 0 ? sizeof(float) * (size_t) rs.n_vocab : sizeof(int32_t));
}

static int run_mid(const opts_t & o) {
    if (o.split <= 0 || o.end <= o.split) die("mid stage needs -s A -e B with 0 < A < B");
    if (o.next.empty()) die("mid stage needs --next");
    if (o.stage_id < 1 || o.stage_id > 3) die("--stage-id must be 1..3");
    llama_model * m = load_stage(o, o.split, o.end, "mid");
    llama_context * c = make_ctx(o, m, true);
    const int n_embd = llama_model_n_embd(m);
    const int n_max  = std::min(o.n_seq * o.n_ctx_seq, 8192);

    std::string host, port;
    split_hostport(o.listen, host, port);
    const int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t) std::atoi(port.c_str()));
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) die("bad listen address");
    if (bind(ls, (sockaddr *) &addr, sizeof(addr)) != 0) die("bind failed");
    if (listen(ls, 1) != 0) die("listen failed");
    printf("[mid] layers [%d, %d), stage-id %d, listening on %s, next %s\n", o.split, o.end, o.stage_id, o.listen.c_str(), o.next.c_str());
    fflush(stdout);

    llama_batch b = llama_batch_init(n_max, n_embd, 1);
    std::vector<float>  embd, hidden;
    std::vector<pp_row> rows;

    bool running = true;
    while (running) {
        const int pfd = accept(ls, nullptr, nullptr);
        if (pfd < 0) continue;
        set_nodelay(pfd);
        const int nfd = connect_to(o.next);
        printf("[mid] client connected, next connected\n");
        fflush(stdout);

        // compute time of each forwarded request that expects a reply, in order (replies come back in order)
        std::mutex mu;
        std::deque<float> t_queue;
        double busy = 0.0;
        const double t_session = now_s();

        // replies: next -> previous, adding this stage's compute time
        std::thread up([&]() {
            pp_rsp rs;
            std::vector<char> payload;
            while (recv_all(nfd, &rs, sizeof(rs))) {
                payload.resize(rsp_payload_bytes(rs));
                if (!payload.empty() && !recv_all(nfd, payload.data(), payload.size())) break;
                float t = 0.0f;
                { std::lock_guard<std::mutex> lk(mu); if (!t_queue.empty()) { t = t_queue.front(); t_queue.pop_front(); } }
                rs.t_mid_ms[o.stage_id - 1] += t;
                if (!send_all(pfd, &rs, sizeof(rs))) break;
                if (!payload.empty() && !send_all(pfd, payload.data(), payload.size())) break;
            }
        });
        auto push_t = [&](float t) { std::lock_guard<std::mutex> lk(mu); t_queue.push_back(t); };

        // requests: previous -> this stage -> next
        pp_req rq;
        while (recv_all(pfd, &rq, sizeof(rq))) {
            if (rq.magic != PP_MAGIC) { fprintf(stderr, "bad magic\n"); break; }
            if (rq.type == PP_RESET || rq.type == PP_SEQCP || rq.type == PP_SEQRM) {
                llama_memory_t mem = llama_get_memory(c);
                if (rq.type == PP_RESET) llama_memory_clear(mem, true);
                if (rq.type == PP_SEQCP) for (int sq = 1; sq < rq.n_tokens; ++sq) llama_memory_seq_cp(mem, 0, sq, -1, -1);
                if (rq.type == PP_SEQRM && !llama_memory_seq_rm(mem, -1, rq.n_tokens, -1)) die("seq_rm (mid) failed");
                push_t(0.0f);
                if (!send_all(nfd, &rq, sizeof(rq))) break;
                continue;
            }
            if (rq.type == PP_QUIT || rq.type == PP_SHUTDOWN) {
                send_all(nfd, &rq, sizeof(rq));
                if (rq.type == PP_SHUTDOWN) running = false;
                break;
            }
            if (rq.type != PP_BATCH || rq.n_embd != n_embd || rq.n_tokens <= 0 || rq.n_tokens > n_max) {
                fprintf(stderr, "bad request\n");
                break;
            }
            const int n = rq.n_tokens;
            embd.resize((size_t) n * n_embd);
            rows.resize(n);
            if (!recv_all(pfd, embd.data(), embd.size() * sizeof(float))) break;
            if (!recv_all(pfd, rows.data(), rows.size() * sizeof(pp_row))) break;

            const double t0 = now_s();
            b.n_tokens = n;
            memcpy(b.embd, embd.data(), embd.size() * sizeof(float));
            for (int i = 0; i < n; ++i) {
                b.pos[i] = rows[i].pos; b.n_seq_id[i] = 1; b.seq_id[i][0] = rows[i].seq; b.logits[i] = 1;
            }
            if (llama_decode(c, b) != 0) { fprintf(stderr, "decode failed\n"); break; }
            hidden.resize((size_t) n * n_embd);
            for (int i = 0; i < n; ++i) memcpy(&hidden[(size_t) i * n_embd], llama_get_embeddings_ith(c, i), sizeof(float) * n_embd);
            const double dt = now_s() - t0;
            busy += dt;
            push_t((float) (dt * 1000.0));
            if (!send_all(nfd, &rq, sizeof(rq)) || !send_all(nfd, hidden.data(), hidden.size() * sizeof(float)) ||
                !send_all(nfd, rows.data(), rows.size() * sizeof(pp_row))) break;
        }
        shutdown(nfd, SHUT_WR);  // the next stage closes after QUIT; the reply thread then ends
        up.join();
        close(nfd);
        close(pfd);
        printf("[mid] client disconnected (compute %.1f s of %.1f s)\n", busy, now_s() - t_session);
        fflush(stdout);
    }
    close(ls);
    llama_batch_free(b);
    llama_free(c);
    llama_model_free(m);
    return 0;
}

static int run_first(const opts_t & o) {
    const ref_t r = load_ref(o.ref);
    const int S = o.split > 0 ? o.split : r.n_layer / 2;
    llama_model * m = load_stage(o, 0, S, "first");
    first_t F(o);
    F.c      = make_ctx(o, m, true);
    F.n_embd = llama_model_n_embd(m);
    F.nv     = r.n_vocab;
    if (F.n_embd != r.n_embd) die("n_embd mismatch with ref");
    F.b = llama_batch_init(std::min(o.n_seq * o.n_ctx_seq, 8192), 0, 1);

    printf("[first] connecting to %s ...\n", o.next.c_str());
    fflush(stdout);
    F.fd = connect_to(o.next);
    printf("[first] connected\n");
    fflush(stdout);

    if (o.verify) verify_pass(F, r);

    std::stringstream ss(o.configs);
    std::string item;
    while (std::getline(ss, item, ',')) {
        int R = 0, M = 0;
        if (sscanf(item.c_str(), "%dx%d", &R, &M) != 2 || R <= 0 || M <= 0) { printf("bad config '%s'\n", item.c_str()); continue; }
        config_pass(F, r, R, M);
    }

    pp_req q = { PP_MAGIC, PP_QUIT, 0, F.n_embd, 0 };
    send_all(F.fd, &q, sizeof(q));
    close(F.fd);
    llama_batch_free(F.b);
    llama_free(F.c);
    llama_model_free(m);
    return 0;
}

int main(int argc, char ** argv) {
    opts_t o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) die(("missing value for " + a).c_str()); return argv[++i]; };
        if      (a == "--role")      o.role      = next();
        else if (a == "-m")          o.model     = next();
        else if (a == "-s")          o.split     = std::atoi(next().c_str());
        else if (a == "-N")          o.n_layer   = std::atoi(next().c_str());
        else if (a == "-e")          o.end       = std::atoi(next().c_str());
        else if (a == "--stage-id")  o.stage_id  = std::atoi(next().c_str());
        else if (a == "-n")          o.n_gen     = std::atoi(next().c_str());
        else if (a == "-t")          o.n_threads = std::atoi(next().c_str());
        else if (a == "--n-seq")     o.n_seq     = std::atoi(next().c_str());
        else if (a == "--ctx-seq")   o.n_ctx_seq = std::atoi(next().c_str());
        else if (a == "--listen")    o.listen    = next();
        else if (a == "--next")      o.next      = next();
        else if (a == "--ref")       o.ref       = next();
        else if (a == "--configs")   o.configs   = next();
        else if (a == "--no-verify") o.verify    = false;
        else if (a == "--prefix")    o.prefix    = std::atoi(next().c_str());
        else if (a == "--distinct")  o.distinct  = true;
        else if (a == "--kv-type")   o.kv_type   = next();
        else if (a == "--no-kvu")    o.kv_unified = false;
        else if (a == "--mmap")      o.mmap      = true;
        else { fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 1; }
    }
    if (o.model.empty()) die("-m model.gguf is required");
    llama_backend_init();
    int rc = 1;
    if      (o.role == "last")  rc = run_last(o);
    else if (o.role == "first") { if (o.next.empty() || o.ref.empty()) die("first stage needs --next and --ref"); rc = run_first(o); }
    else if (o.role == "mid")   rc = run_mid(o);
    else die("--role must be first, mid or last");
    llama_backend_free();
    return rc;
}
