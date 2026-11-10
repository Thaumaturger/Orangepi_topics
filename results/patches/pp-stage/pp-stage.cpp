// llama-pp-stage: run a gemma4 model as two pipeline stages in two processes (or on two boards) over TCP.
//
//   last stage (e.g. node2):  llama-pp-stage --role last  -m model.gguf -s S -N n_layer --listen 0.0.0.0:9000 [-t 8]
//   first stage (e.g. node1): llama-pp-stage --role first -m model.gguf -s S --next 10.0.0.2:9000 --ref ref.bin [-n 64] [-t 8]
//
// The first stage runs layers [0, S) and sends the hidden state (fp32, n_tokens x n_embd) to the last stage,
// which runs layers [S, N) + lm head and sends back either the logits rows (verification) or argmax tokens (timing).
// The first stage replays the tokens of a reference file produced by llama-stage-test --mode ref (teacher forcing):
//   pass 1 "verify": compare every returned logits row with the single-model reference
//   pass 2 "timing": tokens only, report per-token time split into stage A / stage B / network

#include "llama.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static const uint32_t PP_MAGIC = 0x31535050; // "PPS1"

enum pp_msg_type : uint32_t {
    PP_BATCH    = 1, // payload: n_tokens x n_embd floats
    PP_RESET    = 2, // clear the KV cache
    PP_QUIT     = 3, // close this connection
    PP_SHUTDOWN = 4, // close and exit the server
};

#pragma pack(push, 1)
struct pp_req {
    uint32_t magic;
    uint32_t type;
    int32_t  n_tokens;
    int32_t  pos0;
    int32_t  n_embd;
    int32_t  want_logits; // 1: return logits rows for all tokens, 0: return argmax token per token
};
struct pp_rsp {
    uint32_t magic;
    int32_t  n_rows;
    int32_t  n_vocab;     // 0 when tokens are returned
    float    t_compute_ms;
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

struct opts_t {
    std::string role, model, listen = "0.0.0.0:9000", next, ref;
    int split = -1, n_layer = -1, n_gen = 64, n_threads = 8, n_ctx = 1024;
};

static llama_model * load_stage(const opts_t & o, int a, int b, const char * name) {
    llama_model_params mp = llama_model_default_params();
    mp.load_mode   = LLAMA_LOAD_MODE_DIRECT_IO;
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
    cp.n_ctx = o.n_ctx; cp.n_batch = o.n_ctx; cp.n_ubatch = 512; cp.n_seq_max = 1;
    cp.n_threads = o.n_threads; cp.n_threads_batch = o.n_threads;
    cp.embeddings = embeddings; cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    llama_context * c = llama_init_from_model(m, cp);
    if (!c) die("failed to create context");
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

    llama_batch b = llama_batch_init(o.n_ctx, n_embd, 1);
    std::vector<float>   embd;
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
                pp_rsp rs = { PP_MAGIC, 0, 0, 0.0f };
                send_all(fd, &rs, sizeof(rs));
                continue;
            }
            if (rq.type == PP_QUIT) break;
            if (rq.type == PP_SHUTDOWN) { running = false; break; }
            if (rq.type != PP_BATCH || rq.n_embd != n_embd || rq.n_tokens <= 0 || rq.n_tokens > o.n_ctx) {
                fprintf(stderr, "bad request\n");
                break;
            }
            const int n = rq.n_tokens;
            embd.resize((size_t) n * n_embd);
            if (!recv_all(fd, embd.data(), embd.size() * sizeof(float))) break;

            const double t0 = now_s();
            b.n_tokens = n;
            memcpy(b.embd, embd.data(), embd.size() * sizeof(float));
            for (int i = 0; i < n; ++i) {
                b.pos[i] = rq.pos0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
            }
            if (llama_decode(c, b) != 0) { fprintf(stderr, "decode failed\n"); break; }

            pp_rsp rs = { PP_MAGIC, n, rq.want_logits ? n_vocab : 0, 0.0f };
            if (rq.want_logits) {
                out_f.resize((size_t) n * n_vocab);
                for (int i = 0; i < n; ++i) memcpy(&out_f[(size_t) i * n_vocab], llama_get_logits_ith(c, i), sizeof(float) * n_vocab);
            } else {
                out_t.resize(n);
                for (int i = 0; i < n; ++i) out_t[i] = argmax(llama_get_logits_ith(c, i), n_vocab);
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
    for (int attempt = 0; attempt < 600 && fd < 0; ++attempt) { // wait up to ~10 min for the last stage to load
        fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (connect(fd, res->ai_addr, res->ai_addrlen) != 0) { close(fd); fd = -1; sleep(1); }
    }
    freeaddrinfo(res);
    if (fd < 0) die("cannot connect to next stage");
    set_nodelay(fd);
    return fd;
}

struct step_t { double t_a, t_rt, t_b; };

static int run_first(const opts_t & o) {
    const ref_t r = load_ref(o.ref);
    const int S = o.split > 0 ? o.split : r.n_layer / 2;
    llama_model * m = load_stage(o, 0, S, "first");
    llama_context * c = make_ctx(o, m, true);
    const int n_embd = llama_model_n_embd(m);
    const int nv     = r.n_vocab;
    if (n_embd != r.n_embd) die("n_embd mismatch with ref");

    printf("[first] connecting to %s ...\n", o.next.c_str());
    fflush(stdout);
    const int fd = connect_to(o.next);
    printf("[first] connected\n");
    fflush(stdout);

    const int n = (int) r.prompt.size();
    llama_batch b = llama_batch_init(std::max(n, 1), 0, 1);
    std::vector<float> hidden((size_t) std::max(n, 1) * n_embd);
    std::vector<float> logits((size_t) std::max(n, 1) * nv);
    std::vector<int32_t> toks(std::max(n, 1));

    // run tokens through stage A, ship the hidden state, get logits/tokens back
    auto step = [&](const llama_token * tk, int nt, int pos0, bool want_logits, step_t & st) {
        double t0 = now_s();
        b.n_tokens = nt;
        for (int i = 0; i < nt; ++i) { b.token[i] = tk[i]; b.pos[i] = pos0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1; }
        if (llama_decode(c, b) != 0) die("decode (first) failed");
        for (int i = 0; i < nt; ++i) memcpy(&hidden[(size_t) i * n_embd], llama_get_embeddings_ith(c, i), sizeof(float) * n_embd);
        st.t_a = now_s() - t0;

        t0 = now_s();
        pp_req rq = { PP_MAGIC, PP_BATCH, nt, pos0, n_embd, want_logits ? 1 : 0 };
        if (!send_all(fd, &rq, sizeof(rq)) || !send_all(fd, hidden.data(), sizeof(float) * (size_t) nt * n_embd)) die("send failed");
        pp_rsp rs;
        if (!recv_all(fd, &rs, sizeof(rs)) || rs.magic != PP_MAGIC || rs.n_rows != nt) die("bad response");
        const bool ok = want_logits ? recv_all(fd, logits.data(), sizeof(float) * (size_t) nt * nv)
                                    : recv_all(fd, toks.data(), sizeof(int32_t) * (size_t) nt);
        if (!ok) die("recv failed");
        st.t_rt = now_s() - t0;
        st.t_b  = rs.t_compute_ms / 1000.0;
    };

    auto reset = [&]() {
        llama_memory_clear(llama_get_memory(c), true);
        pp_req rq = { PP_MAGIC, PP_RESET, 0, 0, n_embd, 0 };
        pp_rsp rs;
        if (!send_all(fd, &rq, sizeof(rq)) || !recv_all(fd, &rs, sizeof(rs))) die("reset failed");
    };

    // ---- pass 1: verify (logits returned, compared to the single-model reference)
    reset();
    int n_rows = 0, n_top1 = 0;
    double worst = 0.0;
    auto check = [&](const float * ref_row, const float * got_row) {
        double mx = 0.0;
        for (int i = 0; i < nv; ++i) mx = std::max(mx, std::fabs((double) ref_row[i] - (double) got_row[i]));
        worst = std::max(worst, mx);
        n_rows++;
        n_top1 += argmax(ref_row, nv) == argmax(got_row, nv);
    };
    step_t st;
    step(r.prompt.data(), n, 0, true, st);
    printf("[verify] prefill %d tok: stage A %.3f s, round trip %.3f s (stage B %.3f s, network+copy %.3f s)\n",
            n, st.t_a, st.t_rt, st.t_b, st.t_rt - st.t_b);
    for (int i = 0; i < n; ++i) check(&r.L[(size_t) i * nv], &logits[(size_t) i * nv]);
    for (size_t i = 0; i < r.gen.size(); ++i) {
        const llama_token t = r.gen[i];
        step(&t, 1, n + (int) i, true, st);
        check(&r.G[i * nv], logits.data());
    }
    printf("[verify] rows %d  top1 agree %d/%d  worst max_abs %.3e  -> %s\n", n_rows, n_top1, n_rows, worst,
            (n_top1 == n_rows && worst < 1e-3) ? "PASS" : "FAIL");
    fflush(stdout);

    // ---- pass 2: timing (tokens only), greedy with teacher forcing on the reference tokens for the first part
    reset();
    std::vector<step_t> steps;
    step(r.prompt.data(), n, 0, false, st);
    const step_t pre = st;
    llama_token next = toks[n - 1];
    int agree = 0;
    for (int i = 0; i < o.n_gen; ++i) {
        if ((size_t) i < r.gen.size()) agree += next == r.gen[i];
        step(&next, 1, n + i, false, st);
        steps.push_back(st);
        next = toks[0];
    }
    double sa = 0, sr = 0, sb = 0;
    for (const auto & s : steps) { sa += s.t_a; sr += s.t_rt; sb += s.t_b; }
    const int k = (int) steps.size();
    printf("[timing] prefill %d tok: stage A %.3f s, stage B %.3f s, network+copy %.4f s\n", n, pre.t_a, pre.t_b, pre.t_rt - pre.t_b);
    printf("[timing] gen %d tok: per token  stage A %.4f s  stage B %.4f s  network+copy %.4f s  total %.4f s  (%.2f t/s)\n",
            k, sa / k, sb / k, (sr - sb) / k, (sa + sr) / k, k / (sa + sr));
    printf("[timing] greedy tokens matching the single-board reference: %d/%d\n", agree, std::min<int>(k, (int) r.gen.size()));

    pp_req q = { PP_MAGIC, PP_QUIT, 0, 0, n_embd, 0 };
    send_all(fd, &q, sizeof(q));
    close(fd);
    llama_batch_free(b);
    llama_free(c);
    llama_model_free(m);
    return 0;
}

int main(int argc, char ** argv) {
    opts_t o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) die(("missing value for " + a).c_str()); return argv[++i]; };
        if      (a == "--role")   o.role      = next();
        else if (a == "-m")       o.model     = next();
        else if (a == "-s")       o.split     = std::atoi(next().c_str());
        else if (a == "-N")       o.n_layer   = std::atoi(next().c_str());
        else if (a == "-n")       o.n_gen     = std::atoi(next().c_str());
        else if (a == "-t")       o.n_threads = std::atoi(next().c_str());
        else if (a == "-c")       o.n_ctx     = std::atoi(next().c_str());
        else if (a == "--listen") o.listen    = next();
        else if (a == "--next")   o.next      = next();
        else if (a == "--ref")    o.ref       = next();
        else { fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 1; }
    }
    if (o.model.empty()) die("-m model.gguf is required");
    llama_backend_init();
    int rc = 1;
    if      (o.role == "last")  rc = run_last(o);
    else if (o.role == "first") { if (o.next.empty() || o.ref.empty()) die("first stage needs --next and --ref"); rc = run_first(o); }
    else die("--role must be first or last");
    llama_backend_free();
    return rc;
}
