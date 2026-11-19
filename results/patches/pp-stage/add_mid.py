# Adds a middle-stage role (--role mid -s A -e B) to llama-pp-stage, for chains of 3 or more stages:
# first [0,S1) -> mid [S1,S2) -> ... -> last [Sk,N).
import sys
p = sys.argv[1]
s = open(p, encoding='utf-8').read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)


rep("""// both roles also take [--kv-type f16|q8_0|q4_0] [--no-kvu] [--mmap]
""", """//   middle stage (3+ stages):  llama-pp-stage --role mid -m model.gguf -s A -e B --listen host:port --next host:port
//                              [--stage-id 1] [--n-seq 32] [--ctx-seq 256] [-t 8]
//                              runs layers [A, B), forwards the hidden state to --next and passes the replies back;
//                              its compute time goes back in pp_rsp.t_mid_ms[stage-id - 1], so the first stage can
//                              print every stage's busy fraction (stage-id 1..3 in chain order)
// all roles also take [--kv-type f16|q8_0|q4_0] [--no-kvu] [--mmap]
""")
rep("""#include <algorithm>
""", """#include <algorithm>
#include <deque>
#include <mutex>
#include <thread>
""")
rep("""    float    t_compute_ms;
};""", """    float    t_compute_ms;
    float    t_mid_ms[3]; // compute time of middle stages 1..3 for this request (0 when absent)
};""")
rep("""    int split = -1, n_layer = -1,""", """    int split = -1, end = -1, stage_id = 1, n_layer = -1,""")
# every pp_rsp initializer gets the mid times zeroed
s = s.replace("pp_rsp rs = { PP_MAGIC, 0, 0, 0.0f };", "pp_rsp rs = { PP_MAGIC, 0, 0, 0.0f, { 0.0f, 0.0f, 0.0f } };")
rep("pp_rsp rs = { PP_MAGIC, ok ? 0 : -1, 0, 0.0f };", "pp_rsp rs = { PP_MAGIC, ok ? 0 : -1, 0, 0.0f, { 0.0f, 0.0f, 0.0f } };")
rep("pp_rsp rs = { PP_MAGIC, n_out, rq.want_logits ? n_vocab : 0, 0.0f };",
    "pp_rsp rs = { PP_MAGIC, n_out, rq.want_logits ? n_vocab : 0, 0.0f, { 0.0f, 0.0f, 0.0f } };")

# first stage: per-mid busy accounting
rep("""    double busy_a = 0.0, busy_b = 0.0;
""", """    double busy_a = 0.0, busy_b = 0.0;
    double busy_m[3] = { 0.0, 0.0, 0.0 }; // middle stages, in chain order
    void clear_busy() { busy_a = busy_b = 0.0; busy_m[0] = busy_m[1] = busy_m[2] = 0.0; }
""")
rep("""        busy_b += rs.t_compute_ms / 1000.0;
""", """        busy_b += rs.t_compute_ms / 1000.0;
        for (int k = 0; k < 3; ++k) busy_m[k] += rs.t_mid_ms[k] / 1000.0;
""")
rep("""        busy_a = busy_b = 0.0;
    }""", """        clear_busy();
    }""")
rep("""    F.busy_a = F.busy_b = 0.0;
""", """    F.clear_busy();
""")
rep("""    const double tps = (double) R * G / t_gen;
    printf("[config %2dx%d] R %2d  M %d  per-mb %2d  gen %d tok/seq  time %.2f s  -> %.2f t/s total, %.3f t/s per seq  "
           "| busy A %.0f%%  B %.0f%%  | prefix""", """    const double tps = (double) R * G / t_gen;
    std::string mids;
    for (int k = 0; k < 3; ++k) {
        if (F.busy_m[k] > 0.0) { char buf[32]; snprintf(buf, sizeof(buf), "  M%d %.0f%%", k + 1, 100.0 * F.busy_m[k] / t_gen); mids += buf; }
    }
    printf("[config %2dx%d] R %2d  M %d  per-mb %2d  gen %d tok/seq  time %.2f s  -> %.2f t/s total, %.3f t/s per seq  "
           "| busy A %.0f%%%s  B %.0f%%  | prefix""")
rep("""            R, M, R, M, K, G, t_gen, tps, tps / R, 100.0 * F.busy_a / t_gen, 100.0 * F.busy_b / t_gen, P,""",
    """            R, M, R, M, K, G, t_gen, tps, tps / R, 100.0 * F.busy_a / t_gen, mids.c_str(), 100.0 * F.busy_b / t_gen, P,""")

MID = r'''
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

static int run_first('''
rep("\nstatic int run_first(", MID)
rep("""        else if (a == "-N")          o.n_layer   = std::atoi(next().c_str());
""", """        else if (a == "-N")          o.n_layer   = std::atoi(next().c_str());
        else if (a == "-e")          o.end       = std::atoi(next().c_str());
        else if (a == "--stage-id")  o.stage_id  = std::atoi(next().c_str());
""")
rep("""    else die("--role must be first or last");""", """    else if (o.role == "mid")   rc = run_mid(o);
    else die("--role must be first, mid or last");""")
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print("ok")
