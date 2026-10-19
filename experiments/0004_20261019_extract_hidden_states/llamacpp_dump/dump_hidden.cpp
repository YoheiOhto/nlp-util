// dump_hidden: extract per-token hidden states from a GGUF model with llama.cpp.
//
// Usage:
//   dump_hidden <model.gguf> <manifest.tsv> <out_dir> <engine_label> [n_ctx]
//
// manifest.tsv: one line per prompt: "<prompt_id>\t<text_file>\t<ids_file>"
//   text_file : raw prompt text (tokenized with the GGUF vocab for a check)
//   ids_file  : whitespace-separated reference token IDs (from HF); the model
//               is always evaluated on these IDs so engines see identical input.
//
// For each prompt two passes are run on a fresh KV cache:
//   pass A (no eval callback): embeddings with pooling NONE
//          (= llama_get_embeddings_ith = "result_norm", AFTER the final RMSNorm)
//          and logits of the last token.
//   pass B (eval callback):    copies "inp_embd" and every "l_out-<il>" tensor
//          (residual stream after each block, BEFORE the final norm) and
//          "result_norm" from the graph to host memory.
// Outputs in <out_dir>/<engine>/<prompt_id>/:
//   final.npy [T,H] f32 (pass A), logits_last.npy [V] f32 (pass A),
//   layers.npy [n_layer+1,T,H] f32 (pass B), final_cb.npy [T,H] f32 (pass B),
//   llama_ids.json (GGUF tokenization of text_file), meta.json (timings).

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

void mkdirs(const std::string & path) {
    std::string cur;
    std::stringstream ss(path);
    std::string part;
    if (!path.empty() && path[0] == '/') cur = "/";
    while (std::getline(ss, part, '/')) {
        if (part.empty()) continue;
        cur += part + "/";
        mkdir(cur.c_str(), 0755);
    }
}

// Write a float32 C-order .npy file.
void save_npy(const std::string & path, const float * data, const std::vector<size_t> & shape) {
    std::string shp = "(";
    size_t n = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        shp += std::to_string(shape[i]) + (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : ""));
        n *= shape[i];
    }
    shp += ")";
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shp + ", }";
    size_t total = 10 + header.size() + 1;
    size_t pad = (64 - total % 64) % 64;
    header += std::string(pad, ' ') + "\n";
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    const char magic[] = "\x93NUMPY";
    f.write(magic, 6);
    const char ver[2] = {1, 0};
    f.write(ver, 2);
    uint16_t hl = (uint16_t) header.size();
    f.write(reinterpret_cast<const char *>(&hl), 2);
    f.write(header.data(), header.size());
    f.write(reinterpret_cast<const char *>(data), n * sizeof(float));
    if (!f) throw std::runtime_error("write failed " + path);
}

std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<llama_token> read_ids(const std::string & path) {
    std::string s = read_file(path);
    for (char & c : s) if (c == '[' || c == ']' || c == ',') c = ' ';
    std::stringstream ss(s);
    std::vector<llama_token> ids;
    long v;
    while (ss >> v) ids.push_back((llama_token) v);
    if (ids.empty()) throw std::runtime_error("no ids in " + path);
    return ids;
}

std::string ids_json(const std::vector<llama_token> & ids) {
    std::string s = "[";
    for (size_t i = 0; i < ids.size(); ++i) s += std::to_string(ids[i]) + (i + 1 < ids.size() ? ", " : "");
    return s + "]";
}

struct capture_state {
    bool enabled = false;
    int n_layer = 0;
    int n_embd = 0;
    // name -> rows appended (row-major [n_tokens][n_embd]) across ubatches
    std::map<std::string, std::vector<float>> data;
    std::vector<uint8_t> buf;
};

bool wanted(const capture_state & st, const char * name) {
    if (std::strcmp(name, "inp_embd") == 0 || std::strcmp(name, "result_norm") == 0) return true;
    return std::strncmp(name, "l_out-", 6) == 0;
}

// ggml_backend_sched_eval_callback
bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<capture_state *>(user_data);
    if (!st->enabled) return ask ? false : true;
    if (ask) return wanted(*st, t->name);
    if (!wanted(*st, t->name)) return true;
    if (t->ne[0] != st->n_embd || t->ne[2] != 1 || t->ne[3] != 1) {
        std::fprintf(stderr, "unexpected shape for %s: %lld %lld %lld %lld\n", t->name,
                     (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
        std::exit(3);
    }
    const size_t nbytes = ggml_nbytes(t);
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * src;
    if (is_host) {
        src = static_cast<const uint8_t *>(t->data);
    } else {
        st->buf.resize(nbytes);
        ggml_backend_tensor_get(t, st->buf.data(), 0, nbytes);
        src = st->buf.data();
    }
    auto & dst = st->data[t->name];
    const int64_t nr = t->ne[1];
    const int64_t nc = t->ne[0];
    for (int64_t r = 0; r < nr; ++r) {
        for (int64_t c = 0; c < nc; ++c) {
            const uint8_t * p = src + r * t->nb[1] + c * t->nb[0];
            float v;
            switch (t->type) {
                case GGML_TYPE_F32: v = *reinterpret_cast<const float *>(p); break;
                case GGML_TYPE_F16: v = ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(p)); break;
                case GGML_TYPE_BF16: v = ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(p)); break;
                default:
                    std::fprintf(stderr, "unsupported type %s for %s\n", ggml_type_name(t->type), t->name);
                    std::exit(3);
            }
            dst.push_back(v);
        }
    }
    return true;
}

int decode_all(llama_context * ctx, const std::vector<llama_token> & ids) {
    llama_batch batch = llama_batch_init((int32_t) ids.size(), 0, 1);
    batch.n_tokens = (int32_t) ids.size();
    for (size_t i = 0; i < ids.size(); ++i) {
        batch.token[i] = ids[i];
        batch.pos[i] = (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    int rc = llama_decode(ctx, batch);
    llama_synchronize(ctx);
    llama_batch_free(batch);
    return rc;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <model.gguf> <manifest.tsv> <out_dir> <engine> [n_ctx]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string manifest = argv[2];
    const std::string out_dir = argv[3];
    const std::string engine = argv[4];
    const int n_ctx = argc > 5 ? std::atoi(argv[5]) : 8192;

    llama_backend_init();

    auto t_load0 = clk::now();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { std::fprintf(stderr, "failed to load %s\n", model_path.c_str()); return 1; }

    capture_state st;
    st.n_layer = llama_model_n_layer(model);
    st.n_embd = llama_model_n_embd(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = n_ctx;  // one ubatch per prompt
    cp.n_seq_max = 1;
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    // ctx: plain context (pass A). ctx_cb: same settings plus the eval callback (pass B).
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "failed to create context\n"); return 1; }
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = &st;
    llama_context * ctx_cb = llama_init_from_model(model, cp);
    if (!ctx_cb) { std::fprintf(stderr, "failed to create callback context\n"); return 1; }
    const double load_s = secs(t_load0, clk::now());
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::fprintf(stderr, "n_layer=%d n_embd=%d n_vocab=%d n_ctx=%d flash_attn=%s\n", st.n_layer, st.n_embd,
                 n_vocab, n_ctx, llama_flash_attn_type_name(cp.flash_attn_type));

    struct item { std::string id, text_file, ids_file; };
    std::vector<item> items;
    {
        std::stringstream ss(read_file(manifest));
        std::string line;
        while (std::getline(ss, line)) {
            if (line.empty()) continue;
            std::stringstream ls(line);
            item it;
            std::getline(ls, it.id, '\t');
            std::getline(ls, it.text_file, '\t');
            std::getline(ls, it.ids_file, '\t');
            items.push_back(it);
        }
    }

    // Warmup on the first prompt (both passes), not timed.
    {
        auto ids = read_ids(items.at(0).ids_file);
        llama_memory_clear(llama_get_memory(ctx), true);
        decode_all(ctx, ids);
        st.enabled = true;
        llama_memory_clear(llama_get_memory(ctx_cb), true);
        decode_all(ctx_cb, ids);
        st.enabled = false;
        st.data.clear();
    }

    double tot_a = 0, tot_b = 0, tot_a_copy = 0;
    size_t tot_tok = 0;
    for (const auto & it : items) {
        const std::string text = read_file(it.text_file);
        std::vector<llama_token> lt(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), lt.data(), (int32_t) lt.size(),
                               /*add_special*/ false, /*parse_special*/ false);
        if (n < 0) { std::fprintf(stderr, "tokenize failed for %s\n", it.id.c_str()); return 1; }
        lt.resize(n);
        const auto ids = read_ids(it.ids_file);
        const size_t T = ids.size();
        const size_t H = st.n_embd;
        if ((int) T > n_ctx) { std::fprintf(stderr, "prompt %s too long\n", it.id.c_str()); return 1; }
        const std::string d = out_dir + "/" + engine + "/" + it.id;
        mkdirs(d);

        // pass A
        llama_memory_clear(llama_get_memory(ctx), true);
        auto a0 = clk::now();
        if (decode_all(ctx, ids) != 0) { std::fprintf(stderr, "decode A failed\n"); return 1; }
        auto a1 = clk::now();
        std::vector<float> final_a(T * H);
        for (size_t i = 0; i < T; ++i) {
            const float * e = llama_get_embeddings_ith(ctx, (int32_t) i);
            if (!e) { std::fprintf(stderr, "no embedding for token %zu\n", i); return 1; }
            std::memcpy(final_a.data() + i * H, e, H * sizeof(float));
        }
        const float * lg = llama_get_logits_ith(ctx, (int32_t) T - 1);
        if (!lg) { std::fprintf(stderr, "no logits\n"); return 1; }
        std::vector<float> logits(lg, lg + n_vocab);
        auto a2 = clk::now();

        // pass B
        st.data.clear();
        st.enabled = true;
        llama_memory_clear(llama_get_memory(ctx_cb), true);
        auto b0 = clk::now();
        if (decode_all(ctx_cb, ids) != 0) { std::fprintf(stderr, "decode B failed\n"); return 1; }
        auto b1 = clk::now();
        st.enabled = false;

        std::vector<float> layers((st.n_layer + 1) * T * H);
        auto take = [&](const std::string & name, float * dst) {
            auto f = st.data.find(name);
            if (f == st.data.end() || f->second.size() != T * H) {
                std::fprintf(stderr, "missing/short tensor %s (%zu)\n", name.c_str(),
                             f == st.data.end() ? (size_t) 0 : f->second.size());
                std::exit(4);
            }
            std::memcpy(dst, f->second.data(), T * H * sizeof(float));
        };
        take("inp_embd", layers.data());
        for (int il = 0; il < st.n_layer; ++il) take("l_out-" + std::to_string(il), layers.data() + (il + 1) * T * H);
        std::vector<float> final_b(T * H);
        take("result_norm", final_b.data());

        save_npy(d + "/final.npy", final_a.data(), {T, H});
        save_npy(d + "/logits_last.npy", logits.data(), {(size_t) n_vocab});
        save_npy(d + "/layers.npy", layers.data(), {(size_t) st.n_layer + 1, T, H});
        save_npy(d + "/final_cb.npy", final_b.data(), {T, H});
        {
            std::ofstream f(d + "/llama_ids.json");
            f << ids_json(lt);
        }
        const double sa = secs(a0, a1), sac = secs(a1, a2), sb = secs(b0, b1);
        {
            std::ofstream f(d + "/meta.json");
            f << "{\"engine\": \"" << engine << "\", \"prompt_id\": \"" << it.id << "\", \"n_tokens\": " << T
              << ", \"llama_tokenization_equal_hf\": " << (lt == ids ? "true" : "false")
              << ", \"pass_a_decode_s\": " << sa << ", \"pass_a_copy_s\": " << sac
              << ", \"pass_b_decode_with_capture_s\": " << sb << "}\n";
        }
        tot_a += sa; tot_a_copy += sac; tot_b += sb; tot_tok += T;
        std::fprintf(stderr, "%s T=%zu tok_eq=%d A=%.4fs B=%.4fs\n", it.id.c_str(), T, (int) (lt == ids), sa, sb);
    }
    {
        std::ofstream f(out_dir + "/" + engine + "/_throughput.json");
        f << "{\"engine\": \"" << engine << "\", \"model\": \"" << model_path << "\", \"load_s\": " << load_s
          << ", \"n_prompts\": " << items.size() << ", \"n_tokens\": " << tot_tok
          << ", \"pass_a_decode_s\": " << tot_a << ", \"pass_a_copy_s\": " << tot_a_copy
          << ", \"pass_b_decode_with_capture_s\": " << tot_b
          << ", \"tok_per_s_final_only\": " << tot_tok / tot_a
          << ", \"tok_per_s_final_only_plus_copy\": " << tot_tok / (tot_a + tot_a_copy)
          << ", \"tok_per_s_all_layers_capture\": " << tot_tok / tot_b << "}\n";
    }
    llama_free(ctx_cb);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
