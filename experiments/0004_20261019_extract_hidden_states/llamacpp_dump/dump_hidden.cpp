// dump_hidden: extract per-token hidden states from a GGUF model with llama.cpp.
//
// Usage:
//   dump_hidden <model.gguf> <manifest.tsv> <variant_out_dir> <n_ctx> <flash_attn 0|1> <n_gpu_layers>
//
// manifest.tsv: one line per prompt: "<prompt_id>\t<text_file>\t<ids_file>"
//   text_file : raw prompt text (tokenized with the GGUF vocab for an agreement check only)
//   ids_file  : JSON list of reference token IDs (HF tokenizer); the model is always
//               evaluated on exactly these IDs so every engine sees identical input.
//
// For each prompt two passes are run on a cleared KV cache, each as ONE llama_decode
// (n_batch = n_ubatch = n_ctx, every position requests an output):
//   pass A (no eval callback): llama_get_embeddings_ith with pooling NONE
//          (= graph tensor "result_norm", AFTER the final RMSNorm) and the logits of the
//          last prompt token (llama_get_logits_ith).
//   pass B (eval callback cb_eval): copies "inp_embd" (embedding output), every
//          "l_out-<il>" (residual stream after block il, BEFORE the final norm) and
//          "result_norm" from the graph to host memory.
//
// Outputs in <variant_out_dir>/<prompt_id>/ (float32 C-order .npy, layer names shared
// with the HF and vLLM tasks of experiment 0004):
//   emb.npy            [T, H]  pass B inp_embd                 (= HF hidden_states[0])
//   pre_L01..pre_LNN   [T, H]  pass B l_out-(k-1), k = 1..N    (= HF block k output, pre-norm)
//   final_norm.npy     [T, H]  pass A embeddings API           (= HF last_hidden_state)
//   final_norm_cb.npy  [T, H]  pass B result_norm
//   last_logits.npy    [V]     pass A logits of position T-1   (only if available)
//   llama_token_ids.json       GGUF tokenization of text_file
//   dump_meta.json             timings and checks; written LAST (marks the prompt done)
// Any error exits non-zero; prompts already finished keep their dump_meta.json.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cerrno>
#include <chrono>
#include <cmath>
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

[[noreturn]] void die(int code, const std::string & msg) {
    std::fprintf(stderr, "dump_hidden error: %s\n", msg.c_str());
    std::fflush(stderr);
    std::exit(code);
}

void mkdirs(const std::string & path) {
    std::string cur;
    std::stringstream ss(path);
    std::string part;
    if (!path.empty() && path[0] == '/') cur = "/";
    while (std::getline(ss, part, '/')) {
        if (part.empty()) continue;
        cur += part + "/";
        if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) die(2, "mkdir failed: " + cur);
    }
}

// Write a float32 C-order .npy file (format version 1.0) atomically (tmp + rename).
void save_npy(const std::string & path, const float * data, const std::vector<size_t> & shape) {
    std::string shp = "(";
    size_t n = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        shp += std::to_string(shape[i]);
        if (shape.size() == 1) shp += ",";
        else if (i + 1 < shape.size()) shp += ", ";
        n *= shape[i];
    }
    shp += ")";
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shp + ", }";
    const size_t total = 10 + header.size() + 1;
    const size_t pad = (64 - total % 64) % 64;
    header += std::string(pad, ' ') + "\n";
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f) die(2, "cannot open " + tmp);
        const char magic[] = "\x93NUMPY";
        f.write(magic, 6);
        const char ver[2] = {1, 0};
        f.write(ver, 2);
        const uint16_t hl = (uint16_t) header.size();
        f.write(reinterpret_cast<const char *>(&hl), 2);
        f.write(header.data(), (std::streamsize) header.size());
        f.write(reinterpret_cast<const char *>(data), (std::streamsize) (n * sizeof(float)));
        if (!f) die(2, "write failed " + tmp);
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) die(2, "rename failed " + path);
}

void write_text(const std::string & path, const std::string & content) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp);
        if (!f) die(2, "cannot open " + tmp);
        f << content;
        if (!f) die(2, "write failed " + tmp);
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) die(2, "rename failed " + path);
}

std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die(2, "cannot open " + path);
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
    if (!ss.eof()) die(2, "non-integer content in " + path);
    if (ids.empty()) die(2, "no ids in " + path);
    return ids;
}

std::string ids_json(const std::vector<llama_token> & ids) {
    std::string s = "[";
    for (size_t i = 0; i < ids.size(); ++i) s += std::to_string(ids[i]) + (i + 1 < ids.size() ? ", " : "");
    return s + "]";
}

bool all_finite(const std::vector<float> & v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

struct capture_state {
    bool enabled = false;
    int n_embd = 0;
    // tensor name -> rows (row-major [n_tokens][n_embd]) appended across ubatches
    std::map<std::string, std::vector<float>> data;
    std::vector<uint8_t> buf;
};

bool wanted(const char * name) {
    if (std::strcmp(name, "inp_embd") == 0 || std::strcmp(name, "result_norm") == 0) return true;
    return std::strncmp(name, "l_out-", 6) == 0;
}

// ggml_backend_sched_eval_callback: ask==true -> "do you want to observe this node?",
// ask==false -> node computed, copy it.
bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<capture_state *>(user_data);
    if (!st->enabled) return ask ? false : true;
    if (ask) return wanted(t->name);
    if (!wanted(t->name)) return true;
    if (t->ne[0] != st->n_embd || t->ne[2] != 1 || t->ne[3] != 1) {
        die(3, std::string("unexpected shape for ") + t->name + ": " + std::to_string(t->ne[0]) + " " +
               std::to_string(t->ne[1]) + " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]));
    }
    const size_t nbytes = ggml_nbytes(t);
    const uint8_t * src;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        src = static_cast<const uint8_t *>(t->data);
    } else {
        st->buf.resize(nbytes);
        ggml_backend_tensor_get(t, st->buf.data(), 0, nbytes);
        src = st->buf.data();
    }
    auto & dst = st->data[t->name];
    const int64_t nr = t->ne[1];
    const int64_t nc = t->ne[0];
    dst.reserve(dst.size() + (size_t) (nr * nc));
    for (int64_t r = 0; r < nr; ++r) {
        for (int64_t c = 0; c < nc; ++c) {
            const uint8_t * p = src + r * t->nb[1] + c * t->nb[0];
            float v;
            switch (t->type) {
                case GGML_TYPE_F32: v = *reinterpret_cast<const float *>(p); break;
                case GGML_TYPE_F16: v = ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(p)); break;
                case GGML_TYPE_BF16: v = ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(p)); break;
                default:
                    die(3, std::string("unsupported type ") + ggml_type_name(t->type) + " for " + t->name);
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
        batch.logits[i] = 1;  // every position is an output (embeddings for all tokens)
    }
    const int rc = llama_decode(ctx, batch);
    llama_synchronize(ctx);
    llama_batch_free(batch);
    return rc;
}

char layer_name_buf[32];
std::string pre_layer_name(int k) {
    std::snprintf(layer_name_buf, sizeof(layer_name_buf), "pre_L%02d", k);
    return layer_name_buf;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 7) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <manifest.tsv> <variant_out_dir> <n_ctx> <flash_attn 0|1> <n_gpu_layers>\n",
                     argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string manifest = argv[2];
    const std::string out_dir = argv[3];
    const int n_ctx = std::atoi(argv[4]);
    const int fa = std::atoi(argv[5]);
    const int ngl = std::atoi(argv[6]);
    if (n_ctx <= 0) die(1, "n_ctx must be positive");
    if (fa != 0 && fa != 1) die(1, "flash_attn must be 0 or 1");

    llama_backend_init();

    const auto t_load0 = clk::now();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) die(1, "failed to load " + model_path);

    capture_state st;
    const int n_layer = llama_model_n_layer(model);
    st.n_embd = llama_model_n_embd(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    cp.n_batch = (uint32_t) n_ctx;
    cp.n_ubatch = (uint32_t) n_ctx;  // one ubatch per prompt
    cp.n_seq_max = 1;
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    // ctx: plain context (pass A). ctx_cb: same settings plus the eval callback (pass B).
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die(1, "failed to create context");
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = &st;
    llama_context * ctx_cb = llama_init_from_model(model, cp);
    if (!ctx_cb) die(1, "failed to create callback context");
    const double load_s = secs(t_load0, clk::now());
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::fprintf(stderr, "n_layer=%d n_embd=%d n_vocab=%d n_ctx=%d flash_attn=%s n_gpu_layers=%d load_s=%.3f\n",
                 n_layer, st.n_embd, n_vocab, n_ctx, llama_flash_attn_type_name(cp.flash_attn_type), ngl, load_s);

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
            if (it.id.empty() || it.text_file.empty() || it.ids_file.empty()) die(1, "bad manifest line: " + line);
            items.push_back(it);
        }
    }
    if (items.empty()) die(1, "empty manifest");

    // Warm-up on the first prompt (both passes), not timed and not saved.
    {
        const auto ids = read_ids(items.at(0).ids_file);
        llama_memory_clear(llama_get_memory(ctx), true);
        if (decode_all(ctx, ids) != 0) die(5, "warm-up decode A failed");
        st.enabled = true;
        llama_memory_clear(llama_get_memory(ctx_cb), true);
        if (decode_all(ctx_cb, ids) != 0) die(5, "warm-up decode B failed");
        st.enabled = false;
        st.data.clear();
    }

    for (const auto & it : items) {
        const std::string text = read_file(it.text_file);
        std::vector<llama_token> lt(text.size() + 16);
        const int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), lt.data(), (int32_t) lt.size(),
                                     /*add_special*/ false, /*parse_special*/ false);
        if (n < 0) die(5, "tokenize failed for " + it.id);
        lt.resize((size_t) n);
        const auto ids = read_ids(it.ids_file);
        const size_t T = ids.size();
        const size_t H = (size_t) st.n_embd;
        if ((int) T > n_ctx) die(5, "prompt " + it.id + " longer than n_ctx");
        for (llama_token id : ids) {
            if (id < 0 || id >= n_vocab) die(5, "token id out of GGUF vocab range in " + it.id);
        }
        const std::string d = out_dir + "/" + it.id;
        mkdirs(d);

        // pass A: embeddings API (result_norm) and last-token logits
        llama_memory_clear(llama_get_memory(ctx), true);
        const auto a0 = clk::now();
        if (decode_all(ctx, ids) != 0) die(5, "decode A failed for " + it.id);
        const auto a1 = clk::now();
        std::vector<float> final_a(T * H);
        for (size_t i = 0; i < T; ++i) {
            const float * e = llama_get_embeddings_ith(ctx, (int32_t) i);
            if (!e) die(5, "no embedding for token " + std::to_string(i) + " of " + it.id);
            std::memcpy(final_a.data() + i * H, e, H * sizeof(float));
        }
        const float * lg = llama_get_logits_ith(ctx, (int32_t) T - 1);
        std::vector<float> logits;
        if (lg) logits.assign(lg, lg + n_vocab);
        const auto a2 = clk::now();

        // pass B: graph tensors via cb_eval
        st.data.clear();
        st.enabled = true;
        llama_memory_clear(llama_get_memory(ctx_cb), true);
        const auto b0 = clk::now();
        if (decode_all(ctx_cb, ids) != 0) die(5, "decode B failed for " + it.id);
        const auto b1 = clk::now();
        st.enabled = false;

        auto take = [&](const std::string & name) -> const std::vector<float> & {
            auto f = st.data.find(name);
            if (f == st.data.end()) die(4, "tensor " + name + " not observed for " + it.id);
            if (f->second.size() != T * H) {
                die(4, "tensor " + name + " has " + std::to_string(f->second.size()) + " values, expected " +
                       std::to_string(T * H) + " for " + it.id);
            }
            return f->second;
        };

        bool finite = all_finite(final_a) && (logits.empty() || all_finite(logits));
        {
            const auto & v = take("inp_embd");
            finite = finite && all_finite(v);
            save_npy(d + "/emb.npy", v.data(), {T, H});
        }
        for (int il = 0; il < n_layer; ++il) {
            const auto & v = take("l_out-" + std::to_string(il));
            finite = finite && all_finite(v);
            save_npy(d + "/" + pre_layer_name(il + 1) + ".npy", v.data(), {T, H});
        }
        {
            const auto & v = take("result_norm");
            finite = finite && all_finite(v);
            save_npy(d + "/final_norm_cb.npy", v.data(), {T, H});
        }
        save_npy(d + "/final_norm.npy", final_a.data(), {T, H});
        if (!logits.empty()) save_npy(d + "/last_logits.npy", logits.data(), {(size_t) n_vocab});
        write_text(d + "/llama_token_ids.json", ids_json(lt));

        const double sa = secs(a0, a1), sac = secs(a1, a2), sb = secs(b0, b1);
        std::ostringstream m;
        m << "{\"prompt_id\": \"" << it.id << "\", \"n_tokens\": " << T << ", \"n_layer\": " << n_layer
          << ", \"n_embd\": " << H << ", \"n_vocab\": " << n_vocab
          << ", \"llama_tokenization_equal_hf\": " << (lt == ids ? "true" : "false")
          << ", \"logits_available\": " << (logits.empty() ? "false" : "true")
          << ", \"all_finite\": " << (finite ? "true" : "false")
          << ", \"pass_a_decode_s\": " << sa << ", \"pass_a_copy_s\": " << sac
          << ", \"pass_b_decode_with_capture_s\": " << sb << ", \"load_s\": " << load_s
          << ", \"flash_attn\": \"" << llama_flash_attn_type_name(cp.flash_attn_type) << "\""
          << ", \"n_ctx\": " << n_ctx << ", \"n_gpu_layers\": " << ngl << "}\n";
        write_text(d + "/dump_meta.json", m.str());
        std::fprintf(stderr, "%s T=%zu tok_eq=%d finite=%d A=%.4fs B=%.4fs\n", it.id.c_str(), T, (int) (lt == ids),
                     (int) finite, sa, sb);
        std::fflush(stderr);
    }
    llama_free(ctx_cb);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
