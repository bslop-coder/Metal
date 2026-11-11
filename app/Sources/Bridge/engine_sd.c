#include "engine_sd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>

#include "ggml.h"
#include "stable-diffusion.h"

static sd_ctx_t* g_ctx = NULL;
static eng_log_cb g_log_cb = NULL;
static eng_progress_cb g_progress_cb = NULL;
static void* g_user = NULL;

static void emit_log(int level, const char* text) {
    if (g_log_cb && text) {
        g_log_cb(level, text, g_user);
    }
}

static void set_err(char* err, size_t err_size, const char* message) {
    if (err && err_size > 0) {
        snprintf(err, err_size, "%s", message ? message : "unknown error");
    }
}

static const char* nz(const char* s) {
    return (s && s[0]) ? s : NULL;
}

static void sd_log_trampoline(enum sd_log_level_t level, const char* text, void* data) {
    (void)data;
    emit_log((int)level, text);
}

static void sd_generation_progress_trampoline(const sd_generation_progress_t* p, void* data) {
    (void)data;
    if (g_progress_cb && p) {
        g_progress_cb(p->stage ? p->stage : "",
                      p->detail ? p->detail : "",
                      p->completed,
                      p->total,
                      g_user);
    }
}

static void sd_step_progress_trampoline(int step, int steps, float time, void* data) {
    (void)data;
    if (g_progress_cb) {
        char detail[64];
        snprintf(detail, sizeof(detail), "%.2f s per step", (double)time);
        g_progress_cb("sampling", detail, step, steps, g_user);
    }
}

void eng_sd_set_callbacks(eng_log_cb log_cb, eng_progress_cb progress_cb, void* user) {
    g_log_cb = log_cb;
    g_progress_cb = progress_cb;
    g_user = user;
    sd_set_log_callback(sd_log_trampoline, NULL);
    sd_set_progress_callback(sd_step_progress_trampoline, NULL);
}

size_t eng_sd_devices(char* buffer, size_t buffer_size) {
    return sd_list_devices(buffer, buffer_size);
}

const char* eng_sd_commit(void) {
    return sd_commit();
}

const char* eng_sd_version(void) {
    return sd_version();
}

const char* eng_sd_system_info(void) {
    return sd_get_system_info();
}

bool eng_sd_is_loaded(void) {
    return g_ctx != NULL;
}

void eng_sd_unload(void) {
    if (g_ctx) {
        free_sd_ctx(g_ctx);
        g_ctx = NULL;
    }
}

bool eng_sd_load(const eng_sd_load_params* p, char* err, size_t err_size) {
    if (!p) {
        set_err(err, err_size, "no load parameters");
        return false;
    }
    eng_sd_unload();

    sd_ctx_params_t cp;
    sd_ctx_params_init(&cp);

    cp.diffusion_model_path = nz(p->diffusion_path);
    cp.llm_path = nz(p->llm_path);
    cp.llm_vision_path = nz(p->llm_vision_path);
    cp.vae_path = nz(p->vae_path);

    if (nz(p->backend)) cp.backend = p->backend;
    if (nz(p->params_backend)) cp.params_backend = p->params_backend;
    if (nz(p->max_vram)) cp.max_vram = p->max_vram;
    if (nz(p->model_args)) cp.model_args = p->model_args;
    if (nz(p->flash_attn_mode)) cp.flash_attn_mode = p->flash_attn_mode;
    if (nz(p->conv_direct_mode)) cp.conv_direct_mode = p->conv_direct_mode;

    cp.n_threads = p->n_threads > 0 ? p->n_threads : (int)sd_get_num_physical_cores();
    cp.linear_scale = p->linear_scale;
    cp.attn_scale = p->attn_scale;
    cp.enable_mmap = p->enable_mmap;
    cp.auto_fit = p->auto_fit;
    cp.disable_segmented_compute = p->disable_segmented_compute;
    cp.disable_prefetch = p->disable_prefetch;
    cp.eager_load = p->eager_load;

    char* summary = sd_ctx_params_to_str(&cp);
    if (summary) {
        emit_log(2, "context parameters:");
        emit_log(2, summary);
        free(summary);
    }

    g_ctx = new_sd_ctx(&cp);
    if (!g_ctx) {
        set_err(err, err_size, "new_sd_ctx returned NULL; the log above has the reason");
        return false;
    }

    sd_set_generation_progress_callback(g_ctx, sd_generation_progress_trampoline, NULL);

    if (!sd_ctx_supports_image_generation(g_ctx)) {
        eng_sd_unload();
        set_err(err, err_size, "this model does not support image generation");
        return false;
    }

    char line[160];
    snprintf(line, sizeof(line), "model version: %s", sd_get_model_version_name(g_ctx));
    emit_log(2, line);
    return true;
}

static int parse_sigmas(const char* csv, float** out) {
    *out = NULL;
    if (!csv || !csv[0]) {
        return 0;
    }
    int cap = 64;
    int n = 0;
    float* values = (float*)malloc(sizeof(float) * (size_t)cap);
    if (!values) {
        return 0;
    }
    const char* s = csv;
    while (*s) {
        char* end = NULL;
        float v = strtof(s, &end);
        if (end == s) {
            s++;
            continue;
        }
        if (n == cap) {
            cap *= 2;
            float* grown = (float*)realloc(values, sizeof(float) * (size_t)cap);
            if (!grown) {
                break;
            }
            values = grown;
        }
        values[n++] = v;
        s = end;
        while (*s == ',' || *s == ' ' || *s == '\n') {
            s++;
        }
    }
    if (n == 0) {
        free(values);
        return 0;
    }
    *out = values;
    return n;
}

bool eng_sd_generate(const eng_sd_gen_params* g,
                     uint8_t** out_pixels,
                     int* out_width,
                     int* out_height,
                     int* out_channels,
                     char** out_info,
                     char* err,
                     size_t err_size) {
    if (!g_ctx) {
        set_err(err, err_size, "no model loaded");
        return false;
    }
    if (!g || !out_pixels || !out_width || !out_height || !out_channels || !out_info) {
        set_err(err, err_size, "bad arguments");
        return false;
    }
    *out_pixels = NULL;
    *out_info = NULL;

    sd_cancel_generation(g_ctx, SD_CANCEL_RESET);

    sd_img_gen_params_t ip;
    sd_img_gen_params_init(&ip);
    sd_sample_params_init(&ip.sample_params);

    ip.prompt = g->prompt ? g->prompt : "";
    ip.negative_prompt = g->negative_prompt ? g->negative_prompt : "";
    ip.width = g->width;
    ip.height = g->height;
    ip.seed = g->seed;
    ip.batch_count = 1;

    ip.sample_params.sample_steps = g->steps;
    ip.sample_params.guidance.txt_cfg = g->cfg_scale;
    ip.sample_params.sample_method = g->sample_method >= 0
                                         ? (enum sample_method_t)g->sample_method
                                         : sd_get_default_sample_method(g_ctx);
    ip.sample_params.scheduler = g->scheduler >= 0
                                     ? (enum scheduler_t)g->scheduler
                                     : sd_get_default_scheduler(g_ctx, ip.sample_params.sample_method);
    ip.sample_params.extra_sample_args = nz(g->extra_sample_args);

    float* sigmas = NULL;
    int sigma_count = parse_sigmas(g->sigmas_csv, &sigmas);
    if (sigma_count > 0) {
        ip.sample_params.custom_sigmas = sigmas;
        ip.sample_params.custom_sigmas_count = sigma_count;
    }

    char* summary = sd_img_gen_params_to_str(&ip);
    if (summary) {
        emit_log(2, "generation parameters:");
        emit_log(2, summary);
        free(summary);
    }

    sd_image_generation_result_t* result = NULL;
    bool ok = generate_image(g_ctx, &ip, &result);
    free(sigmas);

    if (!ok || !result || result->count < 1 || !result->items || !result->items[0].image.data) {
        if (result) {
            free_sd_image_generation_result(result);
        }
        set_err(err, err_size, "generation failed or was cancelled; the log has the reason");
        return false;
    }

    sd_image_t image = result->items[0].image;
    size_t bytes = (size_t)image.width * (size_t)image.height * (size_t)image.channel;
    uint8_t* pixels = (uint8_t*)malloc(bytes);
    if (!pixels) {
        free_sd_image_generation_result(result);
        set_err(err, err_size, "out of memory while copying the image");
        return false;
    }
    memcpy(pixels, image.data, bytes);

    *out_pixels = pixels;
    *out_width = (int)image.width;
    *out_height = (int)image.height;
    *out_channels = (int)image.channel;
    if (result->items[0].generation_info) {
        size_t info_len = strlen(result->items[0].generation_info) + 1;
        char* info_copy = (char*)malloc(info_len);
        if (info_copy) {
            memcpy(info_copy, result->items[0].generation_info, info_len);
            *out_info = info_copy;
        }
    }

    free_sd_image_generation_result(result);
    return true;
}

void eng_sd_cancel(void) {
    if (g_ctx) {
        sd_cancel_generation(g_ctx, SD_CANCEL_ALL);
    }
}

void eng_free(void* ptr) {
    free(ptr);
}

/* ---------------------------------------------------------------------------
 * Profiler
 * ------------------------------------------------------------------------- */

#define PROF_MAX 512

typedef struct {
    int used;
    int op;
    int type;
    int64_t k;
    int64_t m;
    int64_t n;
    int64_t count;
    double ms;
    double flops;
} prof_entry;

static prof_entry g_prof[PROF_MAX];
static double g_prof_ask_ms = 0.0;
static double g_prof_total_ms = 0.0;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static prof_entry* prof_slot(int op, int type, int64_t k, int64_t m, int64_t n) {
    for (int i = 0; i < PROF_MAX; i++) {
        prof_entry* e = &g_prof[i];
        if (!e->used) {
            e->used = 1;
            e->op = op;
            e->type = type;
            e->k = k;
            e->m = m;
            e->n = n;
            return e;
        }
        if (e->op == op && e->type == type && e->k == k && e->m == m && e->n == n) {
            return e;
        }
    }
    return NULL;
}

static bool prof_eval_callback(struct ggml_tensor* t, bool ask, void* user) {
    (void)user;
    if (ask) {
        g_prof_ask_ms = now_ms();
        return true;
    }
    double dt = now_ms() - g_prof_ask_ms;
    g_prof_total_ms += dt;

    int op = (int)t->op;
    int type = (int)t->type;
    int64_t k = 0;
    int64_t m = 0;
    int64_t n = 0;
    double flops = 0.0;

    if ((t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) && t->src[0]) {
        const struct ggml_tensor* a = t->src[0];
        type = (int)a->type;
        k = a->ne[0];
        m = t->ne[0];
        n = t->ne[1];
        flops = 2.0 * (double)k * (double)m * (double)t->ne[1] * (double)t->ne[2] * (double)t->ne[3];
    } else if (t->op == GGML_OP_FLASH_ATTN_EXT && t->src[0] && t->src[1]) {
        const struct ggml_tensor* q = t->src[0];
        const struct ggml_tensor* kt = t->src[1];
        type = (int)kt->type;
        k = q->ne[0];
        m = kt->ne[1];
        n = q->ne[1];
        flops = 4.0 * (double)q->ne[0] * (double)q->ne[1] * (double)kt->ne[1] * (double)q->ne[2] * (double)q->ne[3];
    }

    prof_entry* e = prof_slot(op, type, k, m, n);
    if (e) {
        e->count += 1;
        e->ms += dt;
        e->flops += flops;
    }
    return true;
}

void eng_sd_profile_start(void) {
    memset(g_prof, 0, sizeof(g_prof));
    g_prof_total_ms = 0.0;
    sd_set_backend_eval_callback(prof_eval_callback, NULL);
}

size_t eng_sd_profile_report(char* buffer, size_t buffer_size) {
    sd_set_backend_eval_callback(NULL, NULL);
    if (!buffer || buffer_size == 0) {
        return 0;
    }
    buffer[0] = '\0';

    int order[PROF_MAX];
    int count = 0;
    for (int i = 0; i < PROF_MAX; i++) {
        if (g_prof[i].used) {
            order[count++] = i;
        }
    }
    if (count == 0) {
        snprintf(buffer, buffer_size, "profile: no graph nodes were reported to the callback");
        return strlen(buffer);
    }
    for (int i = 0; i < count - 1; i++) {
        int best = i;
        for (int j = i + 1; j < count; j++) {
            if (g_prof[order[j]].ms > g_prof[order[best]].ms) {
                best = j;
            }
        }
        int tmp = order[i];
        order[i] = order[best];
        order[best] = tmp;
    }

    size_t used = 0;
    int written = snprintf(buffer + used, buffer_size - used,
                           "profile: %d groups, %.1f s measured in total (inflated by per-node waits)\n",
                           count, g_prof_total_ms / 1000.0);
    if (written > 0) {
        used += (size_t)written;
    }
    int shown = count < 28 ? count : 28;
    for (int i = 0; i < shown && used + 160 < buffer_size; i++) {
        const prof_entry* e = &g_prof[order[i]];
        double seconds = e->ms / 1000.0;
        double share = g_prof_total_ms > 0.0 ? 100.0 * e->ms / g_prof_total_ms : 0.0;
        double gflops = (e->flops > 0.0 && e->ms > 0.0) ? e->flops / (e->ms * 1.0e6) : 0.0;
        written = snprintf(buffer + used, buffer_size - used,
                           "%-14s %-8s K=%lld M=%lld N=%lld count=%lld  %.2f s  %.1f%%  %.0f GFLOPS\n",
                           ggml_op_name((enum ggml_op)e->op),
                           ggml_type_name((enum ggml_type)e->type),
                           (long long)e->k, (long long)e->m, (long long)e->n,
                           (long long)e->count, seconds, share, gflops);
        if (written > 0) {
            used += (size_t)written;
        }
    }
    return used;
}
