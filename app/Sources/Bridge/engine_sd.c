#ifndef ENGINE_SD_H
#define ENGINE_SD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Thin wrapper around stable-diffusion.cpp. The app only ever talks to this
 * header, so a change in the engine's own header only touches engine_sd.c. */

typedef void (*eng_log_cb)(int level, const char* text, void* user);
typedef void (*eng_progress_cb)(const char* stage, const char* detail, int64_t completed, int64_t total, void* user);

typedef struct {
    const char* diffusion_path;
    const char* llm_path;         /* text encoder */
    const char* llm_vision_path;  /* vision projector, editing only */
    const char* vae_path;
    const char* backend;          /* e.g. "diffusion=metal,te=cpu,vae=metal"; NULL or empty = automatic */
    const char* params_backend;   /* e.g. "diffusion=disk,te=disk"; NULL or empty = automatic */
    const char* max_vram;         /* GiB budget as text; NULL or empty = live free memory */
    const char* model_args;       /* e.g. "qwen_image_2_1_prefix_cache=false" */
    const char* flash_attn_mode;  /* "auto", "on", "off" or per-module assignments */
    const char* conv_direct_mode; /* NULL or empty = engine default */
    int n_threads;                /* 0 = physical core count */
    float linear_scale;           /* 0 keeps the model default */
    float attn_scale;             /* 0 keeps the model default */
    bool enable_mmap;
    bool auto_fit;
    bool disable_segmented_compute;
    bool disable_prefetch;
    bool eager_load;
} eng_sd_load_params;

typedef struct {
    const char* prompt;
    const char* negative_prompt;
    int width;
    int height;
    int steps;
    float cfg_scale;
    int64_t seed;
    int sample_method;            /* -1 keeps the model default */
    int scheduler;                /* -1 keeps the model default */
    const char* sigmas_csv;       /* comma separated custom sigmas; NULL or empty = none */
    const char* extra_sample_args;
} eng_sd_gen_params;

void eng_sd_set_callbacks(eng_log_cb log_cb, eng_progress_cb progress_cb, void* user);

/* Writes "name<TAB>description" lines. Returns the bytes needed (without the terminator). */
size_t eng_sd_devices(char* buffer, size_t buffer_size);
const char* eng_sd_commit(void);
const char* eng_sd_version(void);
const char* eng_sd_system_info(void);

bool eng_sd_load(const eng_sd_load_params* params, char* err, size_t err_size);
void eng_sd_unload(void);
bool eng_sd_is_loaded(void);

/* On success the caller owns *out_pixels (8-bit, interleaved) and *out_info; release both with eng_free. */
bool eng_sd_generate(const eng_sd_gen_params* params,
                     uint8_t** out_pixels,
                     int* out_width,
                     int* out_height,
                     int* out_channels,
                     char** out_info,
                     char* err,
                     size_t err_size);

void eng_sd_cancel(void);
void eng_free(void* ptr);

/* Per-operation profiler. While it is on, the engine evaluates every graph node on its own and
 * waits for it, so absolute times are inflated; the shares and the achieved GFLOPS per
 * operation type and shape are what matters. Start it before a generation, read the report after. */
void eng_sd_profile_start(void);
size_t eng_sd_profile_report(char* buffer, size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif
