/*
 * npu_check.c - self-contained RKNN NPU check for Orange Pi CM5 (rk3588s, aarch64).
 * Runtime: librknnrt.so 2.3.0, linked as: -L. -lrknnrt
 * No rknn_api.h on target: types/functions declared here (layout per public RKNN C API v2.x).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

typedef uint64_t rknn_context;

/* ---------------- enum declarations (public rknn_api.h order/values) ---------------- */

typedef enum _rknn_query_cmd {
    RKNN_QUERY_IN_OUT_NUM  = 0,
    RKNN_QUERY_INPUT_ATTR  = 1,
    RKNN_QUERY_OUTPUT_ATTR = 2,
    RKNN_QUERY_SDK_VERSION = 5
} rknn_query_cmd;

typedef enum _rknn_tensor_type {
    RKNN_TENSOR_FLOAT32 = 0,
    RKNN_TENSOR_FLOAT16 = 1,
    RKNN_TENSOR_INT8    = 2,
    RKNN_TENSOR_UINT8   = 3,
    RKNN_TENSOR_INT16   = 4,
    RKNN_TENSOR_UINT16  = 5,
    RKNN_TENSOR_INT32   = 6,
    RKNN_TENSOR_UINT32  = 7,
    RKNN_TENSOR_INT64   = 8,
    RKNN_TENSOR_BOOL    = 9
} rknn_tensor_type;

typedef enum _rknn_tensor_format {
    RKNN_TENSOR_NCHW      = 0, /* official rknn_api.h v2.3.0: NCHW=0, NHWC=1, NC1HWC2=2, UNDEFINED=3 */
    RKNN_TENSOR_NHWC      = 1,
    RKNN_TENSOR_NC1HWC2   = 2,
    RKNN_TENSOR_UNDEFINED = 3
} rknn_tensor_format;

typedef enum _rknn_tensor_qnt_type {
    RKNN_TENSOR_QNT_NONE         = 0,
    RKNN_TENSOR_QNT_DFP          = 1,
    RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC = 2
} rknn_tensor_qnt_type;

typedef enum _rknn_io_pass_through {
    RKNN_IO_PASS_THROUGH_DEFAULT = 0,
    RKNN_IO_PASS_THROUGH_TRUE    = 1,
    RKNN_IO_PASS_THROUGH_FALSE   = 2
} rknn_io_pass_through;

/* ---------------- struct declarations ---------------- */

typedef struct _rknn_input_output_num {
    uint32_t n_input;
    uint32_t n_output;
} rknn_input_output_num;

/* Official rknn_api.h v2.x field layout; final sizes verified by asserts */
typedef struct _rknn_tensor_attr {
    uint32_t index;              /* 0 */
    uint32_t n_dims;             /* 4 */
    uint32_t dims[16];           /* 8..71 */
    char     name[256];          /* 72..327 */
    uint32_t n_elems;            /* 328 */
    uint32_t size;               /* 332 */
    uint64_t size64;             /* 336..343 */
    uint32_t fmt;                /* 344 */
    uint32_t type;               /* 348 */
    uint32_t qnt_type;           /* 352 */
    int8_t   fl;                 /* 356 */
    int32_t  zp;                 /* 360 */
    float    scale;              /* 364 */
    int32_t  w_stride;           /* 368 */
    uint32_t size_with_stride;   /* 372..375 -> sizeof = 376 (reported by environment 06.11) */
} rknn_tensor_attr;

typedef struct _rknn_sdk_version {
    char api_version[256];
    char drv_version[256];
} rknn_sdk_version;

typedef struct _rknn_input {
    uint32_t index;              /* 0 */
    void    *buf;                /* 8..15 */
    uint32_t size;               /* 16 */
    uint8_t  pass_through;       /* 20 */
    int32_t  type;               /* 24 */
    int32_t  fmt;                /* 28 -> sizeof = 32 */
} rknn_input;

typedef struct _rknn_output {
    uint8_t  want_float;         /* 0 */
    uint8_t  is_prealloc;        /* 1 */
    uint32_t index;              /* 4 */
    void    *buf;                /* 8..15 */
    uint32_t size;               /* 16 -> sizeof = 24 */
} rknn_output;

_Static_assert(sizeof(rknn_tensor_attr) == 376, "rknn_tensor_attr must be 376 bytes (runtime 2.3.0)");
_Static_assert(sizeof(rknn_sdk_version) == 512, "rknn_sdk_version must be 512");
_Static_assert(sizeof(rknn_input) == 32, "rknn_input must be 32");
_Static_assert(sizeof(rknn_output) == 24, "rknn_output must be 24");

/* ---------------- external function declarations (names from rknn_symbols.txt) ---------------- */

extern int rknn_init(rknn_context *context, void *model, uint32_t size, uint32_t auto_batch, uint32_t ext_size);
extern int rknn_destroy(rknn_context context);
extern int rknn_query(rknn_context context, rknn_query_cmd cmd, void *info, uint32_t size);
extern int rknn_inputs_set(rknn_context context, uint32_t n_inputs, rknn_input inputs[]);
extern int rknn_run(rknn_context context, void *ext_info);
extern int rknn_outputs_get(rknn_context context, uint32_t n_outputs, rknn_output outputs[], void *ext_info);
extern int rknn_outputs_release(rknn_context context, uint32_t n_outputs, rknn_output outputs[]);

/* ---------------- helpers ---------------- */

static const char *type_name(uint32_t t)
{
    static const char *names[] = {
        "FLOAT32", "FLOAT16", "INT8", "UINT8", "INT16",
        "UINT16", "INT32", "UINT32", "INT64", "BOOL"
    };
    if (t < sizeof(names) / sizeof(names[0])) return names[t];
    return "UNKNOWN";
}

static const char *fmt_name(uint32_t f)
{
    static const char *names[] = { "UNDEFINED", "NCHW", "NHWC", "NC1HWC2", "NFNC" };
    if (f < sizeof(names) / sizeof(names[0])) return names[f];
    return "UNKNOWN";
}

static const char *qnt_name(uint32_t q)
{
    static const char *names[] = { "NONE", "DFP", "AFFINE_ASYMMETRIC" };
    if (q < sizeof(names) / sizeof(names[0])) return names[q];
    return "UNKNOWN";
}

static void die(const char *step, int ret)
{
    fprintf(stderr, "NPU_CHECK_FAIL step=%s ret=%d\n", step, ret);
    exit(1);
}

static void die_msg(const char *msg)
{
    fprintf(stderr, "NPU_CHECK_FAIL %s\n", msg);
    exit(1);
}

static void print_tensor_attr(const char *role, const rknn_tensor_attr *a)
{
    uint32_t i;
    printf("%s[%u] name=%s dims=", role, a->index, a->name);
    for (i = 0; i < a->n_dims && i < 16; i++)
        printf("%s%u", i ? "x" : "", a->dims[i]);
    printf(" n_elems=%u size=%u type=%s fmt=%s qnt=%s zp=%d scale_exp=%d\n",
           a->n_elems, a->size, type_name(a->type), fmt_name(a->fmt),
           qnt_name(a->qnt_type), a->zp, a->fl);
}

static long diff_us(const struct timespec *t0, const struct timespec *t1)
{
    return (long)(t1->tv_sec - t0->tv_sec) * 1000000L
         + (long)(t1->tv_nsec - t0->tv_nsec) / 1000L;
}

static void *read_whole_file(const char *path, uint64_t expect, size_t *out_size)
{
    FILE *f;
    long sz;
    unsigned char *buf;
    size_t got;

    f = fopen(path, "rb");
    if (!f) die_msg("cannot open model file");

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); die_msg("fseek failed"); }
    sz = ftell(f);
    if (sz <= 0) { fclose(f); die_msg("ftell failed or empty file"); }
    rewind(f);

    if (expect != 0 && (uint64_t)sz != expect) {
        fprintf(stderr, "NPU_CHECK_FAIL file size %ld != expected %lu\n", sz, (unsigned long)expect);
        fclose(f);
        exit(1);
    }

    buf = malloc((size_t)sz);
    if (!buf) { fclose(f); die_msg("out of memory"); }
    got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) { free(buf); die_msg("short read"); }

    *out_size = (size_t)sz;
    return buf;
}

int main(int argc, char **argv)
{
    const char *model_path;
    const char *image_path;
    void *model_buf;
    size_t model_size;
    rknn_context ctx = 0;
    rknn_input_output_num io_num;
    rknn_sdk_version sdk;
    rknn_tensor_attr *in_attrs;
    rknn_tensor_attr *out_attrs;
    rknn_input in;
    unsigned char *in_data;
    uint32_t n_in_elems;
    size_t in_bytes;
    rknn_output *outs;
    int ret;
    int i;
    int run_idx;
    long total_us = 0;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.rknn> [raw_input_float32_le.bin]\n", argv[0]);
        return 1;
    }
    model_path = argv[1];
    image_path = (argc >= 3) ? argv[2] : NULL;

    model_buf = read_whole_file(model_path, 0, &model_size);

    ret = rknn_init(&ctx, model_buf, (uint32_t)model_size, 0, 0);
    if (ret != 0) die("rknn_init", ret);
    printf("NPU_CHECK_OK rknn_init model=%s size=%lu\n", model_path, (unsigned long)model_size);

    memset(&sdk, 0, sizeof(sdk));
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk, sizeof(sdk));
    if (ret != 0) die("rknn_query SDK_VERSION", ret);
    printf("NPU_CHECK_OK sdk_version api=%s drv=%s\n", sdk.api_version, sdk.drv_version);

    memset(&io_num, 0, sizeof(io_num));
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != 0) die("rknn_query IN_OUT_NUM", ret);
    printf("NPU_CHECK_OK in_out_num n_input=%u n_output=%u\n", io_num.n_input, io_num.n_output);

    if (io_num.n_input == 0 || io_num.n_output == 0) die_msg("model reports zero inputs or outputs");
    if (io_num.n_input != 1)
        printf("note: only input[0] is fed, model declares %u inputs\n", io_num.n_input);

    in_attrs = calloc(io_num.n_input, sizeof(rknn_tensor_attr));
    if (!in_attrs) die_msg("out of memory");
    for (i = 0; (uint32_t)i < io_num.n_input; i++) {
        memset(&in_attrs[i], 0, sizeof(rknn_tensor_attr));
        in_attrs[i].index = (uint32_t)i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != 0) die("rknn_query INPUT_ATTR", ret);
        print_tensor_attr("input", &in_attrs[i]);
    }
    printf("NPU_CHECK_OK input_attrs\n");

    out_attrs = calloc(io_num.n_output, sizeof(rknn_tensor_attr));
    if (!out_attrs) die_msg("out of memory");
    for (i = 0; (uint32_t)i < io_num.n_output; i++) {
        memset(&out_attrs[i], 0, sizeof(rknn_tensor_attr));
        out_attrs[i].index = (uint32_t)i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != 0) die("rknn_query OUTPUT_ATTR", ret);
        print_tensor_attr("output", &out_attrs[i]);
    }
    printf("NPU_CHECK_OK output_attrs\n");

    n_in_elems = in_attrs[0].n_elems;
    if (n_in_elems == 0) die_msg("input[0] n_elems is zero");
    in_bytes = (size_t)n_in_elems; /* byte uint8 input, as in the official sample (no normalization) */
    in_data = malloc(in_bytes);
    if (!in_data) die_msg("out of memory");

    if (image_path) {
        size_t got_size = 0;
        void *file_in = read_whole_file(image_path, (uint64_t)in_bytes, &got_size);
        memcpy(in_data, file_in, in_bytes);
        free(file_in);
        printf("image loaded from %s (%lu bytes)\n", image_path, (unsigned long)got_size);
    } else {
        uint32_t k;
        for (k = 0; k < n_in_elems; k++)
            in_data[k] = (unsigned char)((k * 3 + k / 224 + k / (224 * 224)) & 0xFFu);
        printf("image synthesized as gradient (%lu bytes)\n", (unsigned long)in_bytes);
    }

    memset(&in, 0, sizeof(in));
    in.index = 0;
    in.type = RKNN_TENSOR_UINT8;
    in.fmt = RKNN_TENSOR_NHWC;
    in.size = (uint32_t)in_bytes;
    in.buf = in_data;
    in.pass_through = 0;

    ret = rknn_inputs_set(ctx, 1, &in);
    if (ret != 0) die("rknn_inputs_set", ret);
    printf("NPU_CHECK_OK inputs_set\n");

    for (run_idx = 0; run_idx < 10; run_idx++) {
        struct timespec t0, t1;
        long us;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ret = rknn_run(ctx, NULL);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (ret != 0) die("rknn_run", ret);
        us = diff_us(&t0, &t1);
        total_us += us;
        printf("NPU_CHECK_OK run[%d] time_us=%ld\n", run_idx, us);
    }
    printf("NPU_CHECK_OK runs_avg avg_time_us=%ld\n", total_us / 10);

    outs = calloc(io_num.n_output, sizeof(rknn_output));
    if (!outs) die_msg("out of memory");
    for (i = 0; (uint32_t)i < io_num.n_output; i++) {
        outs[i].index = (uint32_t)i;
        outs[i].want_float = 1;
        outs[i].is_prealloc = 0;
    }

    ret = rknn_outputs_get(ctx, io_num.n_output, outs, NULL);
    if (ret != 0) die("rknn_outputs_get", ret);
    printf("NPU_CHECK_OK outputs_get\n");

    for (i = 0; (uint32_t)i < io_num.n_output; i++) {
        float *f = (float *)outs[i].buf;
        uint32_t n = out_attrs[i].n_elems;
        uint32_t take = (n < 8) ? n : 8;
        float sum = 0.0f;
        float top_v[3];
        uint32_t top_i[3];
        uint32_t top_cnt = 0;
        uint32_t j, k;
        if (!f) {
            printf("output[%d] empty buffer\n", i);
            continue;
        }
        for (j = 0; j < take; j++) sum += f[j];
        printf("output[%d] name=%s first%u_sum=%.6f\n", i, out_attrs[i].name, take, sum);

        top_v[0] = top_v[1] = top_v[2] = 0.0f;
        top_i[0] = top_i[1] = top_i[2] = 0u;
        for (j = 0; j < n; j++) {
            if (top_cnt < 3) {
                k = top_cnt;
                top_cnt++;
            } else if (f[j] > top_v[2]) {
                k = 2;
            } else {
                continue;
            }
            while (k > 0 && f[top_i[k - 1]] < f[j]) {
                top_v[k] = top_v[k - 1];
                top_i[k] = top_i[k - 1];
                k--;
            }
            top_v[k] = f[j];
            top_i[k] = j;
        }
        for (k = 0; k < top_cnt; k++)
            printf("output[%d] TOP k=%u idx=%u p=%.6f %s\n", i, k + 1, top_i[k], top_v[k],
                   top_v[k] >= 0.10f ? "OK" : "CUT(threshold 10%)");
        {
            uint32_t above = 0;
            for (k = 0; k < top_cnt; k++)
                if (top_v[k] >= 0.10f) above++;
            printf("output[%d] ABOVE_THRESHOLD n=%u (threshold 0.10)\n", i, above);
        }
    }
    printf("NPU_CHECK_OK output_checksums\n");

    ret = rknn_outputs_release(ctx, io_num.n_output, outs);
    if (ret != 0) die("rknn_outputs_release", ret);
    printf("NPU_CHECK_OK outputs_release\n");

    ret = rknn_destroy(ctx);
    if (ret != 0) die("rknn_destroy", ret);
    printf("NPU_CHECK_OK destroy\n");
    printf("NPU_CHECK_OK all\n");

    free(outs);
    free(out_attrs);
    free(in_attrs);
    free(in_data);
    free(model_buf);
    return 0;
}
