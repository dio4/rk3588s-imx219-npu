/*
 * npu_detect.c - object detector (YOLOv5) for Orange Pi CM5 (rk3588s, aarch64).
 *
 * Base: npu_check.c (RKNN API v2.3.0 declarations, uint8+NHWC feeding without normalization) —
 * verified on the board. Box decoding is ported from the official Rockchip example
 * yolov5_demo_postprocess.cc (anchors, thresholds, decoding of 3 outputs, NMS).
 *
 * Outputs are accepted in two variants: INT8 (affine zp/scale quantization, as in the sample)
 * and FLOAT32 (values read directly); decoding formulas are shared, only the value source
 * differs. Any other output type — refusal.
 *
 * Run:     ./npu_detect <model.rknn> <input.raw> [labels.txt]
 *          input.raw = RGB, HxWx3, uint8, exactly the size declared by the model (640x640x3).
 *          labels.txt default: ./coco_80_labels.txt
 * Build:   aarch64-none-linux-gnu-gcc -O2 -Wall -Wextra -o npu_detect npu_detect.c -L. -lrknnrt
 *          (libc only: printf/malloc/fopen; -lm not needed — arithmetic without math.h)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* ================= RKNN API v2.3.0: declarations (no rknn_api.h on the board) ================= */

typedef uint64_t rknn_context;

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
    RKNN_TENSOR_QNT_NONE              = 0,
    RKNN_TENSOR_QNT_DFP               = 1,
    RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC = 2
} rknn_tensor_qnt_type;

typedef enum _rknn_io_pass_through {
    RKNN_IO_PASS_THROUGH_DEFAULT = 0,
    RKNN_IO_PASS_THROUGH_TRUE    = 1,
    RKNN_IO_PASS_THROUGH_FALSE   = 2
} rknn_io_pass_through;

typedef struct _rknn_input_output_num {
    uint32_t n_input;
    uint32_t n_output;
} rknn_input_output_num;

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

extern int rknn_init(rknn_context *context, void *model, uint32_t size, uint32_t auto_batch, uint32_t ext_size);
extern int rknn_destroy(rknn_context context);
extern int rknn_query(rknn_context context, rknn_query_cmd cmd, void *info, uint32_t size);
extern int rknn_inputs_set(rknn_context context, uint32_t n_inputs, rknn_input inputs[]);
extern int rknn_run(rknn_context context, void *ext_info);
extern int rknn_outputs_get(rknn_context context, uint32_t n_outputs, rknn_output outputs[], void *ext_info);
extern int rknn_outputs_release(rknn_context context, uint32_t n_outputs, rknn_output outputs[]);

/* ================= constants of the official sample (Rockchip yolov5 demo) ================= */

/* Derived from the provided files:
 *   OBJ_CLASS_NUM = 80  — coco_80_labels.txt contains exactly 80 labels;
 *   PROP_BOX_SIZE = 5 + 80 — from process() indices in yolov5_demo_postprocess.cc:
 *       channel 4 = objectness, channels 5..5+OBJ_CLASS_NUM-1 = classes (lines 216, 232, 236);
 *   strides 8/16/32 — lines 283, 290, 300; output[l] corresponds to stride[l]. */
#define N_OUT_LAYERS 3
#define OBJ_CLASS_NUM 80
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

/* WARNING: NMS_THRESH / BOX_THRESH / OBJ_NUMB_MAX_SIZE / OBJ_NAME_MAX_SIZE are declared in
 * the sample's postprocess.h, which is NOT among the provided materials. Below are typical
 * values of this Rockchip demo; confirm against postprocess.h. Overridable via compiler
 * flags (-DBOX_THRESH=0.5f etc.) without editing the file. */
#ifndef BOX_THRESH
#define BOX_THRESH 0.25f          /* objectness threshold AND class threshold (same one in the sample) */
#endif
#ifndef NMS_THRESH
#define NMS_THRESH 0.45f          /* IoU threshold when suppressing duplicates */
#endif
#ifndef OBJ_NUMB_MAX_SIZE
#define OBJ_NUMB_MAX_SIZE 128     /* maximum number of boxes in the result */
#endif
#ifndef OBJ_NAME_MAX_SIZE
#define OBJ_NAME_MAX_SIZE 64      /* label length */
#endif

#define DEFAULT_LABELS_PATH "coco_80_labels.txt"

static const int strides[N_OUT_LAYERS] = { 8, 16, 32 };

/* anchors — yolov5_demo_postprocess.cc, lines 30-32, verbatim */
static const int anchor0[6] = { 10, 13, 16, 30, 33, 23 };
static const int anchor1[6] = { 30, 61, 62, 45, 59, 119 };
static const int anchor2[6] = { 116, 90, 156, 198, 373, 326 };
static const int *const anchors[N_OUT_LAYERS] = { anchor0, anchor1, anchor2 };

/* ================= common helpers (from npu_check.c) ================= */

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
    static const char *names[] = { "NCHW", "NHWC", "NC1HWC2", "UNDEFINED" };
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
    fprintf(stderr, "NPU_DETECT_FAIL step=%s ret=%d\n", step, ret);
    exit(1);
}

static void die_msg(const char *msg)
{
    fprintf(stderr, "NPU_DETECT_FAIL %s\n", msg);
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
    if (!f) die_msg("cannot open file");

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); die_msg("fseek failed"); }
    sz = ftell(f);
    if (sz <= 0) { fclose(f); die_msg("ftell failed or empty file"); }
    rewind(f);

    if (expect != 0 && (uint64_t)sz != expect) {
        fprintf(stderr, "NPU_DETECT_FAIL file size %ld != expected %lu (%s)\n",
                sz, (unsigned long)expect, path);
        fclose(f);
        exit(1);
    }

    buf = (unsigned char *)malloc((size_t)sz);
    if (!buf) { fclose(f); die_msg("out of memory"); }
    got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) { free(buf); die_msg("short read"); }

    *out_size = (size_t)sz;
    return buf;
}

/* ================= box decoding: port of yolov5_demo_postprocess.cc =================
 * Without math.h: fmaxf/fminf replaced by ternary operators (as in the sample's clamp()/__clip()),
 * so no linking with -lm is required. sigmoid()/unsigmoid() do not participate in the sample's
 * decoding path — not ported. */

static int clampi(float val, int lo, int hi) /* clamp(), line 34 */
{
    return val > (float)lo ? (val < (float)hi ? (int)val : hi) : lo;
}

static float clipf(float val, float lo, float hi) /* __clip(), lines 188-192 */
{
    return val <= lo ? lo : (val >= hi ? hi : val);
}

/* qnt_f32_to_affine, lines 194-199. In the sample __clip returns int32_t (truncation
 * toward zero) — kept deliberately so that the threshold in the quantized domain matches the reference. */
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + (float)zp;
    return (int8_t)clipf(dst_val, -128.0f, 127.0f);
}

/* deqnt_affine_to_f32, line 201 */
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

typedef struct {
    float x, y, w, h;  /* top-left corner + size in input pixels */
    int   cls;         /* class id 0..OBJ_CLASS_NUM-1 */
} det_box;

/* CalculateOverlap, lines 103-111 (with +1.0 in intersection and union) */
static float calculate_overlap(float xmin0, float ymin0, float xmax0, float ymax0,
                               float xmin1, float ymin1, float xmax1, float ymax1)
{
    float w = (xmax0 < xmax1 ? xmax0 : xmax1) - (xmin0 > xmin1 ? xmin0 : xmin1) + 1.0f;
    float h = (ymax0 < ymax1 ? ymax0 : ymax1) - (ymin0 > ymin1 ? ymin0 : ymin1) + 1.0f;
    float inter, uni;
    if (w < 0.0f) w = 0.0f;
    if (h < 0.0f) h = 0.0f;
    inter = w * h;
    uni = (xmax0 - xmin0 + 1.0f) * (ymax0 - ymin0 + 1.0f)
        + (xmax1 - xmin1 + 1.0f) * (ymax1 - ymin1 + 1.0f) - inter;
    return uni <= 0.0f ? 0.0f : (inter / uni);
}

/* nms, lines 113-149: pass over sorted order[], duplicates marked -1 */
static void nms_boxes(int validCount, const det_box *boxes, int *order, int filterId, float threshold)
{
    int i, j;
    for (i = 0; i < validCount; ++i) {
        int n = order[i];
        if (n == -1 || boxes[n].cls != filterId) continue;
        for (j = i + 1; j < validCount; ++j) {
            int m = order[j];
            float iou;
            if (m == -1 || boxes[m].cls != filterId) continue;
            iou = calculate_overlap(
                boxes[n].x, boxes[n].y, boxes[n].x + boxes[n].w, boxes[n].y + boxes[n].h,
                boxes[m].x, boxes[m].y, boxes[m].x + boxes[m].w, boxes[m].y + boxes[m].h);
            if (iou > threshold) order[j] = -1;
        }
    }
}

/* quick_sort_indice_inverse, lines 151-182: descending sort of scores[]
 * with parallel moving of indices[]. Verbatim port. */
static int quick_sort_indice_inverse(float *input, int left, int right, int *indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right) {
        key_index = indices[left];
        key = input[left];
        while (low < high) {
            while (low < high && input[high] <= key) high--;
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key) low++;
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

/* process, lines 203-258: output layout = [C,H,W], C = PROP_BOX_SIZE*3, map step = grid_len.
 * INT8 branch: threshold compared in the quantized (int8) domain; score = deqnt(class)*deqnt(objectness).
 * Appends candidates to cands[]/scores[] starting at *io_count;
 * return >= 0 — number appended, -1 — not enough room in the buffer. */
static int decode_layer(const int8_t *input, const int *anchor, int grid_h, int grid_w, int stride,
                        float threshold, int32_t zp, float scale,
                        det_box *cands, float *scores, int *io_count, int cap)
{
    int a, i, j, k;
    int base = *io_count;
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);

    for (a = 0; a < 3; a++) {
        for (i = 0; i < grid_h; i++) {
            for (j = 0; j < grid_w; j++) {
                int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_i8) {
                    const int8_t *in_ptr = input + (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    float box_x = deqnt_affine_to_f32(in_ptr[0], zp, scale) * 2.0f - 0.5f;
                    float box_y = deqnt_affine_to_f32(in_ptr[grid_len], zp, scale) * 2.0f - 0.5f;
                    float box_w = deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale) * 2.0f;
                    float box_h = deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale) * 2.0f;
                    int8_t maxClassProbs;
                    int maxClassId;

                    box_x = (box_x + (float)j) * (float)stride;
                    box_y = (box_y + (float)i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= box_w / 2.0f;
                    box_y -= box_h / 2.0f;

                    maxClassProbs = in_ptr[5 * grid_len];
                    maxClassId = 0;
                    for (k = 1; k < OBJ_CLASS_NUM; ++k) {
                        int8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    if (maxClassProbs > thres_i8) {
                        if (base + validCount >= cap) return -1;
                        scores[base + validCount] =
                            deqnt_affine_to_f32(maxClassProbs, zp, scale) *
                            deqnt_affine_to_f32(box_confidence, zp, scale);
                        cands[base + validCount].x = box_x;
                        cands[base + validCount].y = box_y;
                        cands[base + validCount].w = box_w;
                        cands[base + validCount].h = box_h;
                        cands[base + validCount].cls = maxClassId;
                        validCount++;
                    }
                }
            }
        }
    }
    *io_count = base + validCount;
    return validCount;
}

/* FLOAT32 branch of process, lines 203-258: channel layout, passes and formulas are the same as above,
 * but the value source is a float buffer (want_float=1) — deqnt_affine_to_f32/qnt_f32_to_affine
 * and zp/scale do not participate, the threshold is compared in the float domain. Result for NMS — the same
 * cands[].x/y/w/h/cls and scores[]. */
static int decode_layer_float(const float *input, const int *anchor, int grid_h, int grid_w, int stride,
                              float threshold,
                              det_box *cands, float *scores, int *io_count, int cap)
{
    int a, i, j, k;
    int base = *io_count;
    int validCount = 0;
    int grid_len = grid_h * grid_w;

    for (a = 0; a < 3; a++) {
        for (i = 0; i < grid_h; i++) {
            for (j = 0; j < grid_w; j++) {
                float box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= threshold) {
                    const float *in_ptr = input + (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    float box_x = in_ptr[0] * 2.0f - 0.5f;
                    float box_y = in_ptr[grid_len] * 2.0f - 0.5f;
                    float box_w = in_ptr[2 * grid_len] * 2.0f;
                    float box_h = in_ptr[3 * grid_len] * 2.0f;
                    float maxClassProbs;
                    int maxClassId;

                    box_x = (box_x + (float)j) * (float)stride;
                    box_y = (box_y + (float)i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= box_w / 2.0f;
                    box_y -= box_h / 2.0f;

                    maxClassProbs = in_ptr[5 * grid_len];
                    maxClassId = 0;
                    for (k = 1; k < OBJ_CLASS_NUM; ++k) {
                        float prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    if (maxClassProbs > threshold) {
                        if (base + validCount >= cap) return -1;
                        scores[base + validCount] = maxClassProbs * box_confidence;
                        cands[base + validCount].x = box_x;
                        cands[base + validCount].y = box_y;
                        cands[base + validCount].w = box_w;
                        cands[base + validCount].h = box_h;
                        cands[base + validCount].cls = maxClassId;
                        validCount++;
                    }
                }
            }
        }
    }
    *io_count = base + validCount;
    return validCount;
}

/* coco_80_labels.txt: exactly OBJ_CLASS_NUM lines, line i = label of class i */
static int load_labels(const char *path, char out[OBJ_CLASS_NUM][OBJ_NAME_MAX_SIZE])
{
    FILE *f = fopen(path, "r");
    char line[512];
    int i = 0;
    if (!f) {
        fprintf(stderr, "NPU_DETECT_FAIL cannot open labels file %s\n", path);
        return -1;
    }
    while (i < OBJ_CLASS_NUM && fgets(line, (int)sizeof(line), f)) {
        size_t n = strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
        if (n == 0) {
            fprintf(stderr, "NPU_DETECT_FAIL empty label at line %d of %s\n", i + 1, path);
            fclose(f);
            return -1;
        }
        if (n >= OBJ_NAME_MAX_SIZE) n = OBJ_NAME_MAX_SIZE - 1;
        memcpy(out[i], line, n);
        out[i][n] = '\0';
        i++;
    }
    fclose(f);
    if (i < OBJ_CLASS_NUM) {
        fprintf(stderr, "NPU_DETECT_FAIL labels: got %d lines, need %d (%s)\n", i, OBJ_CLASS_NUM, path);
        return -1;
    }
    return 0;
}

/* ================= main ================= */

int main(int argc, char **argv)
{
    const char *model_path;
    const char *image_path;
    const char *labels_path;
    void *model_buf;
    size_t model_size = 0;
    size_t in_bytes = 0;
    size_t file_size = 0;
    rknn_context ctx = 0;
    rknn_input_output_num io_num;
    rknn_sdk_version sdk;
    rknn_tensor_attr in_attr;
    rknn_tensor_attr out_attrs[N_OUT_LAYERS];
    rknn_input in;
    rknn_output *outs;
    unsigned char *in_data;
    char labels[OBJ_CLASS_NUM][OBJ_NAME_MAX_SIZE];
    det_box *cands;
    float *scores;
    int *order;
    int present[OBJ_CLASS_NUM];
    int grid_h[N_OUT_LAYERS];
    int grid_w[N_OUT_LAYERS];
    int layer_cands[N_OUT_LAYERS];
    int in_w = 0, in_h = 0, in_c = 0;
    int n_input, n_output;
    int total_cap = 0, validCount = 0, last_count = 0;
    int ret, i, l;

    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.rknn> <input.raw RGB HxWx3 uint8> [labels.txt]\n", argv[0]);
        return 1;
    }
    model_path = argv[1];
    image_path = argv[2];
    labels_path = (argc >= 4) ? argv[3] : DEFAULT_LABELS_PATH;

    if (load_labels(labels_path, labels) != 0) return 1;
    printf("NPU_DETECT_OK labels path=%s count=%d\n", labels_path, OBJ_CLASS_NUM);

    model_buf = read_whole_file(model_path, 0, &model_size);

    ret = rknn_init(&ctx, model_buf, (uint32_t)model_size, 0, 0);
    if (ret != 0) die("rknn_init", ret);
    printf("NPU_DETECT_OK rknn_init model=%s size=%lu\n", model_path, (unsigned long)model_size);

    memset(&sdk, 0, sizeof(sdk));
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk, sizeof(sdk));
    if (ret != 0) die("rknn_query SDK_VERSION", ret);
    printf("NPU_DETECT_OK sdk_version api=%s drv=%s\n", sdk.api_version, sdk.drv_version);

    memset(&io_num, 0, sizeof(io_num));
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != 0) die("rknn_query IN_OUT_NUM", ret);
    printf("NPU_DETECT_OK in_out_num n_input=%u n_output=%u\n", io_num.n_input, io_num.n_output);

    n_input = (int)io_num.n_input;
    n_output = (int)io_num.n_output;
    if (n_input != 1) die_msg("model must have exactly 1 input");
    if (n_output != N_OUT_LAYERS) die_msg("model must have exactly 3 outputs (strides 8/16/32)");

    memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
    if (ret != 0) die("rknn_query INPUT_ATTR", ret);
    print_tensor_attr("input", &in_attr);
    printf("NPU_DETECT_OK input_attrs\n");

    /* input sizes from attrs — as yolov5_demo_main.cc, lines 212-230 */
    if (in_attr.fmt == RKNN_TENSOR_NCHW) {
        in_c = (int)in_attr.dims[1];
        in_h = (int)in_attr.dims[2];
        in_w = (int)in_attr.dims[3];
    } else if (in_attr.fmt == RKNN_TENSOR_NHWC || in_attr.fmt == RKNN_TENSOR_NC1HWC2) {
        in_h = (int)in_attr.dims[1];
        in_w = (int)in_attr.dims[2];
        in_c = (int)in_attr.dims[3];
    } else {
        die_msg("input fmt is neither NCHW nor NHWC nor NC1HWC2");
    }
    if (in_c != 3) die_msg("input channels must be 3 (RGB)");
    if (in_w <= 0 || in_h <= 0) die_msg("model input size must be positive");
    for (l = 0; l < N_OUT_LAYERS; l++) {
        if (in_h % strides[l] != 0 || in_w % strides[l] != 0)
            die_msg("input size not divisible by stride 8/16/32");
        grid_h[l] = in_h / strides[l];
        grid_w[l] = in_w / strides[l];
    }
    printf("NPU_DETECT_OK model_input height=%d width=%d channel=%d\n", in_h, in_w, in_c);

    in_bytes = (size_t)in_h * (size_t)in_w * (size_t)in_c;
    if (in_bytes != (size_t)in_attr.n_elems) die_msg("input n_elems != H*W*C (uint8 input expected)");

    for (l = 0; l < N_OUT_LAYERS; l++) {
        uint32_t expected_elems;
        memset(&out_attrs[l], 0, sizeof(rknn_tensor_attr));
        out_attrs[l].index = (uint32_t)l;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[l], sizeof(rknn_tensor_attr));
        if (ret != 0) die("rknn_query OUTPUT_ATTR", ret);
        print_tensor_attr("output", &out_attrs[l]);
        /* the Rockchip demo decodes outputs as int8 with affine quantization (zp/scale from attr);
         * FLOAT32 is accepted under the same conditions — values are read directly (decode_layer_float).
         * Any other type — refusal, as before. */
        if (out_attrs[l].type != RKNN_TENSOR_INT8 && out_attrs[l].type != RKNN_TENSOR_FLOAT32) {
            fprintf(stderr, "NPU_DETECT_FAIL step=output_type idx=%d type=%s (expected INT8 or FLOAT32)\n",
                    l, type_name(out_attrs[l].type));
            return 1;
        }
        /* the model outputs 255 channels = PROP_BOX_SIZE*3 (three anchors), so n_elems = PROP_BOX_SIZE*3*grid_h*grid_w */
        expected_elems = (uint32_t)(PROP_BOX_SIZE * 3 * grid_h[l] * grid_w[l]);
        if (out_attrs[l].n_elems != expected_elems) {
            fprintf(stderr, "NPU_DETECT_FAIL step=output_elems idx=%d got=%u expected=%u (PROP_BOX_SIZE=%d grid %dx%d)\n",
                    l, out_attrs[l].n_elems, expected_elems, PROP_BOX_SIZE, grid_h[l], grid_w[l]);
            return 1;
        }
    }
    printf("NPU_DETECT_OK output_attrs\n");

    in_data = (unsigned char *)read_whole_file(image_path, (uint64_t)in_bytes, &file_size);
    printf("NPU_DETECT_OK input_loaded path=%s bytes=%lu\n", image_path, (unsigned long)file_size);

    memset(&in, 0, sizeof(in));
    in.index = 0;
    in.type = RKNN_TENSOR_UINT8;   /* feeding as in npu_check: uint8 + NHWC, no normalization */
    in.fmt = RKNN_TENSOR_NHWC;
    in.size = (uint32_t)in_bytes;
    in.buf = in_data;
    in.pass_through = 0;

    ret = rknn_inputs_set(ctx, 1, &in);
    if (ret != 0) die("rknn_inputs_set", ret);
    printf("NPU_DETECT_OK inputs_set\n");

    {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ret = rknn_run(ctx, NULL);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (ret != 0) die("rknn_run", ret);
        printf("NPU_DETECT_OK run time_us=%ld\n", diff_us(&t0, &t1));
    }

    outs = (rknn_output *)calloc((size_t)n_output, sizeof(rknn_output));
    if (!outs) die_msg("out of memory");
    for (i = 0; i < n_output; i++) {
        outs[i].index = (uint32_t)i;
        /* INT8 — native quantized buffer (want_float=0, as yolov5_demo_main.cc, lines 303-309);
         * FLOAT32 — values are already float (want_float=1). */
        outs[i].want_float = (out_attrs[i].type == RKNN_TENSOR_FLOAT32) ? 1 : 0;
        outs[i].is_prealloc = 0;
    }

    ret = rknn_outputs_get(ctx, io_num.n_output, outs, NULL);
    if (ret != 0) die("rknn_outputs_get", ret);
    printf("NPU_DETECT_OK outputs_get size0=%u size1=%u size2=%u\n",
           outs[0].size, outs[1].size, outs[2].size);
    for (l = 0; l < N_OUT_LAYERS; l++) {
        if (!outs[l].buf) die_msg("output buffer is NULL");
        if (outs[l].size < out_attrs[l].n_elems) die_msg("output buffer smaller than n_elems");
    }

    /* candidate buffer with a hard upper bound: 3 anchors * all grid cells of all outputs */
    for (l = 0; l < N_OUT_LAYERS; l++) total_cap += 3 * grid_h[l] * grid_w[l];
    cands = (det_box *)malloc((size_t)total_cap * sizeof(det_box));
    scores = (float *)malloc((size_t)total_cap * sizeof(float));
    order = (int *)malloc((size_t)total_cap * sizeof(int));
    if (!cands || !scores || !order) die_msg("out of memory");

    validCount = 0;
    for (l = 0; l < N_OUT_LAYERS; l++) {
        int added;
        if (out_attrs[l].type == RKNN_TENSOR_FLOAT32) {
            added = decode_layer_float((const float *)outs[l].buf, anchors[l], grid_h[l], grid_w[l],
                                       strides[l], BOX_THRESH, cands, scores, &validCount, total_cap);
        } else {
            added = decode_layer((const int8_t *)outs[l].buf, anchors[l], grid_h[l], grid_w[l],
                                 strides[l], BOX_THRESH, out_attrs[l].zp, out_attrs[l].scale,
                                 cands, scores, &validCount, total_cap);
        }
        if (added < 0) die_msg("candidate buffer overflow");
        layer_cands[l] = added;
    }
    printf("NPU_DETECT_OK decode box_thresh=%.2f nms_thresh=%.2f layer_s8=%d layer_s16=%d layer_s32=%d valid=%d\n",
           (double)BOX_THRESH, (double)NMS_THRESH, layer_cands[0], layer_cands[1], layer_cands[2], validCount);

    for (i = 0; i < validCount; i++) order[i] = i;
    if (validCount > 0) quick_sort_indice_inverse(scores, 0, validCount - 1, order);

    /* NMS per each present class (the sample uses std::set over classId) */
    memset(present, 0, sizeof(present));
    for (i = 0; i < validCount; i++) present[cands[i].cls] = 1;
    for (i = 0; i < OBJ_CLASS_NUM; i++) {
        if (present[i]) nms_boxes(validCount, cands, order, i, NMS_THRESH);
    }

    /* finalization and printing — sample lines 328-359. Letterbox is not applied: the input is
     * already model-sized, so pads = 0, scale = 1 and boxes stay in input pixels. */
    for (i = 0; i < validCount; i++) {
        int n, id, x, y, r, b;
        float obj_conf;
        if (order[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) continue;
        n = order[i];
        id = cands[n].cls;
        obj_conf = scores[i];
        x = clampi(cands[n].x, 0, in_w);
        y = clampi(cands[n].y, 0, in_h);
        r = clampi(cands[n].x + cands[n].w, 0, in_w);
        b = clampi(cands[n].y + cands[n].h, 0, in_h);
        printf("DETECT label=%s p=%.2f box=%d,%d,%d,%d\n", labels[id], (double)obj_conf, x, y, r - x, b - y);
        last_count++;
    }
    printf("NPU_DETECT_OK nms valid=%d kept=%d\n", validCount, last_count);
    printf("DETECT_TOTAL n=%d\n", last_count);

    ret = rknn_outputs_release(ctx, io_num.n_output, outs);
    if (ret != 0) die("rknn_outputs_release", ret);
    printf("NPU_DETECT_OK outputs_release\n");

    ret = rknn_destroy(ctx);
    if (ret != 0) die("rknn_destroy", ret);
    printf("NPU_DETECT_OK destroy\n");

    free(order);
    free(scores);
    free(cands);
    free(outs);
    free(in_data);
    free(model_buf);
    printf("NPU_DETECT_OK all\n");
    return 0;
}
