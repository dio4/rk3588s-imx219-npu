# NOTES — npu_detect.c (YOLOv5 detector for rk3588)

File: `npu_detect.c`, 659 lines, one translation unit. Base — `npu_check.c`,
box decoding — a port of `yolov5_demo_postprocess.cc`.

## 1. Done (compilation NOT run)
The session environment gave only `Read`/`Write`: `Bash`, `Glob`, `Grep`, `WebFetch`, `Edit` — rejected
("environment is non-interactive"). So REQUIREMENT 5 is partially met: self-check —
manual read-through (file completeness, bracket balance, all functions declared before use, all
declared variables/functions used, format specifiers `%u/%d/%lu/%s/%.2f`
match the types, no sign-comparison issues under `-Wsign-compare`). **Compilation with
`aarch64-none-linux-gnu-gcc -O2 -Wall -Wextra` could not be checked locally — your build is needed.**

## 2. Taken from npu_check.c (unchanged)
- RKNN API v2.3.0 declarations: `rknn_context`, 5 enums, `rknn_tensor_attr` (376 B), `rknn_input` (32),
  `rknn_output` (24), `rknn_sdk_version` (512) + all 4 `_Static_assert` + 7 `extern` functions.
- Helpers: `type_name/fmt_name/qnt_name/die/die_msg/print_tensor_attr/diff_us/read_whole_file`.
- Input feeding: `type=RKNN_TENSOR_UINT8`, `fmt=RKNN_TENSOR_NHWC`, `pass_through=0`, **no normalization**;
  input size = `n_elems` of input[0] (the file must match byte-for-byte).
- Stage/error prefix changed to `NPU_DETECT_OK` / `NPU_DETECT_FAIL step=... ret=...`.

## 3. Taken LINE-BY-LINE from yolov5_demo_postprocess.cc (sample line numbers)
| What | Sample lines | How ported |
|---|---|---|
| Anchors `anchor0/1/2` = {10,13,16,30,33,23} / {30,61,62,45,59,119} / {116,90,156,198,373,326} | 30–32 | verbatim, + `anchors[]` per output |
| Strides 8/16/32 for outputs 0/1/2 | 283, 290, 300 | `strides[]` |
| `clamp` (integer) | 34 | `clampi()` |
| `__clip` (returns **int32_t**, i.e. truncation toward zero) | 188–192 | `clipf()` + truncation when quantizing the threshold — kept deliberately |
| `qnt_f32_to_affine` / `deqnt_affine_to_f32` | 194–201 | verbatim |
| `CalculateOverlap` (IoU with **+1.0** in intersection and union) | 103–111 | `calculate_overlap()` |
| `nms` (pass over sorted `order[]`, duplicates → `-1`, threshold `iou > threshold`) | 113–149 | `nms_boxes()` |
| `quick_sort_indice_inverse` (descending, parallel index array, recursion) | 151–182 | verbatim port |
| `process()` — output layout `[C,H,W]`, `C=PROP_BOX_SIZE*3`, map step `grid_len`, obj channel = 4, classes = 5..84, `x=(v*2-0.5+ j)*stride`, `w=(v*2)^2*anchor` | 203–258 | `decode_layer()`, same `a,i,j` loop order |
| Threshold applied **twice and in the int8 domain**: `box_confidence >= thres_i8` and `maxClassProbs > thres_i8`; score = `deqnt(class)*deqnt(obj)` | 209, 216–217, 243–245 | identical |
| Finalization: sort → NMS per class → skip `-1` → limit `OBJ_NUMB_MAX_SIZE` → `clamp` to frame edges | 313–359 | identical |
| `want_float=0` (native int8) + `zp/scale` from `RKNN_QUERY_OUTPUT_ATTR` | main.cc 303–309, 321–325 | identical |
| Input sizes from attrs (NCHW/NHWC branches) | main.cc 212–230 | identical |

## 4. Constants: what is derivable, and what is NOT verified
Derived from the provided materials (no guessing):
- `OBJ_CLASS_NUM = 80` — `coco_80_labels.txt` contains exactly 80 lines.
- `PROP_BOX_SIZE = 5 + OBJ_CLASS_NUM = 85` — from `process()` index arithmetic: channel 4 =
  objectness (line 216), channels `5+k`, `k<OBJ_CLASS_NUM` = classes (lines 232, 236).
- `N_OUT_LAYERS = 3`, strides 8/16/32, anchors — see the table above.

**NOT VERIFIED: the sample's `postprocess.h` IS ABSENT among the provided materials** (the `*.cc` files only have `#include
"postprocess.h"`), so the values of `BOX_THRESH`, `NMS_THRESH`, `OBJ_NUMB_MAX_SIZE`,
`OBJ_NAME_MAX_SIZE` had no source. Typical values of this Rockchip demo are set and
wrapped in `#ifndef`, so they can be confirmed/overridden with a compiler flag without
editing the file. **Please send `postprocess.h` (or its 4 #define lines) — this is the only
numeric risk in the task.**
```c
#ifndef BOX_THRESH        #define BOX_THRESH 0.25f
#ifndef NMS_THRESH        #define NMS_THRESH 0.45f
#ifndef OBJ_NUMB_MAX_SIZE #define OBJ_NUMB_MAX_SIZE 128
#ifndef OBJ_NAME_MAX_SIZE #define OBJ_NAME_MAX_SIZE 64
```
The "score" threshold has no separate constant in the sample: obj and class are compared with one
`conf_threshold` — done the same way (`BOX_THRESH` for both).

## 5. Differences from the sample (justified)
- **Pure C instead of C++**: `std::vector/set` → flat arrays with a hard upper bound
  `cap = 3 * Σ(grid_h*grid_w)` (= 25200 candidates at 640×640); overflow → an explicit error,
  not undefined behaviour. `std::set<int> class_set` → `present[80]` flag and NMS only over
  present classes (result is equivalent: NMS across classes is independent).
- **No `math.h`** (no `-lm` in the link command): `fmaxf/fminf` replaced with ternary operators,
  `expf/logf` not needed — `sigmoid()/unsigmoid()` do not participate in the sample's decoding path,
  not ported. Only printf/fopen/fgets/malloc/free/memcpy/memset/fread are used.
- **Letterbox removed**: the input is a ready raw 640×640×3, so `pads = 0`, `scale_w = scale_h = 1`;
  RGA/OpenCV not needed. Boxes are printed in 640 input pixels, as REQUIREMENT 3 demands
  (`box=x,y,w,h` = left, top, right-left, bottom-top).
- **Labels are read from a file** (`load_labels`, analog of `loadLabelName`), path — argv[3],
  default `coco_80_labels.txt` (the sample hardcodes `./model/coco_80_labels_list.txt`, which
  does not exist on the board). Exactly 80 non-empty lines, otherwise `NPU_DETECT_FAIL labels: ...`; line i = class i.
- **Top-3 from npu_check not ported**: a detector's outputs are anchor maps (544000 elements),
  "top-3 by raw values" is meaningless and would require `want_float=1`.
  Instead, the meaningful stages `decode ... layer_s8/s16/s32/valid` and `nms valid/kept` are printed.
- One `rknn_run` pass (not 10 as in npu_check) — speed benchmarks are not in the spec.

## 6. Input checks (fail loudly, with code/message)
`n_input==1`; `n_output==3`; input NHWC or NCHW, channels==3; `H%32==0 && W%32==0`;
`n_elems==H*W*C`; raw file size == `in_bytes`; **each output type == INT8**
(otherwise `step=output_type idx=.. type=..` — the port is tailored to the sample's int8 decoding);
`output n_elems == PROP_BOX_SIZE*grid_h*grid_w` (this also catches a swapped 8/16/32 output order);
`outs[i].buf != NULL`; `outs[i].size >= n_elems`.

## 7. Output format
```
NPU_DETECT_OK labels path=coco_80_labels.txt count=80
NPU_DETECT_OK rknn_init model=... size=...
NPU_DETECT_OK sdk_version api=... drv=...
NPU_DETECT_OK in_out_num n_input=1 n_output=3
input[0] name=... dims=... type=... fmt=... zp=... scale_exp=...
NPU_DETECT_OK input_attrs
NPU_DETECT_OK model_input height=640 width=640 channel=3
output[0..2] name=... dims=... 
NPU_DETECT_OK output_attrs
NPU_DETECT_OK input_loaded path=... bytes=1228800
NPU_DETECT_OK inputs_set
NPU_DETECT_OK run time_us=...
NPU_DETECT_OK outputs_get size0=.. size1=.. size2=..
NPU_DETECT_OK decode box_thresh=0.25 nms_thresh=0.45 layer_s8=.. layer_s16=.. layer_s32=.. valid=..
DETECT label=person p=0.87 box=120,64,90,210
DETECT_TOTAL n=1
NPU_DETECT_OK nms valid=.. kept=1
NPU_DETECT_OK outputs_release / destroy / all
```
`DETECT` lines go in descending score order (the sample's sort), in 640 input pixels.

## 8. What to verify on the board (risks covered by messages, not by guessing)
1. **Values of the 4 constants from `postprocess.h`** (see item 4) — priority.
2. `size0/1/2` vs `n_elems`: if `size > n_elems` (w_stride row alignment), the sample's flat
   indexing will not be enough — then tell me and I will add a step over `size_with_stride`.
   The Rockchip sample does not account for this either.
3. Output types: if the model outputs FLOAT — the program fails at `step=output_type`; tell me and
   I will add a float branch (without changing the sample's math).
4. Expected accuracy control: compare with the output of `rknn_yolov5_demo` on the same frame.

## 9. Arguments
`./npu_detect <model.rknn> <input.raw RGB HxWx3 uint8> [labels.txt]` — two required, as in the spec;
the third — path to labels (needed because the sample's path `./model/...` does not exist on the board).
