# Orange Pi CM5 (RK3588S) + IMX219: custom kernel, ISP pipeline and NPU detector

A practical build: an **IMX219** USB camera module (3280×2464 @ 21.16 fps) running through the ISP of an **Orange Pi CM5 (RK3588S)** board with a **custom-built Linux 6.1.99 kernel**, a vendor AIQ image-processing stack, and an on-board **NPU object detector** (RKNPU, librknnrt 2.3.0).

## What's in this repo

| Path | Purpose |
|---|---|
| `npu/npu_check.c` | Minimal NPU smoke test: loads a model, prints SDK/driver versions and tensor attributes, runs 10 timed inferences, prints top-3 classes with a confidence cutoff. Plain C, libc + librknnrt only. |
| `npu/npu_detect.c` | YOLOv5s detector (640×640) on the NPU: feeds a raw RGB frame, decodes the three output maps (anchors/thresholds follow the official Rockchip sample), applies NMS, prints `label / confidence / box` lines. |
| `npu/NOTES.md` | Engineering notes on the detector implementation. |
| `kernel/config` | The exact kernel `.config` used for the custom 6.1.99 build (see below). |
| `docs/btp1_detect_boxes_GIT.png` | A real 3280×2464 frame with detected boxes and confidences. |
| `Makefile` | Cross-build for both programs (aarch64, gcc 11.2, clean under `-Wall -Wextra`). |

## The custom kernel (self-built on top of the stock distro)

The stock Orange Pi Debian image ships Linux 6.1.99; the camera stack in it produced a
**tasklet storm after repeated camera start/stop** and mis-detected our IMX219 module.
We kept the distro kernel sources and rebuilt with:

- **An LSC-related fix inside the `rkisp` driver** — repeated start/stop cycles no longer
  leave a growing TASKLET backlog (measured: `+0` per 2 s window after stop, vs. a storm before).
- **Out-of-tree IMX219 sensor driver** (`imx219-vp`, v00.01.01) loaded as a module,
  with its own device-tree overlay (`cam1-imx219vp-6199`) via `user_overlays=` in `orangepiEnv.txt`.
- **Boot cmdline**: `initcall_blacklist=rkcif_clr_unready_dev,rkisp_clr_unready_dev`
  to skip the vendor "clear unready devices" initcalls that hang when the sensor chain
  is provided by the out-of-tree driver.
- Toolchain: **Arm GNU 11.2-2022.02** (`aarch64-none-linux-gnu-`), `ARCH=arm64`,
  `LOCALVERSION=-rockchip-rk3588`; the resulting `Image` + modules replace the stock ones in `/boot`.

`kernel/config` in this repo is the verbatim `.config` of that build — a good starting
point if you want to reproduce it against the distro sources.

## Verified environment

- Board: Orange Pi CM5 (RK3588S), custom 6.1.99 kernel, glibc 2.35 (Jammy userland).
- NPU: driver **0.9.8** (20240828), runtime **librknnrt 2.3.0** (c949ad889d).
- Models: taken verbatim from the official
  [rknn-toolkit2 v2.3.0](https://github.com/airockchip/rknn-toolkit2/tree/v2.3.0) tree —
  `mobilenet_v1` (classifier) and `yolov5s-640-640` (detector); no conversion needed.

## Measured results

- Smoke test: model loads, input 1×224×224×3, **~2.0–2.1 ms per inference** (≈480–490/s).
- Detector: 640×640 input, **~15.6–15.8 ms per frame** (≈63 fps), three 255-channel output maps.
- Classifier on a real frame: correctly identifies desk objects; a 10 % confidence cutoff
  removes the noisy guesses without any retraining.
- Detector on a real frame: see `docs/btp1_detect_boxes_GIT.png`.

## Pitfalls we hit (so you don't have to)

1. **The demo model shipped in the board image is RKNN format v1** — runtime 2.3.0 rejects
   it (`Invalid RKNN format`). Use models from the toolkit release that matches your runtime.
2. **Feed raw uint8 pixels (0–255), declare NHWC** — the runtime converts internally.
   Don't pre-normalize to floats unless the model explicitly expects them.
3. **The detector's native input layout is NC1HWC2** (format value 2) while `dims` stay
   logical `1×H×W×C` — don't reject the frame at that check.
4. **Detector outputs are 255 channels = 85×3 anchors**; objectness threshold 0.25,
   NMS IoU 0.45 (same as the official sample).

## Building

```bash
make CROSS=<path-to-aarch64-none-linux-gnu-gcc> LIBRKNN=<dir containing librknnrt.so>
# produces ./npu/npu_check and ./npu/npu_detect
```

## Running (on the board)

```bash
./npu_check  <mobilenet_v1.rknn> [frame224.rgb]   # NPU smoke test
./npu_detect <yolov5s-640.rknn> <frame640.rgb>    # prints DETECT lines with boxes
# frame = raw RGB, W*H*3 bytes
```

Download the models from the official toolkit tree (link above) and place them next to
the binaries. `librknnrt.so` 2.3.0 comes from the same release (`rknpu2/runtime/Linux/librknn_api/aarch64`).

## License

Code in this repository: MIT. Kernel config: GPLv2 (as the kernel itself).
Models and the runtime library are governed by their respective owners (Rockchip).
