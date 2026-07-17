# Static OpenCV build

The repository build script creates the static libraries used by release
wheels.

```bash
./scripts/build-opencv-static.sh
```

## Pinned inputs

| Component | Version |
|---|---:|
| OpenCV | 4.11.0 |
| FFmpeg | 6.1.1 |

The versions are defined at the top of
`scripts/build-opencv-static.sh`.

## Required commands

```text
bash
curl
tar with bzip2 support
make
cmake
pkg-config
C and C++ compilers
```

CI also installs clang, libclang, LLVM, nasm, and yasm before running the
script.

## Output

| Path | Contents |
|---|---|
| `third_party/ffmpeg-static/` | FFmpeg headers, archives, pkg-config files, signature |
| `third_party/opencv-static/include/opencv4/` | OpenCV headers |
| `third_party/opencv-static/lib/libopencv_world.a` | OpenCV archive |
| `third_party/opencv-static/lib/` | codec and FFmpeg archives |
| `third_party/opencv-static/build_signature.txt` | OpenCV build signature |

Required archives checked by the script:

```text
libjpeg.a
libpng.a
libtiff.a
libwebp.a
libz.a
libjasper.a
libavcodec.a
libavfilter.a
libavformat.a
libavutil.a
libswresample.a
libswscale.a
```

## OpenCV configuration

| Setting | Value |
|---|---|
| Library form | static `opencv_world` |
| Modules | core, imgproc, imgcodecs, highgui, video, videoio, calib3d, features2d, photo |
| Bundled codecs | JPEG, PNG, TIFF, WebP, zlib, Jasper |
| FFmpeg | enabled |
| Tests, examples, apps, docs | disabled |
| IPP, OpenCL, CUDA, OpenJPEG, OpenEXR, ITT, TBB | disabled |
| GStreamer, V4L, GTK, Qt | disabled |
| Carotene | disabled on macOS and ARM hosts |

## FFmpeg configuration

| Setting | Value |
|---|---|
| Library form | static |
| Decoders | H.264, HEVC, MPEG-4, VP8, VP9 |
| Demuxers | MOV, Matroska, Ogg, WebM, image2 |
| Protocols | file, data, pipe |
| Hardware acceleration | disabled |
| Network | disabled |
| Programs and documentation | disabled |

## Cargo and maturin environment

Linux:

```bash
export OPENCV_INCLUDE_PATHS="$PWD/third_party/opencv-static/include/opencv4"
export OPENCV_LINK_PATHS="$PWD/third_party/opencv-static/lib"
export OPENCV_DISABLE_PROBES="pkg_config,cmake,vcpkg,vcpkg_cmake"
export OPENCV_LINK_LIBS="static=opencv_world,static=avformat,static=avcodec,static=avfilter,static=swresample,static=swscale,static=avutil,static=jpeg,static=png,static=tiff,static=webp,static=z,static=jasper,dylib=stdc++"
```

macOS:

```bash
export OPENCV_INCLUDE_PATHS="$PWD/third_party/opencv-static/include/opencv4"
export OPENCV_LINK_PATHS="$PWD/third_party/opencv-static/lib"
export OPENCV_DISABLE_PROBES="pkg_config,cmake,vcpkg,vcpkg_cmake"
export OPENCV_LINK_LIBS="static=opencv_world,static=avformat,static=avcodec,static=avfilter,static=swresample,static=swscale,static=avutil,static=jpeg,static=png,static=tiff,static=webp,static=jasper,framework=Accelerate,dylib=c++,framework=OpenCL,z"
```

Build commands:

```bash
cargo build --release --no-default-features --features opencv,simd
maturin build --release --no-default-features \
  --features python-bindings,simd,opencv
```

macOS release wheels add the `metal` feature.

## Cache behavior

The script skips a bundle when `libopencv_world.a` exists and
`build_signature.txt` matches the configured signature. A signature mismatch
removes and rebuilds the existing install directory. Temporary OpenCV and
FFmpeg build directories are removed after a successful build.

Release workflow settings are in `.github/workflows/publish.yml`. Test-wheel
settings are in `.github/workflows/ci.yml`.
