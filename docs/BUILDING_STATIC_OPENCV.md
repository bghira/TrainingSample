# Building a statically linked OpenCV bundle

The `opencv` crate expects to find an existing OpenCV toolkit and, by default, it
links against the dynamic libraries that come with a system installation
(`libopencv_core.dylib`, `libopencv_core.so`, …). To ship the `trainingsample`
crate without asking end users to install OpenCV themselves, build a static
OpenCV distribution once and point Cargo at it during compilation.

## 1. Build FFmpeg as static libraries

1. Download FFmpeg sources (6.1+ recommended) to a temporary location:

   ```bash
   curl -sSLo ~/Downloads/ffmpeg-6.1.1.tar.bz2 https://ffmpeg.org/releases/ffmpeg-6.1.1.tar.bz2
   tar -C ~/Downloads -xjf ~/Downloads/ffmpeg-6.1.1.tar.bz2
   ```

2. Configure a static build that keeps only the components required for software
   decoding. Disable the executables and hardware accelerators so the build
   stays self-contained:

   ```bash
   pushd ~/Downloads/ffmpeg-6.1.1
   ./configure \
       --prefix=$(pwd)/../../third_party/ffmpeg-static \
       --pkg-config-flags="--static" \
       --extra-cflags="-fPIC" \
       --extra-cxxflags="-fPIC" \
       --extra-ldflags="-fPIC" \
       --enable-static \
       --disable-shared \
       --disable-doc \
       --disable-debug \
       --disable-autodetect \
       --disable-hwaccels \
       --disable-vulkan \
       --disable-cuvid \
       --disable-nvenc \
       --disable-nvdec \
       --disable-vaapi \
       --disable-vdpau \
       --disable-d3d11va \
       --disable-dxva2 \
       --disable-alsa \
       --disable-sdl2 \
       --disable-libxcb \
       --disable-iconv \
       --disable-libdrm \
       --disable-network \
       --disable-avdevice \
       --disable-postproc \
       --disable-programs \
       --enable-swscale \
       --enable-swresample
   make -j$(nproc)
   make install
   popd
   ```

   The install step writes the static archives and headers to
   `third_party/ffmpeg-static`. The OpenCV build picks them up via pkg-config.
   Copy the resulting libraries into the OpenCV bundle later (`libavcodec.a`,
   `libavfilter.a`, `libavformat.a`, `libavutil.a`, `libswresample.a`,
   `libswscale.a`).

## 2. Build OpenCV as static libraries

1. Download an official OpenCV source archive (4.9+ recommended) and place it
   somewhere outside the repository, e.g. `~/Downloads/opencv-4.10.0`.
2. Configure a *Release* build that turns off shared libraries and only enables
   the modules that `trainingsample` uses. Ensure the FFmpeg pkg-config files
   are visible (e.g. `export PKG_CONFIG_PATH=$(pwd)/third_party/ffmpeg-static/lib/pkgconfig:$PKG_CONFIG_PATH`).
   On macOS or Linux:

   ```bash
   cmake -S ~/Downloads/opencv-4.10.0 \
         -B ~/Downloads/opencv-build-static \
         -DBUILD_LIST=core,imgproc,imgcodecs,highgui,video,videoio,calib3d,features2d,photo \
         -DBUILD_SHARED_LIBS=OFF \
         -DBUILD_opencv_world=ON \
         -DOPENCV_FORCE_3RDPARTY_BUILD=ON \
         -DBUILD_JPEG=ON -DWITH_JPEG=ON \
         -DBUILD_PNG=ON -DWITH_PNG=ON \
         -DBUILD_TIFF=ON -DWITH_TIFF=ON \
         -DBUILD_WEBP=ON -DWITH_WEBP=ON \
         -DBUILD_ZLIB=ON -DWITH_ZLIB=ON \
         -DBUILD_JASPER=ON -DWITH_JASPER=ON \
         -DWITH_FFMPEG=ON -DOPENCV_FFMPEG_USE_FIND_LIBS=ON \
         -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
         -DWITH_IPP=OFF -DWITH_OPENCL=OFF -DWITH_CUDA=OFF -DWITH_OPENEXR=OFF
   cmake --build ~/Downloads/opencv-build-static --config Release --target opencv_world
   cmake --install ~/Downloads/opencv-build-static \
         --config Release \
         --prefix $(pwd)/third_party/opencv-static
   ```

   The `opencv_world` target gives you a single `libopencv_world.a` archive;
   switch it off and install the individual module archives if you prefer.
   The OpenCV build drops the bundled codec libraries (`liblibjpeg-turbo.a`,
   `liblibpng.a`, `liblibtiff.a`, `liblibwebp.a`, `libzlib.a`, `liblibjasper.a`) inside the
   build tree under `build/3rdparty/lib/`. Copy them into
   `third_party/opencv-static/lib/` and add the expected linker aliases, e.g.:

   ```bash
   cp ~/Downloads/opencv-build-static/3rdparty/lib/liblibjpeg-turbo.a third_party/opencv-static/lib/
   ln -sf liblibjpeg-turbo.a third_party/opencv-static/lib/libjpeg.a
   # Repeat for liblibpng.a→libpng.a, liblibtiff.a→libtiff.a, liblibwebp.a→libwebp.a, libzlib.a→libz.a, liblibjasper.a→libjasper.a
   ```

3. After installation you should have:

   ```text
   third_party/opencv-static/include/opencv4/...
   third_party/opencv-static/lib/libopencv_world.a
   third_party/opencv-static/lib/libz.a (and other third-party static deps)
   third_party/opencv-static/lib/libjasper.a
   third_party/opencv-static/lib/libavcodec.a
   third_party/opencv-static/lib/libavfilter.a
   third_party/opencv-static/lib/libavformat.a
   third_party/opencv-static/lib/libavutil.a
    third_party/opencv-static/lib/libswresample.a
    third_party/opencv-static/lib/libswscale.a
   ```

   If your OpenCV build links to optional third-party components (TBB, JPEG,
   PNG, WebP, etc.), install their static archives into the same `lib/` folder
   so Cargo can link them in one pass.

> **macOS note**: Apple does not provide static builds of the C++ standard
> library. Replace `static=stdc++` with `dylib=c++` (or `framework=Accelerate`
> when required) in the linking step below.

## 2. Point Cargo at the static toolchain

Add a `.cargo/config.toml` (kept inside the repo) with the environment variables
that the `opencv` build script understands:

```toml
[env]
# Paths are resolved relative to the workspace root.
OPENCV_INCLUDE_PATHS = { value = "third_party/opencv-static/include", relative = true }
OPENCV_LINK_PATHS    = { value = "third_party/opencv-static/lib",      relative = true }
# Tell the build script to stop probing the system.
OPENCV_DISABLE_PROBES = "pkg_config,cmake,vcpkg"
# Static link OpenCV and the extra libraries it depends on.
# Adjust the list to match the archives present in third_party/opencv-static/lib.
OPENCV_LINK_LIBS = "static=opencv_world,static=avformat,static=avcodec,static=avfilter,static=swresample,static=swscale,static=avutil,static=png,static=jpeg,static=tiff,static=webp,static=z,static=jasper,static=stdc++"
```

The helper script in this repository disables OpenJPEG support
(`-DWITH_OPENJPEG=OFF`), so `opencv_world` does not depend on `openjp2`. If you
enable JPEG2000 support yourself, append the appropriate OpenJPEG archive
(`static=openjp2`) to `OPENCV_LINK_LIBS`.

To avoid stale caches, the script drops a `build_signature.txt` file alongside
the libraries containing the configuration fingerprint (OpenCV version and the
set of disabled features). If you rebuild OpenCV manually, update or remove that
file so subsequent runs do not skip regeneration.

If you elected to install the individual module archives instead of
`opencv_world`, list each one (`static=opencv_core`, `static=opencv_imgproc`,
…). Keep the order roughly from high- to low-level modules so the linker can
resolve symbols in one pass.

For cross-compilation add target-specific sections, e.g.:

```toml
[target.aarch64-apple-darwin.env]
OPENCV_LINK_LIBS = "static=opencv_world,static=avformat,static=avcodec,static=avfilter,static=swresample,static=swscale,static=avutil,static=png,static=jpeg,static=tiff,static=z,static=jasper,dylib=c++"
```

## 3. Build the crate

With the static bundle in place you can now build the crate without touching the
system OpenCV installation:

```bash
cargo build --features opencv --release
```

The resulting `libtrainingsample.{so,dylib}` (or the wheels produced by the
Python bindings) now embed the OpenCV symbols directly, so end users do not need
`opencv_core` on their machines.

## 4. Regenerating the bundle

Whenever you need to update OpenCV:

1. Re-run the CMake configure/build/install steps pointing at the new source
   tree.
2. Verify that the list in `OPENCV_LINK_LIBS` still matches the archives produced.
3. Commit the regenerated contents of `third_party/opencv-static/` if you keep
   it under version control (or upload it to your release pipeline’s artifact
   store).

That is all Cargo needs—no changes to `Cargo.toml` are required beyond enabling
the `opencv` feature when you want the acceleration path.
