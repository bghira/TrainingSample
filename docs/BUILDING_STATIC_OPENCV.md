# Building a statically linked OpenCV bundle

The `opencv` crate expects to find an existing OpenCV toolkit and, by default, it
links against the dynamic libraries that come with a system installation
(`libopencv_core.dylib`, `libopencv_core.so`, …). To ship the `trainingsample`
crate without asking end users to install OpenCV themselves, build a static
OpenCV distribution once and point Cargo at it during compilation.

## 1. Build OpenCV as static libraries

1. Download an official OpenCV source archive (4.9+ recommended) and place it
   somewhere outside the repository, e.g. `~/Downloads/opencv-4.10.0`.
2. Configure a *Release* build that turns off shared libraries and only enables
   the modules that `trainingsample` uses. On macOS or Linux:

   ```bash
   cmake -S ~/Downloads/opencv-4.10.0 \
         -B ~/Downloads/opencv-build-static \
         -DBUILD_LIST=core,imgproc,imgcodecs,highgui,video,videoio,calib3d,features2d,photo \
         -DBUILD_SHARED_LIBS=OFF \
         -DBUILD_opencv_world=ON \
         -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
         -DWITH_IPP=OFF -DWITH_OPENCL=OFF -DWITH_CUDA=OFF
   cmake --build ~/Downloads/opencv-build-static --config Release --target opencv_world
   cmake --install ~/Downloads/opencv-build-static \
         --config Release \
         --prefix $(pwd)/third_party/opencv-static
   ```

   The `opencv_world` target gives you a single `libopencv_world.a` archive;
   switch it off and install the individual module archives if you prefer.

3. After installation you should have:

   ```text
   third_party/opencv-static/include/opencv4/...
   third_party/opencv-static/lib/libopencv_world.a
   third_party/opencv-static/lib/libz.a (and other third-party static deps)
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
OPENCV_LINK_LIBS = "static=opencv_world,static=png,static=jpeg,static=tiff,static=webp,static=z,static=stdc++"
```

If you elected to install the individual module archives instead of
`opencv_world`, list each one (`static=opencv_core`, `static=opencv_imgproc`,
…). Keep the order roughly from high- to low-level modules so the linker can
resolve symbols in one pass.

For cross-compilation add target-specific sections, e.g.:

```toml
[target.aarch64-apple-darwin.env]
OPENCV_LINK_LIBS = "static=opencv_world,static=png,static=jpeg,static=tiff,static=z,dylib=c++"
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
