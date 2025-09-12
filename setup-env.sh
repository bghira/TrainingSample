#!/bin/bash
# Setup environment for OpenCV Rust development

echo "Setting up OpenCV Rust environment..."

# LLVM/Clang paths for OpenCV bindings
export DYLD_LIBRARY_PATH="/opt/homebrew/Cellar/llvm/21.1.1/lib:$DYLD_LIBRARY_PATH"
export LIBCLANG_PATH="/opt/homebrew/Cellar/llvm/21.1.1/lib"
export LLVM_CONFIG_PATH="/opt/homebrew/Cellar/llvm/21.1.1/bin/llvm-config"

# OpenCV paths
export OPENCV_LINK_LIBS="opencv_calib3d,opencv_core,opencv_dnn,opencv_features2d,opencv_flann,opencv_highgui,opencv_imgcodecs,opencv_imgproc,opencv_ml,opencv_objdetect,opencv_photo,opencv_stitching,opencv_video,opencv_videoio"
export OPENCV_LINK_PATHS="/opt/homebrew/lib"
export OPENCV_INCLUDE_PATHS="/opt/homebrew/include/opencv4"

# Rust optimizations
export RUSTFLAGS="-C target-cpu=native"

# macOS deployment target - match current system
export MACOSX_DEPLOYMENT_TARGET="$(sw_vers -productVersion | cut -d. -f1,2)"

echo "Environment variables set:"
echo "  DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"
echo "  LIBCLANG_PATH=$LIBCLANG_PATH"
echo "  OPENCV_INCLUDE_PATHS=$OPENCV_INCLUDE_PATHS"
echo "  MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET"
echo ""
echo "You can now run:"
echo "  cargo clippy --all-features"
echo "  cargo test --features opencv,simd"
echo "  cargo test --all-features  # (requires Python environment)"
echo ""
echo "To make this permanent, add this to your ~/.zshrc or ~/.bashrc:"
echo "  source $(pwd)/setup-env.sh"
