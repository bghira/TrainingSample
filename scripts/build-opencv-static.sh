#!/usr/bin/env bash
set -euo pipefail

# Build static OpenCV bundle for embedding in Python wheels
# This eliminates the need for users to install OpenCV system packages

OPENCV_VERSION="4.12.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/opencv-build-tmp"
INSTALL_DIR="${PROJECT_ROOT}/third_party/opencv-static"

echo "Building static OpenCV ${OPENCV_VERSION}..."
echo "Install directory: ${INSTALL_DIR}"

# Skip if already built
if [ -d "${INSTALL_DIR}/lib" ] && [ -f "${INSTALL_DIR}/lib/libopencv_world.a" ]; then
    echo "Static OpenCV already built at ${INSTALL_DIR}"
    exit 0
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Download OpenCV source if not already present
if [ ! -d "opencv-${OPENCV_VERSION}" ]; then
    echo "Downloading OpenCV ${OPENCV_VERSION}..."
    curl -L "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz" \
        -o "opencv-${OPENCV_VERSION}.tar.gz"
    tar -xzf "opencv-${OPENCV_VERSION}.tar.gz"
fi

# Configure CMake build
echo "Configuring CMake build..."
cmake -S "opencv-${OPENCV_VERSION}" \
      -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_LIST=core,imgproc,imgcodecs,highgui,video,videoio,calib3d,features2d,photo \
      -DBUILD_SHARED_LIBS=OFF \
      -DBUILD_opencv_world=ON \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_opencv_apps=OFF \
      -DBUILD_DOCS=OFF \
      -DWITH_IPP=OFF \
      -DWITH_OPENCL=OFF \
      -DWITH_CUDA=OFF \
      -DWITH_FFMPEG=OFF \
      -DWITH_GSTREAMER=OFF \
      -DWITH_V4L=OFF \
      -DWITH_GTK=OFF \
      -DWITH_QT=OFF \
      -DWITH_ITT=OFF \
      -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"

# Build opencv_world (single unified library)
echo "Building OpenCV (this may take several minutes)..."
cmake --build build --config Release --target opencv_world -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)

# Install to third_party/opencv-static
echo "Installing to ${INSTALL_DIR}..."
cmake --install build --config Release

# Verify installation - check both lib and lib64 (manylinux uses lib64)
if [ -f "${INSTALL_DIR}/lib64/libopencv_world.a" ]; then
    # Move from lib64 to lib for consistency
    mkdir -p "${INSTALL_DIR}/lib"
    mv "${INSTALL_DIR}/lib64"/* "${INSTALL_DIR}/lib/"
    rmdir "${INSTALL_DIR}/lib64"
fi

if [ ! -f "${INSTALL_DIR}/lib/libopencv_world.a" ]; then
    echo "ERROR: libopencv_world.a not found after installation"
    echo "Contents of ${INSTALL_DIR}:"
    ls -R "${INSTALL_DIR}"
    exit 1
fi

echo "Static OpenCV built successfully!"
echo "Library: ${INSTALL_DIR}/lib/libopencv_world.a"
echo "Headers: ${INSTALL_DIR}/include/opencv4/"

# Clean up build directory to save space
cd "${PROJECT_ROOT}"
rm -rf "${BUILD_DIR}"

echo "Build complete!"
