FROM ubuntu:24.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    clang \
    libclang-dev \
    llvm-dev \
    wget \
    cmake \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install LLVM 18
RUN apt-get update && apt-get install -y \
    llvm-18 \
    llvm-18-dev \
    llvm-18-tools \
    libclang1-18 \
    libclang-cpp18 \
    && rm -rf /var/lib/apt/lists/*

# Set LLVM environment variables
ENV LIBCLANG_PATH=/usr/lib/llvm-18/lib
ENV LLVM_CONFIG_PATH=/usr/bin/llvm-config-18
ENV LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:$LD_LIBRARY_PATH

# Build OpenCV 4.12 from source
RUN cd /tmp && \
    wget -q https://github.com/opencv/opencv/archive/4.12.0.tar.gz && \
    tar -xzf 4.12.0.tar.gz && \
    cd opencv-4.12.0 && \
    mkdir build && \
    cd build && \
    cmake \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DWITH_CUDA=OFF \
        -DWITH_OPENGL=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_TBB=OFF \
        -DWITH_IPP=OFF \
        -DWITH_1394=OFF \
        -DWITH_V4L=OFF \
        -DWITH_GTK=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_opencv_java=OFF \
        -DOPENCV_GENERATE_PKGCONFIG=ON \
        .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/opencv-4.12.0

# Set OpenCV environment variables
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.85
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup component add rustfmt clippy

# Install Python dependencies
RUN python3 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install maturin[patchelf] pytest pytest-benchmark numpy pillow

# Add venv to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Verify installations
RUN which rustc && rustc --version
RUN which cargo-clippy && cargo-clippy --version
