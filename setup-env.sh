#!/usr/bin/env bash
# Setup environment for OpenCV Rust development

fail() {
    echo "ERROR: $*" >&2
    return 1 2>/dev/null || exit 1
}

prepend_path_var() {
    local var_name="$1"
    local new_path="$2"
    local current_value

    eval "current_value=\"\${$var_name:-}\""

    if [ -z "${new_path}" ] || [ ! -d "${new_path}" ]; then
        return 0
    fi

    case ":${current_value}:" in
        *":${new_path}:"*) ;;
        *)
            if [ -n "${current_value}" ]; then
                eval "export ${var_name}=\"${new_path}:\$${var_name}\""
            else
                eval "export ${var_name}=\"${new_path}\""
            fi
            ;;
    esac
}

append_rustflag() {
    local flag="$1"

    case " ${RUSTFLAGS:-} " in
        *" ${flag} "*) ;;
        *)
            if [ -n "${RUSTFLAGS:-}" ]; then
                export RUSTFLAGS="${RUSTFLAGS} ${flag}"
            else
                export RUSTFLAGS="${flag}"
            fi
            ;;
    esac
}

find_llvm_config() {
    local candidate

    if [ -n "${LLVM_CONFIG_PATH:-}" ] && [ -x "${LLVM_CONFIG_PATH}" ]; then
        printf '%s\n' "${LLVM_CONFIG_PATH}"
        return 0
    fi

    if command -v llvm-config >/dev/null 2>&1; then
        command -v llvm-config
        return 0
    fi

    for candidate in /opt/homebrew/opt/llvm/bin/llvm-config /usr/local/opt/llvm/bin/llvm-config; do
        if [ -x "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    candidate="$(
        find /usr/lib \
            \( -path '/usr/lib/llvm-*/bin/llvm-config' -o -path '/usr/lib/llvm/*/bin/llvm-config' \) \
            2>/dev/null \
            | head -n 1
    )"
    if [ -n "${candidate}" ] && [ -x "${candidate}" ]; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    return 1
}

find_libclang_path() {
    local candidate

    candidate="$(find_llvm_config || true)"
    if [ -n "${candidate}" ]; then
        candidate="$("${candidate}" --libdir 2>/dev/null || true)"
        if [ -n "${candidate}" ] && [ -d "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    fi

    if [ -n "${LIBCLANG_PATH:-}" ] && [ -d "${LIBCLANG_PATH}" ]; then
        printf '%s\n' "${LIBCLANG_PATH}"
        return 0
    fi

    if [ "$(uname -s)" = "Linux" ] && command -v ldconfig >/dev/null 2>&1; then
        candidate="$(
            ldconfig -p 2>/dev/null \
                | sed -n 's/.* => \(.*\/libclang[^ ]*\)$/\1/p' \
                | head -n 1
        )"
        if [ -n "${candidate}" ]; then
            candidate="$(dirname "${candidate}")"
            if [ -d "${candidate}" ]; then
                printf '%s\n' "${candidate}"
                return 0
            fi
        fi
    fi

    for candidate in /opt/homebrew/opt/llvm/lib /usr/local/opt/llvm/lib /usr/lib64 /usr/lib; do
        if [ -d "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    candidate="$(
        find /usr/lib \
            \( -path '/usr/lib/llvm-*/lib' -o -path '/usr/lib/llvm/*/lib' -o -path '/usr/lib/llvm-*/lib64' -o -path '/usr/lib/llvm/*/lib64' \) \
            2>/dev/null \
            | head -n 1
    )"
    if [ -n "${candidate}" ] && [ -d "${candidate}" ]; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    return 1
}

detect_opencv_from_pkg_config() {
    local libs_raw lib_paths_raw include_paths_raw

    if ! command -v pkg-config >/dev/null 2>&1; then
        return 1
    fi

    if ! pkg-config --exists opencv4; then
        return 1
    fi

    libs_raw="$(pkg-config --libs-only-l opencv4 2>/dev/null || true)"
    lib_paths_raw="$(pkg-config --libs-only-L opencv4 2>/dev/null || true)"
    include_paths_raw="$(pkg-config --cflags-only-I opencv4 2>/dev/null || true)"

    OPENCV_LINK_LIBS="$(
        printf '%s\n' "${libs_raw}" \
            | tr ' ' '\n' \
            | sed -n 's/^-l//p' \
            | paste -sd, -
    )"
    OPENCV_LINK_PATHS="$(
        printf '%s\n' "${lib_paths_raw}" \
            | tr ' ' '\n' \
            | sed -n 's/^-L//p' \
            | paste -sd, -
    )"
    OPENCV_INCLUDE_PATHS="$(
        printf '%s\n' "${include_paths_raw}" \
            | tr ' ' '\n' \
            | sed -n 's/^-I//p' \
            | paste -sd, -
    )"

    export OPENCV_LINK_LIBS
    export OPENCV_LINK_PATHS
    export OPENCV_INCLUDE_PATHS
}

detect_opencv_from_defaults() {
    case "${OS_NAME}" in
        Darwin)
            export OPENCV_LINK_LIBS="opencv_calib3d,opencv_core,opencv_dnn,opencv_features2d,opencv_flann,opencv_highgui,opencv_imgcodecs,opencv_imgproc,opencv_ml,opencv_objdetect,opencv_photo,opencv_stitching,opencv_video,opencv_videoio"

            if [ -d /opt/homebrew/include/opencv4 ]; then
                export OPENCV_INCLUDE_PATHS="/opt/homebrew/include/opencv4"
                export OPENCV_LINK_PATHS="/opt/homebrew/lib"
            elif [ -d /usr/local/include/opencv4 ]; then
                export OPENCV_INCLUDE_PATHS="/usr/local/include/opencv4"
                export OPENCV_LINK_PATHS="/usr/local/lib"
            else
                fail "Could not detect OpenCV via pkg-config or common macOS install paths"
            fi
            ;;
        Linux)
            export OPENCV_LINK_LIBS="opencv_calib3d,opencv_core,opencv_dnn,opencv_features2d,opencv_flann,opencv_highgui,opencv_imgcodecs,opencv_imgproc,opencv_ml,opencv_objdetect,opencv_photo,opencv_stitching,opencv_video,opencv_videoio"

            if [ -d /usr/include/opencv4 ]; then
                export OPENCV_INCLUDE_PATHS="/usr/include/opencv4"
            elif [ -d /usr/local/include/opencv4 ]; then
                export OPENCV_INCLUDE_PATHS="/usr/local/include/opencv4"
            else
                fail "Could not detect OpenCV headers via pkg-config or common Linux include paths"
            fi

            if [ -d /usr/lib/x86_64-linux-gnu ]; then
                export OPENCV_LINK_PATHS="/usr/lib/x86_64-linux-gnu"
            elif [ -d /usr/lib64 ]; then
                export OPENCV_LINK_PATHS="/usr/lib64"
            elif [ -d /usr/lib ]; then
                export OPENCV_LINK_PATHS="/usr/lib"
            else
                fail "Could not detect OpenCV library directory via pkg-config or common Linux library paths"
            fi
            ;;
        *)
            fail "Unsupported operating system: ${OS_NAME}"
            ;;
    esac
}

fill_default_opencv_link_paths() {
    if [ -n "${OPENCV_LINK_PATHS:-}" ]; then
        return 0
    fi

    case "${OS_NAME}" in
        Darwin)
            if [ -d /opt/homebrew/lib ]; then
                export OPENCV_LINK_PATHS="/opt/homebrew/lib"
            elif [ -d /usr/local/lib ]; then
                export OPENCV_LINK_PATHS="/usr/local/lib"
            fi
            ;;
        Linux)
            if [ -d /usr/lib/x86_64-linux-gnu ]; then
                export OPENCV_LINK_PATHS="/usr/lib/x86_64-linux-gnu"
            elif [ -d /usr/lib64 ]; then
                export OPENCV_LINK_PATHS="/usr/lib64"
            elif [ -d /usr/lib ]; then
                export OPENCV_LINK_PATHS="/usr/lib"
            fi
            ;;
    esac
}

echo "Setting up OpenCV Rust environment..."

OS_NAME="$(uname -s)"
case "${OS_NAME}" in
    Darwin|Linux) ;;
    *) fail "Unsupported operating system: ${OS_NAME}" ;;
esac

LIBCLANG_DIR="$(find_libclang_path || true)"
if [ -z "${LIBCLANG_DIR}" ]; then
    fail "Could not detect libclang. Install LLVM/Clang and ensure llvm-config is on PATH."
fi

LLVM_CONFIG_BIN="$(find_llvm_config || true)"
if [ -n "${LLVM_CONFIG_BIN}" ]; then
    export LLVM_CONFIG_PATH="${LLVM_CONFIG_BIN}"
fi

export LIBCLANG_PATH="${LIBCLANG_DIR}"

case "${OS_NAME}" in
    Darwin)
        prepend_path_var DYLD_LIBRARY_PATH "${LIBCLANG_DIR}"
        if command -v sw_vers >/dev/null 2>&1; then
            export MACOSX_DEPLOYMENT_TARGET="$(sw_vers -productVersion | cut -d. -f1,2)"
        fi
        ;;
    Linux)
        prepend_path_var LD_LIBRARY_PATH "${LIBCLANG_DIR}"
        ;;
esac

if ! detect_opencv_from_pkg_config; then
    detect_opencv_from_defaults
fi
fill_default_opencv_link_paths

append_rustflag "-C target-cpu=native"

echo "Environment variables set:"
echo "  OS=${OS_NAME}"
echo "  LIBCLANG_PATH=${LIBCLANG_PATH}"
if [ -n "${LLVM_CONFIG_PATH:-}" ]; then
    echo "  LLVM_CONFIG_PATH=${LLVM_CONFIG_PATH}"
fi
echo "  OPENCV_LINK_LIBS=${OPENCV_LINK_LIBS}"
echo "  OPENCV_LINK_PATHS=${OPENCV_LINK_PATHS}"
echo "  OPENCV_INCLUDE_PATHS=${OPENCV_INCLUDE_PATHS}"
if [ "${OS_NAME}" = "Darwin" ]; then
    echo "  DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH:-}"
    if [ -n "${MACOSX_DEPLOYMENT_TARGET:-}" ]; then
        echo "  MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET}"
    fi
else
    echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
fi
echo "  RUSTFLAGS=${RUSTFLAGS}"
echo ""
echo "You can now run:"
echo "  cargo clippy --all-features"
echo "  cargo test --features opencv,simd"
echo "  cargo test --all-features  # (requires Python environment)"
echo ""
echo "To make this permanent, add this to your ~/.zshrc or ~/.bashrc:"
echo "  source $(pwd)/setup-env.sh"
