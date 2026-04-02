#!/bin/bash

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m' # Reset to default color

# Set options to exit on error and unset variables
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CORE_DIR="${REPO_ROOT}/core"
TEMPLATE_DIR="${SCRIPT_DIR}/templates"
TOOLCHAIN_FILE="${SCRIPT_DIR}/cmake/conda_toolchain.cmake"
OPENSAI_CONDA_ENV="${OPENSAI_CONDA_ENV:-opensai-dev}"

detect_build_jobs() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu 2>/dev/null || echo 4
    else
        echo 4
    fi
}

BUILD_JOBS="${OPENSAI_BUILD_JOBS:-$(detect_build_jobs)}"

die() {
    echo "${RED}${BOLD}$1${RESET}" >&2
    exit 1
}

conda_env_exists() {
    conda env list | awk 'NR > 2 {print $1}' | grep -qx "$OPENSAI_CONDA_ENV"
}

ensure_conda_environment() {
    if ! command -v conda >/dev/null 2>&1; then
        die "Conda is required. Run install_dependencies.sh first."
    fi

    if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" != "$OPENSAI_CONDA_ENV" ]]; then
        if conda_env_exists; then
            exec conda run --no-capture-output -n "$OPENSAI_CONDA_ENV" bash "${SCRIPT_DIR}/install_core_libraries.sh" "$@"
        fi

        die "Conda environment '$OPENSAI_CONDA_ENV' was not found. Run install_dependencies.sh first."
    fi
}

setup_conda_paths() {
    export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${CONDA_PREFIX}/share/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
    export CPATH="${CONDA_PREFIX}/include${CPATH:+:${CPATH}}"
    if [ -d "${CONDA_PREFIX}/include/eigen3" ]; then
        export CPATH="${CONDA_PREFIX}/include/eigen3:${CPATH}"
    fi
    export LIBRARY_PATH="${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
}

sync_conda_cmake_templates() {
    [ -f "${TEMPLATE_DIR}/sai-common.CMakeLists.txt" ] || die "Missing template for sai-common."
    [ -f "${TEMPLATE_DIR}/sai-urdfreader.CMakeLists.txt" ] || die "Missing template for sai-urdfreader."
    [ -f "${TEMPLATE_DIR}/sai-model.CMakeLists.txt" ] || die "Missing template for sai-model."
    [ -f "${TEMPLATE_DIR}/sai-primitives.CMakeLists.txt" ] || die "Missing template for sai-primitives."

    install -m 644 "${TEMPLATE_DIR}/sai-common.CMakeLists.txt" "${CORE_DIR}/sai-common/CMakeLists.txt"
    install -m 644 "${TEMPLATE_DIR}/sai-urdfreader.CMakeLists.txt" "${CORE_DIR}/sai-urdfreader/CMakeLists.txt"
    install -m 644 "${TEMPLATE_DIR}/sai-model.CMakeLists.txt" "${CORE_DIR}/sai-model/CMakeLists.txt"
    install -m 644 "${TEMPLATE_DIR}/sai-primitives.CMakeLists.txt" "${CORE_DIR}/sai-primitives/CMakeLists.txt"
}

cmake_args=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}"
    -DCMAKE_PREFIX_PATH="${CONDA_PREFIX:-}"
    -DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE
    -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=TRUE
    -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE
    -DCMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY=TRUE
    -DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON
)

if [ -d "${CONDA_PREFIX:-}/include/eigen3" ]; then
    cmake_args+=("-DEIGEN3_INCLUDE_DIR=${CONDA_PREFIX}/include/eigen3")
    cmake_args+=("-DEIGEN3_INCLUDE_DIRS=${CONDA_PREFIX}/include/eigen3")
fi

build_cmake_project() {
    local repo_dir="$1"
    shift

    cmake --fresh "${cmake_args[@]}" "$@" -S "$repo_dir" -B "$repo_dir/build"
    cmake --build "$repo_dir/build" --parallel "$BUILD_JOBS"
}

patch_rbdl_for_modern_eigen() {
    local rbdl_header="$CORE_DIR/sai-model/rbdl/include/rbdl/rbdl_eigenmath.h"

    if [[ -f "$rbdl_header" ]] && grep -q "_check_template_params" "$rbdl_header"; then
        perl -0pi -e 's/^[ \t]*Base::_check_template_params\(\);\n//mg' "$rbdl_header"
    fi
}

ensure_conda_environment "$@"
setup_conda_paths

if [ ! -f "$TOOLCHAIN_FILE" ]; then
    die "Missing Conda toolchain file at $TOOLCHAIN_FILE"
fi

# install core libraries
mkdir -p "$CORE_DIR"
cd "$CORE_DIR"

# clone all repositories if needed
if [ ! -d "sai-common" ]; then
    git clone https://github.com/manips-sai-org/sai-common.git
fi
if [ ! -d "sai-urdfreader" ]; then
    git clone https://github.com/manips-sai-org/sai-urdfreader.git
fi
if [ ! -d "sai-model" ]; then
    git clone https://github.com/manips-sai-org/sai-model.git
fi
if [ ! -d "sai-primitives" ]; then
    git clone https://github.com/manips-sai-org/sai-primitives.git
fi

# echo "Cloned all repositories."
echo "${YELLOW}${BOLD}All repositories successfully cloned (or cloning was not needed)${RESET}"
sleep 0.5

sync_conda_cmake_templates
patch_rbdl_for_modern_eigen

# build all the repositories
build_cmake_project "sai-common" "-DBUILD_EXAMPLES=OFF"
echo "${YELLOW}${BOLD}sai-common successfully built${RESET}"
sleep 0.5

build_cmake_project "sai-urdfreader" "-DBUILD_EXAMPLES=OFF"
echo "${YELLOW}${BOLD}sai-urdfreader successfully built${RESET}"
sleep 0.5

build_cmake_project "sai-model/rbdl" "-DRBDL_BUILD_TESTS=OFF"
build_cmake_project "sai-model" "-DSAI-URDF_DIR=${CORE_DIR}/sai-urdfreader/build" "-DBUILD_EXAMPLES=OFF" "-DBUILD_TESTS=OFF"
echo "${YELLOW}${BOLD}sai-model successfully built${RESET}"
sleep 0.5

build_cmake_project "sai-primitives/ruckig" "-DBUILD_EXAMPLES=OFF" "-DBUILD_TESTS=OFF"
build_cmake_project "sai-primitives" "-DSAI-MODEL_DIR=${CORE_DIR}/sai-model/build" "-DBUILD_EXAMPLES=OFF"
echo "${YELLOW}${BOLD}sai-primitives successfully built${RESET}"
sleep 0.5

cd "$REPO_ROOT"
echo "${GREEN}${BOLD}All repositories successfully built. The main OpenSai application can be built${RESET}"
