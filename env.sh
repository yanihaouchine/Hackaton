# source env.sh — active l'environnement du hackathon
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$H/.venv/bin/activate"
export LLAMA_CPU="$H/tools/llama.cpp/build-cpu/bin"     # x86 portable (Intel+AMD, dispatch runtime)
export LLAMA_CUDA="$H/tools/llama.cpp/build-cuda/bin"   # x86 + CUDA (sm_86 : RTX 3060 / A6000)
export LLAMA_ARM64="$H/tools/llama.cpp/build-arm64/bin" # aarch64 (OrangePi/Jetson CPU), test via qemu
export MODELS="$H/models"
export PATH="$LLAMA_CPU:$H/tools/qemu/usr/bin:$H/tools/cuda/bin:$PATH"
# exécuter un binaire ARM64 ici : qemu-aarch64 -L $H/tools/xarm/aarch64-conda-linux-gnu/sysroot $LLAMA_ARM64/llama-completion ...
export QEMU_LD_PREFIX="$H/tools/xarm/aarch64-conda-linux-gnu/sysroot"
