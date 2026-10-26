# Hackaton — LLM basse conso

`source env.sh` pour tout activer.

| Élément | Emplacement |
|---|---|
| venv Python 3.12 (torch 2.14 cu130, transformers, peft, onnxruntime, optimum, llama-cpp-python, lm-eval, faiss, sentence-transformers…) | `.venv/` |
| llama.cpp x86 portable (variantes Intel + AMD, choisie au lancement) | `tools/llama.cpp/build-cpu/bin` |
| llama.cpp x86 + CUDA sm_86 (RTX 3060 / A6000) | `tools/llama.cpp/build-cuda/bin` |
| llama.cpp aarch64 (armv8.0 → armv9.2, OrangePi A76/A55, Jetson CPU) | `tools/llama.cpp/build-arm64/bin` |
| Toolchain CUDA 13.0 + gcc 14 (hôte nvcc) | `tools/cuda` |
| Cross-gcc aarch64 + sysroot | `tools/xarm`, toolchain cmake `tools/aarch64.cmake` |
| qemu-aarch64 (tester les binaires ARM ici) | `tools/qemu/usr/bin` |
| micromamba (installer d'autres paquets sans sudo) | `tools/mamba/bin/micromamba` |
| Modèles GGUF | `models/` |

Tester un binaire ARM : se placer dans `tools/llama.cpp/build-arm64/bin` (sous qemu, les backends sont cherchés dans le répertoire courant) puis
`qemu-aarch64 -L $QEMU_LD_PREFIX ./llama-completion -m <gguf> -p "..."`.

## Éditeurs
Serveurs de langage (dans `~/.local/bin`) : `clangd`, `basedpyright`, `ruff`, `clang-format`, `cmake-language-server`.
Pour qu'un projet C/C++/CMake soit compris par clangd : `cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` puis `ln -s build/compile_commands.json .`

- **VS Code** : `~/.config/Code/User/settings.json` + `.vscode/` (venv, débogage Python/lldb, tâches llama-server/bench/pytest).
- **Emacs 30** : `~/.config/emacs/init.el`. Raccourcis principaux :
  `C-s` chercher dans le buffer, `M-s r` ripgrep, `C-x b` buffers, `M-g i` imenu, `C-.` embark,
  `C-c l r/a/f` renommer / actions / formater (eglot), `M-n`/`M-p` erreur suivante/précédente,
  `C-x g` magit, `C-c c` compile, `F5` recompile, `C-c t` terminal, `C-c G` / `C-c g` chat/envoi vers llama-server local (port 8080),
  `C->` multi-curseurs, `C-=` expand-region, `M-j` avy.
