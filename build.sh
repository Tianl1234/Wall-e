#!/usr/bin/env bash
# Build Wall-E 2 -> chess.exe
#
# IMPORTANT: -static is not optional. Without it the exe needs libstdc++-6.dll /
# libgcc_s_seh-1.dll from MSYS2, which are not on the system PATH, so any GUI
# (En Croissant, Arena, Cute Chess) spawns it and it dies instantly.
set -euo pipefail

# Prefer the requested w64devkit toolchain, and add its bin directory to PATH
# so GCC can find its assembler and linker.
W64DEVKIT_HOME="${W64DEVKIT_HOME:-/c/Users/Administrator/Desktop/w64devkit}"
if [[ -z "${CXX:-}" && -x "$W64DEVKIT_HOME/bin/g++.exe" ]]; then
    CXX="$W64DEVKIT_HOME/bin/g++.exe"
    export PATH="$W64DEVKIT_HOME/bin:$PATH"
else
    CXX="${CXX:-g++}"
fi
"$CXX" --version | head -1

# Native Zen 5 build: AVX-512 NNUE plus the existing PEXT bitboard backend.
# Strip release symbols so only executable code/data and the external NNUE remain.
"$CXX" -B"$W64DEVKIT_HOME/bin/" -O3 -march=znver5 -std=c++20 -DNDEBUG -s \
       -static -static-libgcc -static-libstdc++ \
       -o chess.exe chess.cpp nnue.cpp

echo "built chess.exe ($(du -h chess.exe | cut -f1), statically linked)"
