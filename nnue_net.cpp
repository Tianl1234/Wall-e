// Embeds the NNUE network into the executable's read-only data so the engine
// ships as a single self-contained file (no separate nnue/ folder to ship).
//
// The payload is pulled in at assembly time from NNUE_EMBEDDED_FILE, which is
// relative to the directory the build is run from (build.sh cd's to the
// project root for this reason).
#include <cstddef>

#ifndef NNUE_EMBEDDED_FILE
#define NNUE_EMBEDDED_FILE "nnue/fingolfin.nnue"
#endif

// NOTE: the trailing ".text" is required. This is file-scope asm, so without
// it the assembler stays in .rodata and every function emitted afterwards in
// this translation unit would land in a non-executable section (and crash).
__asm__(
    ".section .rodata\n"
    ".globl kEmbeddedNet\n"
    ".globl kEmbeddedNetEnd\n"
    ".balign 64\n"
    "kEmbeddedNet:\n"
    ".incbin \"" NNUE_EMBEDDED_FILE "\"\n"
    "kEmbeddedNetEnd:\n"
    ".text\n");

extern "C" const unsigned char kEmbeddedNet[];
extern "C" const unsigned char kEmbeddedNetEnd[];

namespace Nnue {
const unsigned char* embedded_network(std::size_t& size) {
    size = static_cast<std::size_t>(kEmbeddedNetEnd - kEmbeddedNet);
    return kEmbeddedNet;
}
} // namespace Nnue
