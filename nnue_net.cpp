// Embeds the NNUE networks into the executable's read-only data so the engine
// ships as a single self-contained file (no separate nnue/ folder to ship).
//
// The payloads are pulled in at assembly time from paths relative to the
// directory the build is run from (build.sh cd's to the project root for this
// reason). Slot 0 (fingolfin) is always embedded; the endgame (finarfin) and
// aggressive (feanor) networks let the engine pick an evaluator per search.
#include <cstddef>

#include "nnue.h"

#ifndef NNUE_EMBEDDED_FILE
#define NNUE_EMBEDDED_FILE "nnue/fingolfin.nnue"
#endif
#ifndef NNUE_EMBEDDED_ENDGAME_FILE
#define NNUE_EMBEDDED_ENDGAME_FILE "nnue/finarfin.nnue"
#endif
#ifndef NNUE_EMBEDDED_AGGRESSIVE_FILE
#define NNUE_EMBEDDED_AGGRESSIVE_FILE "nnue/feanor.nnue"
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
    ".globl kEmbeddedEndgame\n"
    ".globl kEmbeddedEndgameEnd\n"
    ".balign 64\n"
    "kEmbeddedEndgame:\n"
    ".incbin \"" NNUE_EMBEDDED_ENDGAME_FILE "\"\n"
    "kEmbeddedEndgameEnd:\n"
    ".globl kEmbeddedAggressive\n"
    ".globl kEmbeddedAggressiveEnd\n"
    ".balign 64\n"
    "kEmbeddedAggressive:\n"
    ".incbin \"" NNUE_EMBEDDED_AGGRESSIVE_FILE "\"\n"
    "kEmbeddedAggressiveEnd:\n"
    ".text\n");

extern "C" const unsigned char kEmbeddedNet[];
extern "C" const unsigned char kEmbeddedNetEnd[];
extern "C" const unsigned char kEmbeddedEndgame[];
extern "C" const unsigned char kEmbeddedEndgameEnd[];
extern "C" const unsigned char kEmbeddedAggressive[];
extern "C" const unsigned char kEmbeddedAggressiveEnd[];

namespace Nnue {
const unsigned char* embedded_network(int slot, std::size_t& size) {
    switch (slot) {
        case MIDDLEGAME_NET:
            size = static_cast<std::size_t>(kEmbeddedNetEnd - kEmbeddedNet);
            return kEmbeddedNet;
        case ENDGAME_NET:
            size = static_cast<std::size_t>(kEmbeddedEndgameEnd - kEmbeddedEndgame);
            return kEmbeddedEndgame;
        case AGGRESSIVE_NET:
            size = static_cast<std::size_t>(kEmbeddedAggressiveEnd - kEmbeddedAggressive);
            return kEmbeddedAggressive;
        default:
            size = 0;
            return nullptr;
    }
}
} // namespace Nnue
