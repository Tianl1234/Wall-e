#pragma once

#include <cstdint>
#include <string>

namespace Nnue {
constexpr int HIDDEN_SIZE = 1024;
constexpr int INPUT_SIZE = 768;

// Bundled networks, addressed by slot because the "strong" package carries all
// three of them (see nnue_embedded.cpp) and picks one per search:
//   0 = fingolfin (middlegame), 1 = finarfin (endgame), 2 = feanor (aggressive).
// The plain package only ever fills slot 0, so loading a higher slot fails there.
constexpr int NUM_NETS = 3;
constexpr int MIDDLEGAME_NET = 0;
constexpr int ENDGAME_NET = 1;
constexpr int AGGRESSIVE_NET = 2;

// 16-bit accumulators use the network's specified saturating updates.
struct alignas(64) Accumulator {
    alignas(64) int16_t values[2][HIDDEN_SIZE]{};
};

bool load(const std::string& networkPath);
// Network linked into the binary instead of read from disk (Android package).
bool load_embedded();
bool load_embedded(int slot);
bool available();
// Makes a loaded network the active evaluator; false when that slot is empty.
bool select(int slot);
struct PieceChange { int color; int type; int square; bool add; };
void apply_changes(Accumulator& accumulator, const PieceChange* changes, int count);
void reset(Accumulator& accumulator, const uint8_t board[64]);
bool consistent(const Accumulator& accumulator, const uint8_t board[64]);
int evaluate(const Accumulator& accumulator, int sideToMove);
} // namespace Nnue
