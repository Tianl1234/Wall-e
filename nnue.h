#pragma once

#include <cstdint>
#include <string>

namespace Nnue {
constexpr int HIDDEN_SIZE = 1024;
constexpr int INPUT_SIZE = 768;

// 16-bit accumulators use the network's specified saturating updates.
struct alignas(64) Accumulator {
    alignas(64) int16_t values[2][HIDDEN_SIZE]{};
};

bool load(const std::string& networkPath);
bool available();
struct PieceChange { int color; int type; int square; bool add; };
void apply_changes(Accumulator& accumulator, const PieceChange* changes, int count);
void reset(Accumulator& accumulator, const uint8_t board[64]);
bool consistent(const Accumulator& accumulator, const uint8_t board[64]);
int evaluate(const Accumulator& accumulator, int sideToMove);
} // namespace Nnue
