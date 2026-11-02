#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace Nnue {
constexpr int HIDDEN_SIZE = 1024;
constexpr int INPUT_SIZE = 768;

// The three networks bundled with Patricia (MIT, Adam Kulju), addressed by slot:
//   0 = fingolfin (strongest general net), 1 = finarfin (endgame),
//   2 = feanor (aggressive; only reached through an explicit EvalFile).
// The desktop executable embeds all three. PhaseNets selects finarfin once the
// root position has little material left, fingolfin otherwise; EvalFile pins a
// single network and turns the automatic choice off.
constexpr int NUM_NETS = 3;
constexpr int MIDDLEGAME_NET = 0;
constexpr int ENDGAME_NET = 1;
constexpr int AGGRESSIVE_NET = 2;

// 16-bit accumulators use the network's specified saturating updates.
struct alignas(64) Accumulator {
    alignas(64) int16_t values[2][HIDDEN_SIZE]{};
};

bool load(const std::string& networkPath);
// Loads network `slot` compiled into the executable (see nnue_net.cpp).
bool load_embedded();
bool load_embedded(int slot);
bool available();
// Makes a loaded network the active evaluator; false when that slot is empty.
bool select(int slot);
// Network bytes compiled into the executable for `slot`; size is 0 if the build
// did not embed one.
const unsigned char* embedded_network(int slot, std::size_t& size);

// A single move changes at most two features per perspective (a castle moves
// king and rook, a capture plus promotion removes two pieces). Collecting the
// adds and subs lets one pass load the accumulator, apply both, and store it
// once, instead of streaming it once per changed feature. Structure borrowed
// from Patricia (MIT, Adam Kulju), which trains the bundled networks.
struct PieceChange { int color; int type; int square; };
struct Update {
    PieceChange add[2];
    PieceChange sub[2];
    int adds = 0;
    int subs = 0;

    void addPiece(int color, int type, int square) {
        if (adds < 2) add[adds++] = {color, type, square};
    }
    void subPiece(int color, int type, int square) {
        if (subs < 2) sub[subs++] = {color, type, square};
    }
};
// Writes src + update into dst (dst may alias src). The search keeps one
// accumulator per ply and only ever computes forwards, like Patricia (MIT):
// unmake then costs nothing because the parent slot is still intact.
void apply_update(Accumulator& dst, const Accumulator& src, const Update& update);
// dst = src - update; used when a move is taken back outside the search stack.
void revert_update(Accumulator& dst, const Accumulator& src, const Update& update);
// Pull the feature rows of a not-yet-applied update into L1.
void prefetch_update(const Update& update);

void reset(Accumulator& accumulator, const uint8_t board[64]);
bool consistent(const Accumulator& accumulator, const uint8_t board[64]);
int evaluate(const Accumulator& accumulator, int sideToMove);
} // namespace Nnue
