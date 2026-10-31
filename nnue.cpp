#include "nnue.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace Nnue {
namespace {
constexpr int QA = 255;
constexpr int SCALE = 400;
constexpr std::size_t NETWORK_BYTES =
    (std::size_t(INPUT_SIZE) * HIDDEN_SIZE + HIDDEN_SIZE * 3 + 1) * sizeof(int16_t);

struct alignas(64) Network {
    alignas(64) int16_t featureWeights[INPUT_SIZE][HIDDEN_SIZE];
    alignas(64) int16_t bias[HIDDEN_SIZE];
    alignas(64) int16_t outputWeights[2][HIDDEN_SIZE];
    int16_t outputBias;
};
std::unique_ptr<Network> network;

bool network_sane();

bool parse_network(const int16_t* raw, const char* label) {
    auto candidate = std::make_unique<Network>();
    std::memcpy(candidate->featureWeights, raw, std::size_t(INPUT_SIZE) * HIDDEN_SIZE * sizeof(int16_t));
    std::size_t offset = std::size_t(INPUT_SIZE) * HIDDEN_SIZE;
    std::memcpy(candidate->bias, raw + offset, HIDDEN_SIZE * sizeof(int16_t));
    offset += HIDDEN_SIZE;
    std::memcpy(candidate->outputWeights, raw + offset, HIDDEN_SIZE * 2 * sizeof(int16_t));
    offset += HIDDEN_SIZE * 2;
    candidate->outputBias = raw[offset];

    // The SIMD inference path multiplies clipped activations (0..255) by
    // signed weights in 16-bit lanes before widening. Keep the product within
    // int16_t so a replacement/retrained network cannot silently wrap.
    for (const auto& perspective : candidate->outputWeights)
        for (int16_t weight : perspective)
            if (weight < -128 || weight > 128) return false;

    network = std::move(candidate);
    // A network whose material ordering is nonsense, or that does not match
    // this code's feature layout, would silently ruin every evaluation.
    // Reject it so the caller can fall back to the handcrafted evaluation.
    if (!network_sane()) {
        network.reset();
        std::cerr << "info string NNUE: " << label
                  << " fails material sanity checks; rejected\n";
        return false;
    }
    return true;
}

bool load_network(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    std::vector<int16_t> raw(NETWORK_BYTES / sizeof(int16_t));
    file.read(reinterpret_cast<char*>(raw.data()), NETWORK_BYTES);
    if (file.gcount() != static_cast<std::streamsize>(NETWORK_BYTES)) return false;
    if (file.peek() != std::char_traits<char>::eof()) return false;
    return parse_network(raw.data(), path.c_str());
}

inline int feature_index(int piece, int square, int perspective) {
    const int color = piece >> 3;
    const int type = piece & 7;
    const int orientedSquare = square ^ (perspective ? 56 : 0);
    return ((color ^ perspective) * 6 + type - 1) * 64 + orientedSquare;
}

void update_row(int16_t* dst, const int16_t* row, bool add) {
#if defined(__AVX512BW__) && defined(__AVX512F__)
    for (int i = 0; i < HIDDEN_SIZE; i += 32) {
        __m512i value = _mm512_load_si512(reinterpret_cast<const __m512i*>(dst + i));
        const __m512i weights = _mm512_load_si512(reinterpret_cast<const __m512i*>(row + i));
        value = add ? _mm512_adds_epi16(value, weights) : _mm512_subs_epi16(value, weights);
        _mm512_store_si512(reinterpret_cast<__m512i*>(dst + i), value);
    }
#elif defined(__AVX2__)
    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        __m256i value = _mm256_load_si256(reinterpret_cast<const __m256i*>(dst + i));
        const __m256i weights = _mm256_load_si256(reinterpret_cast<const __m256i*>(row + i));
        value = add ? _mm256_adds_epi16(value, weights) : _mm256_subs_epi16(value, weights);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i), value);
    }
#else
    const int sign = add ? 1 : -1;
    for (int i = 0; i < HIDDEN_SIZE; ++i)
        dst[i] = static_cast<int16_t>(std::clamp(
            static_cast<int>(dst[i]) + sign * row[i], -32768, 32767));
#endif
}

int infer(const Accumulator& acc, int stm) {
    const int16_t* us = acc.values[stm];
    const int16_t* them = acc.values[stm ^ 1];
    int64_t total = static_cast<int64_t>(network->outputBias) * QA;

#if defined(__AVX512BW__) && defined(__AVX512F__)
    __m512i sum = _mm512_setzero_si512();
    const __m512i zero = _mm512_setzero_si512();
    const __m512i maxValue = _mm512_set1_epi16(QA);
    for (int i = 0; i < HIDDEN_SIZE; i += 32) {
        __m512i a = _mm512_load_si512(reinterpret_cast<const __m512i*>(us + i));
        __m512i b = _mm512_load_si512(reinterpret_cast<const __m512i*>(them + i));
        a = _mm512_min_epi16(_mm512_max_epi16(a, zero), maxValue);
        b = _mm512_min_epi16(_mm512_max_epi16(b, zero), maxValue);
        const __m512i wa = _mm512_load_si512(reinterpret_cast<const __m512i*>(network->outputWeights[0] + i));
        const __m512i wb = _mm512_load_si512(reinterpret_cast<const __m512i*>(network->outputWeights[1] + i));
        sum = _mm512_add_epi32(sum, _mm512_madd_epi16(_mm512_mullo_epi16(a, wa), a));
        sum = _mm512_add_epi32(sum, _mm512_madd_epi16(_mm512_mullo_epi16(b, wb), b));
    }
    alignas(64) int32_t lanes[16];
    _mm512_store_si512(reinterpret_cast<__m512i*>(lanes), sum);
    for (int lane : lanes) total += lane;
#elif defined(__AVX2__)
    __m256i sum = _mm256_setzero_si256();
    const __m256i zero = _mm256_setzero_si256();
    const __m256i maxValue = _mm256_set1_epi16(QA);
    for (int i = 0; i < HIDDEN_SIZE; i += 16) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(us + i));
        __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(them + i));
        a = _mm256_min_epi16(_mm256_max_epi16(a, zero), maxValue);
        b = _mm256_min_epi16(_mm256_max_epi16(b, zero), maxValue);
        const __m256i wa = _mm256_load_si256(reinterpret_cast<const __m256i*>(network->outputWeights[0] + i));
        const __m256i wb = _mm256_load_si256(reinterpret_cast<const __m256i*>(network->outputWeights[1] + i));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_mullo_epi16(a, wa), a));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_mullo_epi16(b, wb), b));
    }
    alignas(32) int32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), sum);
    for (int lane : lanes) total += lane;
#else
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        const int32_t a = std::clamp(static_cast<int>(us[i]), 0, QA);
        const int32_t b = std::clamp(static_cast<int>(them[i]), 0, QA);
        total += static_cast<int64_t>(a) * a * network->outputWeights[0][i];
        total += static_cast<int64_t>(b) * b * network->outputWeights[1][i];
    }
#endif
    return static_cast<int>((total * SCALE) /
                            (static_cast<int64_t>(QA) * QA * 64));
}

// Cheap sanity probe: evaluate a bare-kings position and the same position with
// one extra white piece (pawn/knight/bishop/rook/queen on d4), all with white
// to move. A usable network must return a sane, monotone material ordering;
// anything else (a mismatched or corrupt file) is refused by the loader so the
// engine keeps the handcrafted evaluation instead of a broken one.
bool network_sane() {
    if (!network) return false;

    static const int probeType[5] = {1, 2, 3, 4, 5};  // pawn, knight, bishop, rook, queen

    uint8_t bare[64] = {};
    bare[4]  = 6;    // white king e1
    bare[60] = 14;   // black king e8
    Accumulator baseline{};
    reset(baseline, bare);
    const int zero = infer(baseline, 0);

    int val[5];
    for (int t = 0; t < 5; ++t) {
        uint8_t board[64] = {};
        board[4]  = 6;
        board[60] = 14;
        board[27] = static_cast<uint8_t>(probeType[t]);   // extra white piece on d4
        Accumulator acc{};
        reset(acc, board);
        val[t] = infer(acc, 0) - zero;
    }

    if (val[0] < 20) return false;                                   // a pawn must be worth something
    if (val[1] <= val[0] || val[2] <= val[0]) return false;           // minors beat a pawn
    if (val[3] <= val[1] || val[3] <= val[2]) return false;           // a rook beats the minors
    if (val[4] <= val[3]) return false;                               // a queen beats a rook
    return val[4] >= 300 && val[4] <= 2500;                           // and everything is on a sane scale
}
} // namespace

bool load(const std::string& networkPath) { return load_network(networkPath); }

bool load_embedded() {
    std::size_t size = 0;
    const unsigned char* bytes = embedded_network(size);
    if (!bytes || size != NETWORK_BYTES) return false;
    std::vector<int16_t> raw(NETWORK_BYTES / sizeof(int16_t));
    std::memcpy(raw.data(), bytes, NETWORK_BYTES);
    return parse_network(raw.data(), "<embedded network>");
}
bool available() { return static_cast<bool>(network); }

void apply_changes(Accumulator& acc, const PieceChange* changes, int count) {
    if (!network || count <= 0) return;
    for (int change = 0; change < count; ++change) {
        const PieceChange& piece = changes[change];
        if (piece.type < 1 || piece.type > 6 || piece.square < 0 || piece.square >= 64) continue;
        const int pc = (piece.color << 3) | piece.type;
        for (int perspective = 0; perspective < 2; ++perspective) {
            const int index = feature_index(pc, piece.square, perspective);
            update_row(acc.values[perspective], network->featureWeights[index], piece.add);
        }
    }
}

void reset(Accumulator& acc, const uint8_t board[64]) {
    if (!network) return;
    for (int perspective = 0; perspective < 2; ++perspective) {
        int16_t* dst = acc.values[perspective];
        std::copy_n(network->bias, HIDDEN_SIZE, dst);
        for (int square = 0; square < 64; ++square) {
            const int piece = board[square];
            if (!piece) continue;
            const int index = feature_index(piece, square, perspective);
            update_row(dst, network->featureWeights[index], true);
        }
    }
}

bool consistent(const Accumulator& accumulator, const uint8_t board[64]) {
    if (!network) return true;
    Accumulator expected{};
    reset(expected, board);
    return std::memcmp(&expected, &accumulator, sizeof(expected)) == 0;
}

int evaluate(const Accumulator& accumulator, int sideToMove) {
    return network ? infer(accumulator, sideToMove) : 0;
}
} // namespace Nnue
