#include "qr.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace {

using grid = std::vector<std::vector<bool>>;

// level M: error correction codewords per block, and blocks, versions 1-10
const int ECC_PER_BLOCK[11] = { 0, 10, 16, 26, 18, 24, 16, 18, 22, 22, 26 };
const int BLOCKS[11] = { 0, 1, 1, 1, 2, 2, 4, 4, 4, 5, 5 };

int raw_modules(int v) {   // modules left for data and error correction
    int r = (16 * v + 128) * v + 64;
    if (v >= 2) {
        const int na = v / 7 + 2;
        r -= (25 * na - 10) * na - 55;
        if (v >= 7) r -= 36;
    }
    return r;
}
int data_codewords(int v) { return raw_modules(v) / 8 - ECC_PER_BLOCK[v] * BLOCKS[v]; }

uint8_t gf_mul(uint8_t x, uint8_t y) {   // GF(256) with x^8 + x^4 + x^3 + x^2 + 1
    int z = 0;
    for (int i = 7; i >= 0; i--) {
        z = (z << 1) ^ ((z >> 7) * 0x11D);
        z ^= ((y >> i) & 1) * x;
    }
    return (uint8_t) z;
}
std::vector<uint8_t> rs_divisor(int degree) {
    std::vector<uint8_t> r(degree, 0);
    r[degree - 1] = 1;
    uint8_t root = 1;
    for (int i = 0; i < degree; i++) {
        for (int j = 0; j < degree; j++) {
            r[j] = gf_mul(r[j], root);
            if (j + 1 < degree) r[j] ^= r[j + 1];
        }
        root = gf_mul(root, 0x02);
    }
    return r;
}
std::vector<uint8_t> rs_remainder(const std::vector<uint8_t> & data, const std::vector<uint8_t> & div) {
    std::vector<uint8_t> r(div.size(), 0);
    for (uint8_t b : data) {
        const uint8_t f = b ^ r[0];
        r.erase(r.begin());
        r.push_back(0);
        for (size_t i = 0; i < r.size(); i++) r[i] ^= gf_mul(div[i], f);
    }
    return r;
}

struct code {
    int v, n;
    grid dark, fixed;
    explicit code(int ver) : v(ver), n(4 * ver + 17), dark(n, std::vector<bool>(n)), fixed(n, std::vector<bool>(n)) {}
    void set(int x, int y, bool d) { dark[y][x] = d; fixed[y][x] = true; }

    void finder(int cx, int cy) {
        for (int dy = -4; dy <= 4; dy++)
            for (int dx = -4; dx <= 4; dx++) {
                const int d = std::max(std::abs(dx), std::abs(dy)), x = cx + dx, y = cy + dy;
                if (x >= 0 && x < n && y >= 0 && y < n) set(x, y, d != 2 && d != 4);
            }
    }
    void align(int cx, int cy) {
        for (int dy = -2; dy <= 2; dy++)
            for (int dx = -2; dx <= 2; dx++) set(cx + dx, cy + dy, std::max(std::abs(dx), std::abs(dy)) != 1);
    }
    std::vector<int> align_pos() const {
        if (v == 1) return {};
        const int na = v / 7 + 2, step = (v * 4 + na * 2 + 1) / (na * 2 - 2) * 2;
        std::vector<int> r;
        for (int i = 0, p = v * 4 + 10; i < na - 1; i++, p -= step) r.insert(r.begin(), p);
        r.insert(r.begin(), 6);
        return r;
    }
    void format(int mask) {
        const int data = 0 << 3 | mask;   // level M is 00
        int rem = data;
        for (int i = 0; i < 10; i++) rem = (rem << 1) ^ ((rem >> 9) * 0x537);
        const int bits = (data << 10 | rem) ^ 0x5412;
        auto bit = [&](int i) { return ((bits >> i) & 1) != 0; };
        for (int i = 0; i <= 5; i++) set(8, i, bit(i));
        set(8, 7, bit(6));
        set(8, 8, bit(7));
        set(7, 8, bit(8));
        for (int i = 9; i < 15; i++) set(14 - i, 8, bit(i));
        for (int i = 0; i < 8; i++) set(n - 1 - i, 8, bit(i));
        for (int i = 8; i < 15; i++) set(8, n - 15 + i, bit(i));
        set(8, n - 8, true);
    }
    void version() {
        if (v < 7) return;
        int rem = v;
        for (int i = 0; i < 12; i++) rem = (rem << 1) ^ ((rem >> 11) * 0x1F25);
        const long bits = (long) v << 12 | rem;
        for (int i = 0; i < 18; i++) {
            const bool b = ((bits >> i) & 1) != 0;
            const int a = n - 11 + i % 3, c = i / 3;
            set(a, c, b);
            set(c, a, b);
        }
    }
    void function_patterns() {
        for (int i = 0; i < n; i++) { set(6, i, i % 2 == 0); set(i, 6, i % 2 == 0); }
        finder(3, 3);
        finder(n - 4, 3);
        finder(3, n - 4);
        const std::vector<int> ap = align_pos();
        const int na = (int) ap.size();
        for (int i = 0; i < na; i++)
            for (int j = 0; j < na; j++)
                if (!((i == 0 && j == 0) || (i == 0 && j == na - 1) || (i == na - 1 && j == 0))) align(ap[i], ap[j]);
        format(0);
        version();
    }
    void codewords(const std::vector<uint8_t> & data) {
        size_t i = 0;
        for (int right = n - 1; right >= 1; right -= 2) {
            if (right == 6) right = 5;
            for (int vert = 0; vert < n; vert++)
                for (int j = 0; j < 2; j++) {
                    const int x = right - j;
                    const bool up = ((right + 1) & 2) == 0;
                    const int y = up ? n - 1 - vert : vert;
                    if (!fixed[y][x] && i < data.size() * 8) {
                        dark[y][x] = ((data[i >> 3] >> (7 - (i & 7))) & 1) != 0;
                        i++;
                    }
                }
        }
    }
    void apply_mask(int m) {
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++) {
                bool inv = false;
                switch (m) {
                    case 0: inv = (x + y) % 2 == 0; break;
                    case 1: inv = y % 2 == 0; break;
                    case 2: inv = x % 3 == 0; break;
                    case 3: inv = (x + y) % 3 == 0; break;
                    case 4: inv = (x / 3 + y / 2) % 2 == 0; break;
                    case 5: inv = x * y % 2 + x * y % 3 == 0; break;
                    case 6: inv = (x * y % 2 + x * y % 3) % 2 == 0; break;
                    case 7: inv = ((x + y) % 2 + x * y % 3) % 2 == 0; break;
                }
                if (!fixed[y][x] && inv) dark[y][x] = !dark[y][x];
            }
    }
    long penalty() const {
        long p = 0;
        auto at = [&](int x, int y, bool rows) { return rows ? dark[y][x] : dark[x][y]; };
        for (int pass = 0; pass < 2; pass++) {
            const bool rows = pass == 0;
            for (int a = 0; a < n; a++) {
                int run = 1;   // runs of 5 or more of one colour
                for (int b = 1; b < n; b++) {
                    if (at(b, a, rows) == at(b - 1, a, rows)) run++;
                    else { if (run >= 5) p += 3 + (run - 5); run = 1; }
                }
                if (run >= 5) p += 3 + (run - 5);
                for (int b = 0; b + 11 <= n; b++) {   // a finder-like 1:1:3:1:1 with 4 light on one side
                    static const bool pat[7] = { 1, 0, 1, 1, 1, 0, 1 };
                    bool mid = true;
                    for (int k = 0; k < 7 && mid; k++) mid = at(b + k + 4, a, rows) == pat[k];
                    bool before = true, after = true;
                    for (int k = 0; k < 4; k++) before = before && !at(b + k, a, rows);
                    if (mid && before) p += 40;
                    mid = true;
                    for (int k = 0; k < 7 && mid; k++) mid = at(b + k, a, rows) == pat[k];
                    for (int k = 7; k < 11; k++) after = after && !at(b + k, a, rows);
                    if (mid && after) p += 40;
                }
            }
        }
        for (int y = 0; y + 1 < n; y++)   // 2x2 blocks of one colour
            for (int x = 0; x + 1 < n; x++) {
                const bool c = dark[y][x];
                if (c == dark[y][x + 1] && c == dark[y + 1][x] && c == dark[y + 1][x + 1]) p += 3;
            }
        long d = 0;
        for (auto & r : dark) for (bool b : r) d += b;
        const long total = (long) n * n;
        const long k = (std::labs(d * 20 - total * 10) + total - 1) / total - 1;   // how far from half dark, in 5% steps
        p += std::max(0L, k) * 10;
        return p;
    }
};

}   // namespace

std::vector<std::vector<bool>> nunchi_qr(const std::string & text) {
    int v = 1;
    const size_t len = text.size();
    for (; v <= 10; v++) {
        const int count_bits = v <= 9 ? 8 : 16;
        if ((int) (4 + count_bits + 8 * len) <= data_codewords(v) * 8) break;
    }
    if (v > 10) return {};
    const int cap = data_codewords(v) * 8;
    std::vector<bool> bits;
    auto put = [&](uint32_t val, int nb) { for (int i = nb - 1; i >= 0; i--) bits.push_back(((val >> i) & 1) != 0); };
    put(0x4, 4);
    put((uint32_t) len, v <= 9 ? 8 : 16);
    for (unsigned char c : text) put(c, 8);
    put(0, std::min(4, cap - (int) bits.size()));
    put(0, (8 - (int) bits.size() % 8) % 8);
    for (uint8_t pad = 0xEC; (int) bits.size() < cap; pad ^= 0xEC ^ 0x11) put(pad, 8);
    std::vector<uint8_t> data(bits.size() / 8, 0);
    for (size_t i = 0; i < bits.size(); i++) data[i >> 3] |= (uint8_t) (bits[i] << (7 - (i & 7)));
    // split into blocks, add error correction, interleave
    const int nb = BLOCKS[v], ecc = ECC_PER_BLOCK[v], raw = raw_modules(v) / 8;
    const int n_short = nb - raw % nb, short_len = raw / nb;
    const std::vector<uint8_t> div = rs_divisor(ecc);
    std::vector<std::vector<uint8_t>> blocks;
    for (int i = 0, k = 0; i < nb; i++) {
        const int dl = short_len - ecc + (i < n_short ? 0 : 1);
        std::vector<uint8_t> b(data.begin() + k, data.begin() + k + dl);
        k += dl;
        const std::vector<uint8_t> e = rs_remainder(b, div);
        if (i < n_short) b.push_back(0);
        b.insert(b.end(), e.begin(), e.end());
        blocks.push_back(b);
    }
    std::vector<uint8_t> all;
    for (size_t i = 0; i < blocks[0].size(); i++)
        for (int j = 0; j < nb; j++)
            if (i != (size_t) (short_len - ecc) || j >= n_short) all.push_back(blocks[j][i]);
    code base(v);
    base.function_patterns();
    base.codewords(all);
    long best = -1;
    code pick = base;
    for (int m = 0; m < 8; m++) {
        code c = base;
        c.apply_mask(m);
        c.format(m);
        const long p = c.penalty();
        if (best < 0 || p < best) { best = p; pick = c; }
    }
    return pick.dark;
}

std::string nunchi_qr_svg(const std::string & text) {
    const auto g = nunchi_qr(text);
    if (g.empty()) return "";
    const int n = (int) g.size(), w = n + 8;
    std::string s = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 " + std::to_string(w) + " " + std::to_string(w) +
                    "\" shape-rendering=\"crispEdges\"><rect width=\"100%\" height=\"100%\" fill=\"#fff\"/><path fill=\"#000\" d=\"";
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
            if (g[y][x]) s += "M" + std::to_string(x + 4) + " " + std::to_string(y + 4) + "h1v1h-1z";
    return s + "\"/></svg>";
}
