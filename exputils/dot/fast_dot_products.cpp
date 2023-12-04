/**
 * @file fast_dot_products.cpp
 * @author hari64boli64 (hari64boli64@gmail.com)
 * @brief This file contains the implementation of the following.
 *        - Generation of all pure stabilizer states in the matrix representation.
 *        - Fast computation of the dot product of stabilizer states.
 * @arg ./a.out n_qubit K is_random rho_file output_file
 *                 n: the number of qubits of the rho_vector
 *                 K: the proportion of the number of stabilizer states to return
 *                    (K=1 -> return all the stabilizer states,
 *                     K=0 -> return the stabilizer states of largest and smallest
 *                            dot products if not is_random)
 *         is_random: 0 -> return the stabilizer states sorted by the dot product
 *                         with the rho_vector
 *                    1 -> return randomly selected stabilizer states
 *          rho_file: the file name of the rho_vector
 *       output_file: the file name of the output
 * @version 0.1
 * @date 2023-11-25
 * @copyright Copyright (c) 2023 Nobuyuki Yoshioka
 */

// In this case, it seems that it would be
// faster not to use this option.
// #pragma GCC optimize("unroll-loops")

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "cnpy.cpp"  // for save Amat as npz file
#include "omp.h"     // for parallel processing

// ================================ my template ======================================

// clang-format off
using namespace std;
template <typename T> using vc = vector<T>;
template <typename T> using vvc = vector<vc<T>>;
template <typename T> using vvvc = vector<vvc<T>>;
template <int n> using Array = array<double, (1<<(n))>;
template <int n> using ArrayInt = array<int, (1<<(n))>;
using ll = long long;
using vi = vc<int>; using vvi = vvc<int>; using vvvi = vvvc<int>;
using vll = vc<ll>; using vvll = vvc<ll>; using vvvll = vvvc<ll>;
using vb = vc<bool>; using vvb = vvc<bool>; using vvvb = vvvc<bool>;
using vd = vc<double>; using vvd = vvc<double>; using vvvd = vvvc<double>;
#define ALL(x) begin(x), end(x)

struct Timer{
    void start(){_start=chrono::system_clock::now();}
    void stop(){_end=chrono::system_clock::now();sum+=chrono::duration_cast<chrono::nanoseconds>(_end-_start).count();}
    inline int ms()const{const chrono::system_clock::time_point now=chrono::system_clock::now();return static_cast<int>(chrono::duration_cast<chrono::microseconds>(now-_start).count()/1000);}
    inline int ns()const{const chrono::system_clock::time_point now=chrono::system_clock::now();return static_cast<int>(chrono::duration_cast<chrono::microseconds>(now-_start).count());}
    string report(){return to_string(sum/1000000)+"[ms]";}
    void reset(){_start=chrono::system_clock::now();sum=0;}
    private: chrono::system_clock::time_point _start,_end;long long sum=0;
} timer;

struct Xor128{  // period 2^128 - 1
    uint32_t x,y,z,w;
    Xor128(uint32_t seed=0):x(123456789),y(362436069),z(521288629),w(88675123+seed){}
    uint32_t operator()(){uint32_t t=x^(x<<11);x=y;y=z;z=w;return w=(w^(w>>19))^(t^(t>>8));}
    uint32_t operator()(uint32_t l,uint32_t r){return((*this)()%(r-l))+l;}
    uint32_t operator()(uint32_t r){return(*this)()%r;}};
struct Rand {  // https://docs.python.org/ja/3/library/random.html
    Rand(int seed):gen(seed){};
    template<class T>
    void shuffle(vector<T>&x){for(int i=x.size(),j;i>1;){j=gen(i);swap(x[j],x[--i]);}}
   private:
    Xor128 gen;
} myrand(0);
// clang-format on

// ================================ utility ======================================

vvi generate_combinations(int n, int k) {
    // https://stackoverflow.com/questions/9430568/generating-combinations-in-c
    vb v(n);
    fill(v.begin(), v.begin() + k, true);
    vvi combinations;
    do {
        vi combination;
        for (int i = 0; i < n; i++)
            if (v[i]) combination.push_back(i);
        combinations.push_back(combination);
    } while (prev_permutation(v.begin(), v.end()));
    ll nCk = 1;
    for (ll i = 1; i <= k; i++) {
        // Check overflow
        assert(nCk <= numeric_limits<ll>::max() / (n - k + i));
        nCk *= n - k + i;
        nCk /= i;
    }
    assert(int(combinations.size()) == nCk);
    return combinations;
}

ll q_factorial(int k) {
    // https://mathworld.wolfram.com/q-Factorial.html
    // q_factorial where q=2
    // [k]_q! = \prod_{i=1}^k (1 + q + q^2 + ... + q^{i-1})
    assert(k >= 0);
    ll ret = 1;
    for (int i = 1; i <= k; ++i) ret *= (1ll << i) - 1;
    vll small_k_results = {1, 1, 3, 21, 315, 9765};
    if (k <= 5) assert(ret == small_k_results[k]);
    return ret;
}

ll q_binomial(int n, int k) {
    // https://mathworld.wolfram.com/q-BinomialCoefficient.html
    // q_binomial where q=2
    // [n k]_q = \frac{[n]_q!}{[k]_q! [n-k]_q!}
    ll ret1 = q_factorial(n) / (q_factorial(k) * q_factorial(n - k));
    ll ret2 = 1;
    for (int i = 0; i < k; i++) {
        ret2 *= 1 - (1ll << (n - i));
        ret2 /= 1 - (1ll << (i + 1));
    }
    assert(ret1 == ret2);
    return ret1;
}

template <int n>
constexpr ll total_stabilizer_group_size() {
    // https://arxiv.org/pdf/1711.07848.pdf
    // The number of n qubit pure stabilizer states
    // |S_n| = 2^n \prod_{k=1}^{n} (2^k + 1)
    ll ret = 1ll << n;
    for (int k = 0; k < n; ++k) {
        ret *= (1ll << (n - k)) + 1;
    }
    return ret;
}

vvi sylvesters(int n) {
    // https://en.wikipedia.org/wiki/Hadamard_matrix
    assert(n >= 0);
    if (n == 0) return {{1}};
    if (n == 1) return {{1, 1}, {1, -1}};
    vvi prev = sylvesters(n - 1);
    vvi ret;
    ret.resize(1 << n, vi(1 << n, 0));
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < (1 << (n - 1)); j++) {
            for (int p = 0; p < 2; p++) {
                for (int q = 0; q < (1 << (n - 1)); q++) {
                    ret[i * (1 << (n - 1)) + j][p * (1 << (n - 1)) + q] =
                        prev[j][q] * (i == 1 && p == 1 ? -1 : 1);
                }
            }
        }
    }
    return ret;
}

template <int n>
void FWHT(const Array<n>& As, Array<n>& inlineAs) {
    // https://en.wikipedia.org/wiki/Hadamard_transform
    // A radix-4 variant is used to speed up the calculation.
    constexpr int size = 1 << n;
    inlineAs = As;
    for (int h = 1; h < size / 2; h <<= 2) {
        for (int i = 0; i < size; i += (h << 2)) {
            for (int j = i; j < i + h; ++j) {
                double x = inlineAs[j];
                double y = inlineAs[j + h];
                double z = inlineAs[j + 2 * h];
                double w = inlineAs[j + 3 * h];
                double xpy = x + y, xmy = x - y;
                double zpw = z + w, zmw = z - w;
                inlineAs[j] = xpy + zpw;
                inlineAs[j + h] = xmy + zmw;
                inlineAs[j + 2 * h] = xpy - zpw;
                inlineAs[j + 3 * h] = xmy - zmw;
            }
        }
    }

    if constexpr (n % 2) {
        constexpr int h = size >> 1;
        for (int j = 0; j < h; j++) {
            double x = inlineAs[j];
            double y = inlineAs[j + h];
            inlineAs[j] = x + y;
            inlineAs[j + h] = x - y;
        }
    }

    // for (int i = 0; i < size; ++i) inlineAs[i] /= size;
}

int compute_pauli_dot_phase(int i1_X, int i1_Z, int i2_X, int i2_Z) {
    // https://arxiv.org/pdf/1711.07848.pdf
    //   | I   X   Z   Y
    //  -----------------
    // I | I   X   Z   Y
    // X | X   I -iY  iZ
    // Z | Z  iY   I -iX
    // Y | Y -iZ  iX   I

    // Example: i1_X = 0b11, i1_Z=0b00, i2_X = 0b10, i2_Z = 0b11
    // These are the representation of Pauli operators.
    // so i1 is XX and i2 is YZ.
    // Since X*Y=iZ and X*Z=-iY (Pauli matrix multiplication),
    // the i1*i2 = (X*Y) ⊗ (X*Z) = (iZ) ⊗ (-iY) = ZY.

    // Now, we only want to know the phase of the product
    // (The main term itself can be easily computed by bitwise xor).
    // Thus, the actual calculation needed here is (+i)*(-i)=(+1).
    // By converting +1->0, +i->1, -1->2, -i->3,
    // we can compute the phase by integer addition mod 4.

    int plus_1_or_3 = (i1_X & i2_Z) ^ (i2_X & i1_Z);
    // int plus_1 = ((i1_X) & (~i1_Z) & (i2_X) & (i2_Z)) |   // X*Y=+iZ
    //              ((~i1_X) & (i1_Z) & (i2_X) & (~i2_Z)) |  // Z*X=+iY
    //              ((i1_X) & (i1_Z) & (~i2_X) & (i2_Z));    // Y*Z=+iX
    int plus_3 = ((i1_X) & (~i1_Z) & (~i2_X) & (i2_Z)) |  // X*Z=-iY
                 ((~i1_X) & (i1_Z) & (i2_X) & (i2_Z)) |   // Z*Y=-iX
                 ((i1_X) & (i1_Z) & (i2_X) & (~i2_Z));    // Y*X=-iZ
    // assert(__builtin_popcount(plus_1_or_3) ==
    //        __builtin_popcount(plus_1) + __builtin_popcount(plus_3));
    return (__builtin_popcount(plus_1_or_3) + 2 * __builtin_popcount(plus_3)) % 4;
}

void print_matrix(const vvi& matrix) {
    // Print the matrix as a grid.
    // Matrix should contain only 0, 1, -1.
    for (int i = 0; i < int(matrix.size()); i++) {
        for (int j = 0; j < int(matrix[i].size()); j++) {
            if (matrix[i][j] == 0)
                cout << ".";
            else if (matrix[i][j] == 1)
                cout << "+";
            else if (matrix[i][j] == -1)
                cout << "-";
            else
                assert(false);
        }
        cout << endl;
    }
}

// ================================ Stabilizer ======================================

template <int n>
struct Stabilizer {
    // This struct is for generating the all pure stabilizer states.
    // How we generate the states is the following.
    // 0. Every pure stabilizer state can be represented with
    //    F_2^{n times 2n} matrix, which is called "stabilizer representation matrix".
    //    The matrix is composed of two parts, X side and Z side.
    //    In addition, it can be shown that the matrix is in the following form.
    //
    //        (X side)    (Z side)
    //     (   I  X1   |   Z1    0   )  |   k
    //     (   0  0    |   X1^T  I   )  | n-k
    //    -----------------------------  [rows]
    //        k  n-k       k  n-k  [cols]
    //
    //    where X1 is the subset of RREF matrix of size k times n, and Z1 is symmetric.
    //    The detailed explanation is in our paper.
    //    Be aware that the identity matrix is not always the first k columns.
    //    The position of the identity matrix is determined by "col_idxs_list".
    //
    // 1. Generate all combinations of the representation with X1=Z1=0.
    //    This is called "default matrix". This is indexed by "d_idx".
    //
    // 2. For each default matrix, generate all combinations of X1 and Z1.
    //    These pattern are indexed by "x_idx", and "z_idx".
    //    There are 2^(X1s.size()) patterns of X1, and 2^(Z1s.size()) patterns of Z1.
    //
    // 3. For each matrix representation of stabilizer states,
    //    generate the dot data of the stabilizer state.
    //    The dot data means the necessary information to compute the dot product.

    const int maskForPOP;      // mask for phase_of_product (0b11...11 (n ones))
    vi ks;                     // k per d_idx
    vvi col_idxs_lists;        // col_idxs_list per d_idx
    vll prefix_sum_per_d_idx;  // prefix sum of the number of stabilizer states
    vvi default_matrixes;      // default matrix per d_idx
    vvll X1s;                  // X1 per d_idx
    vvll Z1s;                  // Z1 per d_idx

    Stabilizer() : maskForPOP((1 << n) - 1) {
        assert(1 <= n && n <= 8);
        test();

        ll cnt_total = 0;

        prefix_sum_per_d_idx.push_back(0);
        for (int k = 0; k <= n; k++) {
            ll cnt = 0;
            const vvi combinations = generate_combinations(n, k);
            for (const vi& col_idxs_list : combinations) {
                ks.push_back(k);
                col_idxs_lists.push_back(col_idxs_list);

                int X1size = 0;
                int Z1size = 0;
                {
                    // the following code's explanation is in
                    // "make_default_matrix_and_XZ"
                    set<int> col_idxs_set(ALL(col_idxs_list));
                    vi complement_of_col_idxs_list;
                    for (int col = 0; col < n; ++col)
                        if (col_idxs_set.find(col) == col_idxs_set.end())
                            complement_of_col_idxs_list.push_back(col);
                    assert(int(complement_of_col_idxs_list.size()) == n - k);
                    for (int row = 0; row < k; row++)
                        for (int i = 0; i < int(complement_of_col_idxs_list.size());
                             i++)
                            if (complement_of_col_idxs_list[i] > col_idxs_list[row])
                                X1size++;
                    Z1size = k * (k + 1) / 2;
                }

                cnt += 1ll << X1size;
                cnt_total += 1ll << (X1size + Z1size + n);
                prefix_sum_per_d_idx.push_back(prefix_sum_per_d_idx.back() +
                                               (1ll << (X1size + Z1size + n)));
            }
            assert(cnt == q_binomial(n, k));
        }
        assert(cnt_total == total_stabilizer_group_size<n>());
        assert(prefix_sum_per_d_idx.back() == total_stabilizer_group_size<n>());

        default_matrixes.resize(col_idxs_lists.size());
        X1s.resize(col_idxs_lists.size());
        Z1s.resize(col_idxs_lists.size());
    }

    void make_default_matrix_and_XZ(int d_idx) {
        assert(0 <= d_idx && d_idx < (ll)default_matrixes.size());
        assert(default_matrixes[d_idx].empty());

        // https://mathlandscape.com/rref-matrix

        // which columns are (0, ..., 1, ..., 0)^T of RREF in X side
        vi col_idxs_list = col_idxs_lists[d_idx];  // the size is k
        set<int> col_idxs_set(ALL(col_idxs_list));

        // which columns are not (0, ..., 1, ..., 0)^T of RREF in X side
        vi complement_of_col_idxs_list;
        for (int col = 0; col < n; ++col)
            if (col_idxs_set.find(col) == col_idxs_set.end())
                complement_of_col_idxs_list.push_back(col);

        // k times n matrix (rref matrix only with X side identity matrix)
        int k = ks[d_idx];
        vi rref_matrix(k);
        for (int row = 0; row < k; ++row) rref_matrix[row] |= (1 << col_idxs_list[row]);

        // n times 2n matrix (stabilizer representation matrix)
        vi default_matrix = rref_matrix;
        default_matrix.resize(n);

        // Z side identity
        for (int row2 = k; row2 < n; ++row2) {
            default_matrix[row2] += 1 << (n + complement_of_col_idxs_list[row2 - k]);
        }

        // ((row1,col1),(row2,col2)) of X1 in our paper
        vll X1;
        for (int row = 0; row < k; row++) {
            for (int i = 0; i < int(complement_of_col_idxs_list.size()); i++) {
                int col = complement_of_col_idxs_list[i];
                if (col <= col_idxs_list[row]) continue;
                ll row1 = row, col1 = col;
                ll row2 = k + i, col2 = col_idxs_list[row] + n;
                X1.push_back(compress_row_col(row1, col1, row2, col2));
            }
        }
        assert(int(X1.size()) <= 64);  // check for overflow

        // ((row1,col1),(row2,col2)) of Z1 in our paper
        vll Z1;
        for (int row = 0; row < k; row++) {
            for (int i = row; i < int(col_idxs_list.size()); i++) {
                int col = col_idxs_list[i];
                ll row1 = row, col1 = col + n;
                ll row2 = i, col2 = col_idxs_list[row] + n;
                Z1.push_back(compress_row_col(row1, col1, row2, col2));
            }
        }
        assert(int(Z1.size()) == k * (k + 1) / 2);
        assert(int(Z1.size()) <= 64);  // check for overflow

        X1s[d_idx] = X1;
        Z1s[d_idx] = Z1;
        default_matrixes[d_idx] = default_matrix;
    }

    void delete_default_matrix_and_XZ(int d_idx) {
        // This function is for saving memory.
        default_matrixes[d_idx].clear();
        X1s[d_idx].clear();
        Z1s[d_idx].clear();
    }

    array<int, n> make_matrix_representation(ll d_idx, ll x_idx, ll z_idx) const {
        // Every matrix representation is indexed by (d_idx, x_idx, z_idx).
        // This function returns the matrix indicated by the arguments.
        assert(0 <= d_idx && d_idx < (ll)default_matrixes.size());
        assert(0 <= x_idx && x_idx < (1ll << X1s[d_idx].size()));
        assert(0 <= z_idx && z_idx < (1ll << Z1s[d_idx].size()));
        array<int, n> ret;
        for (int i = 0; i < n; i++) {
            assert(default_matrixes[d_idx][i] < numeric_limits<int>::max());
            ret[i] = default_matrixes[d_idx][i];
        }
        const vll& X1 = X1s[d_idx];
        for (int i = 0; i < int(X1.size()); i++) {
            if (x_idx & (1ll << i)) {
                auto [row1, col1, row2, col2] = decompress_row_col(X1[i]);
                assert(col1 < 32 && col2 < 32);
                ret[row1] |= 1 << col1;
                ret[row2] |= 1 << col2;
            }
        }
        const vll& Z1 = Z1s[d_idx];
        for (int i = 0; i < int(Z1.size()); i++) {
            if (z_idx & (1ll << i)) {
                auto [row1, col1, row2, col2] = decompress_row_col(Z1[i]);
                assert(col1 < 32 && col2 < 32);
                ret[row1] |= 1 << col1;
                ret[row2] |= 1 << col2;
            }
        }
        return ret;
    }

    pair<ArrayInt<n>, ArrayInt<n>> make_arranged_rho_vec(const array<int, n>& matrix,
                                                         Array<2 * n>& rho_vec,
                                                         Array<n>& arranged_rho_vec) {
        // Example: n=3, matrix={0b011'110, 0b100'010, 0b000'011}
        //
        // 0b011'110 = [Z:011, X:110] = XYZ
        // 0b100'010 = [Z:100, X:010] = ZXI
        // 0b000'011 = [Z:000, X:011] = IXX
        //
        // Generate all elements of the stabilizer group by multiplying the generators.
        //
        //   idx Pauli lexical order  relationship  phase
        //   --- ----- -------------  -------------- -----
        //   000  +III       0                        1
        // * 001  +XYZ      27                        1
        // * 010  +ZXI      52                        1
        //   011  -YZZ      47       = 27 ^ 52       -1
        // * 100  +IXX       5                        1
        //   101  +XZY      30       = 27      ^ 5    1
        //   110  +ZIX      49       =      52 ^ 5    1
        //   111  +YYY      42       = 27 ^ 52 ^ 5    1
        //
        // Gray code
        // https://en.wikipedia.org/wiki/Gray_code
        //   i    gray  bit_pos
        // --------------------
        //  000   000     0      if i>=1,
        //  001   001     0      gray[i]
        //  010   011     1     =gray[i-1]^(1<<bit_pos[i])
        //  011   010     0
        //  100   110     2
        //  101   111     0
        //  110   101     1
        //  111   100     0

        // It is ok to use int, since n <= 8.
        ArrayInt<n> rho_vec_idxs, rho_vec_phases;
        rho_vec_idxs[0] = 0;
        rho_vec_phases[0] = 0;
        arranged_rho_vec[0] = rho_vec[0];
        int gray, bit_pos, row = 0, phase = 0;
        for (int i = 1; i < (1 << n); i++) {
            gray = i ^ (i >> 1);
            bit_pos = __builtin_ctz(i);
            phase += phase_of_product(row, matrix[bit_pos]);
            phase &= 0b11;
            assert(phase == 0 || phase == 2);
            row ^= matrix[bit_pos];
            arranged_rho_vec[gray] = rho_vec[row] * ((phase == 0) ? 1 : -1);
            rho_vec_idxs[gray] = row;
            rho_vec_phases[gray] = phase;
        }

        return {rho_vec_idxs, rho_vec_phases};
    }

    void update_arranged_rho_vec(int d_idx, int bit_pos, array<int, n>& mat_rep,
                                 const Array<2 * n>& rho_vec, ArrayInt<n>& rho_vec_idxs,
                                 ArrayInt<n>& rho_vec_phases,
                                 Array<n>& arranged_rho_vec) {
        // This function is for updating the arranged_rho_vec.
        // The gray code is also important here. The modified position is limited
        // only to the bit_pos thanks to the gray code.

        // Check the arguments
        assert(0 <= d_idx && d_idx < (ll)default_matrixes.size());
        assert(0 <= bit_pos && bit_pos < int(X1s[d_idx].size() + Z1s[d_idx].size()));
        for (int r = 0; r < n; r++) {
            assert(rho_vec_idxs[1 << r] == mat_rep[r]);
            assert(rho_vec_phases[1 << r] == 0);
        }

        // This is the updated (row,col) of the matrix representation.
        // One bit_pos is corresponding to two (row,col).
        int _bp = (bit_pos < int(Z1s[d_idx].size())) ? bit_pos
                                                     : bit_pos - int(Z1s[d_idx].size());
        const auto [row1, col1, row2, col2] = decompress_row_col(
            bit_pos < int(Z1s[d_idx].size()) ? Z1s[d_idx][_bp] : X1s[d_idx][_bp]);
        const int col1_shift = 1 << col1;
        const int col2_shift = 1 << col2;
        const int col12_shift = col1_shift | col2_shift;
        const int phase_diff1 = (4 - phase_of_product_1bit(mat_rep[row1], col1)) & 0b11;
        const int phase_diff2 = (4 - phase_of_product_1bit(mat_rep[row2], col2)) & 0b11;
        mat_rep[row1] ^= col1_shift;
        if (row1 != row2 || col1 != col2) mat_rep[row2] ^= col2_shift;

        // Update rho_vec_idxs and rho_vec_phases by (1<<col)
        auto updateData1 = [&](int i) -> void {
            rho_vec_phases[i] = (rho_vec_phases[i] + phase_diff1 +
                                 phase_of_product_1bit(rho_vec_idxs[i], col1)) &
                                0b11;
            rho_vec_idxs[i] ^= col1_shift;
        };
        auto updateData2 = [&](int i) -> void {
            rho_vec_phases[i] = (rho_vec_phases[i] + phase_diff2 +
                                 phase_of_product_1bit(rho_vec_idxs[i], col2)) &
                                0b11;
            rho_vec_idxs[i] ^= col2_shift;
        };
        auto updateData12 = [&](int i) -> void {
            // If both row1 and row2 are in i, the extra flip is needed.
            // For example, mat_rep[0]=YZ, mat_rep[1]=ZY,
            // col1 = 3 -> IZ, col2 = 2 -> ZI, and i=0b11.
            // Then, the original one is YZ*ZY=XX,
            //   and the correct one is (YZ*IZ)*(ZY*ZI)=YI*IY=YY.
            // However, if we do not flip the phase, the result is
            // XX*IZ*ZI=(-i XY)*ZI=-YY.
            rho_vec_phases[i] = (rho_vec_phases[i] + phase_diff1 + phase_diff2 +
                                 phase_of_product_1bit(rho_vec_idxs[i], col1) +
                                 phase_of_product_1bit(rho_vec_idxs[i], col2) + 2) &
                                0b11;
            rho_vec_idxs[i] ^= col12_shift;
        };
        // Assign to arranged_rho_vec with the updated infomation
        auto assignData = [&](int i) -> void {
            assert(rho_vec_phases[i] == 0 || rho_vec_phases[i] == 2);
            arranged_rho_vec[i] = (1 - rho_vec_phases[i]) * rho_vec[rho_vec_idxs[i]];
        };

        if (row1 == row2 && col1 == col2) {
            int row_shift = 1 << row1;
            // Loop over all the i where i & (1 << row1) != 0
            for (int i = row_shift; i < (1 << n); i++, i |= row_shift) {
                updateData1(i);
                assignData(i);
            }
        } else {
            int row1_shift = 1 << row1;
            int row2_shift = 1 << row2;
            int row12_shift = row1_shift | row2_shift;
            // Loop over all the i where i & (1 << row1) != 0 and i & (1 << row2) != 0
            for (int i = row12_shift; i < (1 << n); i++, i |= row12_shift) {
                updateData12(i);
                assignData(i);
                updateData1(i ^ row2_shift);
                assignData(i ^ row2_shift);
                updateData2(i ^ row1_shift);
                assignData(i ^ row1_shift);
            }
        }
    }

    void subroutine_of_restore_Amat(ll idx, array<int, n>& mat_rep,
                                    vi& rows_in_lexical_order, vb& is_minus,
                                    int& d_idx_now, ll& last_ub, ll& ub,
                                    const vvi& walsh) {
        bool is_d_idx_updated = false;
        while (idx >= ub) {
            delete_default_matrix_and_XZ(d_idx_now);
            d_idx_now++;
            is_d_idx_updated = true;
            last_ub = ub;
            ub = prefix_sum_per_d_idx[d_idx_now + 1];
        }
        if (is_d_idx_updated) make_default_matrix_and_XZ(d_idx_now);
        ll idx2 = idx - last_ub;
        ll d_idx = d_idx_now;
        ll wal_idx = idx2 % (1ll << n);
        idx2 /= (1ll << n);
        idx2 = idx2 ^ (idx2 >> 1);  // due to gray code
        ll x_idx = idx2 / (1ll << Z1s[d_idx].size());
        ll z_idx = idx2 % (1ll << Z1s[d_idx].size());
        mat_rep = make_matrix_representation(d_idx, x_idx, z_idx);
        rows_in_lexical_order[0] = 0;
        is_minus[0] = false;
        int row = 0;
        int phase = 0;
        for (int i = 1; i < (1 << n); i++) {
            int gray = i ^ (i >> 1);
            int bit_pos = __builtin_ctz(i);
            phase += phase_of_product(row, mat_rep[bit_pos]);
            phase &= 0b11;
            row ^= mat_rep[bit_pos];
            rows_in_lexical_order[gray] = to_lexical_order(row);
            assert(phase == 0 || phase == 2);
            is_minus[gray] = (phase == 2);
        }
        for (int j = 0; j < (1 << n); j++)
            is_minus[j] = is_minus[j] ^ (walsh[j][wal_idx] == -1);
    }

    pair<vvi, vvb> restore_Amat(vll& idxs) {
        // For given idxs, restore the Amat.
        // This is corresponding to the function "Amat_by_dot".
        sort(ALL(idxs));
        array<int, n> mat_rep;
        vvi walsh = sylvesters(n);
        vi rows_in_lexical_order(1 << n);
        vb is_minus(1 << n);
        int d_idx_now = 0;
        make_default_matrix_and_XZ(d_idx_now);
        ll last_ub = 0;
        ll ub = prefix_sum_per_d_idx[d_idx_now + 1];
        vvi Amat_rows(idxs.size(), vi(1 << n));
        vvb Amat_is_minus(idxs.size(), vb(1 << n));
        std::cerr << "[start] restore the Amat..." << flush;
        for (int i = 0; i < int(idxs.size()); i++) {
            subroutine_of_restore_Amat(idxs[i], mat_rep, rows_in_lexical_order,
                                       is_minus, d_idx_now, last_ub, ub, walsh);
            Amat_rows[i] = rows_in_lexical_order;
            Amat_is_minus[i] = is_minus;
        }
        std::cerr << " done" << endl;
        delete_default_matrix_and_XZ(d_idx_now);
        assert(all_of(ALL(default_matrixes), [](const vi& v) { return v.empty(); }));
        return {Amat_rows, Amat_is_minus};
    }

    int to_lexical_order(int bit) const {
        // Example: n=4, bit = 0b11001010
        // The bit is corresponding to the matrix representation of a Pauli operator,
        // which means [Z side: (1100), X side: (1010)].
        // Thus, the indicated Pauli operator is YZXI,
        // and this is in reverse order, the actual one is IXZY.
        // Therefore, the lexical idx is 0(I)*64 + 1(X)*16 + 3(Z)*4 + 2(Y)*1 = 30.
        int ret = 0;
        for (int _ = 0; _ < n; _++) {
            int isZ = (bit >> n) & 1;
            ret <<= 2;
            ret += ((bit & 1) ^ isZ) + (isZ << 1);
            bit >>= 1;
        }
        return ret;
    }

    void vis_Amat() {
        // Only for debug. Print the Amat.
        vvi Amat = make_Amat();
        std::cerr << "n: " << n << " Amat size: " << Amat.size() << " x "
                  << Amat[0].size() << endl;
        print_matrix(Amat);
    }

    template <bool is_dual>
    void check_values(const vd& rho_vec_lexical_order, double max_value,
                      double min_value) {
        // Only for debug. Compute the maximum value and minimum value
        // of the dot products by naive O(4^n |S_n|) time complexity solution.
        std::cerr << "----- check with naive solution -----" << endl;
        if (n >= 6) {
            std::cerr << "\033[1;31mWARNING: it might take very long time. "
                         "If not intended, please add -DNDEBUG to the compile options."
                      << "\033[0m" << endl;
        }
        ll t_s_g_s = total_stabilizer_group_size<n>();
        vvi Amat = make_Amat();
        double max_value_slow = -numeric_limits<double>::max();
        double min_value_slow = +numeric_limits<double>::max();
        for (int j = 0; j < t_s_g_s; j++) {
            double dot = 0;
            for (int i = 0; i < (1 << (2 * n)); i++) {
                dot += Amat[i][j] * rho_vec_lexical_order[i];
            }
            if constexpr (is_dual) dot = abs(dot);
            max_value_slow = max(max_value_slow, dot);
            min_value_slow = min(min_value_slow, dot);
        }
        std::cerr << "max value (naive solution): " << max_value_slow << endl;
        std::cerr << "max value (  our solution): " << max_value << endl;
        std::cerr << "min value (naive solution): " << min_value_slow << endl;
        std::cerr << "min value (  our solution): " << min_value << endl;
        if (abs(max_value - max_value_slow) > 1e-5 ||
            abs(min_value - min_value_slow) > 1e-5) {
            throw runtime_error(
                "The error of fast_dot_products.cpp itself. We are sorry.\n"
                "Please report this error to us.");
        }
    }

   private:
    int phase_of_product(int i1, int i2) const {
        // Compute the phase of product of two Pauli operators.
        int i1_X = i1 & maskForPOP;
        int i1_Z = (i1 >> n) & maskForPOP;
        int i2_X = i2 & maskForPOP;
        int i2_Z = (i2 >> n) & maskForPOP;
        return compute_pauli_dot_phase(i1_X, i1_Z, i2_X, i2_Z);
    }

    int phase_of_product_1bit(int i1, int col) const {
        // The result is equivalent to phase_of_product(i1, 1 << col).
        if (col < n) {
            // X (Z*X=+iY, Y*X=-iZ)
            bool isZ = i1 & (1 << (n + col));
            if (!isZ) return 0;
            bool isX = i1 & (1 << col);
            return isX ? 3 : 1;
        } else {
            // Z (X*Z=-iY, Y*Z=+iX)
            bool isX = i1 & (1 << (col - n));
            if (!isX) return 0;
            bool isZ = i1 & (1 << col);
            return isZ ? 1 : 3;
        }
    }

    // Only for speed up.
    ll compress_row_col(ll row1, ll col1, ll row2, ll col2) const {
        assert(row1 < (1 << 16) && col1 < (1 << 16) && row2 < (1 << 16) &&
               col2 < (1 << 16));
        return (row1 << 48) + (col1 << 32) + (row2 << 16) + col2;
    }
    tuple<int, int, int, int> decompress_row_col(ll compressed) const {
        int row1 = (compressed >> 48);
        int col1 = (compressed >> 32) & ((1 << 16) - 1);
        int row2 = (compressed >> 16) & ((1 << 16) - 1);
        int col2 = compressed & ((1 << 16) - 1);
        return make_tuple(row1, col1, row2, col2);
    }

    // Only for debug.
    vvi make_Amat() {
        vvi Amat(1 << (2 * n), vi(total_stabilizer_group_size<n>()));
        vll all_cols(total_stabilizer_group_size<n>());
        iota(ALL(all_cols), 0);
        auto [rows, is_minus] = restore_Amat(all_cols);
        for (int i = 0; i < int(rows.size()); i++) {
            for (int j = 0; j < int(rows[i].size()); j++) {
                Amat[rows[i][j]][i] = is_minus[i][j] ? -1 : 1;
            }
        }
        return Amat;
    }

    // Only for debug.
    void test() {
        if constexpr (n == 2) {
            if (phase_of_product(0b00'11, 0b11'10) != 0)
                throw runtime_error("test failed (phase_of_product)");
            for (int col = 0; col < 2 * n; col++)
                if (phase_of_product(0b00'11, 1 << col) !=
                    phase_of_product_1bit(0b00'11, col))
                    throw runtime_error("test failed (phase_of_product_1bit)");
        }
        if constexpr (n == 4) {
            if (to_lexical_order(0b1100'1010) != 30)
                throw runtime_error("test failed (to_lexical_order)");
        }
        if constexpr (n == 3) {
            vi rows(1 << n);
            vb is_minus(1 << n);
            Array<2 * n> rho_vec;
            Array<n> arranged_rho_vec{};
            iota(ALL(rho_vec), 0);
            make_arranged_rho_vec(array<int, n>{0b011'110, 0b100'010, 0b000'011},
                                  rho_vec, arranged_rho_vec);
            Array<n> ans = {0b000'000, 0b011'110, 0b100'010, -(0b111'100),  //
                            0b000'011, 0b011'101, 0b100'001, 0b111'111};
            if (arranged_rho_vec != ans)
                throw runtime_error("test failed (arranged_rho_vec)");
        }
    }
};

// ================================ Solver ======================================

template <int n>
vll divide_to_chunk() {
    // In order to save memory and do parallelization,
    // we divide the calculation to chunks.
    assert(n <= 8);
    ll CHUNK_CNT = vll{1, 1, 1, 8, 8, 8, 8, 400, 200000}[n];
    vll idx_range_per_chunk = {0};
    ll t_s_g_s = total_stabilizer_group_size<n>();
    for (ll i = 1; i <= CHUNK_CNT; i++)
        idx_range_per_chunk.push_back(((ll)(double(t_s_g_s) * i / CHUNK_CNT) >> n)
                                      << n);
    assert(idx_range_per_chunk.front() == 0);
    assert(idx_range_per_chunk.back() == t_s_g_s);
    assert(is_sorted(ALL(idx_range_per_chunk)));
    return idx_range_per_chunk;
}

template <int n, bool is_dual>
void print_log(int chunk_id, double progress, double now, double topK_value,
               size_t values_top_sz, [[maybe_unused]] double botK_value,
               [[maybe_unused]] size_t values_bot_sz) {
    if constexpr (n <= 5) return;
    if constexpr (n == 7)
        if (chunk_id % 40 != 39) return;
    if constexpr (n == 8)
        if (chunk_id % 1000 != 999) return;
    int total = ((now / 1000) / progress) * (1 - progress);
    int hour = total / 3600;
    int min = (total - hour * 3600) / 60;
    int sec = total - hour * 3600 - min * 60;
    string m = (min < 10 ? "0" : "") + to_string(min);
    string s = (sec < 10 ? "0" : "") + to_string(sec);
    std::cerr << (chunk_id % 5000 == 4999 ? "" : "\e[1F\e[0K")
              << "\033[1;34m chunk_id\033[0m = " << chunk_id
              << " |\033[1;36m expected remain time\033[0m = "  //
              << hour << "[h] " << m << "[m] " << s << "[s]"
              << " |\033[1;32m topK\033[0m = " << topK_value
              << " |\033[1;32m vt.sz()\033[0m = " << values_top_sz;
    if constexpr (!is_dual) {
        std::cerr << " |\033[1;31m botK\033[0m = " << botK_value
                  << " |\033[1;31m vb.sz()\033[0m = " << values_bot_sz;
    }
    std::cerr << endl;
}

void save_Amat_as_npz(int n, const pair<vvi, vvb>& Amat,
                      const string& output_file_name) {
    // npz is a file format of numpy.
    // This function saves the calculated Amat as npz.
    auto [Amat_rows, Amat_is_minus] = Amat;
    assert(Amat_rows.size() == Amat_is_minus.size());
    vi Amat_rows_flat(Amat_rows.size() << n);
    vi Amat_data_flat(Amat_is_minus.size() << n);
    for (int i = 0; i < int(Amat_rows.size()); i++) {
        for (int j = 0; j < (1 << n); j++) {
            int idx = i * (1 << n) + j;
            Amat_rows_flat[idx] = Amat_rows[i][j];
            Amat_data_flat[idx] = Amat_is_minus[i][j] ? -1 : 1;
        }
    }
    // If the data is empty and loaded in Python,
    // zipfile.BadZipFile error will occur.
    if (Amat_rows_flat.empty()) {
        assert(Amat_rows.empty());
        Amat_rows_flat.push_back(-1);
        Amat_data_flat.push_back(0);
    }
    cnpy::npz_save(output_file_name + ".npz", "Amat_rows", Amat_rows_flat.data(),
                   {Amat_rows_flat.size()}, "w");
    cnpy::npz_save(output_file_name + ".npz", "Amat_data", Amat_data_flat.data(),
                   {Amat_data_flat.size()}, "a");
}

template <int n, bool is_dual>
void update_topK_botK(const double K, double& topK_value,
                      [[maybe_unused]] double& botK_value,
                      vc<pair<double, ll>>& values_top,
                      [[maybe_unused]] vc<pair<double, ll>>& values_bot,
                      const vc<pair<double, ll>>& local_values_top,
                      [[maybe_unused]] const vc<pair<double, ll>>& local_values_bot) {
    // Subroutine of "Amat_by_dot".
    // Update the topK and botK values.
    ll group_size = total_stabilizer_group_size<n>();
    size_t value_size = max(1ll, ll(round(double(group_size) * K / (is_dual ? 1 : 2))));

    vc<pair<double, ll>> temp_res_top;
    assert(is_sorted(ALL(values_top), greater<pair<double, ll>>()));
    merge(ALL(values_top), ALL(local_values_top), std::back_inserter(temp_res_top),
          greater<pair<double, ll>>());
    swap(values_top, temp_res_top);
    if (values_top.size() > value_size) values_top.resize(value_size);
    if (!values_top.empty()) {
        topK_value = values_top.back().first;
        assert(topK_value == (*min_element(ALL(values_top))).first);
    }

    if constexpr (!is_dual) {
        vc<pair<double, ll>> temp_res_bot;
        assert(is_sorted(ALL(values_bot), less<pair<double, ll>>()));
        merge(ALL(values_bot), ALL(local_values_bot), std::back_inserter(temp_res_bot),
              less<pair<double, ll>>());
        swap(values_bot, temp_res_bot);
        if (values_bot.size() > value_size) values_bot.resize(value_size);
        if (!values_bot.empty()) {
            botK_value = values_bot.back().first;
            assert(botK_value == (*max_element(ALL(values_bot))).first);
        }
    }
}

template <int n>
tuple<ll, ll, ll, ll> get_xz_idx(ll idx_start, int d_idx, int d_idx_start,
                                 const Stabilizer<n>& stab) {
    // Convert idx to (x_idx, z_idx).
    if (d_idx == d_idx_start) {
        ll idx_diff = idx_start - stab.prefix_sum_per_d_idx[d_idx];
        assert(idx_diff % (1 << n) == 0);
        idx_diff >>= n;
        ll gray = idx_diff ^ (idx_diff >> 1);
        assert(0 <= gray &&
               gray < (1ll << (stab.X1s[d_idx].size() + stab.Z1s[d_idx].size())));
        ll x_idx = gray / (1ll << stab.Z1s[d_idx].size());
        ll z_idx = gray % (1ll << stab.Z1s[d_idx].size());
        return {x_idx, z_idx, idx_diff, gray};
    } else {
        return {0, 0, 0, 0};
    }
}

template <int n>
vll generate_random_idxs(ll idxs_size) {
    // Generate random indexes.
    // The random means that the indexes are uniformly sampled
    // from the all the stabilizer group.
    ll t_s_g_s = total_stabilizer_group_size<n>();
    idxs_size = min(idxs_size, t_s_g_s);
    ll step = t_s_g_s / idxs_size;
    vll result(idxs_size);
    for (ll i = 0; i < idxs_size; i++) result[i] = i * step;
    return result;
}

template <int n>
pair<double, double> get_top_and_bot_of_random(double K,
                                               const vd& rho_vec_lexical_order) {
    // Get the top and bottom values of the dot product of random indexes.
    // This function is to find appropriate thresholds for the "Amat_by_dot" function.
    assert(K < 0.01);
    vll idxs = generate_random_idxs<n>(
        K == 0.0 ? 1000000ll : min(1000000ll, max(3ll, ll((1 / K) * 0.1))));
    array<int, n> mat_rep;
    vvi walsh = sylvesters(n);
    vi rows_in_lexical_order(1 << n);
    vb is_minus(1 << n);
    int d_idx_now = 0;
    Stabilizer<n> stab;
    stab.make_default_matrix_and_XZ(d_idx_now);
    ll last_ub = 0;
    ll ub = stab.prefix_sum_per_d_idx[d_idx_now + 1];
    vd dots(idxs.size());
    for (int i = 0; i < int(idxs.size()); i++) {
        stab.subroutine_of_restore_Amat(idxs[i], mat_rep, rows_in_lexical_order,
                                        is_minus, d_idx_now, last_ub, ub, walsh);
        double dot = 0;
        for (int j = 0; j < (1 << n); j++) {
            dot += rho_vec_lexical_order[rows_in_lexical_order[j]] *
                   (is_minus[j] ? -1 : 1);
        }
        dots[i] = dot;
    }
    stab.delete_default_matrix_and_XZ(d_idx_now);
    assert(all_of(ALL(stab.default_matrixes), [](const vi& v) { return v.empty(); }));
    sort(ALL(dots));
    return {dots[int(dots.size()) - 2], dots[1]};
}

template <int n>
pair<vvi, vvb> Amat_by_random(const double K) {
    ll t_s_g_s = total_stabilizer_group_size<n>();
    ll values_size = max(1ll, ll(round(t_s_g_s * K)));
    ll step = t_s_g_s / values_size;
    vll result(values_size);
    for (ll i = 0; i < values_size; i++) result[i] = i * step;
    Stabilizer<n> stab;
    return stab.restore_Amat(result);
}

template <int n, bool is_dual>
pair<vvi, vvb> Amat_by_dot(const double K, const vd& rho_vec_lexical_order) {
    // This is the main function of this program.
    // Find the indexes of the topK and botK elements of the dot product.
    // (dot product = inner product ≒ overlap)
    timer.start();

    // The following variables are for store the topK and botK columns.
    ll t_s_g_s = total_stabilizer_group_size<n>();
    vc<pair<double, ll>> values_top;
    vc<pair<double, ll>> values_bot;

    ll values_size = max(1ll, ll(round(t_s_g_s * K / (is_dual ? 1 : 2))));
    if constexpr (is_dual) {
        values_top.reserve(values_size);
        values_top.reserve(0);
    } else {
        values_top.reserve(values_size);
        values_bot.reserve(values_size);
    }

    // The following variables are for parallelization.
    Stabilizer<n> stab_original;
    vll idx_range_per_chunk = divide_to_chunk<n>();

    // Originally, the rho_vec is ordered in the lexical order.
    // We convert this to the order of the bit.
    assert(int(rho_vec_lexical_order.size()) == 1 << (2 * n));
    assert(abs(rho_vec_lexical_order[0] - 1.0) < 1e-8);  // I^\otimes{n} element
    Array<2 * n> rho_vec;
    for (int bit = 0; bit < (1 << (2 * n)); bit++)
        rho_vec[bit] = rho_vec_lexical_order[stab_original.to_lexical_order(bit)];

    // The threshold of the dot product.
    double topK_value, botK_value;
    if (K >= 0.01) {
        if constexpr (is_dual) {
            // If the rho_vec is the dual variable,
            // then what we want to find is the violated constraints,
            // which means the absolute value of dot products are larger than 1.
            // In order to utilize the column generation, we set the threshold as 0.8.
            topK_value = +0.8;
            botK_value = -9999;
        } else {
            // If the rho_vec is the Pauli vector of appropriate density matrix,
            // it is guaranteed that every dot products are in [0, 2^n],
            //                the 1/2^n of dot products are in [0,1],
            //            and the 1/2^n of dot products aer in [1,2^n].
            // Therefore, we can assume that topK >= 1.0 and botK <= 1.0.
            topK_value = 1.0;
            botK_value = 1.0;
        }
    } else {
        // For the sake of speed, we first find the topK and botK values
        // of the random indexes. We cannot assume that those thresholds
        // are the same as the actual ones, but practically it is ok.
        tie(topK_value, botK_value) =
            get_top_and_bot_of_random<n>(K, rho_vec_lexical_order);
        if constexpr (is_dual) {
            // If the rho_vec is the dual variable,
            // we only use the topK value.
            botK_value = -9999;
            std::cerr << "init topK_value: " << topK_value << "\n" << endl;
        } else {
            cerr << "init topK_value: " << topK_value << ", "
                 << "init botK_value: " << botK_value << "\n"
                 << endl;
        }
    }

    const ll CHUNK_CNT = idx_range_per_chunk.size() - 1;
    vll randomized_chunk_ids(CHUNK_CNT);
    iota(ALL(randomized_chunk_ids), 0);
    myrand.shuffle(randomized_chunk_ids);
    ll total_violated_count = 0;

    timer.start();
#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
    for (ll chunk_id = 0; chunk_id < CHUNK_CNT; chunk_id++) {
        ll randomized_chunk_id = randomized_chunk_ids[chunk_id];
        ll idx_start = idx_range_per_chunk[randomized_chunk_id];
        ll idx_end = idx_range_per_chunk[randomized_chunk_id + 1];
        vc<pair<double, ll>> local_values_top;
        vc<pair<double, ll>> local_values_bot;
        if constexpr (is_dual) {
            local_values_top.reserve(values_size / CHUNK_CNT);
            local_values_bot.reserve(0);
        } else {
            local_values_top.reserve(values_size / CHUNK_CNT);
            local_values_bot.reserve(values_size / CHUNK_CNT);
        }
        double threshold_top = topK_value, threshold_bot = botK_value;

        Stabilizer<n> stab = stab_original;
        int d_idx_start =
            -1 + distance(stab.prefix_sum_per_d_idx.begin(),
                          upper_bound(ALL(stab.prefix_sum_per_d_idx), idx_start));
        assert(0 <= d_idx_start && d_idx_start < (int)stab.prefix_sum_per_d_idx.size());
        assert(stab.prefix_sum_per_d_idx[d_idx_start] <= idx_start &&
               idx_start < stab.prefix_sum_per_d_idx[d_idx_start + 1]);

        ll local_violated_count = 0;
        ll idx = idx_start;
        Array<n> arranged_rho_vec{};
        Array<n> fwht_result{};
        ArrayInt<n> rho_vec_idxs{};
        ArrayInt<n> rho_vec_phases{};
        array<int, n> mat_rep{};

        for (int d_idx = d_idx_start;
             d_idx < int(stab.default_matrixes.size()) && idx < idx_end; d_idx++) {
            stab.make_default_matrix_and_XZ(d_idx);

            ll x_idx, z_idx, idx_diff;
            [[maybe_unused]] ll gray;
            tie(x_idx, z_idx, idx_diff, gray) =
                get_xz_idx<n>(idx_start, d_idx, d_idx_start, stab);
            mat_rep = stab.make_matrix_representation(d_idx, x_idx, z_idx);
            ll i_max = 1ll << (stab.X1s[d_idx].size() + stab.Z1s[d_idx].size());

            {
                tie(rho_vec_idxs, rho_vec_phases) =
                    stab.make_arranged_rho_vec(mat_rep, rho_vec, arranged_rho_vec);
                FWHT<n>(arranged_rho_vec, fwht_result);
                for (double& v : fwht_result) {
                    if constexpr (is_dual) {
                        if (abs(v) > 1.0) local_violated_count++;
                        if (abs(v) >= threshold_top)
                            local_values_top.emplace_back(abs(v), idx);
                    } else {
                        if (v >= threshold_top)
                            local_values_top.emplace_back(v, idx);
                        else if (v <= threshold_bot)
                            local_values_bot.emplace_back(v, idx);
                    }
                    idx++;
                }
            }

            for (ll i = idx_diff + 1; i < i_max && idx < idx_end; i++) {
                // Here is the core of "Amat_by_dot",
                // which uses FWHT to calculate the 2^n dot products of
                // stabilizer states (same up to sign) and rho_vec.
                int bit_pos = __builtin_ctzl(i);
                stab.update_arranged_rho_vec(d_idx, bit_pos, mat_rep, rho_vec,
                                             rho_vec_idxs, rho_vec_phases,
                                             arranged_rho_vec);
                FWHT<n>(arranged_rho_vec, fwht_result);
                for (double& v : fwht_result) {
                    if constexpr (is_dual) {
                        if (abs(v) > 1.0) local_violated_count++;
                        if (abs(v) >= threshold_top)
                            local_values_top.emplace_back(abs(v), idx);
                    } else {
                        if (v >= threshold_top)
                            local_values_top.emplace_back(v, idx);
                        else if (v <= threshold_bot)
                            local_values_bot.emplace_back(v, idx);
                    }
                    idx++;
                }
            }
            stab.delete_default_matrix_and_XZ(d_idx);
        }
        assert(idx == idx_end);

        size_t local_values_top_sz = local_values_top.size();
        size_t local_values_bot_sz = local_values_bot.size();
        sort(ALL(local_values_top), greater<pair<double, ll>>());
        sort(ALL(local_values_bot), less<pair<double, ll>>());

// Update topK and boK
#pragma omp critical
        {
            total_violated_count += local_violated_count;
            update_topK_botK<n, is_dual>(K, topK_value, botK_value, values_top,
                                         values_bot, local_values_top,
                                         local_values_bot);
            double progress = 1.0 * idx_range_per_chunk[chunk_id + 1] / t_s_g_s;
            print_log<n, is_dual>(chunk_id, progress, timer.ms(), topK_value,
                                  local_values_top_sz, botK_value, local_values_bot_sz);
        }
    }
    timer.stop();
    std::cerr << "\e[1F\e[0K total time: " << timer.ms() << "[ms]" << endl;

    vll result;
    result.reserve(values_top.size() + values_bot.size());
    for (auto& [_v, idx] : values_top) result.push_back(idx);
    if constexpr (!is_dual)
        for (auto& [_v, idx] : values_bot) result.push_back(idx);

    // In the case of values are empty, we add dummy values.
    if (values_top.empty()) values_top.emplace_back(-9999, -1);
    if constexpr (!is_dual)
        if (values_bot.empty()) values_bot.emplace_back(+9999, -1);

    std::cerr << "----- the obtained value ranges -----" << endl;
    std::cerr << "\033[1;32m max of topK\033[0m: "
              << (*max_element(ALL(values_top))).first << endl;
    std::cerr << "\033[1;32m min of topK\033[0m: "
              << (*min_element(ALL(values_top))).first << endl;
    if constexpr (!is_dual) {
        std::cerr << "\033[1;31m max of botK\033[0m: "
                  << (*max_element(ALL(values_bot))).first << endl;
        std::cerr << "\033[1;31m min of botK\033[0m: "
                  << (*min_element(ALL(values_bot))).first << endl;
    } else {
        std::cerr << "\033[1;33m violated_count\033[0m: " << total_violated_count
                  << endl;
        assert(values_bot.empty() && botK_value == -9999);
    }

    // Check with the naive solution
#ifndef NDEBUG
    stab_original.check_values(rho_vec_lexical_order, is_dual,
                               (*max_element(ALL(values_top))).first,
                               is_dual ? (*min_element(ALL(values_top))).first
                                       : (*min_element(ALL(values_bot))).first);
#endif

    if (values_top.back().second == -1) values_top.pop_back();
    if constexpr (!is_dual)
        if (values_bot.back().second == -1) values_bot.pop_back();

    return stab_original.restore_Amat(result);
}

int main(int argc, char** argv) {
    // * For debug
    // Stabilizer<1> stab1;
    // stab1.vis_Amat();
    // Stabilizer<2> stab2;
    // stab2.vis_Amat();
    // return 0;

    int n_qubit;
    double K;
    bool is_dual;
    bool is_random;
    string rho_file;
    string output_file;

    if (argc == 7) {
        n_qubit = atoi(argv[1]);
        K = atof(argv[2]);
        is_dual = (atoi(argv[3]) == 1);
        is_random = (atoi(argv[4]) == 1);
        rho_file = argv[5];
        output_file = argv[6];
    } else {
        cout << "Usage: ./a.out n_qubit K is_dual is_random rho_file output_file\n"
             << "Please see the comment at the top of this file "
                "(fast_dot_products.cpp) for the details."
             << endl;
        return 1;
    }
    if (n_qubit < 1 || 8 < n_qubit) throw runtime_error("n_qubit must be in [1, 8]");

    cnpy::NpyArray rho_npy = cnpy::npz_load(rho_file + ".npz")["rho_vec"];
    assert(int(rho_npy.shape.size()) == 1);
    assert(int(rho_npy.shape[0]) == 1 << (2 * n_qubit));
    vd rho_vec(rho_npy.shape[0]);
    for (int i = 0; i < int(rho_npy.shape[0]); i++)
        rho_vec[i] = rho_npy.data<double>()[i];

    std::cerr << fixed << setprecision(10);
    std::cerr << "[start] n_qubit: " << n_qubit << " K: " << K << endl;

    pair<vvi, vvb> Amat;

    if (is_random) {
        if (n_qubit == 1) Amat = Amat_by_random<1>(K);
        if (n_qubit == 2) Amat = Amat_by_random<2>(K);
        if (n_qubit == 3) Amat = Amat_by_random<3>(K);
        if (n_qubit == 4) Amat = Amat_by_random<4>(K);
        if (n_qubit == 5) Amat = Amat_by_random<5>(K);
        if (n_qubit == 6) Amat = Amat_by_random<6>(K);
        if (n_qubit == 7) Amat = Amat_by_random<7>(K);
        if (n_qubit == 8) Amat = Amat_by_random<8>(K);
    } else if (is_dual) {
        if (n_qubit == 1) Amat = Amat_by_dot<1, true>(K, rho_vec);
        if (n_qubit == 2) Amat = Amat_by_dot<2, true>(K, rho_vec);
        if (n_qubit == 3) Amat = Amat_by_dot<3, true>(K, rho_vec);
        if (n_qubit == 4) Amat = Amat_by_dot<4, true>(K, rho_vec);
        if (n_qubit == 5) Amat = Amat_by_dot<5, true>(K, rho_vec);
        if (n_qubit == 6) Amat = Amat_by_dot<6, true>(K, rho_vec);
        if (n_qubit == 7) Amat = Amat_by_dot<7, true>(K, rho_vec);
        if (n_qubit == 8) Amat = Amat_by_dot<8, true>(K, rho_vec);
    } else {
        if (n_qubit == 1) Amat = Amat_by_dot<1, false>(K, rho_vec);
        if (n_qubit == 2) Amat = Amat_by_dot<2, false>(K, rho_vec);
        if (n_qubit == 3) Amat = Amat_by_dot<3, false>(K, rho_vec);
        if (n_qubit == 4) Amat = Amat_by_dot<4, false>(K, rho_vec);
        if (n_qubit == 5) Amat = Amat_by_dot<5, false>(K, rho_vec);
        if (n_qubit == 6) Amat = Amat_by_dot<6, false>(K, rho_vec);
        if (n_qubit == 7) Amat = Amat_by_dot<7, false>(K, rho_vec);
        if (n_qubit == 8) Amat = Amat_by_dot<8, false>(K, rho_vec);
    }

    save_Amat_as_npz(n_qubit, Amat, output_file);

    return 0;
}
