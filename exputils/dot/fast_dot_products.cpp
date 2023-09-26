#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
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
using ll = long long;
using vi = vc<int>; using vvi = vvc<int>; using vvvi = vvvc<int>;
using vll = vc<ll>; using vvll = vvc<ll>; using vvvll = vvvc<ll>;
using vb = vc<bool>; using vvb = vvc<bool>; using vvvb = vvvc<bool>;
using vd = vc<double>; using vvd = vvc<double>; using vvvd = vvvc<double>;
using pii = pair<int, int>;
#define ALL(x) begin(x), end(x)

struct Timer{
    void start(){_start=chrono::system_clock::now();}
    void stop(){_end=chrono::system_clock::now();sum+=chrono::duration_cast<chrono::nanoseconds>(_end-_start).count();}
    inline int ms()const{const chrono::system_clock::time_point now=chrono::system_clock::now();return static_cast<int>(chrono::duration_cast<chrono::microseconds>(now-_start).count()/1000);}
    inline int ns()const{const chrono::system_clock::time_point now=chrono::system_clock::now();return static_cast<int>(chrono::duration_cast<chrono::microseconds>(now-_start).count());}
    string report(){return to_string(sum/1000000)+"[ms]";}
    void reset(){_start=chrono::system_clock::now();sum=0;}
    private: chrono::system_clock::time_point _start,_end;long long sum=0;
}timer;
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
        // check for overflow
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

ll total_stabilizer_group_size(int n) {
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

vd FWHT(int N, const vd& As) {
    // https://en.wikipedia.org/wiki/Hadamard_transform
    int size = 1 << N;
    assert(int(As.size()) == size);
    vd inplaceAs = As;
    for (int h = 1; h < size; h <<= 1) {
        for (int i = 0; i < size; i += (h << 1)) {
            for (int j = i; j < i + h; ++j) {
                double x = inplaceAs[j];
                double y = inplaceAs[j + h];
                inplaceAs[j] = x + y;
                inplaceAs[j + h] = x - y;
            }
        }
    }
    // for (int i = 0; i < size; ++i) inplaceAs[i] /= size;
    return inplaceAs;
}

void print_matrix(const vvi& matrix) {
    // Print matrix as a grid.
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

struct PauliDotPhaseTable {
    int phase_table[16] = {
        // https://arxiv.org/pdf/1711.07848.pdf
        //                   (+i = 1, -i = 3)
        //                   | I   X   Y   Z
        //                  -----------------
        0, 0,  0,  0,   // I | I   X   Y   Z
        0, 0,  +1, +3,  // X | X   I  iZ -iY
        0, +3, 0,  +1,  // Y | Y -iZ   I  iX
        0, +1, +3, 0,   // Z | Z  iY -iX   I
    };

    // 65536 = (4^4) × (4^4)
    // The reason why we use 4 is that the number of qubits is at most 8.
    // We cannot hold (4^8) × (4^8) table in memory,
    // so we use 4^4 table instead and compute the phase by using the table.
    int phase_table_4[65536];

    constexpr PauliDotPhaseTable() : phase_table_4() {
        int size = 256;  // = 4^4
        for (int i1 = 0; i1 < size; i1++) {
            for (int i2 = 0; i2 < size; i2++) {
                phase_table_4[i1 * size + i2] = pauli_product_phase(i1, i2);
            }
        }
    }

   private:
    constexpr int pauli_product_phase(int i1, int i2) {
        // Example: i1 = 0b0101, i2 = 0b1011 where n=2
        // (Actually, n is fixed to 4 in this function.)
        // i1 and i2 are "Amat_row_idx", (cf. "to_Amat_row_idx")
        // so i1 is XX and i2 is YZ.
        // Since X*Y=iZ and X*Z=-iY (Pauli matrix multiplication),
        // the i1*i2 = (X*Y) ⊗ (X*Z) = (iZ) ⊗ (-iY) = ZY.
        // Now, we only want to know the phase of the product,
        // so the actual calculation needed here is (+i)*(-i)=(+1).
        // By converting +1->0, +i->1, -1->2, -i->3,
        // we can compute the phase by integer addition mod 4.
        int ret = 0;
        for (int i = 0; i < 4; i++) {
            ret += phase_table[((i1 & 0b11) << 2) + (i2 & 0b11)];
            i1 >>= 2;
            i2 >>= 2;
        }
        return ret % 4;
    }
} pauli_dot_phase_table;

// ================================ Stabilizer ======================================

struct Stabilizer {
    // This struct is for generating the all pure stabilizer states.
    // How we generate the states is the following.

    // 0. Every pure stabilizer state can be represented with
    //    F_2 n times 2n matrix, which is called "stabilizer representation matrix".
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

    const int n;               // the number of qubits
    vi ks;                     // k per d_idx
    vvi col_idxs_lists;        // col_idxs_list per d_idx
    vll prefix_sum_per_d_idx;  // prefix sum of the number of stabilizer states
    vvi default_matrixes;      // default matrix per d_idx
    vvll X1s;                  // X1 per d_idx
    vvll Z1s;                  // Z1 per d_idx

    Stabilizer(int n) : n(n) {
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

                // the following code's explanation is in "make_default_matrix_and_XZ"
                int X1size = 0;
                int Z1size = 0;
                {
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
        assert(cnt_total == total_stabilizer_group_size(n));
        assert(prefix_sum_per_d_idx.back() == total_stabilizer_group_size(n));

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

    vi make_matrix_representation(ll d_idx, ll x_idx, ll z_idx) const {
        // Every matrix representation is indexed by (d_idx, x_idx, z_idx).
        // This function returns the matrix indicated by the arguments.
        assert(0 <= d_idx && d_idx < (ll)default_matrixes.size());
        assert(0 <= x_idx && x_idx < (1ll << X1s[d_idx].size()));
        assert(0 <= z_idx && z_idx < (1ll << Z1s[d_idx].size()));
        vi ret = default_matrixes[d_idx];
        const vll& X1 = X1s[d_idx];
        for (int i = 0; i < int(X1.size()); i++) {
            if (x_idx & (1 << i)) {
                auto [row1, col1, row2, col2] = decompress_row_col(X1[i]);
                ret[row1] |= 1 << col1;
                ret[row2] |= 1 << col2;
            }
        }
        const vll& Z1 = Z1s[d_idx];
        for (int i = 0; i < int(Z1.size()); i++) {
            if (z_idx & (1 << i)) {
                auto [row1, col1, row2, col2] = decompress_row_col(Z1[i]);
                ret[row1] |= 1 << col1;
                ret[row2] |= 1 << col2;
            }
        }
        return ret;
    }

    void modify_matrix_representation(ll d_idx, int bit_pos, vi& mat_rep) const {
        // We can speed up making matrix representation by using this function.
        // If the change to the matrix representation is small,
        // we can modify the matrix representation without making a new one.
        //
        // The key of speed up is gray code.
        // https://en.wikipedia.org/wiki/Gray_code
        assert(0 <= d_idx && d_idx < (ll)default_matrixes.size());
        assert(0 <= bit_pos && bit_pos < int(X1s[d_idx].size() + Z1s[d_idx].size()));
        if (bit_pos < int(Z1s[d_idx].size())) {
            const vll& Z1 = Z1s[d_idx];
            int i = bit_pos;
            auto [row1, col1, row2, col2] = decompress_row_col(Z1[i]);
            mat_rep[row1] ^= 1 << col1;
            if (row1 != row2 || col1 != col2) mat_rep[row2] ^= 1 << col2;
        } else {
            const vll& X1 = X1s[d_idx];
            int i = bit_pos - int(Z1s[d_idx].size());
            auto [row1, col1, row2, col2] = decompress_row_col(X1[i]);
            mat_rep[row1] ^= 1 << col1;
            mat_rep[row2] ^= 1 << col2;
        }
    }

    int gen_idxs[8];  // Temporary variable. It is guaranteed that n <= 8.
    void generate_dot_data(const vi& matrix, vi& rows, vb& is_minus) {
        // Example: n_qubit=3, matrix=[0b110011, 0b001010, 0b000110]
        //
        // The following three Pauli operators are generators of the stabilizer group.
        // 0b110011 = [Z:110, X:011] = ZYX(in reverse order) = XYZ
        // 0b001010 = [Z:001, X:010] = IXZ(in reverse order) = ZXI
        // 0b000110 = [Z:000, X:110] = XXI(in reverse order) = IXX
        //
        // Generate all elements of the stabilizer group by multiplying the generators.
        //
        //   idx Pauli idx  relationship  phase
        //   --- ----- --- -------------- -----
        //   000  +III   0                  1
        // * 100  +XYZ  27                  1
        // * 010  +ZXI  52                  1
        //   110  -YZZ  47 = 27 ^ 52       -1
        // * 001  +IXX   5                  1
        //   101  +XZY  30 = 27      ^ 5    1
        //   011  +ZIX  49 =      52 ^ 5    1
        //   111  +YYY  42 = 27 ^ 52 ^ 5    1
        //
        // The result will be stored in rows and is_minus.
        // Also refer to the function "test".

        for (int i = 0; i < n; i++) gen_idxs[i] = to_Amat_row_idx(matrix[i]);

        rows[0] = 0;
        is_minus[0] = false;

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

        // It is ok to use int, since the 2*n <= 32(int size)
        int row = 0;
        int phase = 0;
        for (int i = 1; i < (1 << n); i++) {
            int gray = i ^ (i >> 1);
            int bit_pos = __builtin_ctz(i);
            phase += phase_of_product(row, gen_idxs[bit_pos]);
            phase %= 4;
            row ^= gen_idxs[bit_pos];
            rows[gray] = row;
            assert(phase == 0 || phase == 2);
            is_minus[gray] = (phase == 2);
        }
    }

    pair<vvi, vvb> restore_Amat(vll& selected_cols) {
        // For given selected_cols, restore the Amat.
        // This is corresponding to the function "compute_topK_botK_dots".
        vi mat_rep;
        vi rows(1 << n);
        vb is_minus(1 << n);
        vvi Amat_rows(selected_cols.size(), vi(1 << n));
        vvb Amat_is_minus(selected_cols.size(), vb(1 << n));
        vvi walsh = sylvesters(n);
        int d_idx_now = 0;
        ll last_ub = 0;
        make_default_matrix_and_XZ(d_idx_now);
        ll ub = prefix_sum_per_d_idx[d_idx_now + 1];
        sort(ALL(selected_cols));
        std::cerr << "start: restore_Amat..." << endl;
        for (int i = 0; i < int(selected_cols.size()); i++) {
            ll idx = selected_cols[i];
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
            idx2 = idx2 ^ (idx2 >> 1);
            ll x_idx = idx2 / (1ll << Z1s[d_idx].size());
            ll z_idx = idx2 % (1ll << Z1s[d_idx].size());
            mat_rep = make_matrix_representation(d_idx, x_idx, z_idx);
            generate_dot_data(mat_rep, rows, is_minus);
            for (int i = 0; i < (1 << n); i++)
                is_minus[i] = is_minus[i] ^ (walsh[i][wal_idx] == -1);
            Amat_rows[i] = rows;
            Amat_is_minus[i] = is_minus;
        }
        delete_default_matrix_and_XZ(d_idx_now);
        assert(all_of(ALL(default_matrixes), [](const vi& v) { return v.empty(); }));
        std::cerr << "done" << endl;

        return {Amat_rows, Amat_is_minus};
    }

    void vis_Amat() {
        // Only for debug. Make the Amat.
        assert(n <= 4);
        vvi Amat(1 << (2 * n), vi(total_stabilizer_group_size(n), 0));
        vvi walsh = sylvesters(n);
        ll cnt = 0;

        for (ll d_idx = 0; d_idx < (ll)default_matrixes.size(); d_idx++) {
            make_default_matrix_and_XZ(d_idx);
            for (ll x_idx = 0; x_idx < (1ll << X1s[d_idx].size()); x_idx++) {
                for (ll z_idx = 0; z_idx < (1ll << Z1s[d_idx].size()); z_idx++) {
                    // First, make the matrix representation.
                    vi mat_rep = make_matrix_representation(d_idx, x_idx, z_idx);
                    // Second, convert the matrix representation to the Amat format.
                    vi rows(1 << n);
                    vb is_minus(1 << n);
                    generate_dot_data(mat_rep, rows, is_minus);
                    // Third, put the matrix representation to the Amat.
                    // Since every stabilizer state has 2^n states
                    // which is the same up to sign,
                    // we need to put the all of them by using the walsh matrix.
                    for (int wal_idx = 0; wal_idx < (1 << n); wal_idx++) {
                        for (int i = 0; i < (1 << n); i++) {
                            Amat[rows[i]][cnt] =
                                (is_minus[i] ? -1 : 1) * (walsh[i][wal_idx]);
                        }
                        cnt++;
                    }
                }
            }
            delete_default_matrix_and_XZ(d_idx);
        }
        assert(cnt == total_stabilizer_group_size(n));

        std::cerr << "n: " << n << endl;
        std::cerr << "Amat size: " << Amat.size() << " x " << Amat[0].size() << endl;
        print_matrix(Amat);
    }

    //    private:
    int to_Amat_row_idx(int bit) const {
        // Example: n=4, bit = 0b11001010
        // The bit is corresponding to the matrix representation of a Pauli operator,
        // which means [Z side: (1100), X side: (1010)].
        // Thus, the indicated Pauli operator is YZXI,
        // and this is in reverse order, the actual one is IXZY.
        // We defined the "Amat_row_idx" as the lexical order of the Pauli operators.
        // Therefore, the Amat_row_idx is 0(I)*64 + 1(X)*16 + 3(Z)*4 + 2(Y)*1 = 30.
        int ret = 0;
        for (int _ = 0; _ < n; _++) {
            int isZ = (bit >> n) & 1;
            ret <<= 2;
            ret += ((bit & 1) ^ isZ) + (isZ << 1);
            bit >>= 1;
        }
        return ret;
    }

    int phase_of_product(int i1, int i2) const {
        // Compute the phase of product of two Pauli operators.
        // By dividing the Pauli operator into
        // first 4 qubits and last 4 qubits,
        // we can compute the phase by looking up the "phase_table_4".
        int i1_fir = i1 >> (2 * 4);
        int i1_sec = i1 & 0b11111111;
        int i2_fir = i2 >> (2 * 4);
        int i2_sec = i2 & 0b11111111;
        int p1 = pauli_dot_phase_table.phase_table_4[(i1_fir << (2 * 4)) + i2_fir];
        int p2 = pauli_dot_phase_table.phase_table_4[(i1_sec << (2 * 4)) + i2_sec];
        return (p1 + p2) % 4;
    }

    // only for speed up.
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

    // only for debug.
    void test() {
        if (n == 2) assert(phase_of_product(0b0101, 0b1011) == 0);
        if (n == 4) assert(to_Amat_row_idx(0b11001010) == 30);
        if (n == 3) {
            vi rows(1 << 3);
            vb is_minus(1 << 3);
            generate_dot_data({0b110011, 0b001010, 0b000110}, rows, is_minus);
            assert(rows == vi({0, 27, 52, 47, 5, 30, 49, 42}));
            assert(is_minus == vb({0, 0, 0, 1, 0, 0, 0, 0}));
        }
    }
};

// ================================ Solver ======================================

double get_topK_value(int n_qubit, double K, const vd& values, bool is_top) {
    // This function is similar to np.partition in numpy.
    ll group_size = total_stabilizer_group_size(n_qubit);
    ll k_index = round(double(group_size) * K / 2 - 1);
    assert(k_index < (ll)values.size());
    vd v2;

    // The following is a heuristic for speed up.
    // It is guaranteed that every dot products are in [0, 2^n],
    //                the 1/2^n of dot products are in [0,1],
    //            and the 1/2^n of dot products aer in [1,2^n].
    // Therefore, we can assume that topK > 1.3 and botK < 0.7
    // with a high probability. The parameters are tunable.
    {
        for (double v : values)
            if (is_top ? v > 1.3 : v < 0.7) v2.push_back(v);
        if (k_index >= (ll)v2.size()) v2 = values;
    }

    if (is_top)
        nth_element(v2.begin(), v2.begin() + k_index, v2.end(), greater<double>());
    else
        nth_element(v2.begin(), v2.begin() + k_index, v2.end(), less<double>());
    return v2[k_index];
}

vd calculate_dot_with_yield_dot_data(int n_qubit, const vd& rho_vec, const vi& rows,
                                     const vb& is_minus) {
    int size = 1 << n_qubit;
    vd arranged_rho_vec(size, 0.0);
    for (int i = 0; i < size; ++i)
        arranged_rho_vec[i] = rho_vec[rows[i]] * (is_minus[i] ? -1 : 1);
    return FWHT(n_qubit, arranged_rho_vec);
}

void save_Amat_as_npz(int n_qubit, const vd& rho_vec, const vvi& Amat_rows,
                      const vvb& Amat_is_minus, const vd& values,
                      const string& output_file_name) {
    assert(Amat_rows.size() == Amat_is_minus.size());
    vi Amat_rows_flat(Amat_rows.size() << n_qubit);
    vi Amat_data_flat(Amat_is_minus.size() << n_qubit);
    double max_dot = numeric_limits<double>::min();
    for (int i = 0; i < int(Amat_rows.size()); i++) {
        double dot = 0;
        for (int j = 0; j < (1 << n_qubit); j++) {
            int idx = i * (1 << n_qubit) + j;
            Amat_rows_flat[idx] = Amat_rows[i][j];
            Amat_data_flat[idx] = Amat_is_minus[i][j] ? -1 : 1;
            dot += rho_vec[Amat_rows_flat[idx]] * Amat_data_flat[idx];
        }
        max_dot = max(max_dot, dot);
    }
    assert(abs(max_dot - *max_element(ALL(values))) < 1e-5);
    cnpy::npz_save(output_file_name + ".npz", "Amat_rows", Amat_rows_flat.data(),
                   {Amat_rows_flat.size()}, "w");
    cnpy::npz_save(output_file_name + ".npz", "Amat_data", Amat_data_flat.data(),
                   {Amat_data_flat.size()}, "a");
}

pair<ll, vll> divide_to_chunk(int n_qubit) {
    // In order to save memory and do parallelization,
    // we divide the calculation into CHUNK_CNT pieces.
    // The overhead of this is not zero, but negligible.
    assert(n_qubit <= 8);
    ll CHUNK_CNT = vll{8, 8, 8, 8, 8, 8, 8, 1024, 1048576}[n_qubit];
    vll idx_range_per_chunk(CHUNK_CNT + 1);
    ll tsgs = total_stabilizer_group_size(n_qubit);
    for (ll i = 0; i <= CHUNK_CNT; i++)
        idx_range_per_chunk[i] = ((tsgs >> n_qubit) * i / CHUNK_CNT) << n_qubit;
    assert(idx_range_per_chunk.back() == tsgs);
    return {CHUNK_CNT, idx_range_per_chunk};
}

void print_log(int n_qubit, int chunk_id, int CHUNK_CNT, double now) {
    if (n_qubit <= 6) return;
    int total = now / (chunk_id + 1) * (CHUNK_CNT - chunk_id) / 1000;
    int hour = total / 3600;
    int min = (total - hour * 3600) / 60;
    int sec = total - hour * 3600 - min * 60;
    string m = (min < 10 ? "0" : "") + to_string(min);
    string s = (sec < 10 ? "0" : "") + to_string(sec);
    std::cerr << "start:\033[1;34m chunk_id\033[0m = " << chunk_id
              << " |\033[1;36m expected remain time\033[0m = " << hour << "[h] " << m
              << "[m] " << s << "[s]" << endl;
}

void update_topK_botK(int n_qubit, double K, ll idx_start, double& topK_value,
                      double& botK_value, vd& values_top, vd& values_bot,
                      vll& indexes_top, vll& indexes_bot, const vd& values) {
    // Subroutine of compute_topK_botK_dots.

    if (K == 1.0) {
        for (int i = 0; i < int(values.size()); i++) {
            values_top.push_back(values[i]);
            indexes_top.push_back(i + idx_start);
        }
        return;
    }

    if (K == 0.0) {
        assert(int(values_top.size()) <= 1 && int(values_bot.size()) <= 1);
        assert(values_top.empty() || topK_value == values_top[0]);
        assert(values_bot.empty() || botK_value == values_bot[0]);
        ll max_idx = distance(values.begin(), max_element(ALL(values)));
        double max_val = values[max_idx];
        if (max_val > topK_value) {
            topK_value = max_val;
            values_top = {topK_value};
            indexes_top = {max_idx + idx_start};
        }
        ll min_idx = distance(values.begin(), min_element(ALL(values)));
        double min_val = values[min_idx];
        if (min_val < botK_value) {
            botK_value = min_val;
            values_bot = {botK_value};
            indexes_bot = {min_idx + idx_start};
        }
        return;
    }

    for (int i = 0; i < int(values.size()); i++) {
        if (values[i] >= topK_value) {
            values_top.push_back(values[i]);
            indexes_top.push_back(i + idx_start);
        } else if (values[i] <= botK_value) {
            values_bot.push_back(values[i]);
            indexes_bot.push_back(i + idx_start);
        }
    }
    ll group_size = total_stabilizer_group_size(n_qubit);
    ll k_index = round(double(group_size) * K / 2 - 1);
    topK_value = int(values_top.size()) > k_index
                     ? get_topK_value(n_qubit, K, values_top, true)
                     : 1.0;
    botK_value = int(values_bot.size()) > k_index
                     ? get_topK_value(n_qubit, K, values_bot, false)
                     : 1.0;
    vd values_top2;
    vd values_bot2;
    vll indexes_top2;
    vll indexes_bot2;
    values_top2.reserve(k_index + 1);
    values_bot2.reserve(k_index + 1);
    indexes_top2.reserve(k_index + 1);
    indexes_bot2.reserve(k_index + 1);
    assert(values_top.size() == indexes_top.size());
    for (int i = 0; i < int(values_top.size()); i++) {
        if (values_top[i] >= topK_value) {
            values_top2.push_back(values_top[i]);
            indexes_top2.push_back(indexes_top[i]);
        }
    }
    assert(values_bot.size() == indexes_bot.size());
    for (int i = 0; i < int(values_bot.size()); i++) {
        if (values_bot[i] <= botK_value) {
            values_bot2.push_back(values_bot[i]);
            indexes_bot2.push_back(indexes_bot[i]);
        }
    }
    swap(values_top, values_top2);
    swap(values_bot, values_bot2);
    swap(indexes_top, indexes_top2);
    swap(indexes_bot, indexes_bot2);
}

tuple<ll, ll, ll, ll> get_xz_idx(int n_qubit, ll idx_start, int d_idx, int d_idx_start,
                                 const Stabilizer& stab) {
    if (d_idx == d_idx_start) {
        ll idx_diff = idx_start - stab.prefix_sum_per_d_idx[d_idx];
        assert(idx_diff % (1 << n_qubit) == 0);
        idx_diff >>= n_qubit;
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

void compute_topK_botK_dots(int n_qubit, double K, const vd& rho_vec,
                            const string& output_file_name) {
    // Find the indexes of the topK and botK elements of the dot (inner) product.
    assert(n_qubit <= 8);

    vd values_top;
    vd values_bot;
    vll indexes_top;
    vll indexes_bot;
    double topK_value = 1.0;
    double botK_value = 1.0;

    Stabilizer stab_original(n_qubit);

    auto [CHUNK_CNT, idx_range_per_chunk] = divide_to_chunk(n_qubit);

    timer.start();
#pragma omp parallel for schedule(dynamic, 1)
    for (int chunk_id = 0; chunk_id < CHUNK_CNT; chunk_id++) {
        print_log(n_qubit, chunk_id, CHUNK_CNT, timer.ms());

        ll idx_start = idx_range_per_chunk[chunk_id];
        ll idx_end = idx_range_per_chunk[chunk_id + 1];
        vd values(idx_end - idx_start);

        Stabilizer stab = stab_original;

        int d_idx_start =
            -1 + distance(stab.prefix_sum_per_d_idx.begin(),
                          upper_bound(ALL(stab.prefix_sum_per_d_idx), idx_start));
        assert(0 <= d_idx_start && d_idx_start < (int)stab.prefix_sum_per_d_idx.size());
        assert(stab.prefix_sum_per_d_idx[d_idx_start] <= idx_start &&
               idx_start < stab.prefix_sum_per_d_idx[d_idx_start + 1]);

        ll idx = 0;
        vi rows(1 << n_qubit);
        vb is_minus(1 << n_qubit);

        auto compute_dots = [&](const vi& mat_rep) {
            // This function is the core of "compute_topK_botK_dots",
            // which uses FWHT to calculate the 2^n dot products of
            // stabilizer states (same up to sign) and rho_vec.
            stab.generate_dot_data(mat_rep, rows, is_minus);
            vd dots =
                calculate_dot_with_yield_dot_data(n_qubit, rho_vec, rows, is_minus);
            for (int i = 0; i < (1 << n_qubit); i++) values[idx + i] = dots[i];
            idx += 1 << n_qubit;
        };

        for (int d_idx = d_idx_start;
             d_idx < int(stab.default_matrixes.size()) && idx < idx_end - idx_start;
             d_idx++) {
            stab.make_default_matrix_and_XZ(d_idx);

            ll x_idx, z_idx, idx_diff;
            [[maybe_unused]] ll gray;
            tie(x_idx, z_idx, idx_diff, gray) =
                get_xz_idx(n_qubit, idx_start, d_idx, d_idx_start, stab);
            vi mat_rep = stab.make_matrix_representation(d_idx, x_idx, z_idx);
            compute_dots(mat_rep);

            for (ll i = idx_diff + 1;
                 i < (1ll << (stab.X1s[d_idx].size() + stab.Z1s[d_idx].size())) &&
                 idx < idx_end - idx_start;
                 i++) {
                int bit_pos = __builtin_ctzl(i);
                stab.modify_matrix_representation(d_idx, bit_pos, mat_rep);
                compute_dots(mat_rep);
                // * test
                // gray ^= 1 << bit_pos;
                // assert(gray == (i ^ (i >> 1)));
                // ll x_idx = gray / (1ll << stab.Z1s[d_idx].size());
                // ll z_idx = gray % (1ll << stab.Z1s[d_idx].size());
                // assert(mat_rep ==
                //        stab.make_matrix_representation(d_idx, x_idx, z_idx));
            }
            stab.delete_default_matrix_and_XZ(d_idx);
        }
        assert(idx == idx_end - idx_start);

// update topK and botK
#pragma omp critical
        {
            update_topK_botK(n_qubit, K, idx_start, topK_value, botK_value, values_top,
                             values_bot, indexes_top, indexes_bot, values);
        }
    }

    vll result;
    result.reserve(indexes_top.size() + indexes_bot.size());
    result.insert(result.end(), ALL(indexes_top));
    result.insert(result.end(), ALL(indexes_bot));

    auto [Amat_rows, Amat_is_minus] = stab_original.restore_Amat(result);

    save_Amat_as_npz(n_qubit, rho_vec, Amat_rows, Amat_is_minus, values_top,
                     output_file_name);
}

int main(int argc, char** argv) {
    // for (int n = 1; n <= 2; n++) {
    //     Stabilizer stab(n);
    //     stab.vis_Amat();
    // }
    // return 0;

    timer.start();

    int n_qubit;
    double K;
    string rho_file;
    string output_file;

    if (argc == 5) {
        n_qubit = atoi(argv[1]);
        K = atof(argv[2]);
        rho_file = argv[3];
        output_file = argv[4];
    } else if (argc != 5) {
        cout << "Usage: ./a.out <n_qubit> <K> <rho_file> <output_file>" << endl;
        return 0;
    }

    cnpy::NpyArray rho_npy = cnpy::npz_load(rho_file + ".npz")["rho_vec"];
    assert(int(rho_npy.shape.size()) == 1);
    assert(int(rho_npy.shape[0]) == 1 << (2 * n_qubit));
    vd rho_vec(rho_npy.shape[0]);
    for (int i = 0; i < int(rho_npy.shape[0]); ++i)
        rho_vec[i] = rho_npy.data<double>()[i];

    compute_topK_botK_dots(n_qubit, K, rho_vec, output_file);

    timer.stop();

    std::cerr << "time: " << timer.ms() << "[ms]" << endl;

    return 0;
}
