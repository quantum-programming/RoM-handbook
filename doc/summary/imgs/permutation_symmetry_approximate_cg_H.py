import numpy as np
from scipy.sparse import hstack

from exputils.eff_kron_perm_Amat import eff_kron_perm_Amat
from exputils.perm_Amat import (
    get_group_to_perm,
    get_perm_Amat,
    tensor_product_in_perm_basis,
)
from exputils.perm_dot import compute_all_dot_products_perm
from exputils.RoM.custom import calculate_RoM_custom
from exputils.RoM.dot import get_topK_indices
from exputils.state.random import make_random_quantum_state
from exputils.state.tensor import make_random_tensor_product_state

####################################
# To calculate the RoM with n = 17,
# turn off caching of perm Amat.
####################################


class PermOptRes:
    def __init__(self, n, RoM, coeff, opt_perm_Amat):
        self.n = n
        self.RoM = RoM
        self.coeff = coeff
        self.opt_perm_Amat = opt_perm_Amat

    def __mul__(self, other):
        n = self.n + other.n
        RoM = self.RoM * other.RoM
        coeff = np.kron(self.coeff, other.coeff)
        opt_perm_Amat = eff_kron_perm_Amat(
            self.n, self.opt_perm_Amat, other.n, other.opt_perm_Amat
        )
        return self.__class__(n, RoM, coeff, opt_perm_Amat)

    def __repr__(self):
        return f"PermOptRes(n={(str(self.n) + ',').ljust(3)} RoM={self.RoM:.7f}, shape={self.opt_perm_Amat.shape})"


def optimize_RoM_only(n, perm_Amat, single_rho):
    perm_rho = tensor_product_in_perm_basis(single_rho, n)
    RoM, _ = calculate_RoM_custom(
        perm_Amat,
        perm_rho,
        method="gurobi",
        presolve=True,
        crossover=False,
        verbose=True,
    )
    return RoM


def optimize(n, perm_Amat, single_rho):
    eps = 0
    perm_rho = tensor_product_in_perm_basis(single_rho, n)
    RoM, coeff = calculate_RoM_custom(
        perm_Amat, perm_rho, method="gurobi", presolve=True, verbose=True
    )
    basic_indices = abs(coeff) > eps
    basic_coeff = coeff[basic_indices]
    opt_perm_Amat = perm_Amat[:, basic_indices]
    return PermOptRes(n, RoM, basic_coeff, opt_perm_Amat)


# precalculation with n = 7
def cg(
    perm_opt_res_list,
    single_rho,
    n=7,
    K=0.005,
    iter_max=100,
    violation_max=100000,
    discard_current_threshold=0.9,
):
    eps = 10**-12

    perm_Amat = get_perm_Amat(n)
    perm_rho = tensor_product_in_perm_basis(single_rho, n)

    (
        perm_idxs,
        perm_data,
        perm_unique_indices,
    ) = get_group_to_perm(n)

    rho_dots = compute_all_dot_products_perm(
        n,
        perm_rho,
        perm_idxs,
        perm_data,
        perm_unique_indices,
    )
    indices = get_topK_indices(rho_dots, K)
    del rho_dots
    current_Amat = perm_Amat[:, indices]
    del indices
    for it in range(iter_max):
        print(f"{it = }, # of columns = {current_Amat.shape[1]}")
        RoM, coeff, dual = calculate_RoM_custom(
            current_Amat, perm_rho, method="gurobi", return_dual=True
        )
        print(f"{RoM = }")
        dual_dots = np.abs(
            compute_all_dot_products_perm(
                n,
                dual,
                perm_idxs,
                perm_data,
                perm_unique_indices,
            )
        )
        dual_violated_indices = dual_dots > 1 + eps
        del dual_dots
        violated_count = np.sum(dual_violated_indices)

        print(f"# of violations: {violated_count}")
        if violated_count == 0:
            break
        elif violated_count <= violation_max:
            indices = np.where(dual_violated_indices)[0]
            extra_Amat = perm_Amat[:, indices]
            del indices
        else:
            raise Exception("Too many violations")

        # restrict current Amat
        nonbasic_indices = np.abs(coeff) > eps
        critical_indices = np.abs(dual @ current_Amat) >= (
            discard_current_threshold - eps
        )
        remain_indices = np.logical_or(nonbasic_indices, critical_indices)
        restricted_current_Amat = current_Amat[:, remain_indices]

        current_Amat = hstack((restricted_current_Amat, extra_Amat))

    del dual_violated_indices
    del perm_Amat
    del perm_idxs, perm_data, perm_unique_indices

    basic_indices = abs(coeff) > eps
    basic_coeff = coeff[basic_indices]
    opt_perm_Amat = current_Amat[:, basic_indices]
    res = PermOptRes(n, RoM, basic_coeff, opt_perm_Amat)
    perm_opt_res_list.append(res)


# extension
def make_approx_Amat(perm_opt_res_list):
    n = len(perm_opt_res_list)
    kron_Amats = []
    for smaller_n in range(1, n // 2 + 1):
        larger_n = n - smaller_n
        kron_opt_perm_Amat = perm_opt_res_list[smaller_n] * perm_opt_res_list[larger_n]
        kron_Amats.append(kron_opt_perm_Amat.opt_perm_Amat)
    hstacked_Amat = hstack(kron_Amats)
    del kron_Amats
    # eliminate duplicate columns
    rng = np.random.default_rng(0)
    rv = rng.integers(low=1e-18, high=1e18, size=hstacked_Amat.shape[0])
    dots = rv @ hstacked_Amat
    values, indices = np.unique(dots, return_index=True)
    del values
    del dots
    perm_Amat = hstacked_Amat[:, indices]
    del indices
    return perm_Amat


def save_perm_opt_res(res: PermOptRes):
    file = f"coeff_Pmat/H_{res.n}.npz"
    np.savez_compressed(file, coeff=res.coeff, opt_perm_Amat=res.opt_perm_Amat)


tough = 21


def main():
    seed = 0
    # single_rho = make_random_quantum_state("mixed", 1, seed)
    # single_rho = make_random_quantum_state("pure", 1, seed)
    single_rho = np.array([1, 2**-0.5, 2**-0.5, 0])

    # precalculation with 1 <= n <= 6

    perm_opt_res_list = [None]
    for n in range(1, 6 + 1):
        perm_Amat = get_perm_Amat(n)
        res = optimize(n, perm_Amat, single_rho)
        perm_opt_res_list.append(res)

    cg(perm_opt_res_list, single_rho)

    for res in perm_opt_res_list[1:]:
        save_perm_opt_res(res)

    exact = len(perm_opt_res_list) - 1
    print(f"{exact = }")

    RoMs = [res.RoM for res in perm_opt_res_list[1:]]
    while len(perm_opt_res_list) < 26:
        perm_Amat = make_approx_Amat(perm_opt_res_list, single_rho)
        if len(perm_opt_res_list) >= tough:
            RoM = optimize_RoM_only(n, perm_Amat, single_rho)
            RoMs.append(RoM)
            with open("result.txt", "w") as f:
                print(*RoMs, sep="\n", file=f)
            res = optimize(n, perm_Amat, single_rho)
            perm_opt_res_list.append(res)
            save_perm_opt_res(perm_opt_res_list[-1])
        else:
            res = optimize(n, perm_Amat, single_rho)
            perm_opt_res_list.append(res)
            RoMs.append(res.RoM)
            with open("result.txt", "w") as f:
                print(*RoMs, sep="\n", file=f)
            save_perm_opt_res(perm_opt_res_list[-1])


if __name__ == "__main__":
    main()
