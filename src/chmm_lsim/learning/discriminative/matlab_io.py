"""MATLAB interop: packed LSIM models (lsim_disc_pack / lsim_pack_para) -> LSIMParams."""

from __future__ import annotations

import numpy as np

from chmm_lsim.models.lsim.params import LSIMParams
from chmm_lsim.core.structure import LSIMStructure


def packed_to_params(d: dict) -> LSIMParams:
    """``d``: a packed model struct read with scipy.io.loadmat(simplify_cells=True) (fields pi_0,
    transitions_matrices, coupling_tetha_IM, P_all, mu_all, sigma_all, channel_num_states,
    num_gmm_component, dim_observation, sigma_diag)."""
    S = np.atleast_1d(np.asarray(d["channel_num_states"], int)).ravel()
    K = np.atleast_1d(np.asarray(d["num_gmm_component"], int)).ravel()
    dim = np.atleast_1d(np.asarray(d["dim_observation"], int)).ravel()
    st = LSIMStructure(tuple(S), tuple(dim), tuple(K))
    diag = bool(np.asarray(d.get("sigma_diag", 1)).item())
    C, maxS, maxK, D = st.C, st.max_M, st.max_K, st.D
    P_all = np.asarray(d["P_all"], float).reshape(C, maxS, maxK)
    mu_all = np.asarray(d["mu_all"], float).reshape(D, maxS, maxK)
    sig = np.asarray(d["sigma_all"], float)
    sig = sig.reshape(D, maxS, maxK) if diag else sig.reshape(D, st.max_d, maxS, maxK)
    return LSIMParams.from_packed(
        st, pi_0=np.asarray(d["pi_0"], float).ravel(),
        transition=np.asarray(d["transitions_matrices"], float).reshape(st.S, st.S),
        coupling=np.asarray(d["coupling_tetha_IM"], float).reshape(C, C),
        P_all=P_all, mu_all=mu_all, sigma_all=sig, sigma_diag=diag)


def load_packed_model(path: str, name: str) -> LSIMParams:
    from scipy.io import loadmat

    return packed_to_params(loadmat(path, simplify_cells=True)[name])
