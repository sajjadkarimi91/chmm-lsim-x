% CHMM  Full coupled hidden Markov models (exact inference on the joint state space).
%
%   em_chmm                     - maximum-likelihood EM for a CHMM with GMM emissions
%   forward_backward_chmm       - exact forward-backward for a CHMM
%   viterbi_chmm                - exact CHMM Viterbi (4 inputs) or LSIM per-channel Viterbi (5 inputs)
%   chmm_joint_model            - joint-state (equivalent single-chain HMM) form of a CHMM
%   chmm_exact_forward_backward - scaled forward-backward on a joint-state chain
%   chmm_cartesian_product      - convert a CHMM to its equivalent HMM over prod(M) states
%   generate_chmm_time_series   - sample hidden states and observations from a CHMM
