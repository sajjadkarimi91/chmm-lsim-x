function obs = disc_sample(model, N, T, missing_frac)
% N sequences of lengths T (scalar or N-vector) from an LSIM in the em_lsim
% init_model format, using the repository's own sampler path:
% im_para_eqhmm(..., flag_chmms = 1) -> per-channel CHMM transitions, then
% generate_chmm_time_series. Returns a C x N cell in em_lsim's format.
% missing_frac (default 0) sets that fraction of entries to NaN at random.
if nargin < 4
    missing_frac = 0;
end
C = numel(model.pi_0_lsim);
if isscalar(T)
    T = T*ones(N, 1);
end
S = zeros(1, C);
d = zeros(1, C);
for c = 1:C
    S(c) = numel(model.pi_0_lsim{c});
    d(c) = numel(model.gmm_para_lsim{c}.gmm_para(1).mu(1).x);
end
[~, ~, ~, ~, ~, pi_0_chmm, transition_chmm, chmm_gmm_para] = im_para_eqhmm( ...
    model.pi_0_lsim, model.gmm_para_lsim, model.coupling_tetha_IM, ...
    model.transition_matrices_IM_cell, 1);
obs = cell(C, N);
for n = 1:N
    seq = generate_chmm_time_series(T(n), S, d, chmm_gmm_para, transition_chmm, pi_0_chmm);
    for c = 1:C
        x = seq{c};
        if missing_frac > 0
            x(rand(size(x)) < missing_frac) = NaN;
        end
        obs{c, n} = x;
    end
end
end
