function model = disc_random_model(S, d, K, opts)
% A random, well-conditioned LSIM in the em_lsim init_model format, for tests.
%   S, d, K : 1 x C state counts, feature dims, mixture counts
%   opts    : .mean_sep (default 1.5), .var_range ([0.4 1.2]),
%             .self_coupling (default 0.6; theta(z,z) before normalization),
%             .left2right (default false: dense transitions; true: upper
%             bidiagonal S(c) x S(z) blocks with exact structural zeros),
%             .missing_frac (unused here; the sampler applies it)
if nargin < 4
    opts = struct();
end
if ~isfield(opts, 'mean_sep'), opts.mean_sep = 1.5; end
if ~isfield(opts, 'var_range'), opts.var_range = [0.4 1.2]; end
if ~isfield(opts, 'self_coupling'), opts.self_coupling = 0.6; end
if ~isfield(opts, 'left2right'), opts.left2right = false; end

C = numel(S);
model.pi_0_lsim = cell(C, 1);
model.transition_matrices_IM_cell = cell(C, C);
model.gmm_para_lsim = cell(C, 1);

theta = rand(C, C);
theta = theta + opts.self_coupling*C*eye(C);
model.coupling_tetha_IM = theta./sum(theta, 1);

for z = 1:C
    pz = rand(S(z), 1) + 0.2;
    if opts.left2right
        pz(:) = 0;
        pz(1) = 1;
    end
    model.pi_0_lsim{z, 1} = pz/sum(pz);
    for c = 1:C
        A = rand(S(c), S(z)) + 0.3*eye(S(c), S(z))*S(z);
        if opts.left2right
            mask = zeros(S(c), S(z));
            for i = 1:S(c)
                j1 = min(i, S(z));
                mask(i, j1) = 1;
                mask(i, min(j1+1, S(z))) = 1;
            end
            A = A.*mask;
        end
        model.transition_matrices_IM_cell{c, z} = A./sum(A, 2);
    end
    gmm_para = [];
    for i = 1:S(z)
        w = rand(K(z), 1) + 0.5;
        w = w/sum(w);
        for k = 1:K(z)
            gmm_para(i).P(k, 1) = w(k); %#ok<AGROW>
            gmm_para(i).mu(k).x = opts.mean_sep*(i - (S(z)+1)/2) + 0.5*randn(d(z), 1) + 0.3*(k-1);
            gmm_para(i).sigma(k).x = opts.var_range(1) + diff(opts.var_range)*rand(d(z), 1);
        end
    end
    model.gmm_para_lsim{z, 1}.gmm_para = gmm_para;
end
end
