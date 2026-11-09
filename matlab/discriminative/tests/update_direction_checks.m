function out = update_direction_checks(seed, n_per_class, iterations)
%UPDATE_DIRECTION_CHECKS First-order check of the applied MMI update (Jacobi schedule).
%
%   out = update_direction_checks(seed, n_per_class, iterations)
%
% On T3 (C = 1 HMM, exact score) and T4 (C = 2 LSIM, approximate score), runs MMI with
% mmi_recipe = 'legacy' (score scale 1) for max(iterations) iterations, capturing the models
% after every iteration. All models of iteration t are updated from the same context: every
% model at iterate t-1. For each checked t and model k, along the APPLIED update
% Delta = p_k(t) - p_k(t-1), it compares
%   predicted = <dJ/dtheta (from the weighted statistics at the context), Delta>
%   fd        = [J(p_k + e Delta) - J(p_k - e Delta)] / (2e), e = 1e-4, other models at t-1.
% The toys:
%   T3  C = 1 HMM, two states, emissions N(-1.5,1) / N(1.5,1) shared; class 1
%       A = [0.9 0.1; 0.1 0.9], class 2 A = [0.6 0.4; 0.4 0.6]; T = 50.
%   T4  C = 2 LSIM, two states per channel, 1-D emissions N(-1.5,1) / N(1.5,1) identical across
%       classes and channels; local transitions [0.9 0.1; 0.1 0.9]; coupling
%       theta = [1-e e; e 1-e] with e = 0.4 (class 1) or 0.05 (class 2); T = 50.
global DISC_SNAPS %#ok<GVMIS>
if nargin < 2 || isempty(n_per_class), n_per_class = 30; end
if nargin < 3 || isempty(iterations), iterations = 1:3; end
fbc = lsim_fb_constants();
out = struct('label', {}, 'predicted', {}, 'fd', {}, 'rel_err', {}, 'same_sign', {});
toys = {'T3', 2, 1; 'T4', [2 2], [1 1]};
for ti = 1:size(toys, 1)
    [class_obs, truth] = toy_data(toys{ti, 1}, n_per_class, seed + ti);
    S = toys{ti, 2};
    K = toys{ti, 3};
    data = lsim_disc_prepare_data(class_obs);
    init = truth;   % start from the truth, perturbed, so the first steps are non-trivial
    for k = 1:numel(init)
        p = lsim_disc_pack(init{k});
        p.mu_all = p.mu_all + 0.3*randn(size(p.mu_all));
        init{k} = lsim_disc_unpack(p);
    end
    M = numel(init);
    ex = struct('objective', 'mmi', 'mmi_recipe', 'legacy');
    cfg = struct('objective', 'mmi', 'score_scale', 1, 'mmi_recipe', 'legacy', 'log_prior', log(ones(M, 1)/M));
    wscale = 1;   % legacy weights = a_n (1[k=y] - q) are exactly dJ/dL
    ex.verbose = 0;
    ex.iteration_hook = @(itr, p, info) store_snap(itr, p);
    DISC_SNAPS = {};
    [~, ~, dg] = em_lsim_discriminative(class_obs, S, K, max(iterations), ex, init);
    snaps = DISC_SNAPS;
    for t = iterations
        if t+1 > numel(snaps), break; end
        ctx = snaps{t};
        L = score_all(ctx, data.Y, fbc);
        [~, pc] = lsim_disc_objective(L, data.label, data.T, cfg);
        for k = 1:M
            pa = ctx{k};
            pb = snaps{t+1}{k};
            if max_diff(pa, pb) == 0
                continue   % rejected step
            end
            [num, den] = lsim_disc_weighted_counts(pa, data, pc.W.num(:, k), pc.W.den(:, k), fbc);
            g = raw_gradient(pa, num, den, dg.domain{k});
            pred = wscale*inner(g, pa, pb);
            e = 1e-4;
            fd = (acc_obj(move(pa, pb, e), k, L, data, cfg, fbc) - acc_obj(move(pa, pb, -e), k, L, data, cfg, fbc))/(2*e);
            lab = sprintf('%s mmi itr %d model %d', toys{ti, 1}, t, k);
            out(end+1) = struct('label', lab, 'predicted', pred, 'fd', fd, ...
                'rel_err', abs(fd - pred)/max(abs(fd), 1e-12), 'same_sign', sign(fd) == sign(pred)); %#ok<AGROW>
        end
    end
end
for r = 1:numel(out)
    fprintf('%-34s predicted %+.6g  fd %+.6g  rel err %.3g\n', out(r).label, out(r).predicted, out(r).fd, out(r).rel_err);
end
end

function [class_obs, truth] = toy_data(name, n_per_class, seed)
rng(seed, 'twister');
T = 50;
switch name
    case 'T3'
        mus = [-1.5 1.5];
        truth = {hmm_model(mus, [0.9 0.1; 0.1 0.9]); hmm_model(mus, [0.6 0.4; 0.4 0.6])};
    case 'T4'
        truth = {lsim2(0.4); lsim2(0.05)};
end
class_obs = {disc_sample(truth{1}, n_per_class, T); disc_sample(truth{2}, n_per_class, T)};
end

function m = hmm_model(mus, A)
S = numel(mus);
m.pi_0_lsim = {ones(S, 1)/S};
m.coupling_tetha_IM = 1;
m.transition_matrices_IM_cell = {A};
gp = [];
for i = 1:S
    gp(i).P = 1; %#ok<AGROW>
    gp(i).mu(1).x = mus(i);
    gp(i).sigma(1).x = 1;
end
m.gmm_para_lsim = {struct('gmm_para', gp)};
end

function m = lsim2(e)
A = [0.9 0.1; 0.1 0.9];
m.pi_0_lsim = {[0.5; 0.5]; [0.5; 0.5]};
m.coupling_tetha_IM = [1-e e; e 1-e];
m.transition_matrices_IM_cell = {A, A; A, A};
gp = [];
mus = [-1.5 1.5];
for i = 1:2
    gp(i).P = 1; %#ok<AGROW>
    gp(i).mu(1).x = mus(i);
    gp(i).sigma(1).x = 1;
end
m.gmm_para_lsim = {struct('gmm_para', gp); struct('gmm_para', gp)};
end

function store_snap(itr, p)
global DISC_SNAPS %#ok<GVMIS>
DISC_SNAPS{itr+1} = p;
end

function A = acc_obj(pk, k, L, data, cfg, fbc)
L(:, k) = lsim_disc_score(pk, data.Y, fbc);
A = lsim_disc_objective(L, data.label, data.T, cfg);
end

function L = score_all(p, Y, fbc)
L = zeros(numel(Y), numel(p));
for k = 1:numel(p)
    L(:, k) = lsim_disc_score(p{k}, Y, fbc);
end
end

function d = max_diff(a, b)
f = {'pi_0', 'transitions_matrices', 'coupling_tetha_IM', 'P_all', 'mu_all', 'sigma_all'};
d = 0;
for j = 1:numel(f)
    d = max(d, max(abs(a.(f{j})(:) - b.(f{j})(:))));
end
end

function q = move(pa, pb, e)
q = pa;
f = {'pi_0', 'transitions_matrices', 'coupling_tetha_IM', 'P_all', 'mu_all', 'sigma_all'};
for j = 1:numel(f)
    q.(f{j}) = pa.(f{j}) + e*(pb.(f{j}) - pa.(f{j}));
end
end

function v = inner(g, pa, pb)
f = {'pi_0', 'transitions_matrices', 'coupling_tetha_IM', 'P_all', 'mu_all', 'sigma_all'};
v = 0;
for j = 1:numel(f)
    v = v + sum(g.(f{j})(:).*(pb.(f{j})(:) - pa.(f{j})(:)));
end
end

function g = raw_gradient(p, num, den, dom)
% dJ/d(raw parameter) implied by signed counts: count/p for probabilities,
% (s - n mu)/v for means, (dJ/dlog v)/v for variances
d = @(f) num.(f) - den.(f);
g.pi_0 = zeros(size(p.pi_0)); g.transitions_matrices = zeros(size(p.transitions_matrices));
g.coupling_tetha_IM = zeros(size(p.coupling_tetha_IM)); g.P_all = zeros(size(p.P_all));
on = dom.support_pi; x = d('pi'); g.pi_0(on) = x(on)./p.pi_0(on);
on = dom.support_A; x = d('A'); g.transitions_matrices(on) = x(on)./p.transitions_matrices(on);
on = dom.support_theta; x = d('theta'); g.coupling_tetha_IM(on) = x(on)./p.coupling_tetha_IM(on);
on = dom.support_w; x = d('occ'); g.P_all(on) = x(on)./p.P_all(on);
n = d('n'); s = d('s'); r = d('r');
mu = p.mu_all; v = p.sigma_all;
g.mu_all = (s - n.*mu)./v;
g.sigma_all = 0.5*((r - 2*mu.*s + n.*mu.^2)./v - n)./v;
pad = ~(n ~= 0 | s ~= 0 | r ~= 0);
g.mu_all(pad) = 0;
g.sigma_all(pad) = 0;
end
