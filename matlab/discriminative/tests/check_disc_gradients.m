% Gradient checks, before any optimization uses the statistics.
%   A. C = 1 (HMM reduction): the gradient built from the kernel's posterior
%      statistics (lsim_disc_gradient) equals central finite differences of the
%      kernel score, for EVERY parameter family -- initial probabilities,
%      transition rows, mixture weights, means, log variances -- with d = 2,
%      K = 2 and 10% missing entries.
%   B. C = 2 exact oracle: the gradient built from the oracle's exact
%      statistics (including the latent source-selection responsibilities that
%      the coupling needs) equals finite differences of the exact
%      log-likelihood -- Fisher's identity, verified rather than assumed.
%   C. C = 2 kernel: the SED statistics are compared with (i) the exact
%      statistics and (ii) finite differences of the kernel score. They are an
%      approximate smoother, so this is REPORTED as agreement, not asserted as
%      equality; the trainer treats them as a surrogate direction.
here = fileparts(mfilename('fullpath'));
addpath(genpath(fileparts(fileparts(here))));
fbc0 = lsim_fb_constants('sed', 'pi0_reg', 0);
h = 1e-5;

% ------------------------------------------------------------------ A. C = 1
rng(21, 'twister');
m1 = disc_random_model(3, 2, 2);
p1 = lsim_disc_pack(m1);
obs = disc_sample(m1, 3, [25 1 40], 0.1);     % includes a singleton sequence
Y1 = obs(1, :);
kernel_score = @(p) sum(lsim_disc_score(p, Y1, fbc0));
g1 = counts_gradient(p1, Y1, @(p, Y) lsim_disc_sequence_stats(p, Y, fbc0));
res1 = disc_fd_compare(p1, g1, kernel_score, h);
print_res('A. C=1 kernel stats vs FD(kernel score)', res1);
assert(all([res1.relerr] < 1e-5), 'C = 1 analytic gradient must match finite differences');

% ------------------------------------------------------------------ B/C. C = 2
rng(22, 'twister');
m2 = disc_random_model([2 2], [1 2], [1 2], struct('self_coupling', 0.3));
p2 = lsim_disc_pack(m2);
obs2 = disc_sample(m2, 3, [30 20 1], 0.1);
Y2 = cell(1, 3);
for n = 1:3
    Y2{n} = [obs2{1, n}; obs2{2, n}];
end

exact_score = @(p) sum(cellfun(@(Y) lsim_disc_exact_oracle(p, Y), Y2));
g2_exact = counts_gradient(p2, Y2, @(p, Y) second_output(@lsim_disc_exact_oracle, p, Y));
res2 = disc_fd_compare(p2, g2_exact, exact_score, h);
print_res('B. C=2 exact stats vs FD(exact log-likelihood)', res2);
assert(all([res2.relerr] < 1e-5), 'exact oracle statistics must give the exact gradient');

kernel_score2 = @(p) sum(lsim_disc_score(p, Y2, fbc0));
g2_sed = counts_gradient(p2, Y2, @(p, Y) lsim_disc_sequence_stats(p, Y, fbc0));
res3 = disc_fd_compare(p2, g2_sed, kernel_score2, h);
print_res('C. C=2 SED stats vs FD(kernel score)   [surrogate: reported, not asserted]', res3);
g2_fd_exact = res2;   %#ok<NASGU>

% statistics themselves: SED vs exact
bc_sed = lsim_disc_counts('zero', p2);
bc_ex = bc_sed;
for n = 1:3
    bc_sed = lsim_disc_counts('add', bc_sed, lsim_disc_sequence_stats(p2, Y2{n}, fbc0), 1, p2);
    [~, ste] = lsim_disc_exact_oracle(p2, Y2{n});
    bc_ex = lsim_disc_counts('add', bc_ex, ste, 1, p2);
end
fprintf('C. SED vs exact expected counts (relative L1 difference):\n');
f = {'pi', 'A', 'theta', 'occ', 's'};
for j = 1:numel(f)
    a = bc_sed.(f{j})(:);
    b = bc_ex.(f{j})(:);
    fprintf('     %-6s %.3e\n', f{j}, sum(abs(a-b))/sum(abs(b)));
end
fprintf('   coupling counts theta(c,z): SED = %s, exact = %s\n', mat2str(bc_sed.theta, 4), mat2str(bc_ex.theta, 4));
fprintf('PASS check_disc_gradients\n');

% =====================================================================
function out = second_output(f, varargin)
[~, out] = f(varargin{:});
end

function g = counts_gradient(p, Ylist, statfun)
bc = lsim_disc_counts('zero', p);
for n = 1:numel(Ylist)
    bc = lsim_disc_counts('add', bc, statfun(p, Ylist{n}), 1, p);
end
dom = struct('support_pi', p.pi_0 > 0, 'support_A', p.transitions_matrices > 0, ...
    'support_theta', p.coupling_tetha_IM > 0, 'support_w', p.P_all > 0);
g = lsim_disc_gradient(p, bc, dom);
end

function print_res(title, res)
fprintf('%s\n', title);
for r = 1:numel(res)
    fprintf('     %-5s n=%3d  max rel err %.3e   cosine %.6f\n', res(r).family, res(r).n, ...
        res(r).relerr, res(r).cosine);
end
end
