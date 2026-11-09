%DEMO_DISCRIMINATIVE_LSIM  Discriminative (MMI) refinement of class-specific LSIMs.
%
%   1. Simulate a 2-class, 2-channel problem: both classes share the emissions and differ
%      only in their dynamics (channel coupling and transition persistence).
%   2. Fit one LSIM per class by EM (em_lsim), with fewer states than the true models.
%   3. Refine both models jointly by maximum mutual information (em_lsim_discriminative),
%      keeping the iteration with the best validation balanced accuracy.
%   4. Compare test accuracy of the generative and the MMI models (lsim_discriminative_predict).

clc; clear; close all
addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));   % the matlab/ folder
rng(7)

%% 1. Two LSIM classes and simulated sequences (train / validation / test)
S_true = [3 3];  d = [1 1];  T = 50;
truth = {make_lsim(S_true, 0.30, 0.85), make_lsim(S_true, 0.10, 0.80)};   % (cross-coupling, persistence)
n = struct('train', 60, 'val', 60, 'test', 100);                          % sequences per class
for set = {'train', 'val', 'test'}
    data.(set{1}) = cell(2, 1);
    for k = 1:2
        data.(set{1}){k} = sample_lsim(truth{k}, n.(set{1}), T, S_true, d);
    end
end

%% 2. Generative training: one EM-fitted LSIM per class (2 states per channel)
S = [2 2];  K = [1 1];
em_extra = struct('plot', 0, 'check_convergence', 0, 'sigma_diag', 1);
gen_models = cell(2, 1);
for k = 1:2
    [pi0, theta, A, gmm] = em_lsim(data.train{k}, S, K, 30, em_extra);
    gen_models{k} = struct('pi_0_lsim', {pi0}, 'coupling_tetha_IM', theta, ...
        'transition_matrices_IM_cell', {A}, 'gmm_para_lsim', {gmm});
end

%% 3. MMI refinement (published recipe, 8 iterations), checkpoint on the validation set
opts = struct('objective', 'mmi', 'validation_observations', {data.val}, 'verbose', 0);
[mmi_models, history, diagnostics] = em_lsim_discriminative(data.train, S, K, [], opts, gen_models);
fprintf('MMI objective per iteration: %s\n', mat2str(history.objective', 4));
fprintf('validation balanced accuracy: %s (checkpoint: iteration %d)\n', ...
    mat2str(history.val_balanced_accuracy', 3), diagnostics.checkpoint_iteration);

%% 4. Test accuracy
test_obs = [data.test{1}, data.test{2}];
test_label = [ones(n.test, 1); 2*ones(n.test, 1)];
[~, pred_gen] = lsim_discriminative_predict(gen_models, test_obs);
[~, pred_mmi] = lsim_discriminative_predict(mmi_models, test_obs);
fprintf('test accuracy: generative (EM) %.3f, MMI %.3f\n', mean(pred_gen == test_label), ...
    mean(pred_mmi == test_label));

%% helpers ------------------------------------------------------------------------------
function m = make_lsim(S, cross, stay)
% C = 2 LSIM with shared 1-D emissions N(mu_i, 1); classes differ in coupling and persistence.
C = numel(S);
m.coupling_tetha_IM = (1 - cross)*eye(C) + cross*(ones(C) - eye(C));   % columns sum to 1
for z = 1:C
    m.pi_0_lsim{z, 1} = ones(S(z), 1)/S(z);
    for c = 1:C
        A = (1 - stay)/(S(z) - 1)*ones(S(c), S(z)) + (stay - (1 - stay)/(S(z) - 1))*eye(S(c), S(z));
        m.transition_matrices_IM_cell{c, z} = A./sum(A, 2);
    end
    mus = linspace(-1.5, 1.5, S(z));
    for i = 1:S(z)
        gp(i) = struct('P', 1, 'mu', struct('x', mus(i)), 'sigma', struct('x', 1)); %#ok<AGROW>
    end
    m.gmm_para_lsim{z, 1}.gmm_para = gp;
end
end

function obs = sample_lsim(m, N, T, S, d)
% N sequences of length T, sampled through the equivalent CHMM (C x N cell, em_lsim format)
[~, ~, ~, ~, ~, pi_chmm, trans_chmm, gmm_chmm] = im_para_eqhmm(m.pi_0_lsim, m.gmm_para_lsim, ...
    m.coupling_tetha_IM, m.transition_matrices_IM_cell, 1);
obs = cell(numel(S), N);
for i = 1:N
    obs(:, i) = generate_chmm_time_series(T, S, d, gmm_chmm, trans_chmm, pi_chmm);
end
end
