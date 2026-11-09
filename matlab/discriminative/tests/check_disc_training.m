% End-to-end training contracts of em_lsim_discriminative (MMI, C = 2).
%   a. 2 classes, MMI legacy and published, 5% missing entries, singleton (T = 1)
%      sequences, a left-to-right initial model with exact structural zeros:
%      the accepted objective is non-decreasing (legacy), the returned model's objective
%      recomputed from scratch equals the history entry, every model stays in
%      the compact domain, structural zeros stay exactly zero, scores are finite.
%   b. 4 classes, MMI published and MMI legacy, generative (em_lsim)
%      initialization and validation checkpointing: same contracts (published MMI: the
%      objective is logged only, so non-decrease is not required), plus the checkpoint's
%      recomputed objective equals its history entry.
%   c. statistics from the 'sed' and 'fast' variants (the default is raw 'poise');
%      MMI legacy with score scale 0.1.
%   d. full-covariance initial models are rejected explicitly.
%   e. lsim_discriminative_predict: scores, predictions, normalized scores.
here = fileparts(mfilename('fullpath'));
addpath(genpath(fileparts(fileparts(here))));

% ------------------------------------------------------------ a
rng(41, 'twister');
S = [3 3]; d = [1 2]; K = [1 1];
truth = {disc_random_model(S, d, K, struct('self_coupling', 0.3)), ...
    disc_random_model(S, d, K, struct('self_coupling', 0.3))};
truth{2}.gmm_para_lsim = truth{1}.gmm_para_lsim;   % classes differ in dynamics only
train = cell(2, 1);
for i = 1:2
    train{i} = [disc_sample(truth{i}, 10, 40, 0.05), disc_sample(truth{i}, 2, 1, 0)];
end
init = {disc_random_model(S, d, K, struct('left2right', true)), ...
    disc_random_model(S, d, K, struct('left2right', true))};
ex = struct('objective', 'mmi', 'mmi_recipe', 'legacy', 'verbose', 0);
[models, hist, diag] = em_lsim_discriminative(train, S, K, 25, ex, init);
check_contracts('a. 2-class mmi legacy, missing + T=1 + structural zeros', train, models, hist, diag, init);
ex = struct('objective', 'mmi', 'verbose', 0);
[models, hist, diag] = em_lsim_discriminative(train, S, K, 10, ex, init);
check_contracts('a. 2-class mmi published, missing + T=1 + structural zeros', train, models, hist, diag, init);

% ------------------------------------------------------------ b
rng(42, 'twister');
M = 4;
base = disc_random_model(S, d, K, struct('self_coupling', 0.3));
truth4 = cell(M, 1);
train4 = cell(M, 1);
val4 = cell(M, 1);
for i = 1:M
    ti = disc_random_model(S, d, K, struct('self_coupling', 0.3));
    ti.gmm_para_lsim = base.gmm_para_lsim;          % shared, overlapping emissions
    truth4{i} = ti;
    train4{i} = disc_sample(ti, 8, 30);
    val4{i} = disc_sample(ti, 8, 30);
end
for obj = {struct('objective', 'mmi'), struct('objective', 'mmi', 'mmi_recipe', 'legacy')}
    rng(43, 'twister');
    ex = obj{1};
    ex.verbose = 0;
    ex.validation_observations = val4;
    ex.generative_max_iteration = 30;
    [models, hist, diag] = em_lsim_discriminative(train4, S, K, 15, ex);
    tag = 'mmi published';
    if isfield(ex, 'mmi_recipe'), tag = ['mmi ' ex.mmi_recipe]; end
    check_contracts(sprintf('b. 4-class %s, em_lsim init, validation checkpoint', tag), ...
        train4, models, hist, diag, diag.generative_models);
end

% ------------------------------------------------------------ c: other variants, MMI score scale
for v = {'sed', 'fast'}
    for rc = {'legacy', 'published'}
        ex = struct('objective', 'mmi', 'mmi_recipe', rc{1}, 'verbose', 0, 'fb_constants', lsim_fb_constants(v{1}));
        [models, hist, diag] = em_lsim_discriminative(train, S, K, 8, ex, init);
        check_contracts(sprintf('c. mmi %s, statistics from the ''%s'' variant', rc{1}, v{1}), train, models, hist, diag, init);
    end
end
ex = struct('objective', 'mmi', 'mmi_recipe', 'legacy', 'score_scale', 0.1, 'verbose', 0);
[models, hist, diag] = em_lsim_discriminative(train, S, K, 8, ex, init);
check_contracts('c. MMI legacy, score scale 0.1', train, models, hist, diag, init);

% ------------------------------------------------------------ d
full_init = init;
for i = 1:2
    for z = 2
        for st = 1:S(z)
            full_init{i}.gmm_para_lsim{z}.gmm_para(st).sigma(1).x = eye(d(z));
        end
    end
end
try
    em_lsim_discriminative(train, S, K, 2, struct('verbose', 0), full_init);
    error('full covariance was accepted');
catch err
    assert(strcmp(err.identifier, 'em_lsim_discriminative:fullCovarianceUnsupported'), err.message);
end
fprintf('d. full covariance rejected with %s\n', err.identifier);

% ------------------------------------------------------------ e
[scores, pred, ns] = lsim_discriminative_predict(models, [train{1}, train{2}]);
assert(all(isfinite(scores(:))) && all(abs(sum(ns, 2) - 1) < 1e-12) && numel(pred) == size(scores, 1));
fprintf('e. predict: %d sequences x %d classes, normalized scores sum to 1\n', size(scores));
fprintf('PASS check_disc_training\n');

% =====================================================================
function check_contracts(title, train, models, hist, diag, init)
J = hist.objective;
dJ = diff(J);
assert(all(isfinite(J)), 'non-finite accepted objective');
if ~(strcmp(diag.objective, 'mmi') && strcmp(diag.options.mmi_recipe, 'published'))
    % legacy MMI accepts only measured non-decreasing steps; published MMI is not monotone
    assert(all(dJ >= 0), 'accepted objective decreased');
end
assert(diag.returned_objective_recomputed == diag.returned_objective_in_history, ...
    'returned model objective does not match its history entry');
M = numel(models);
data = lsim_disc_prepare_data(train);
for k = 1:M
    for mset = {models{k}, diag.final_models{k}}
        p = lsim_disc_pack(mset{1});
        assert(lsim_disc_domain('check', p, diag.domain{k}, 1e-10), 'model left the domain');
        assert(all(isfinite(lsim_disc_score(p, data.Y, lsim_fb_constants()))), 'non-finite score');
        p0 = lsim_disc_pack(init{k});
        assert(all(p.transitions_matrices(p0.transitions_matrices == 0) == 0), 'structural zero changed');
        assert(all(p.pi_0(p0.pi_0 == 0) == 0), 'structural zero changed');
    end
end
fprintf('%s\n   iterations %d, J %.4f -> %.4f , accepted steps %d/%d, checkpoint itr %d\n   stop: %s\n', ...
    title, diag.iterations_run, J(1), J(end), nnz(hist.accepted(2:end, :)), numel(hist.accepted(2:end, :)), ...
    diag.checkpoint_iteration, diag.stop_reason);
if any(~isnan(hist.val_balanced_accuracy))
    fprintf('   validation balanced accuracy by iteration: %s\n', mat2str(hist.val_balanced_accuracy', 3));
end
fprintf('   train balanced accuracy: %.3f -> %.3f\n', hist.train_balanced_accuracy(1), hist.train_balanced_accuracy(end));
end
