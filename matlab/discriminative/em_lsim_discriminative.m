function [models, history, diagnostics] = em_lsim_discriminative( ...
    class_observations, channel_num_states, num_gmm_component, max_iteration, extra, init_models)
%EM_LSIM_DISCRIMINATIVE Discriminative (MMI) training of class-specific LSIMs.
%
%   [models, history, diagnostics] = em_lsim_discriminative( ...
%       class_observations, channel_num_states, num_gmm_component, max_iteration, extra, init_models)
%
% class_observations : M x 1 cell (M >= 2); class_observations{i} is a C x N_i cell in em_lsim's
%                      format ({c,n} is d(c) x T_n, NaN = missing, T_n may vary)
% channel_num_states, num_gmm_component : C x 1, as for em_lsim
% max_iteration      : number of iterations; [] = default (published recipe 8, legacy 40)
% extra              : options, validated by lsim_disc_options (see `help lsim_disc_options` for
%                      the full table); unknown fields raise an error
% init_models        : optional M x 1 cell of em_lsim init_model structs (normally EM models);
%                      omitted = em_lsim on each class's own training data
%
% models{k}   : em_lsim init_model-compatible struct of class k (plus .meta) at the returned
%               iterate: the validation checkpoint (extra.selection_metric, default the argmax
%               balanced accuracy) when validation data are given, else the final iterate.
%               Usable with forward_backward_lsim / em_lsim / lsim_discriminative_predict.
% history     : one struct per run, one row per iteration (row 1 = projected initialization);
%               see "HISTORY" below
% diagnostics : stop reason, checkpoint, final / initial / generative models, domain, options,
%               timing
%
% THE OBJECTIVE (lsim_disc_objective): maximum mutual information (Bahl et al. 1986; Woodland &
% Povey 2002; Povey et al. 2008). L(n,k) = LSIM score of sequence n under model k: the exact
% log-likelihood for C = 1, an approximate factorized-filter score for C > 1 (Karimi &
% Shamsollahi, IEEE TSP 2020). y_n = class of n, pi = fixed class priors (extra.log_prior),
%     J = sum_n a_n log q(n,y_n),  q = softmax(κ L + log pi),  a_n = 1/(M N_{y_n}),
% with score scale κ = extra.score_scale. Classification rule: argmax_k κ L(n,k) + log pi_k.
%
% THE UPDATE. Every iteration computes, for every model k and every training sequence n, the
% expected sufficient statistics UNDER the current model k (lsim_disc_sequence_stats: the
% E-step of em_lsim, including the coupling / transition source-selection counts of Karimi &
% Shamsollahi, IEEE TPAMI 2023), weighted by the objective's numerator / denominator weights
% (lsim_disc_weighted_counts). All models are updated from the same scores (Jacobi).
%   mmi_recipe 'published' (default): the full update every iteration, no step search
%     (lsim_disc_published_update): Gaussian EBW with D = max(ebw_E gamma_den, 2 D_min),
%     I-smoothing (tau frame units, ML or previous back-off), weak-sense updates of the discrete
%     blocks (lsim_disc_weak_sense); J is logged only and may decrease.
%   mmi_recipe 'legacy': extended Baum-Welch proposal (lsim_disc_ebw_proposal) with
%     D = max(damping_E * gamma_den, 2 D_min); all models move together along the segment to
%     their proposals, halving the step until the measured objective does not decrease, so J
%     is non-decreasing. The damping multiplier E (damping_E) halves after a full accepted step
%     (floor damping_E_min) and is multiplied by 4 after a rejection.
% Every model stays in a compact, training-data-derived domain (lsim_disc_domain): probability
% floor, mean box, variance bounds. For C > 1 the statistics come from an approximate smoother,
% so the direction is a surrogate for the score's gradient (tests/check_disc_gradients.m).
%
% VALIDATION (extra.validation_observations, same format as class_observations, e.g. a
% subject-wise split): every iteration records the validation balanced accuracy of
% argmax(κ L + log pi), macro-F1, AUROC, AUPRC, max-F1, best-threshold balanced accuracy and the
% validation objective (two classes: detection score κ (L_pos - L_neg) + log prior ratio,
% extra.positive_class = 2 by default; lsim_disc_score_metrics). The checkpoint is the iteration
% with the highest extra.selection_metric (ties: higher validation objective, then earlier);
% extra.early_stopping_patience = k stops once it has not improved by more than
% extra.early_stopping_min_delta for k iterations. Metric functions for any scores:
% lsim_disc_binary_metrics, lsim_disc_score_metrics, lsim_disc_tune_threshold,
% lsim_disc_threshold_metrics, lsim_disc_evaluate (train / val / test in one call).
%
% HISTORY fields (rows = iterations 0..I; M columns = models): objective, step_tau, backtracks,
% accepted, damping_E, D_max, D_median, mean_own_score (own-class log-likelihood per model),
% mean_other_score, frac_prob_at_floor, frac_var_at_lower, frac_var_at_cap, frac_mean_at_bound,
% train_balanced_accuracy, train_auroc, train_auprc, train_max_f1, val_balanced_accuracy,
% val_macro_f1, val_objective, val_auroc, val_auprc, val_max_f1, val_ba_best_threshold,
% val_selection_value, time_s, frac_q_confident_wrong / _uncertain / _right (of q(n,y_n));
% published recipe only: projection_changed, weak_sense_iterations, weak_sense_monotone.
% Fields that do not apply to the run's recipe are NaN.
%
% See also lsim_disc_options, lsim_discriminative_predict, lsim_disc_evaluate, em_lsim.

if nargin < 5 || isempty(extra)
    extra = struct();
end
opts = lsim_disc_options(extra);
published = strcmp(opts.mmi_recipe, 'published');
if nargin < 4 || isempty(max_iteration)
    if published
        max_iteration = 8;
    else
        max_iteration = 40;
    end
end

data = lsim_disc_prepare_data(class_observations);
M = data.M;
C = data.C;
if M < 2
    error('em_lsim_discriminative:needTwoClasses', 'Discriminative training needs M >= 2 classes.');
end
if any(data.N_per_class < 1)
    error('em_lsim_discriminative:emptyClass', 'Every class needs at least one training sequence.');
end
channel_num_states = channel_num_states(:);
num_gmm_component = num_gmm_component(:);
if numel(channel_num_states) ~= C || numel(num_gmm_component) ~= C
    error('em_lsim_discriminative:badSizes', ...
        'channel_num_states and num_gmm_component must have one entry per channel (C = %d).', C);
end
if isempty(opts.log_prior)
    opts.log_prior = log(ones(M, 1)/M);
end
if numel(opts.log_prior) ~= M
    error('em_lsim_discriminative:badPrior', 'log_prior must have one entry per class (M = %d).', M);
end

N = numel(data.Y);
label = data.label;
Tn = data.T;
fbc = opts.fb_constants;
if opts.parallel
    % per-sequence statistics and scores are computed in parfor and accumulated serially in
    % sequence order, so results are identical to the serial path
    fbc.parallel = true;
end

% ---------------------------------------------------------------- initialization
t_start = tic;
generative_models = cell(M, 1);
if nargin >= 6 && ~isempty(init_models)
    if numel(init_models) ~= M
        error('em_lsim_discriminative:badInit', 'init_models must have one model per class (M = %d).', M);
    end
    generative_models = init_models(:);
else
    for k = 1:M
        if opts.verbose
            fprintf('em_lsim_discriminative: generative initialization of class %d/%d (em_lsim)\n', k, M);
        end
        gx = opts.generative_extra;
        gx.fb_constants = fbc;
        [pi0, theta, A, gmm] = em_lsim(data.class_obs{k}, channel_num_states, ...
            num_gmm_component, opts.generative_max_iteration, gx);
        generative_models{k} = struct('pi_0_lsim', {pi0}, 'coupling_tetha_IM', theta, ...
            'transition_matrices_IM_cell', {A}, 'gmm_para_lsim', {gmm});
    end
end

p = cell(M, 1);
dom = cell(M, 1);
for k = 1:M
    pk = lsim_disc_pack(generative_models{k});
    if ~pk.sigma_diag
        error('em_lsim_discriminative:fullCovarianceUnsupported', ...
            ['Full-covariance emissions are not supported by the discriminative trainer ', ...
            '(only diagonal covariances are implemented and verified).']);
    end
    if pk.C ~= C || any(pk.channel_num_states ~= channel_num_states) || ...
            any(pk.num_gmm_component ~= num_gmm_component) || any(pk.dim_observation ~= data.dim)
        error('em_lsim_discriminative:initMismatch', ...
            'Initial model %d does not match the data / channel_num_states / num_gmm_component.', k);
    end
    dom{k} = lsim_disc_domain('build', data, pk, opts);
    p{k} = lsim_disc_domain('project', pk, dom{k});
end
initial_models = p;
init_time = toc(t_start);

cfg = struct('objective', opts.objective, 'log_prior', opts.log_prior);
cfg.score_scale = opts.score_scale;
cfg.mmi_recipe = opts.mmi_recipe;
cfg.N_stat = N;      % published: weights a_n N (mean sequence weight 1, frame units)
ebw_opts = struct('update_blocks', {opts.update_blocks});
if published
    pub_opts = struct('ebw_E', opts.ebw_E, 'ismooth_tau', opts.ismooth_tau, ...
        'ismooth_backoff', opts.ismooth_backoff, 'ismooth_tau_discrete', opts.ismooth_tau_discrete, ...
        'weak_sense_iters', opts.weak_sense_iters, 'weak_sense_tol', opts.weak_sense_tol);
    pub_opts.update_blocks = opts.update_blocks;
end
monotone = ~published;     % legacy: the measured objective can only increase

has_val = ~isempty(opts.validation_observations);
if ~has_val && (~isempty(opts.early_stopping_patience) || ~strcmp(opts.selection_metric, 'balanced_accuracy'))
    error('em_lsim_discriminative:needValidation', ...
        'selection_metric / early_stopping_patience need extra.validation_observations.');
end
if any(strcmp(opts.selection_metric, {'max_f1', 'ba_best_threshold'})) && M ~= 2
    error('em_lsim_discriminative:needTwoClasses', 'selection_metric ''%s'' needs exactly two classes.', ...
        opts.selection_metric);
end
if opts.positive_class > M
    error('em_lsim_discriminative:badPositiveClass', 'positive_class must be a class index (1..%d).', M);
end
cfg.positive_class = opts.positive_class;
cfg.selection_metric = opts.selection_metric;
cfg.es_min_delta = opts.early_stopping_min_delta;
es = struct('best', -Inf, 'since', 0);
if has_val
    vdata = lsim_disc_prepare_data(opts.validation_observations, 'extra.validation_observations');
    if vdata.M ~= M || vdata.C ~= C || any(vdata.dim ~= data.dim)
        error('em_lsim_discriminative:badValidation', 'Validation data must match the training classes and channels.');
    end
end

% ---------------------------------------------------------------- iteration 0
L = score_all(p, data.Y, fbc);
[J, parts] = lsim_disc_objective(L, label, Tn, cfg);
if ~isfinite(J)
    error('em_lsim_discriminative:nonfiniteInit', ...
        'The (projected) initial models give a non-finite objective; some training sequence has zero score.');
end
if published
    damping_E = NaN(M, 1);   % the published recipe has no adaptive step damping (fixed ebw_E)
else
    damping_E = opts.damping_E*ones(M, 1);
end
history = init_history(M, max_iteration);
history = record(history, 1, J, parts, L, label, M, opts, NaN(M, 1), zeros(M, 1), true(M, 1), ...
    damping_E, cell(M, 1), p, dom, 0);
best = struct('iteration', 0, 'p', {p}, 'value', -Inf, 'val_obj', -Inf);
if has_val
    [history, best, es] = validate_checkpoint(history, 1, 0, p, vdata, fbc, cfg, best, M, es);
end
if opts.verbose
    fprintf('itr %3d  J = %.6f\n', 0, J);
end
call_hook(opts, 0, p, J, parts, NaN(M, 1), zeros(M, 1), true(M, 1), damping_E, cell(M, 1));

% ---------------------------------------------------------------- iterations
stop_reason = 'max_iteration';
small_count = 0;
stalled = false;
itr = 0;
for itr = 1:max_iteration
    t_itr = tic;
    J_old = J;
    tau = NaN(M, 1);
    backtracks = zeros(M, 1);
    accepted = false(M, 1);
    pinfo = cell(M, 1);
    props = cell(M, 1);
    W = parts.W;
    for k = 1:M
        % statistics of every model under its CURRENT parameters, weighted by the objective
        % (all models from the same scores: Jacobi)
        [num, den] = lsim_disc_weighted_counts(p{k}, data, W.num(:, k), W.den(:, k), fbc);
        if published
            [props{k}, pinfo{k}] = lsim_disc_published_update(p{k}, num, den, dom{k}, pub_opts);
        else
            [props{k}, pinfo{k}] = lsim_disc_ebw_proposal(p{k}, num, den, dom{k}, damping_E(1), ebw_opts);
        end
    end
    if published
        p = props;          % the full update, objective monitored only
        tau(:) = 1;
        accepted(:) = true;
    else
        % legacy: step search on the measured training objective
        A_old = J;
        for bt = 0:opts.max_backtracks
            step = 0.5^bt;
            cand = cell(M, 1);
            for k = 1:M
                cand{k} = interpolate(p{k}, props{k}, step, opts.update_blocks);
            end
            Lc = score_all(cand, data.Y, fbc);
            A_new = lsim_disc_objective(Lc, label, Tn, cfg);
            backtracks(:) = bt;
            if isfinite(A_new) && A_new >= A_old
                p = cand;
                tau(:) = step;
                accepted(:) = true;
                break
            end
        end
        [damping_E_new, stalled] = adapt_damping_E(damping_E(1), accepted(1), tau(1), opts);
        damping_E(:) = damping_E_new;
    end

    L = score_all(p, data.Y, fbc);
    [J, parts] = lsim_disc_objective(L, label, Tn, cfg);
    history = record(history, itr+1, J, parts, L, label, M, opts, tau, backtracks, accepted, ...
        damping_E, pinfo, p, dom, toc(t_itr));
    if has_val
        [history, best, es] = validate_checkpoint(history, itr+1, itr, p, vdata, fbc, cfg, best, M, es);
    end
    if opts.verbose
        fprintf('itr %3d  J = %.6f  step %s  damping_E %s\n', itr, J, mat2str(tau(1), 3), mat2str(damping_E(1), 3));
    end
    call_hook(opts, itr, p, J, parts, tau, backtracks, accepted, damping_E, pinfo);

    if monotone
        % accepted steps were measured, so the recomputed objective cannot be lower
        if J < J_old
            error('em_lsim_discriminative:objectiveDecreased', ...
                'Objective decreased from %.12g to %.12g after accepted steps.', J_old, J);
        end
    end
    if stalled
        stop_reason = 'stalled: no step along the EBW segment raised the measured objective';
        break
    end
    if ~published   % the published recipe runs the requested number of iterations
        rel = (J - J_old)/max(1, abs(J_old));
        if rel < opts.tol_rel
            small_count = small_count + 1;
        else
            small_count = 0;
        end
        if small_count >= opts.patience
            stop_reason = sprintf('converged: relative change < %g for %d iterations', opts.tol_rel, opts.patience);
            break
        end
    end
    if has_val && ~isempty(opts.early_stopping_patience) && es.since >= opts.early_stopping_patience
        stop_reason = sprintf('early stopping: validation %s did not improve by more than %g for %d iterations', ...
            opts.selection_metric, opts.early_stopping_min_delta, opts.early_stopping_patience);
        break
    end
end
history = trim_history(history, itr+1);

% ---------------------------------------------------------------- outputs
if has_val
    returned = best.p;
    checkpoint_iteration = best.iteration;
    checkpoint_rule = sprintf('validation %s, ties by validation objective, then earliest', opts.selection_metric);
else
    returned = p;
    checkpoint_iteration = itr;
    checkpoint_rule = 'final iterate (no validation data supplied)';
end
semantics = score_semantics(C, fbc);
models = cell(M, 1);
final_models = cell(M, 1);
for k = 1:M
    models{k} = lsim_disc_unpack(returned{k});
    models{k}.meta = struct('class_index', k, 'objective', opts.objective, ...
        'iteration', checkpoint_iteration, 'score_semantics', semantics.score, ...
        'direction_semantics', semantics.direction);
    final_models{k} = lsim_disc_unpack(p{k});
end
L_ret = score_all(returned, data.Y, fbc);
diagnostics = struct();
diagnostics.objective = opts.objective;
diagnostics.score_semantics = semantics.score;
diagnostics.direction_semantics = semantics.direction;
diagnostics.stop_reason = stop_reason;
diagnostics.iterations_run = itr;
diagnostics.checkpoint_iteration = checkpoint_iteration;
diagnostics.checkpoint_rule = checkpoint_rule;
diagnostics.checkpoint_validation_value = NaN;
if has_val
    diagnostics.checkpoint_validation_value = best.value;
end
diagnostics.returned_objective_recomputed = lsim_disc_objective(L_ret, label, Tn, cfg);
diagnostics.returned_objective_in_history = history.objective(checkpoint_iteration+1);
diagnostics.final_models = final_models;
diagnostics.initial_models = cellfun(@lsim_disc_unpack, initial_models, 'UniformOutput', false);
diagnostics.generative_models = generative_models;
diagnostics.domain = dom;
diagnostics.options = opts;
diagnostics.init_time_s = init_time;
diagnostics.total_time_s = toc(t_start);
end

% =====================================================================
function call_hook(opts, itr, p, J, parts, tau, backtracks, accepted, damping_E, pinfo)
% Optional logging hook (opts.iteration_hook): receives the packed models and step diagnostics
% after the iteration; nothing it returns is used.
if isempty(opts.iteration_hook)
    return
end
info = struct('J', J, 'tau', tau, 'backtracks', backtracks, 'accepted', accepted, 'damping_E', damping_E);
info.parts = parts;
info.proposal_info = pinfo;
opts.iteration_hook(itr, p, info);
end

function L = score_all(p, Y, fbc)
M = numel(p);
L = zeros(numel(Y), M);
for k = 1:M
    L(:, k) = lsim_disc_score(p{k}, Y, fbc);
end
end

function c = interpolate(p, q, tau, update_blocks)
% the segment p + tau (q - p): convex combination, so it stays in the (convex) domain.
% Families frozen through update_blocks are copied exactly.
c = p;
f = {'pi_0', 'transitions_matrices', 'coupling_tetha_IM', 'P_all', 'mu_all', 'sigma_all'};
fam = {'pi', 'A', 'theta', 'w', 'mu', 'var'};
for j = 1:numel(f)
    if ~isempty(update_blocks) && ~ismember(fam{j}, update_blocks)
        continue
    end
    c.(f{j}) = (1-tau)*p.(f{j}) + tau*q.(f{j});
end
end

function [damping_E, stalled] = adapt_damping_E(damping_E, accepted, tau, opts)
% The EBW damping multiplier E is adapted, never assumed: a FULL accepted step (tau = 1) halves
% it, down to damping_E_min (the step was evidently conservative); a rejected iteration
% multiplies it by 4, which turns the proposal towards the statistics' gradient direction.
% D never drops below 2 D_min, so positivity holds for every E, and the measured-objective
% check still decides acceptance. The run is stalled only once E is at its cap and still no
% point on the segment helped.
stalled = false;
if accepted
    if tau == 1
        damping_E = max(opts.damping_E_min, damping_E/2);
    end
else
    if damping_E >= opts.damping_E_max
        stalled = true;
    end
    damping_E = min(opts.damping_E_max, 4*damping_E);
end
end

function h = init_history(M, max_iteration)
n = max_iteration + 1;
h.iteration = (0:max_iteration)';
h.objective = NaN(n, 1);
h.step_tau = NaN(n, M);
h.backtracks = NaN(n, M);
h.accepted = false(n, M);
h.damping_E = NaN(n, M);
h.D_max = NaN(n, M);
h.D_median = NaN(n, M);
h.mean_own_score = NaN(n, M);
h.mean_other_score = NaN(n, M);
h.frac_prob_at_floor = NaN(n, M);
h.frac_var_at_lower = NaN(n, M);
h.frac_var_at_cap = NaN(n, M);
h.frac_mean_at_bound = NaN(n, M);
h.train_balanced_accuracy = NaN(n, 1);
h.val_balanced_accuracy = NaN(n, 1);
h.val_macro_f1 = NaN(n, 1);
h.val_objective = NaN(n, 1);
h.val_auroc = NaN(n, 1);
h.val_auprc = NaN(n, 1);
h.val_max_f1 = NaN(n, 1);
h.val_ba_best_threshold = NaN(n, 1);
h.val_selection_value = NaN(n, 1);
h.train_auroc = NaN(n, 1);
h.train_auprc = NaN(n, 1);
h.train_max_f1 = NaN(n, 1);
h.time_s = NaN(n, 1);
h.frac_q_confident_wrong = NaN(n, 1);
h.frac_q_uncertain = NaN(n, 1);
h.frac_q_confident_right = NaN(n, 1);
h.projection_changed = NaN(n, M);
h.weak_sense_iterations = NaN(n, M);
h.weak_sense_monotone = true(n, M);
h.note = ['objective: training objective after each iteration (non-decreasing for the legacy ', ...
    'recipe); val_*: validation metrics, used only for checkpoint selection'];
end

function h = record(h, row, J, parts, L, label, M, opts, tau, backtracks, accepted, damping_E, pinfo, p, dom, secs)
h.objective(row) = J;
h.step_tau(row, :) = tau';
h.backtracks(row, :) = backtracks';
h.accepted(row, :) = accepted';
h.damping_E(row, :) = damping_E';
for k = 1:M
    h.mean_own_score(row, k) = mean(L(label == k, k));
    h.mean_other_score(row, k) = mean(L(label ~= k, k));
    b = lsim_disc_domain('at_bounds', p{k}, dom{k});
    h.frac_prob_at_floor(row, k) = b.prob_at_floor;
    h.frac_var_at_lower(row, k) = b.var_at_lower;
    h.frac_var_at_cap(row, k) = b.var_at_cap;
    h.frac_mean_at_bound(row, k) = b.mean_at_bound;
    if ~isempty(pinfo{k})
        h.D_max(row, k) = pinfo{k}.max_D;
        h.D_median(row, k) = pinfo{k}.median_D;
        if isfield(pinfo{k}, 'projection_changed_total')
            h.projection_changed(row, k) = pinfo{k}.projection_changed_total;
            h.weak_sense_iterations(row, k) = pinfo{k}.weak_sense_max_iterations;
            h.weak_sense_monotone(row, k) = pinfo{k}.weak_sense_monotone;
        end
    end
end
if isfield(parts, 'frac_q_uncertain')
    h.frac_q_confident_wrong(row) = parts.frac_q_confident_wrong;
    h.frac_q_uncertain(row) = parts.frac_q_uncertain;
    h.frac_q_confident_right(row) = parts.frac_q_confident_right;
end
h.train_balanced_accuracy(row) = balanced_accuracy(L, label, opts.log_prior, decision_scale(opts), M);
tm = lsim_disc_score_metrics(L, label, opts.log_prior, decision_scale(opts), opts.positive_class);
h.train_auroc(row) = tm.auroc;
h.train_auprc(row) = tm.auprc;
h.train_max_f1(row) = tm.max_f1;
h.time_s(row) = secs;
end

function s = decision_scale(opts)
% the decision rule argmax_k(κ L + log pi), κ = score_scale
s = opts.score_scale;
end

function ba = balanced_accuracy(L, label, log_prior, kap, M)
Ln = L;
if kap ~= 1
    Ln = kap*Ln;
end
[~, pred] = max(Ln + log_prior', [], 2);
m = lsim_disc_metrics(label, pred, M);
ba = m.balanced_accuracy;
end

function h = trim_history(h, n)
f = fieldnames(h);
for j = 1:numel(f)
    v = h.(f{j});
    if ~ischar(v) && size(v, 1) > n
        h.(f{j}) = v(1:n, :);
    end
end
end

function [h, best, es] = validate_checkpoint(h, row, itr, p, vdata, fbc, cfg, best, M, es)
% validation metrics of the iteration, the checkpoint (cfg.selection_metric; ties: higher validation
% objective, then earlier) and the early-stopping counter (improvement > cfg.es_min_delta)
Lv = score_all(p, vdata.Y, fbc);
Lvn = Lv;
kap = 1;
if cfg.score_scale ~= 1
    Lvn = cfg.score_scale*Lvn;   % decision rule argmax(κ L + log pi)
    kap = cfg.score_scale;
end
[~, pred] = max(Lvn + cfg.log_prior', [], 2);
m = lsim_disc_metrics(vdata.label, pred, M);
cfg_v = cfg;
cfg_v.N_stat = numel(vdata.Y);
Jv = lsim_disc_objective(Lv, vdata.label, vdata.T, cfg_v);
sm = lsim_disc_score_metrics(Lv, vdata.label, cfg.log_prior, kap, cfg.positive_class);
h.val_balanced_accuracy(row) = m.balanced_accuracy;
h.val_macro_f1(row) = m.macro_f1;
h.val_objective(row) = Jv;
h.val_auroc(row) = sm.auroc;
h.val_auprc(row) = sm.auprc;
h.val_max_f1(row) = sm.max_f1;
h.val_ba_best_threshold(row) = sm.ba_best_threshold;
switch cfg.selection_metric
    case 'balanced_accuracy'
        v = m.balanced_accuracy;
    case 'objective'
        v = Jv;
    otherwise
        v = sm.(cfg.selection_metric);
end
if ~isfinite(v)
    v = -Inf;
end
h.val_selection_value(row) = v;
if v > best.value || (v == best.value && Jv > best.val_obj)
    best.iteration = itr;
    best.p = p;
    best.value = v;
    best.val_obj = Jv;
end
if v > es.best + cfg.es_min_delta
    es.best = v;
    es.since = 0;
else
    es.since = es.since + 1;
end
end

function s = score_semantics(C, fbc)
if C == 1
    s.score = 'exact HMM log-likelihood (C = 1)';
    s.direction = sprintf(['exact gradient: posterior statistics of the exact C = 1 ', ...
        'forward-backward (%s), Fisher identity'], fbc.variant);
else
    s.score = ['approximate LSIM forward score: per-channel factorized (assumed-density) ', ...
        'filter, NOT the exact joint log-likelihood of the coupled model'];
    s.direction = sprintf(['surrogate direction: posterior statistics of the approximate ''%s'' ', ...
        'smoother, not the gradient of the score'], fbc.variant);
end
end
