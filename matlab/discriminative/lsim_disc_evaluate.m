function ev = lsim_disc_evaluate(models, sets, options)
%LSIM_DISC_EVALUATE Score models on several data sets and report every metric (threshold tuned on one).
%
%   ev = lsim_disc_evaluate(models, sets, options)
%
% models  : M x 1 cell of em_lsim init_model structs (e.g. em_lsim_discriminative output)
% sets    : struct of data sets in the class-structured format of em_lsim_discriminative, e.g.
%           struct('train', {tr_obs}, 'val', {va_obs}, 'test', {te_obs}); every field is an M x 1
%           cell of C x N_i cells (labels = cell index)
% options : .tune_on ('val'), .threshold_criterion ('ba' | 'f1'), .log_prior ([] = uniform),
%           .score_scale (κ, 1), .positive (2), .fb_constants (lsim_fb_constants()),
%           .parallel (false: parfor over sequences)
% ev.<set> : lsim_disc_score_metrics fields; for M = 2 also *_at_tuned = lsim_disc_threshold_metrics at
%            the threshold tuned on options.tune_on and applied unchanged to every set.
% ev.threshold, ev.scores.<set> (N x M), ev.labels.<set>. Python twin:
% chmm_lsim.discriminative.evaluate_models.
if nargin < 3, options = struct(); end
d = struct('tune_on', 'val', 'threshold_criterion', 'ba', 'log_prior', [], 'score_scale', 1, 'positive', 2, ...
    'fb_constants', lsim_fb_constants(), 'parallel', false);
f = fieldnames(d);
for j = 1:numel(f)
    if ~isfield(options, f{j}), options.(f{j}) = d.(f{j}); end
end
M = numel(models);
if isempty(options.log_prior), options.log_prior = log(ones(M, 1)/M); end
fbc = options.fb_constants;
fbc.parallel = logical(options.parallel);
names = fieldnames(sets);
ev = struct('threshold', [], 'scores', struct(), 'labels', struct());
packed = cellfun(@lsim_disc_pack, models(:), 'UniformOutput', false);
for j = 1:numel(names)
    data = lsim_disc_prepare_data(sets.(names{j}), names{j});
    L = zeros(numel(data.Y), M);
    for k = 1:M
        L(:, k) = lsim_disc_score(packed{k}, data.Y, fbc);
    end
    ev.scores.(names{j}) = L;
    ev.labels.(names{j}) = data.label;
end
thr = [];
if M == 2 && isfield(sets, options.tune_on)
    s = lsim_disc_detection_score(ev.scores.(options.tune_on), options.log_prior, options.score_scale, options.positive);
    thr = lsim_disc_tune_threshold(s, ev.labels.(options.tune_on) == options.positive, options.threshold_criterion);
end
ev.threshold = thr;
for j = 1:numel(names)
    L = ev.scores.(names{j});
    y = ev.labels.(names{j});
    m = lsim_disc_score_metrics(L, y, options.log_prior, options.score_scale, options.positive);
    if ~isempty(thr)
        t = lsim_disc_threshold_metrics(lsim_disc_detection_score(L, options.log_prior, options.score_scale, ...
            options.positive), y == options.positive, thr);
        for g = {'balanced_accuracy', 'f1', 'sensitivity', 'specificity', 'precision', 'tp', 'fp', 'fn', 'tn'}
            m.([g{1} '_at_tuned']) = t.(g{1});
        end
    end
    ev.(names{j}) = m;
end
end
