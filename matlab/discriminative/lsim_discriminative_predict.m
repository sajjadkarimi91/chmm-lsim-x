function [scores, predicted, normalized_scores] = lsim_discriminative_predict(models, observations, options)
%LSIM_DISCRIMINATIVE_PREDICT Class scores and predictions from M LSIM models.
%
%   [scores, predicted, normalized_scores] = lsim_discriminative_predict(models, observations, options)
%
% models       : M x 1 cell of em_lsim init_model-format structs (e.g. the output
%                of em_lsim_discriminative, or em_lsim models wrapped likewise)
% observations : C x N cell, one sequence per column ({c,n} is d(c) x T_n)
% options      : .log_prior (M x 1, default uniform), .score_scale (κ, default 1:
%                the MMI decision rule argmax_k κ L + log pi), .fb_constants
%
% scores(n,k)  : LSIM score of sequence n under model k (lsim_disc_score):
%                the exact log-likelihood when C = 1, an approximate
%                factorized-filter score when C > 1, times κ.
% predicted(n) : argmax_k scores(n,k) + log_prior(k)
% normalized_scores : softmax over classes of scores + log_prior. For exact
%                likelihoods and correct priors these are Bayes class posteriors;
%                for C > 1 they are NORMALIZED SCORES, not calibrated
%                probabilities, and are labeled as such.

if nargin < 3
    options = struct();
end
M = numel(models);
if ~isfield(options, 'log_prior') || isempty(options.log_prior)
    options.log_prior = log(ones(M, 1)/M);
end
if isfield(options, 'score_normalization')
    error('lsim_discriminative_predict:removedOption', ...
        'options.score_normalization is not supported; use options.score_scale.');
end
if ~isfield(options, 'score_scale') || isempty(options.score_scale)
    options.score_scale = 1;
end
if ~isfield(options, 'fb_constants')
    options.fb_constants = lsim_fb_constants();
end

data = lsim_disc_prepare_data({observations}, 'observations');
scores = zeros(numel(data.Y), M);
for k = 1:M
    scores(:, k) = lsim_disc_score(lsim_disc_pack(models{k}), data.Y, options.fb_constants);
end
if options.score_scale ~= 1
    scores = options.score_scale*scores;
end
Z = scores + options.log_prior(:)';
[~, predicted] = max(Z, [], 2);
mx = max(Z, [], 2);
normalized_scores = exp(Z - mx);
normalized_scores = normalized_scores ./ sum(normalized_scores, 2);

end
