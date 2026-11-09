function s = lsim_disc_detection_score(L, log_prior, score_scale, positive)
%LSIM_DISC_DETECTION_SCORE Two-class detection score s = κ (L_pos - L_neg) + log pi_pos - log pi_neg.
%
%   s = lsim_disc_detection_score(L, log_prior, score_scale, positive)
%
% L : N x 2 LSIM scores; log_prior ([] = uniform), score_scale κ (default 1), positive class index
% (default 2, e.g. seizure). AUROC / AUPRC of s do not depend on κ or the priors.
if size(L, 2) ~= 2
    error('lsim_disc_detection_score:needTwoClasses', 'the detection score needs exactly two classes.');
end
if nargin < 2 || isempty(log_prior), log_prior = [0; 0]; end
if nargin < 3 || isempty(score_scale), score_scale = 1; end
if nargin < 4 || isempty(positive), positive = 2; end
neg = 3 - positive;
s = score_scale*(L(:, positive) - L(:, neg)) + (log_prior(positive) - log_prior(neg));
end
