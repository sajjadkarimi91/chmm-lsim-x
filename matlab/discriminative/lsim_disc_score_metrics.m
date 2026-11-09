function m = lsim_disc_score_metrics(L, label, log_prior, score_scale, positive)
%LSIM_DISC_SCORE_METRICS Every classification metric from an N x M score matrix.
%
%   m = lsim_disc_score_metrics(L, label, log_prior, score_scale, positive)
%
% L : N x M LSIM scores (lsim_disc_score / lsim_discriminative_predict), label : N x 1 in 1..M,
% log_prior : M x 1 ([] = uniform), score_scale κ (default 1), positive class (default 2).
% m.balanced_accuracy / .macro_f1 / .accuracy : argmax_k(κ L + log pi) rule (lsim_disc_metrics);
% M = 2: .auroc, .auprc, .max_f1(_threshold), .ba_best_threshold (+ .best_ba_threshold) of the
%        detection score (lsim_disc_detection_score);
% M > 2: .auroc / .auprc = one-vs-rest macro averages over the normalized scores; max-F1 and the
%        best-threshold BA are NaN.
[N, M] = size(L);
if nargin < 3 || isempty(log_prior), log_prior = log(ones(M, 1)/M); end
if nargin < 4 || isempty(score_scale), score_scale = 1; end
if nargin < 5 || isempty(positive), positive = 2; end
label = label(:);
Z = L;
if score_scale ~= 1
    Z = score_scale*Z;
end
Z = Z + log_prior(:)';
[~, pred] = max(Z, [], 2);
c = lsim_disc_metrics(label, pred, M);
m = struct('balanced_accuracy', c.balanced_accuracy, 'macro_f1', c.macro_f1, 'accuracy', c.accuracy, ...
    'auroc', NaN, 'auprc', NaN, 'max_f1', NaN, 'max_f1_threshold', NaN, 'ba_best_threshold', NaN, ...
    'best_ba_threshold', NaN);
if M == 2
    b = lsim_disc_binary_metrics(lsim_disc_detection_score(L, log_prior, score_scale, positive), label == positive);
    for f = {'auroc', 'auprc', 'max_f1', 'max_f1_threshold', 'ba_best_threshold', 'best_ba_threshold'}
        m.(f{1}) = b.(f{1});
    end
else
    Zf = Z;
    Zf(isnan(Zf)) = -Inf;
    Pn = exp(Zf - max(Zf, [], 2));
    Pn = Pn./sum(Pn, 2);
    a = []; p = [];
    for k = 1:M
        if any(label == k) && any(label ~= k)
            b = lsim_disc_binary_metrics(Pn(:, k), label == k);
            a(end+1) = b.auroc; %#ok<AGROW>
            p(end+1) = b.auprc; %#ok<AGROW>
        end
    end
    m.auroc = mean(a);
    m.auprc = mean(p);
end
if N == 0
    m.balanced_accuracy = NaN;
end
end
