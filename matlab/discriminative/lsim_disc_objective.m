function [J, parts] = lsim_disc_objective(L, label, T, cfg)
%LSIM_DISC_OBJECTIVE The MMI objective and its statistic weights.
%
%   [J, parts] = lsim_disc_objective(L, label, T, cfg)
%
% L     : N x M scores, L(n,k) = LSIM score of sequence n under model k (lsim_disc_score;
%         exact log-likelihood for C = 1, approximate factorized-filter score for C > 1;
%         Karimi & Shamsollahi 2020)
% label : N x 1 true class, T : N x 1 sequence lengths (not used by MMI)
% cfg   : .objective ('mmi'), .log_prior (M x 1, log pi), .score_scale (κ),
%         .mmi_recipe ('published' | 'legacy'), .N_stat (published: number of training
%         sequences N, the frame-unit scale of the weights)
%
% parts.W.num / parts.W.den (N x M, >= 0) are the per-sequence statistic weights of every
% model: the trainer multiplies sequence n's expected sufficient statistics under model k by
% W.num(n,k) (numerator) and W.den(n,k) (denominator) before accumulating them
% (lsim_disc_weighted_counts). They are the literature's uncanceled numerator / denominator
% weights.
%
% MMI   (Bahl et al. 1986; Woodland & Povey 2002; Povey et al. 2008)
%   q(n,k) = softmax_k(κ L(n,k) + log pi_k),  a_n = 1/(M N_{y_n})
%   J = sum_n a_n log q(n, y_n)
%   dJ/dL(n,k) = a_n κ (1[k = y_n] - q(n,k))
%   Weights (uncanceled): W.num(n,y) = c_n a_n, W.den(n,k) = c_n a_n q(n,k) for every k,
%     published: c_n = N (frame units: mean sequence weight 1, κ only inside q),
%     legacy:    c_n = κ (the exact gradient scale, W.num - W.den = dJ/dL).
%
% A non-finite log q(n,y_n) gives J = -Inf (a rejected trial point); -Inf COMPETING
% scores are tolerated and only make q = 0.
[N, M] = size(L);
label = label(:);
lp = cfg.log_prior(:)';
own = sub2ind([N, M], (1:N)', label);
if ~strcmp(cfg.objective, 'mmi')
    error('em_lsim_discriminative:badObjective', 'Unknown objective %s (only ''mmi'' is supported).', cfg.objective);
end
[J, parts] = mmi(L, label, cfg, lp, own, N, M);
end

% ---------------------------------------------------------------------
function [J, parts] = mmi(L, label, cfg, lp, own, N, M)
if cfg.score_scale ~= 1
    Z = cfg.score_scale*L + lp;
else
    Z = L + lp;                             % κ = 1: no multiplication
end
mx = max(Z, [], 2);
lse = mx + log(sum(exp(Z - mx), 2));
N_per_class = accumarray(label, 1, [M, 1]);
a = 1./(M*N_per_class(label));
terms = Z(own) - lse;
if any(~isfinite(terms))
    J = -Inf;
else
    J = sum(a.*terms);
end
q = exp(Z - lse);
q(~isfinite(q)) = 0;
switch cfg.mmi_recipe
    case 'published'
        c = cfg.N_stat*ones(N, 1);
    case 'legacy'
        c = ones(N, 1);
        if cfg.score_scale ~= 1
            c = cfg.score_scale*c;
        end
end
ca = c.*a;
W.num = zeros(N, M);
W.den = zeros(N, M);
for k = 1:M
    W.num(:, k) = ca.*(label == k);
    W.den(:, k) = ca.*q(:, k);
end
qy = q(own);
parts.W = W;
parts.q = q;
parts.a = a;
parts.log_posterior_own = terms;
parts.frac_q_confident_wrong = mean(qy < 0.01);
parts.frac_q_uncertain = mean(qy >= 0.01 & qy <= 0.99);
parts.frac_q_confident_right = mean(qy > 0.99);
end
