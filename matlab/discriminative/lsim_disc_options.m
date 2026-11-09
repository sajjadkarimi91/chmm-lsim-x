function opts = lsim_disc_options(extra)
%LSIM_DISC_OPTIONS Validated options of em_lsim_discriminative (MMI training).
%
%   opts = lsim_disc_options(extra)
%
% extra is a scalar struct of options; every field must be one of the options below and
% unknown fields raise an error (inputParser, never ignored). The objective is maximum mutual
% information ('mmi'); mmi_recipe selects the update.
%
% COMMON OPTIONS                                                                default
%   objective              'mmi'                                                'mmi'
%   log_prior              M x 1 fixed log class priors                         [] (uniform)
%   validation_observations M x 1 cell, enables the validation checkpoint       []
%   update_blocks          subset of {'pi','A','theta','w','mu','var'} to update   {} (all)
%   prob_floor             probability floor of the compact domain                  1e-4
%   mean_margin            mean box = [min - m*span, max + m*span] of the training data 0.5
%   var_lo_frac, var_hi_frac  variance bounds as fractions of the pooled training variance  1e-3, 10 (Inf allowed)
%   tol_rel, patience      legacy recipe: stop when the relative change of J is below
%                          tol_rel for `patience` iterations                       1e-6, 3
%   selection_metric       validation checkpoint metric: 'balanced_accuracy' (argmax rule) | 'auroc' |
%                          'auprc' | 'max_f1' | 'ba_best_threshold' | 'objective'  'balanced_accuracy'
%   positive_class         detection (positive) class for the two-class metrics      2
%   early_stopping_patience  stop when the validation selection metric has not improved by more
%                          than early_stopping_min_delta for this many iterations    [] (off)
%   early_stopping_min_delta                                                         0
%   fb_constants           forward-backward constants of the statistics (scores use the forward
%                          pass, identical for every variant)   lsim_fb_constants('poise') (raw POISE)
%   parallel               parfor over sequences                                    false
%   generative_max_iteration, generative_extra   em_lsim initialization when no init_models  100, struct
%   iteration_hook         function(itr, packed_models, info), logging only         []
%   verbose                                                                          1
%
% MMI OPTIONS
%   mmi_recipe             'published' (full EBW update with I-smoothing and weak-sense
%                          discrete updates, Woodland & Povey 2002) | 'legacy' (EBW proposal
%                          with step search and adaptive damping, no I-smoothing)  'published'
%   score_scale            score scale κ in q = softmax(κ L + log pi)               1
%   ebw_E                  fixed EBW damping multiplier E (published)               2
%   ismooth_tau            I-smoothing pseudo-counts, frame units (published)       100
%   ismooth_backoff        'ml' | 'previous' (published)                            'ml'
%   ismooth_tau_discrete   tau_d * p_ML added to discrete numerator counts (published)  0
%   weak_sense_iters, weak_sense_tol  inner weak-sense iterations (published)       100, 1e-10
%   damping_E              initial adaptive EBW damping multiplier E (legacy)       2
%   damping_E_min, damping_E_max  bounds of E (legacy)                               1e-2, 1e6
%   max_backtracks         step halvings before an iteration is rejected (legacy)   12
% Published-only options raise an error with mmi_recipe = 'legacy' and vice versa.
%
% Default max_iteration (when em_lsim_discriminative gets []): published 8, legacy 40.
if nargin < 1 || isempty(extra)
    extra = struct();
end
if ~isstruct(extra) || ~isscalar(extra)
    error('em_lsim_discriminative:badOptions', 'extra must be a scalar struct of options.');
end
objective = 'mmi';
if isfield(extra, 'objective')
    objective = extra.objective;
end
if ~(ischar(objective) || (isstring(objective) && isscalar(objective))) || ~strcmp(objective, 'mmi')
    error('em_lsim_discriminative:badObjective', ...
        'extra.objective must be ''mmi'' (maximum mutual information), the only objective of this toolbox.');
end

ip = inputParser;
ip.FunctionName = 'em_lsim_discriminative';
ip.KeepUnmatched = false;
ip.PartialMatching = false;
ip.StructExpand = true;
pos = @(x) isnumeric(x) && isscalar(x) && isreal(x) && isfinite(x) && x > 0;
nonneg = @(x) isnumeric(x) && isscalar(x) && isreal(x) && isfinite(x) && x >= 0;
addParameter(ip, 'objective', 'mmi', @(x) strcmp(x, 'mmi'));
addParameter(ip, 'log_prior', [], @(x) isempty(x) || (isnumeric(x) && isvector(x) && all(isfinite(x))));
addParameter(ip, 'validation_observations', [], @(x) isempty(x) || iscell(x));
addParameter(ip, 'update_blocks', {}, @(x) iscell(x) && all(cellfun(@(b) any(strcmp(b, {'pi', 'A', 'theta', 'w', 'mu', 'var'})), x)));
addParameter(ip, 'prob_floor', 1e-4, @(x) pos(x) && x < 1);
addParameter(ip, 'mean_margin', 0.5, nonneg);
addParameter(ip, 'var_lo_frac', 1e-3, nonneg);
addParameter(ip, 'var_hi_frac', 10, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(ip, 'tol_rel', 1e-6, nonneg);
addParameter(ip, 'patience', 3, @(x) pos(x) && x == round(x));
addParameter(ip, 'selection_metric', 'balanced_accuracy', @(x) any(strcmp(x, {'balanced_accuracy', 'auroc', ...
    'auprc', 'max_f1', 'ba_best_threshold', 'objective'})));
addParameter(ip, 'positive_class', 2, @(x) pos(x) && x == round(x));
addParameter(ip, 'early_stopping_patience', [], @(x) isempty(x) || (pos(x) && x == round(x)));
addParameter(ip, 'early_stopping_min_delta', 0, nonneg);
addParameter(ip, 'fb_constants', lsim_fb_constants('poise'), @isstruct);   % raw POISE (no refinement): fastest
addParameter(ip, 'parallel', false, @(x) islogical(x) || isequal(x, 0) || isequal(x, 1));
addParameter(ip, 'generative_max_iteration', 100, @(x) pos(x) && x == round(x));
addParameter(ip, 'generative_extra', struct('plot', 0, 'check_convergence', 1, 'sigma_diag', 1), @isstruct);
addParameter(ip, 'iteration_hook', [], @(x) isempty(x) || isa(x, 'function_handle'));
addParameter(ip, 'verbose', 1, @(x) isscalar(x));
addParameter(ip, 'mmi_recipe', 'published', @(x) any(strcmp(x, {'published', 'legacy'})));
addParameter(ip, 'score_scale', 1, pos);
addParameter(ip, 'ebw_E', 2, pos);
addParameter(ip, 'ismooth_tau', 100, nonneg);
addParameter(ip, 'ismooth_backoff', 'ml', @(x) any(strcmp(x, {'ml', 'previous'})));
addParameter(ip, 'ismooth_tau_discrete', 0, nonneg);
addParameter(ip, 'weak_sense_iters', 100, @(x) pos(x) && x == round(x));
addParameter(ip, 'weak_sense_tol', 1e-10, pos);
addParameter(ip, 'damping_E', 2, pos);
addParameter(ip, 'damping_E_min', 1e-2, pos);
addParameter(ip, 'damping_E_max', 1e6, pos);
addParameter(ip, 'max_backtracks', 12, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x == round(x));
try
    parse(ip, extra);
catch err
    if strcmp(err.identifier, 'MATLAB:InputParser:UnmatchedParameter')
        error('em_lsim_discriminative:unknownOption', '%s (see help lsim_disc_options).', err.message);
    end
    error('em_lsim_discriminative:badOptionValue', '%s', err.message);
end
opts = ip.Results;
opts.parallel = logical(opts.parallel);
if ~isempty(opts.log_prior)
    opts.log_prior = opts.log_prior(:);
end
given = fieldnames(extra);
pub_only = intersect(given, {'ebw_E', 'ismooth_tau', 'ismooth_backoff', 'ismooth_tau_discrete', ...
    'weak_sense_iters', 'weak_sense_tol'});
leg_only = intersect(given, {'damping_E', 'damping_E_min', 'damping_E_max', 'max_backtracks'});
if strcmp(opts.mmi_recipe, 'legacy') && ~isempty(pub_only)
    error('em_lsim_discriminative:publishedOptionOnly', 'option(s) %s apply only to mmi_recipe = ''published''.', ...
        strjoin(pub_only(:)', ', '));
end
if strcmp(opts.mmi_recipe, 'published') && ~isempty(leg_only)
    error('em_lsim_discriminative:legacyOptionOnly', ['option(s) %s apply only to mmi_recipe = ''legacy'' ', ...
        '(the published recipe uses the fixed ebw_E and no step search).'], strjoin(leg_only(:)', ', '));
end
if isfield(opts.generative_extra, 'sigma_diag') && opts.generative_extra.sigma_diag == 0
    error('em_lsim_discriminative:fullCovarianceUnsupported', ...
        'Full-covariance emissions are not supported by the discriminative trainer.');
end
end
