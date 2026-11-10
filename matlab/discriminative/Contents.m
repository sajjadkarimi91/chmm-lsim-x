% DISCRIMINATIVE  Discriminative (MMI) training of class-specific LSIMs.
%
%   em_lsim_discriminative      - MMI training of M class LSIMs (published or legacy recipe)
%   lsim_discriminative_predict - class scores, predictions and normalized scores
%   lsim_disc_options           - validated options of em_lsim_discriminative
%   lsim_disc_objective         - MMI objective and its numerator/denominator weights
%   lsim_disc_score             - per-sequence LSIM score (log-likelihood approximation)
%   lsim_disc_sequence_stats    - expected sufficient statistics of one sequence
%   lsim_disc_weighted_counts   - numerator/denominator block counts of one model
%   lsim_disc_counts            - block-level expected counts
%   lsim_disc_gradient          - gradient in unconstrained coordinates from block counts
%   lsim_disc_ebw_proposal      - extended Baum-Welch proposal (legacy recipe)
%   lsim_disc_published_update  - one published-recipe update (EBW, I-smoothing, weak-sense)
%   lsim_disc_weak_sense        - weak-sense update of one probability vector
%   lsim_disc_domain            - compact feasible parameter domain (build, project, check)
%   lsim_disc_pack              - em_lsim model struct to packed arrays
%   lsim_disc_unpack            - packed arrays to em_lsim model struct
%   lsim_disc_prepare_data      - validate and flatten class-structured data
%   lsim_disc_metrics           - balanced accuracy, macro-F1, accuracy, confusion matrix
%   lsim_disc_binary_metrics    - AUROC, AUPRC, max-F1, best-threshold balanced accuracy
%   lsim_disc_score_metrics     - every classification metric from a score matrix
%   lsim_disc_detection_score   - two-class detection score (scaled log-likelihood ratio)
%   lsim_disc_tune_threshold    - threshold maximizing balanced accuracy or F1
%   lsim_disc_threshold_metrics - metrics at a given threshold
%   lsim_disc_evaluate          - score models on several data sets, all metrics
%
% Tests: discriminative/tests (runtests('TestMMI'), runtests('TestValidationSelection'),
% check_disc_training, check_disc_gradients).
