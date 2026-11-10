"""Model families (internal module; import from chmm_lsim).

One subpackage per transition model, each with `params`, `inference`, `em`,
`init`, `simulate` and `viterbi` modules that share `chmm_lsim.learning` (EM
loop, `EmResult`) and `chmm_lsim.emissions` (emission models):

* `lsim`: Latent Structure Influence Models (coupled HMMs with
  coupling-weighted pairwise transitions) and their approximate
  forward-backward recursions;
* `chmm`: full coupled HMMs with exact joint-state inference.
"""
