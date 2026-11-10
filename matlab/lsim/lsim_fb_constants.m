function fb_constants = lsim_fb_constants( varargin )
%LSIM_FB_CONSTANTS  Variant and numerical constants of the LSIM forward-backward.
%
%   fbc = lsim_fb_constants()                       % raw POISE (the default)
%   fbc = lsim_fb_constants('sed')                  % SED
%   fbc = lsim_fb_constants('fast')                 % fast (SVI)
%   fbc = lsim_fb_constants('poise','poise_refinement',true)  % POISE + refinement
%   fbc = lsim_fb_constants('fast','pi0_reg',1e-6)  % variant, then name/value overrides
%   fbc = lsim_fb_constants('<name>')               % a user backward recursion
%
% Used by forward_backward_lsim(_sed/_fast/_poise), em_lsim(_fast/_poise) and
% em_lsim_discriminative (extra.fb_constants).
%
% Fields
%   variant          backward recursion; the kernel calls lsim_backward_<variant>.
%                    Built-in (lsim/backward):
%                      'poise'  power-corrected outgoing-influence recursion
%                      'sed'    pairwise-conditional message, delta weights solved
%                               per channel from x = ( diag(f-h) + h*ones(C,C) ) \ f
%                      'fast'   fixed coupling weights, mixture raised to the power
%                               w_opt = sum_w( svi_eps + theta(zee,w) )
%                    Any name for which a function lsim_backward_<name> is on the
%                    path is accepted (see lsim/backward/Contents.m).
%   beta_floor       sed: lower clamp on the mixed backward message        (1e-9)
%   pi0_reg          additive regularization of pi_0 per channel, then
%                    renormalized; pi0_reg <= 0 leaves pi_0 unchanged       (1e-10)
%   f_reg            sed: additive regularization of f in the delta solver  (1e-6)
%   svi_eps          fast: guard in the coupling-weight denominator and in
%                    the exponent w_opt                                     (1e-8)
%   poise_refinement poise: apply the one-shot observation-conditioned
%                    refinement (weight max(0, 0.75-theta_out), so it only
%                    acts on weakly coupled channels)                       (false)
% Constants needed by a user recursion can be added as fields of the returned
% struct; the struct is passed to lsim_backward_<variant> unchanged.

fb_constants.variant         = 'poise';
fb_constants.beta_floor      = 10^-9;
fb_constants.pi0_reg         = 10^-10;
fb_constants.f_reg           = 10^-6;
fb_constants.svi_eps         = 10^-8;
fb_constants.poise_refinement = false;

args = varargin;

% optional leading variant name
if ~isempty(args) && ischar(args{1}) && ~any(strcmp(args{1} , fieldnames(fb_constants)))
    fb_constants.variant = args{1};
    args = args(2:end);
end

if rem(length(args),2)~=0
    error('lsim_fb_constants: overrides must be name/value pairs')
end

valid_names = fieldnames(fb_constants);

for i = 1:2:length(args)

    if ~ischar(args{i}) || ~any(strcmp(args{i} , valid_names))
        error('lsim_fb_constants: unknown option ''%s''', num2str(args{i}))
    end

    fb_constants.(args{i}) = args{i+1};

end

if isstring(fb_constants.variant) && isscalar(fb_constants.variant)
    fb_constants.variant = char(fb_constants.variant);
end

% a variant is valid when its backward recursion lsim_backward_<variant> exists
if ~( ischar(fb_constants.variant) && isrow(fb_constants.variant) && ...
        isvarname(['lsim_backward_' fb_constants.variant]) && ...
        any(exist(['lsim_backward_' fb_constants.variant], 'file') == [2 3 6]) )
    error(['lsim_fb_constants: variant must be ''sed'', ''fast'', ''poise'' or a name ' ...
        'with a function lsim_backward_<name> on the path, got ''%s'''], num2str(fb_constants.variant))
end

if ~(islogical(fb_constants.poise_refinement) && isscalar(fb_constants.poise_refinement))
    error('lsim_fb_constants: poise_refinement must be a scalar logical')
end

end
