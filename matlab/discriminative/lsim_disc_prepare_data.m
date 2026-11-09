function data = lsim_disc_prepare_data(class_observations, name)
% Validates the class-structured data contract and flattens it into a list of
% sequences, WITHOUT ever concatenating trials (each sequence keeps its own
% boundaries; no transition is created between trials).
%
%   class_observations : M x 1 cell. class_observations{i} is a C x N_i cell in
%                        em_lsim's format: {c,n} is d(c) x T_n, NaN = missing,
%                        every channel of sequence n has the same T_n.
%
% Returns data with fields
%   M, C, dim (C x 1), N_per_class (M x 1), Y (1 x Ntot cell of sum(d) x T),
%   label (Ntot x 1), T (Ntot x 1), class_obs (the input, for em_lsim init).

if nargin < 2
    name = 'class_observations';
end

if ~iscell(class_observations) || isempty(class_observations)
    error('em_lsim_discriminative:badData', '%s must be a non-empty M x 1 cell.', name);
end
class_observations = class_observations(:);
M = numel(class_observations);

C = [];
dim = [];
Y = {};
label = [];
T = [];
N_per_class = zeros(M, 1);

for i = 1:M
    obs = class_observations{i};
    if ~iscell(obs) || isempty(obs)
        error('em_lsim_discriminative:badData', '%s{%d} must be a non-empty C x N cell.', name, i);
    end
    if isempty(C)
        C = size(obs, 1);
        dim = zeros(C, 1);
        for c = 1:C
            dim(c) = size(obs{c, 1}, 1);
        end
    elseif size(obs, 1) ~= C
        error('em_lsim_discriminative:badData', ...
            '%s{%d} has %d channels; class 1 has %d.', name, i, size(obs, 1), C);
    end
    N_per_class(i) = size(obs, 2);

    for n = 1:size(obs, 2)
        Tn = size(obs{1, n}, 2);
        if Tn < 1
            error('em_lsim_discriminative:badData', '%s{%d}{1,%d} is empty.', name, i, n);
        end
        Yn = zeros(sum(dim), Tn);
        off = 0;
        for c = 1:C
            x = obs{c, n};
            if ~isnumeric(x) || ~isreal(x)
                error('em_lsim_discriminative:badData', '%s{%d}{%d,%d} must be real numeric.', name, i, c, n);
            end
            if size(x, 1) ~= dim(c)
                error('em_lsim_discriminative:badData', ...
                    '%s{%d}{%d,%d} has dimension %d; expected %d.', name, i, c, n, size(x, 1), dim(c));
            end
            if size(x, 2) ~= Tn
                error('em_lsim_discriminative:badData', ...
                    '%s{%d}: channels of sequence %d are not synchronized (%d vs %d samples).', ...
                    name, i, n, size(x, 2), Tn);
            end
            if any(isinf(x(:)))
                error('em_lsim_discriminative:badData', ...
                    '%s{%d}{%d,%d} contains Inf; use NaN for missing values.', name, i, c, n);
            end
            Yn(off+1:off+dim(c), :) = double(x);
            off = off + dim(c);
        end
        Y{end+1} = Yn; %#ok<AGROW>
        label(end+1, 1) = i; %#ok<AGROW>
        T(end+1, 1) = Tn; %#ok<AGROW>
    end
end

data = struct('M', M, 'C', C, 'dim', dim, 'N_per_class', N_per_class, ...
    'label', label, 'T', T);
data.Y = Y;
data.class_obs = class_observations;

end
