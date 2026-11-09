function packed = lsim_disc_pack(model)
% Packs one LSIM model (the em_lsim init_model cell API) into the flat arrays
% forward_backward_lsim_kernel consumes, by delegating to lsim_pack_para so the
% two representations can never drift apart.
%
%   model.pi_0_lsim                     C x 1 cell, S(c) x 1
%   model.coupling_tetha_IM             C x C, columns sum to 1 (source in rows)
%   model.transition_matrices_IM_cell   C x C cell, {c,z} is S(c) x S(z),
%                                       source state in rows, rows sum to 1
%   model.gmm_para_lsim                 C x 1 cell, .gmm_para(i).P/.mu(k).x/.sigma(k).x
%
% The returned struct is lsim_pack_para's output without the data fields.

C = numel(model.pi_0_lsim);
Y_dummy = cell(C, 1);
for c = 1:C
    Y_dummy{c, 1} = zeros(numel(model.gmm_para_lsim{c}.gmm_para(1).mu(1).x), 1);
end

packed = lsim_pack_para(model.pi_0_lsim, model.coupling_tetha_IM, ...
    model.transition_matrices_IM_cell, model.gmm_para_lsim, Y_dummy);
packed = rmfield(packed, {'channel_time_series', 'T'});

end
