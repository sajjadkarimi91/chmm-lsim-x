function model = lsim_disc_unpack(p)
% Inverse of lsim_disc_pack: packed arrays -> the em_lsim init_model cell API
% (pi_0_lsim, coupling_tetha_IM, transition_matrices_IM_cell, gmm_para_lsim).
% Mirrors the output conversion at the end of em_lsim.m, so a returned model
% can be passed straight back to em_lsim as init_model or to
% forward_backward_lsim.

C = p.C;
sni = p.state_numbers_index;
dni = p.dimension_numbers_index;

model = struct();
model.pi_0_lsim = cell(C, 1);
model.coupling_tetha_IM = p.coupling_tetha_IM;
model.transition_matrices_IM_cell = cell(C, C);
model.gmm_para_lsim = cell(C, 1);

for z = 1:C
    zi = sni(z)+1:sni(z+1);
    di = dni(z)+1:dni(z+1);

    model.pi_0_lsim{z, 1} = p.pi_0(zi);

    gmm_para = [];
    for i = 1:p.channel_num_states(z)
        for k = 1:p.num_gmm_component(z)
            gmm_para(i).P(k, 1) = p.P_all(z, i, k); %#ok<AGROW>
            gmm_para(i).sigma(k).x = p.sigma_all(di, i, k);
            gmm_para(i).mu(k).x = p.mu_all(di, i, k);
        end
    end
    model.gmm_para_lsim{z, 1}.gmm_para = gmm_para;

    for c = 1:C
        model.transition_matrices_IM_cell{c, z} = p.transitions_matrices(sni(c)+1:sni(c+1), zi);
    end
end

end
