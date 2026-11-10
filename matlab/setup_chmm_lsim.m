function setup_chmm_lsim()
%SETUP_CHMM_LSIM  Add the CHMM-LSIM toolbox folders to the MATLAB path.
%
%   setup_chmm_lsim
%
% Adds lsim, lsim/backward, chmm, common, emissions, discriminative and demos
% (relative to this file). Use savepath afterwards to keep them for later sessions.

root = fileparts(mfilename('fullpath'));
folders = {'lsim', fullfile('lsim', 'backward'), 'chmm', 'common', 'emissions', ...
    'discriminative', 'demos'};
for i = numel(folders):-1:1
    addpath(fullfile(root, folders{i}));
end

end
