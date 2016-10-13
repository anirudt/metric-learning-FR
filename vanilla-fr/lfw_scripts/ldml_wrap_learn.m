close all;
clear all;

% Add the path for the mildml files
addpath /home/anirudt/Projects/big_projects/BTP/mtp/vanilla-fr/lfw_scripts/mildml_files/

% Add the ldml_learn wrapper
X_tr = load('X_tr.mat');
y_train = load('y_train.mat');

[L b info] = ldml_learn(X_tr, y_train, 3, 1000);

save('L.mat', L);
