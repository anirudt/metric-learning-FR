close all;
clear all;

% Add the path for the mildml files
addpath /home/anirudt/Projects/FR-research/vanilla-fr/lfw_scripts/mildml_files/

% Add the ldml_learn wrapper
X_tr = csvread('X_tr.csv');
y_train = csvread('y_train.csv');
size(X_tr)
size(y_train)

[L b info] = ldml_learn(X_tr, y_train);

% Write back the transform for use in Python side
csvwrite('L.csv', L');
