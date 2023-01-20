addpath src

% Data construction
seed_shuffle = 13;
[num_classes,dv_in,dv_tg,tr_in,tr_tg,vl_in,vl_tg,ts_in,ts_tg]=dataConstruction(seed_shuffle);

% Hyper-parameters
omega_in = 0.4; %input scaling
rho = 0.9; %spectral radius
dns = 1; %connectivity
a = 0.1; %leaking rate
lambda_r = 0.1; %regularization
ws = 0; %transient

Nh = 2; %num. hidden neurons for starting population
num_pop = 2; %number of populations
gen = 8; %number of generations

% Grid-Search
[~,omega_in_best,rho_best,Nh_best,dns_best,a_best,lambda_r_best,ws_best, MAK_tr, MAK_vl] = gridSearch(num_classes,tr_in,tr_tg,vl_in,vl_tg,dv_in,dv_tg,omega_in,rho,Nh,dns,a,lambda_r,ws);

% Training on DV set
seed = 4;
hidden_dv = zeros(Nh_best*(num_pop)^gen,0);
for sample=1:size(dv_in,3)
    [~, sequence_dv] = evolrc(dv_in(:,:,sample), seed, omega_in_best, rho_best, Nh_best, num_pop, gen, a_best, ws_best);
    hidden_dv = cat(2,hidden_dv,sequence_dv);
end
W_out_best = trainOffline(hidden_dv,dv_tg, lambda_r_best, ws_best);

% Assessment on the test set
disp('Assessment')
seed = 4;
one_hot = eye(num_classes);
% hidden_ts = zeros(Nh_best,0);
% for sample=1:size(ts_in,3)
%     [~, sequence_ts] = rc(ts_in(:,:,sample), seed, omega_in_best, rho_best, Nh_best, dns_best, a_best);
%     hidden_ts = cat(2,hidden_ts, sequence_ts);
% end
hidden_ts = zeros(Nh_best*(num_pop)^gen,0);
for sample=1:size(ts_in,3)
    [~, sequence_ts] = evolrc(ts_in(:,:,sample), seed, omega_in_best, rho_best, Nh_best, num_pop, gen, a_best);
    hidden_ts = cat(2,hidden_ts, sequence_ts);
end

y_ts = readout(hidden_ts, W_out_best);
[~, argmax_ts] = max(y_ts,[],1);
ts_pr = one_hot(:,argmax_ts);
[~, accuracy_K_ts, accuracy_ts, accuracy_av_ts, F1_ts, F1_macro_ts] = evaluation(ts_tg, ts_pr);

% Plot Confusion Matrix
[classes_target, ~] = find(ts_tg);
[classes_predict, ~] = find(ts_pr);
gcf = figure;
confusionchart(classes_target, classes_predict);
title("Confusion Matrix (TS set)")

% Save plot and net structure
saveas(gcf, fullfile('results', strcat('confusionMatrix', '.png')))
save(fullfile('results', strcat('hyperparameters', '.mat')), 'seed', 'omega_in_best', 'rho_best', 'Nh_best', 'dns_best', 'a_best', 'lambda_r_best', 'ws_best')
save(fullfile('results', strcat('readOutWeights', '.mat')), 'W_out_best')

% Save performance
save(fullfile('results', strcat('performanceTR', '.mat')), 'MAK_tr')
save(fullfile('results', strcat('performanceVL', '.mat')), 'MAK_vl')
save(fullfile('results', strcat('performanceTS', '.mat')), 'accuracy_K_ts', 'accuracy_ts', 'accuracy_av_ts', 'F1_ts', 'F1_macro_ts')