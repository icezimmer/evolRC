function [W_out_best,omega_in_best,rho_best,Nh_best,dns_best,a_best,lambda_r_best,ws_best, MAK_tr, MAK_vl] = gridSearch(num_classes,tr_in,tr_tg,vl_in,vl_tg,dv_in,dv_tg,omega_in,rho,Nh,dns,a,lambda_r,ws)

tot = length(omega_in)*length(rho)*length(Nh)*length(dns)*length(a)*length(lambda_r)*length(ws);
r_guesses = 5;
one_hot = eye(num_classes);

% Model selection (by Grid search)
disp('Grid Search: 0%')
MAK_vl = -Inf;
MAK_tr = -Inf;
config = 0;
for i = 1:length(omega_in)
    for j = 1:length(rho)
        for k = 1:length(Nh)
            for l = 1:length(dns)
                for m = 1:length(a)
                    for n = 1:length(lambda_r)
                        for o = 1:length(ws)
                            meanAccuracy_K_vl = 0;
                            meanAccuracy_K_tr = 0;
                            for seed = 1:r_guesses
                                % TRAINING
                                hidden_tr = zeros(Nh(k),0);
    
                                for sample=1:size(tr_in,3)
                                    [~, sequence_tr] = rc(tr_in(:,:,sample), seed, omega_in(i), rho(j), Nh(k), dns(l), a(m), ws(o));
                                    hidden_tr = cat(2,hidden_tr,sequence_tr);
                                end
                                W_out = trainOffline(hidden_tr,tr_tg, lambda_r(n), ws(o));
    
                                y_tr = readout(hidden_tr,W_out);
                                [~, argmax_tr] = max(y_tr,[],1);
                                tr_pr = one_hot(:,argmax_tr);
                                [~, accuracy_K_tr] = evaluation(washout(tr_tg,ws(o)), tr_pr);
    
                                meanAccuracy_K_tr = meanAccuracy_K_tr + (accuracy_K_tr / r_guesses);
                                
                                % VALIDATION
                                hidden_vl = zeros(Nh(k),0);
                                
                                for sample=1:size(vl_in,3)
                                    [~, sequence_vl] = rc(vl_in(:,:,sample), seed, omega_in(i), rho(j), Nh(k), dns(l), a(m));
                                    hidden_vl = cat(2,hidden_vl,sequence_vl);
                                end
    
                                y_vl = readout(hidden_vl, W_out);
                                [~, argmax_vl] = max(y_vl,[],1);
                                vl_pr = one_hot(:,argmax_vl);
                                
                                [~, accuracy_K_vl] = evaluation(vl_tg, vl_pr);
                                
                                meanAccuracy_K_vl = meanAccuracy_K_vl + (accuracy_K_vl / r_guesses);
                            end
                            config = config+1;
                            disp(['Grid Search: ',num2str(100*(config/tot)),'%'])
                            if meanAccuracy_K_vl > MAK_vl
                                MAK_tr = meanAccuracy_K_tr;
                                MAK_vl = meanAccuracy_K_vl;
                                omega_in_best = omega_in(i);
                                rho_best = rho(j);
                                Nh_best = Nh(k);
                                dns_best = dns(l);
                                a_best = a(m);
                                ws_best = ws(o);
                                lambda_r_best = lambda_r(n);
                            end
                        end
                    end
                end
            end
        end
    end
end

% % Refit on the development set
% disp('Refit')
% start=tic;
% hidden_dv = zeros(Nh_best,0);
% for sample=1:size(dv_in,3)
%     [~, sequence_dv] = rc(dv_in(:,:,sample), seed, omega_in_best, rho_best, Nh_best, dns_best, a_best, ws_best);
%     hidden_dv = cat(2,hidden_dv, sequence_dv);
% end
% W_out_best = trainOffline(hidden_dv, dv_tg, lambda_r_best, ws_best);
% timeTrain=toc(start);
% disp(['Refit best configuration time: ', num2str(timeTrain)])
% end