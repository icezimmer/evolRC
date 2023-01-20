function [num_classes,dv_in,dv_tg,tr_in,tr_tg,vl_in,vl_tg,ts_in,ts_tg]=dataConstruction(seed_shuffle)

[input_data, target_data] = dataLoader();
[~, len, num] = size(input_data);
[num_classes, ~] = size(target_data);

%shuffle the dataset
rng(seed_shuffle)
shuffle = randperm(num);
input_data = input_data(:,:,shuffle);
target_data = target_data(:,shuffle);

dv_index = 1:64; % from 1 to 64
tr_index = 1:50; % from 1 to 50
vl_index = 51:64; % from 51 to 64
ts_index = 65:88; % from 65 to 88

dv_in = input_data(:,:,dv_index);
tr_in = input_data(:,:,tr_index);
vl_in = input_data(:,:,vl_index);
ts_in = input_data(:,:,ts_index);

dv_tg = target_data(:,dv_index);
dv_tg = kron(dv_tg, ones(1,len));
tr_tg = target_data(:,tr_index);
tr_tg = kron(tr_tg, ones(1,len));
vl_tg = target_data(:,vl_index);
vl_tg = kron(vl_tg, ones(1,len));
ts_tg = target_data(:,ts_index);
ts_tg = kron(ts_tg, ones(1,len));
end