function y = readout(x, W_out)

[~, num_samples] = size(x);
y = W_out * [x; ones(1, num_samples)];
end

