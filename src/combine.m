function W = combine(pop, seed)
rng(seed)
Nh = length(pop{1});
num_pop = length(pop);
size = Nh * num_pop;
W = blkdiag(pop{:});
P = speye(size);
P = P(randperm(size),:);
W = P * sparse(W) * P';
end