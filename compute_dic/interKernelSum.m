% one sample is a column vector
% refer to beyond the euclidean distance: ...

function dises = interKernelSum(samples, hikTables, idxes)
    
    samples = samples + 1;
    [d, n] = size(samples);
    dicSize = length(hikTables);
    dises = zeros(dicSize, n);
    for ii = 1:dicSize
        for jj = 1:d
            dises(ii, :) = dises(ii, :) + hikTables{ii}(jj, samples(jj, :));
        end
        %dises(ii, :) = dises(ii, :) / length(hikTables{ii});
        dises(ii, :) = dises(ii, :) / length(idxes{ii});
    end
end
