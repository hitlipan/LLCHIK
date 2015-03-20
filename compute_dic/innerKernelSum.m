function dis = innerKernelSum(oneClusterSamples)
    
    [m, n] = size(oneClusterSamples);
    
    d = 0;
    for jj = 1:n
        curSift = oneClusterSamples(:, jj);
        curSiftRep = repmat(curSift, [1, n]);
        t = min(curSiftRep, oneClusterSamples);
        t = sum(t, 1);
        d = d + sum(t);
    end
    dis =  d / (n * n);
end 
