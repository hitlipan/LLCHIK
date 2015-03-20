function table = buildTable(samples, maxH)
%建立表格
[d, n] = size(samples);
table = zeros(d, maxH + 1);
for j = 1:d
    currLine = samples(j, :);
    sortCL = sort(currLine, 'ascend');
    for k = 1:maxH + 1
        idx = bSearch(k - 1, sortCL);
        table(j, k) = sum(sortCL(1:(idx - 1)));
        table(j, k) = table(j, k) + (k - 1) * (n - idx + 1);
    end
end

end



function idx = bSearch(key, X)
    len = length(X);
    low = 1;
    high = len;
    while low <= high
        middle = floor((low + high) / 2);
        if X(middle) == key
            idx = middle;
            return;
        else
            if X(middle) < key
                low = middle + 1;
            else
                high = middle - 1;
            end
        end
    end
    idx = low;
end
