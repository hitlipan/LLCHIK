%=========================================================================
%
% Accurate llchik coding
% Input 
%     B: the dictionay in mapping space
%     X: the feaSet, histograms which belongs to one image
%     knn: number of neighbors
%     iniDises: innerKernel sum
%     hikTables: tables to compute kernel reducing the time complexity to
%                 O(d)
%     D: the ij-th element is mi_T * mj, used in later coding percedure
%     Writtern by Li Pan, hit
% Output
%     Coeff -dicSize * n ,where n is the nubmer of X
%========================================================================

function [Coeff] = LLCHIK_coding(B,  X,  knn,  iniDises, hikTables, D)

    n = length(B);
    [p q] = size(X);
    
    %对X中的每个样本，找到最近邻的knn个，然后使用marginal regression
    %  对该样本进行编码
    dises = interKernelSum(X, hikTables, B);
    for jj = 1:n
        dises(jj, :) = -2 * dises(jj, :);
    end
    iniDises = repmat(iniDises, [1, q]);
    dises = dises + iniDises;
    
    %计算X中每个histogram到@dicSize 个基的距离排序，并将结果保存到 $Y, $I
    [Y, I] = sort(dises, 1, 'ascend');
    Coeff = computeCoeff(X, hikTables, B, I, knn, D);
end

%
%c = inv(D'D)*(D'fai(h) + lamda * 1)
%With normalizing
%这里采用的是LLE中的方法
function Coeff = computeCoeff(samples, hikTables, idxes, I, knn, D)
    
    [d, n] = size(samples);
    dicSize = length(hikTables);
    Coeff = zeros(dicSize, n);
    smallD = zeros(knn, knn);
    %全部计算出来，然后需要时，直接取
    laterTerms = interKernelSum(samples, hikTables, idxes);
    
    for jj = 1:1:n
        relatedBases = I(1:knn,jj);
        %生成smallD，仅由近邻base组成
        for pp = 1:1:knn
            for qq = 1:1:knn
                smallD(pp, qq) = D(relatedBases(pp), relatedBases(qq));
            end
        end
        C = smallD' * smallD;
        invC = inv(C);
        DTFaih = laterTerms(relatedBases, jj);
        alpha = 1;
        for ii = 1:1:knn
            for kk = 1:1:knn
                alpha = alpha - invC(ii, kk) * laterTerms(relatedBases(kk), jj);
            end
        end
        beta = sum(sum(invC));
        lamda = alpha / beta;
        Coeff(I(1:knn, jj), jj) = invC * (DTFaih + lamda);
    end
end
