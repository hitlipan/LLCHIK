
%==========================================================================
%Inputs 
%   feaSet: the histograms corresponding to one image
%   B: the dictionary in mapping space
%   pyramid: the spatial pyramid structure
%   knn: the nubmer of neighbors for LLC coding
%   dises: innerKernelSum
%   hikTables: tables to compute kernel reduing the time complexity to O(d)
%   multiply: the factor of translation, A = multiply * A
%   D: the ij-th element is mi_T * mj, used in later coding percedure
%
%  Written by Li Pan lipan@hit.edu.cn
%==========================================================================
function [beta] = LLCHIK_pooling(feaSet, B, pyramid, knn, dises, hikTables, multiply, D)

dSize = length(B);

nSmp = size(feaSet.feaArr, 2);

img_width = feaSet.width;
img_height = feaSet.height;
idxBin = zeros(nSmp, 1);

% llchik coding
feaSet.feaArr = multiply * feaSet.feaArr;
feaSet.feaArr = round(feaSet.feaArr - 0.4);
llc_codes = LLCHIK_coding(B,  feaSet.feaArr, knn, dises, hikTables, D);
%llc_codes = llc_codes';

% spatial levels
pLevels = length(pyramid);
% spatial bins on each level
pBins = pyramid.^2;
% total spatial bins
tBins = sum(pBins);

beta = zeros(dSize, tBins);
bId = 0;

for iter1 = 1:pLevels,
    
    nBins = pBins(iter1);
    
    wUnit = img_width / pyramid(iter1);
    hUnit = img_height / pyramid(iter1);
    
    % find to which spatial bin each local descriptor belongs
    xBin = ceil(feaSet.x / wUnit);
    yBin = ceil(feaSet.y / hUnit);
    idxBin = (yBin - 1)*pyramid(iter1) + xBin;
    
    for iter2 = 1:nBins,     
        bId = bId + 1;
        sidxBin = find(idxBin == iter2);
        if isempty(sidxBin),
            continue;
        end      
        %这里按照绝对值大小进行max pooling,并且取原来的值
        subLLCCodes = llc_codes(:, sidxBin);
        [Y, I] = max(abs(subLLCCodes), [], 2);
        
        subLLCCodesT = subLLCCodes';
        [p, q] = size(subLLCCodesT);
        x = eye(q);
        x = p * x;
        y = 0:1:(q - 1);
        y = y';
        I = I + x * y;
        beta(:, bId)= subLLCCodesT(I);
    end
end

if bId ~= tBins,
    error('Index number error!');
end

beta = beta(:);
beta = beta./sqrt(sum(beta.^2));
