function main
% driver

path(path, '.\compute_dic\PG_SPBOW');
path(path, './compute_dic/PG_SPBOW');
path(path, 'compute_dic');
pyramid = [1 2 4];
knn =40;

nRounds = 2;
dicSize = 1024;
%dicSize = 1;

mem_block = 3500; 

datasetName = 'Caltech101';
fid = fopen('log.txt', 'a+');

img_dir = fullfile('image', datasetName);
data_dir = fullfile('data', datasetName);
fea_dir = fullfile('features', [datasetName, '_knn40']);
dic_dir = fullfile('dictionary', datasetName);

%skip_extr_sift = false;%需要提取sift
skip_extr_sift = true;
if ~skip_extr_sift
    extr_sift(img_dir, data_dir);
end

%该函数获取所有图像的数量，类别等等
%database.
%         imnum, cname, label, path, nclass
database = retr_database_dir(data_dir);
if (isempty(database))
    error('Please extracting sifts!');
end

%skip_compu_dic = false;
skip_compu_dic = true;
Bpath = fullfile(dic_dir, [datasetName, '_HIK_postIdxes.mat']);
hikTablePath = fullfile('aux_file', [datasetName, '_hikTable.mat']);
innerKernelSumPath = fullfile('aux_file', [datasetName, '_innerKernelSum.mat']);
DForAccCodingPath = fullfile('aux_file', [datasetName, '_D_for_accurate_coding.mat']);
trainSiftsPath = fullfile('train_samples', [datasetName, '_trainSifts.mat']);
%将allSifts映射到整数，以便建立table
maxH = 512;
multiply = maxH;

if ~skip_compu_dic    
    %调用rand-sampling，因为如果考虑所有训练数据的sift特征，太多了，不可计算
    num_smp = 200000;
    %num_smp = 6000;
    
    disp(['Sampling......']);
    fprintf(fid, 'Sampling...............................................\n');
    [allSifts, imageSeparateRecord] = rand_sampling(database, num_smp);
    [siftDimension, siftNumber] = size(allSifts);
    allSifts = multiply * allSifts;
    allSifts = round(allSifts - 0.4);
    
    %保存采样的样本，这样程序有什么改动，直接读取保存的样本，并能保证后面的字典，
    %   hikTable等变量和保存的样本对应
    save(trainSiftsPath, 'allSifts');
    
    disp('Computing hik kmeans......');
    fprintf(fid, 'Computing hik kmeans..................................\n');
    [postIdxes] = HIK_KMeans(dicSize, allSifts, imageSeparateRecord, maxH);
    
    save(Bpath, 'postIdxes');
    disp('Computing hik kmeans done!--------------------------------------\n');
    fprintf(fid, 'Computing hik kmeans done!------------------------------------\n');
    hikTables = cell(dicSize, 1);
    for ii = 1:dicSize
        hikTables{ii} = buildTable(allSifts(:, postIdxes{ii}), maxH);
    end
    save(hikTablePath, 'hikTables');
    
    %这里预先计算每个聚类内部的核求和
    %   sigma_j,k in PI_i K(hj, hk) / |PI_i| ^ 2
    dises = zeros(dicSize, 1);
    for ii = 1:dicSize
        dises(ii) = innerKernelSum(allSifts(:, postIdxes{ii}));
    end
    save(innerKernelSumPath, 'dises');
    
    %计算矩阵D,该矩阵 ij-th 元素是m[i]_T * m[j]的值，以便后面精确编码的时候用
    D = zeros(dicSize, dicSize);
    fprintf(1, 'Computing D.............................................\n');
    fprintf(fid, 'Computing D.............................................\n');
    for ii = 1:1:dicSize
        %获取第 i-th, j-th base的所有histogram
        indexInPaiI = postIdxes{ii};
        for jj = 1:1:dicSize
            ijValue = 0;
            jTable = hikTables{jj};
            for kk = 1:1:length(indexInPaiI)
                curSift = allSifts(:, indexInPaiI(kk));
                curSift = curSift + 1; %这里是实现Wu的方法时需要有加1操作
                for pp = 1:1:siftDimension
                    ijValue = ijValue + jTable(pp, curSift(pp));
                end
            end
            D(ii, jj) = ijValue;
            if (~mod(ii, 100))
                fprintf(1, 'Has computed %d lines of %d lines----------------------------\n', ii, dicSize);
                fprintf(fid, 'Has computed %d lines of %d lines----------------------------\n', ii, dicSize);
                fprintf(fid, datestr(now));
                datestr(now)
            end
        end
    end
    %保存D，在后面编码时要用
    save(DForAccCodingPath, 'D');
    fprintf(1, 'Computing D done!-------------------------------------------\n');
    fprintf(fid, 'Computing D done!------------------------------------------\n');
else
    load(Bpath);
    %load(hikTablePath);
    %load(innerKernelSumPath);
    %load(DForAccCodingPath);
    %load(trainSiftsPath);
end
nCodeBook = dicSize;
B = postIdxes;

%提取稀疏特征
dFea = sum(nCodeBook * pyramid .^ 2);
nFea = length(database.path);

fdatabase = struct;
fdatabase.path = cell(nFea, 1);
fdatabase.label = zeros(nFea, 1);

%skip_compu_sc = false;
skip_compu_sc = true;

%编码
disp(['Coding......']);
fprintf(fid, 'Coding......\n');
for iter1 = 1:nFea
    if ~mod(iter1, 5)
        fprintf('.');
    end
    if ~mod(iter1, 100)
        fprintf('coding %d images processed\n', iter1);
        fprintf(fid, 'Coding %d images processed\n', iter1);
    end
    imageSiftPath = database.path{iter1};
    imageLabel = database.label(iter1);

   
    [rtpath, fname] = fileparts(imageSiftPath);
    feaPath = fullfile(fea_dir, num2str(imageLabel), [fname, '.mat']);
    if ~skip_compu_sc
        load(imageSiftPath);
        
        fea = LLCHIK_pooling(feaSet, B,  pyramid, knn, dises, hikTables, multiply, D);
        label = imageLabel;
        if ~isdir(fullfile(fea_dir, num2str(imageLabel))),
            mkdir(fullfile(fea_dir, num2str(imageLabel)));
        end      
        save(feaPath, 'fea', 'label');
    end
    fdatabase.label(iter1) = imageLabel;
    fdatabase.path{iter1} = feaPath;    
end

clabel = unique(fdatabase.label);
nclass = length(clabel);

fclose(fid);
% -------------------------------------------------------------------------
% 使用liblinear分类，并统计分类准确率
fprintf('Clasifying ......');
fid = fopen('result.txt', 'a+');
clabel = unique(fdatabase.label);
nclass = length(clabel);
disp(nclass);
fea_accuracy = zeros(nRounds, 1);

tr_nums = [5 10 15 20 25 30];
for tt = 1:length(tr_nums)
    tr_num = tr_nums(tt);
for ii = 1:nRounds,
    fprintf('Round: %d...\n', ii);
    fprintf(fid, 'Round : %d...\n', ii);
    tr_idx = [];
    ts_idx = [];
    
    for jj = 1:nclass,
        idx_label = find(fdatabase.label == clabel(jj));
        num = length(idx_label);
        
        idx_rand = randperm(num);
        
        tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];
        ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:end))];
    end
    
    fprintf('Training number: %d\n', length(tr_idx));
    fprintf('Testing number:%d\n', length(ts_idx));
     fprintf(fid, 'Training number: %d\n', length(tr_idx));
    fprintf(fid, 'Testing number:%d\n', length(ts_idx));
    
    % load the training features 
    tr_fea = zeros(length(tr_idx), dFea);
    tr_label = zeros(length(tr_idx), 1);
    
    for jj = 1:length(tr_idx),
        fpath = fdatabase.path{tr_idx(jj)};
        load(fpath, 'fea', 'label');
        tr_fea(jj, :) = fea';
        
        tr_label(jj) = label;
    end
    
    %options = ['-s 1 -c 10 -B 1 -q'];
    options = ['-s 2 -c 5 -q'];
    model_fea = train(double(tr_label), sparse(tr_fea), options);
    disp('Train doone!');
    fprintf(fid, 'Train done!\n');
    %model_fea = libsvmtrain(tr_label, tr_fea, '-t 0 -q');
    clear tr_fea;
    
    % load the testing features
    ts_num = length(ts_idx);
    ts_label = [];
    
    if ts_num < mem_block,
        % load the testing features directly into memory for testing
        ts_fea = zeros(length(ts_idx), dFea);
        
        ts_label = zeros(length(ts_idx), 1);

        for jj = 1:length(ts_idx),
            fpath = fdatabase.path{ts_idx(jj)};
            load(fpath, 'fea', 'label');
            ts_fea(jj, :) = fea';
       
            ts_label(jj) = label;
        end
        %[C1] = libsvmpredict(ts_label, ts_fea, model_fea);
        [C1] = predict(ts_label, sparse(ts_fea), model_fea);
 
    else
        % load the testing features block by block
        num_block = floor(ts_num/mem_block);
        rem_fea = rem(ts_num, mem_block);
        
        curr_ts_fea = zeros(mem_block, dFea);
      
        curr_ts_label = zeros(mem_block, 1);
        
        C1 = [];
        
        for jj = 1:num_block,
            block_idx = (jj-1)*mem_block + (1:mem_block);
            curr_idx = ts_idx(block_idx); 
            
            % load the current block of features
            for kk = 1:mem_block,
                fpath = fdatabase.path{curr_idx(kk)};
                load(fpath, 'fea', 'label');
                curr_ts_fea(kk, :) = fea';
                
           
                curr_ts_label(kk) = label;
            end    
            
            % test the current block features
            ts_label = [ts_label; curr_ts_label];
            [curr_C1] = predict(curr_ts_label, sparse(curr_ts_fea), model_fea);
            %[curr_C1] = libsvmpredict(curr_ts_label, curr_ts_fea, model_fea);
            C1 = [C1; curr_C1];

        end
        
        curr_ts_fea = zeros(rem_fea, dFea);
     
        curr_ts_label = zeros(rem_fea, 1);
        curr_idx = ts_idx(num_block*mem_block + (1:rem_fea));
        
        for kk = 1:rem_fea,
           fpath = fdatabase.path{curr_idx(kk)};
           load(fpath, 'fea', 'label');
           curr_ts_fea(kk, :) = fea';
    
           curr_ts_label(kk) = label;
        end  
        
        ts_label = [ts_label; curr_ts_label];
        [curr_C1] = predict(curr_ts_label, sparse(curr_ts_fea), model_fea);
        %[curr_C1] = libsvmpredict(curr_ts_label, curr_ts_fea, model_fea);
        C1 = [C1; curr_C1];
    end
 
    % normalize the classification accuracy by averaging over different
    % classes
    fea_acc = zeros(nclass, 1);
  
    total_fea_right = 0;

    for jj = 1 : nclass,
        c = clabel(jj);
        idx = find(ts_label == c);
        fea_curr_pred_label = C1(idx);
      
        
        curr_gnd_label = ts_label(idx);    
        fea_acc(jj) = length(find(fea_curr_pred_label == curr_gnd_label))/length(idx);
        total_fea_right = total_fea_right + length(find(fea_curr_pred_label == curr_gnd_label));
       
    end

    fea_accuracy(ii) = total_fea_right / length(ts_label); 
    fprintf('fea Classification accuracy for round %d: %f\n', ii, fea_accuracy(ii));
     fprintf(fid, 'fea Classification accuracy for round %d: %f\n', ii, fea_accuracy(ii));
end

Ravg = mean(fea_accuracy);                  % average recognition rate
Rstd = std(fea_accuracy);                   % standard deviation of the recognition rate

fprintf('==================fea=============================');
fprintf('Average classification accuracy: %f\n', Ravg);
fprintf(fid, 'Average classification accuracy: %f\n', Ravg);
 
fprintf('Standard deviation: %f\n', Rstd);
fprintf(fid, 'Standard deviation: %f\n', Rstd);

fprintf('===============================================');
end
fclose(fid);
end