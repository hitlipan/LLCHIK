function [B] = computeKMeans(dicSize, allSifts, imageSeparateRecord)
%输入：sift目录
%输出：Kmeans文件， $15Scenes_SIFT_Kmeans_1024.mat

path(path, './PG_SPBOW');
path(path, '.\PG_SPBOW');

disp(size(allSifts));
disp(['Computing k means......']);
disp('Transposing......');
allSifts = allSifts';
disp('After transpose!');
%[idx, C] = kmeans(allSifts, 1024, 'OnlinePhase', 'on');
%调用GP_SPBOW的kmenas
%disp(imageSeparateRecord);
C = calculateDictionaryLP(dicSize, allSifts, imageSeparateRecord);
B = C;
end
