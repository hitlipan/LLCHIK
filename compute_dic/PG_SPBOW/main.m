%% initialize the settings
display('*********** start BOW *********')
clc;
clear;

%% Set Path

rootpath='E:\workspace\matlab\work\PG_SPBOW\';

images=strcat(rootpath,'images');
data=strcat(rootpath,'data');

mypath.imgpath=images; % image path
mypath.datapath=data;

% local and global data paths
mypath.localdatapath=sprintf('%s/local',mypath.datapath);
mypath.globaldatapath=sprintf('%s/global',mypath.datapath);

imgpath=strcat(mypath.imgpath,'/*.jpg');
fileNames = dir(imgpath);
nimages=length(fileNames); % number of images in data set
mypath.nimages=nimages;

make_directory_structure(mypath);  % mkdir data/local and data/global




for i=1:nimages
    image_names(1,i)={fileNames(i).name};
end
save ([mypath.globaldatapath,'/','image_names'], 'image_names');
% initialize the image names
mypath.image_names=sprintf('%s/image_names.mat',mypath.globaldatapath);



%% parameters

descriptor_opts=[];dictionary_opts=[];assignment_opts=[];


%% Descriptors

descriptor_opts.type='sift';                                                    % name descriptor
descriptor_opts.name=['DES',descriptor_opts.type];                              % output name (combines detector and descrtiptor name)
descriptor_opts.patchSize=16;                                                   % normalized patch size
descriptor_opts.gridSpacing=8;
descriptor_opts.maxImageSize=1000;
descriptor_opts.featureName='sift_features';  
GenerateSiftDescriptors(mypath,descriptor_opts);


%% Create the texton dictionary

dictionary_opts.dictionarySize = 200;
dictionary_opts.name='dictionary';
dictionary_opts.featureName=descriptor_opts.featureName;
CalculateDictionary(mypath, dictionary_opts);

%% assignment

assignment_opts.type='1nn';                                 % name of assignment method
assignment_opts.descriptor_name=descriptor_opts.name;       % name of descriptor (input)
assignment_opts.dictionary_name=dictionary_opts.name;       % name of dictionary
assignment_opts.name=['BOW_',descriptor_opts.type];         % name of assignment output
assignment_opts.featureName=dictionary_opts.featureName;
assignment_opts.texton_name='texton_ind';
do_assignment(mypath,assignment_opts);

%% CompilePyramid
pyramid_opts.name='spatial_pyramid';
pyramid_opts.dictionarySize=dictionary_opts.dictionarySize;
pyramid_opts.pyramidLevels=4;                               % pyramidLevels > 0
pyramid_opts.texton_name=assignment_opts.texton_name;
CompilePyramid(mypath,pyramid_opts);
