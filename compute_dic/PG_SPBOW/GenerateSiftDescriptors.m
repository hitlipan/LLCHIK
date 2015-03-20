function [] = GenerateSiftDescriptors(opts,descriptor_opts)
%function [] = GenerateSiftDescriptors( opts,descriptor_opts )
%
%Generate the dense grid of sift descriptors for each
%
fprintf('Building Sift Descriptors\n\n');

%% parameters
descriptor_flag=1;
maxImageSize = descriptor_opts.maxImageSize;
gridSpacing = descriptor_opts.gridSpacing;
patchSize = descriptor_opts.patchSize;

try
    descriptor_opts2=getfield(load([opts.globaldatapath,'/',descriptor_opts.name,'_settings']),'descriptor_opts');
    if(isequal(descriptor_opts,descriptor_opts2))
        descriptor_flag=0;
        display('descriptor has already been computed for this settings');
    else
        display('Overwriting descriptor with same name, but other descriptor settings !!!!!!!!!!');
    end
end


if(descriptor_flag)
    
    %% load image
    imgpath=strcat(opts.imgpath,'/*.jpg');
    fileNames = dir(imgpath); % load image in data set
    nimages=length(fileNames);           % number of images in data set
    
    h = waitbar(0,'compute SIFT...');
    for f = 1:nimages
        imgpath=[opts.imgpath,'/',fileNames(f).name];
        I=load_image(imgpath);
        
        tic
        
        [hgt wid] = size(I);
        if min(hgt,wid) > maxImageSize
            I = imresize(I, maxImageSize/min(hgt,wid), 'bicubic');
            fprintf('Loaded %s: original size %d x %d, resizing to %d x %d\n', ...
                fileNames(f).name, wid, hgt, size(I,2), size(I,1));
            [hgt wid] = size(I);
        end
        
        %% make grid (coordinates of upper left patch corners)
        remX = mod(wid-patchSize,gridSpacing);% the right edge
        offsetX = floor(remX/2)+1;
        remY = mod(hgt-patchSize,gridSpacing);
        offsetY = floor(remY/2)+1;
        
        [gridX,gridY] = meshgrid(offsetX:gridSpacing:wid-patchSize+1, offsetY:gridSpacing:hgt-patchSize+1);
        
        fprintf('Processing %s: wid %d, hgt %d, grid size: %d x %d, %d patches\n', ...
            fileNames(f).name, wid, hgt, size(gridX,2), size(gridX,1), numel(gridX));
        
        %% find SIFT descriptors
        siftArr = find_sift_grid(I, gridX, gridY, patchSize, 0.8);
        siftArr = normalize_sift(siftArr);
        
        features.data = siftArr;
        features.x = gridX(:) + patchSize/2 - 0.5;
        features.y = gridY(:) + patchSize/2 - 0.5;
        features.wid = wid;
        features.hgt = hgt;
        
        toc
        waitbar(f/nimages,h);
        
        image_dir=sprintf('%s/%s/',opts.localdatapath,num2string(f,3)); % location descriptor
        save ([image_dir,'/',descriptor_opts.featureName], 'features');           % save the descriptors
        
        fprintf('The SFIT of %d images...\n',f);
        
        
        
    end % for
    close(h);
    save ([opts.globaldatapath,'/',descriptor_opts.name,'_settings'],'descriptor_opts');      % save the settings of descriptor in opts.globaldatapath
end % if

end% function
