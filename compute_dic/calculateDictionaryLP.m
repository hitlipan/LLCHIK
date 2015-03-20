function dictionary = calculateDictionaryLP(dicSize, allSifts, imageSeparateRecord)

    [m, n] = size(allSifts); %allSifts row vector,
    dictionarySize = dicSize;
    nimages = length(imageSeparateRecord);
    disp(['image num : ', num2str(nimages)]);
    niters = 20;
    centres = zeros(dictionarySize, n);
    data = allSifts(1:2048, :);
    [ndata, data_dim] = size(data);
    [ncentres, dim] = size(centres);
    
    %initialization
    perm = randperm(ndata);
    perm = perm(1:ncentres);
    centres = data(perm, :);
    old_centres = centres;
    display('Run k-means');
    
    for n = 1:niters
        e2 = max(max(abs(centres - old_centres)));
        old_centres = centres;
        tempc = zeros(ncentres, dim);
        num_points=zeros(1,ncentres);
        
        for f = 1:nimages
            if f == 1
             fprintf('The %d th interation the %d th image. eCenter=%f \n',n,f,e2); 
            end
           id = eye(ncentres);
           if (f > 1)
                data = allSifts( ((imageSeparateRecord(f - 1) + 1): imageSeparateRecord(f)),:);
           else
                data = allSifts(1 : imageSeparateRecord(1), :);
           end
           d2 = EuclideanDistance(data, centres);
           
           [minvals, index] = min(d2', [], 1);
           post = id(index, :);
           num_points = num_points + sum(post, 1);
           
           for j = 1:ncentres
            tempc(j, :) = tempc(j, :) + sum(data(find(post(:, j)), :), 1);
           end
           
        end
        for j = 1:ncentres
            if num_points(j) > 0
                centres(j, :) = tempc(j, :) / num_points(j);
            end
        end
        
        if n > 1
            ThrError = 0.1;
            if max(max(abs(centres - old_centres))) <ThrError
                %dictionary= centres;
               % fprintf('Saving  dictionary......\n');
                %save ('15Scenes_KMeans_1024.mat', 'dictionary')     % save the settings of descriptor in opts.globaldatapath
                break;
            end
            
            fprintf('The %d th interation finished \n',n);
       
        end
       
    end
    dictionary = centres;
end

