function postProcessing(folderName, datasetName, seqNumber, maskTH, markerTH, bwTH)

    maskPath = strcat(folderName, datasetName, seqNumber, 'Mask/');
    markerPath = strcat(folderName, datasetName, seqNumber,'Marker/');

    labelPath = strcat(folderName, datasetName, seqNumber, 'LabelMat/');
    labelColorPath = strcat(folderName, datasetName, seqNumber, 'LabelMat_Vis/');

    if (0==isdir(labelPath))
        mkdir(labelPath);
    end

    if (0==isdir(labelColorPath))
        mkdir(labelColorPath);
    end

    flist=dir(fullfile(maskPath,'*.png'));
    n = length(flist);

    for fr = 1 : n

        if contains(flist(fr).name, '._')
            continue;
        end

        nName = strip(flist(fr).name,'right','g');
        nName = strip(nName,'right','n');
        nName = strip(nName,'right','p');
        fileName = strip(nName,'right','.');

        mask = imread(fullfile(maskPath, flist(fr).name));
        marker = imread(fullfile(markerPath, flist(fr).name));

        % threshold mask to binary
        imgMask = im2bw(mask, maskTH); 
        % remove small blobs
        imgMask = bwareaopen(imgMask, bwTH);

        % threshold marker to binary
        imgMarker = im2bw(marker, markerTH);
        % remove small blobs
        imgMarker = bwareaopen(imgMarker, bwTH);

        % apply watershed algorithm
        watershedResult = watershed(~imgMarker);
        % figure; imshow(watershedResult,[]);

        boundriesWatershed = watershedResult == 0;
        % figure; imshow(boundriesWatershed);

        imgMask(boundriesWatershed == 1) = 0;
        % figure; imshow(imgMask);
        
        % remove small blobs
        imgMask = bwareaopen(imgMask, bwTH); 
        % fill blobs with small holes inside
        imgMask = 1-bwareaopen(1-imgMask, bwTH);
        % figure; imshow(imgMask);

        % label mask
        labelMask = bwlabel(imgMask);
        % labelMaskColor = label2rgb(labelMask,'jet','black','shuffle');
        % figure; imshow(labelMaskColor);  

        maxN = max(max(labelMask));
        finalMask = zeros(size(labelMask));
        se = strel('disk', 1);

        % figure
        % imshow(se.Neighborhood)

        % dilate labeled masks
        for nM = 1 : maxN
            tempM = zeros(size(labelMask));
            tempM (labelMask == nM) = 1;
            % figure; imshow(tempM);
            dImg = imdilate(tempM,se);
            % figure; imshow(dImg);
            finalMask(dImg == 1) = nM;
        end

        finalMaskColor = label2rgb(finalMask,'jet','black','shuffle');
        % figure; imshow(finalMaskColor); 

        bST = uint8(imgMask*255);
        shapeMarker = uint8(imgMarker*255);

        rgb = zeros(size(bST,1), size(bST,2), 3);

        rgb(:,:,1) = bST;
        rgb(:,:,2) = shapeMarker;

        bImg = cat(3, bST, bST, bST);
        sImg = cat(3, shapeMarker, shapeMarker, shapeMarker);

        combined = [bImg sImg;
            rgb finalMaskColor];

        fullfile(labelPath, [fileName, '.tif'])
        
        imwrite(uint16(finalMask), fullfile(labelPath, [fileName, '.tif']));
        imwrite(combined, fullfile(labelColorPath, flist(fr).name));
    end

end


 

