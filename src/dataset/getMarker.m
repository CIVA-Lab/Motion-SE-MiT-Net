function getMarker(folderName, bwTh, rbwTH)

    stPath = strcat(folderName,'SEG/');

    bSTPath = strcat(folderName,'BSEG/');
    markerPath = strcat(folderName,'MARKER/');

    if (0==isdir(bSTPath))
        mkdir(bSTPath);
    end

    if (0==isdir(markerPath))
        mkdir(markerPath);
    end

    flist=dir(fullfile(stPath,'*.tif'));
    n = length(flist);

    for fr = 1 : n

        if contains(flist(fr).name, '._')
            continue;
        end

        nName = strip(flist(fr).name,'right','f');
        nName = strip(nName,'right','i');
        nName = strip(nName,'right','t');
        fileName = strip(nName,'right','.');

        st = imread(fullfile(stPath, flist(fr).name));

        bST = imbinarize(st, 0.0);
        bST = bST * 255;

        areaCC = regionprops(st,'Area');
        totalNumberOfCells = size(areaCC,1);

        sCell = zeros(size(st));
        mImgAll = zeros(size(st));

        % getting area of each cell
        for cell = 1: totalNumberOfCells
            cellArea = areaCC(cell).Area;

            sCell (st == cell) = 1;

            % erode according to cell area
            if (cellArea ~= 0)
                
               if (contains(folderName, 'MSC'))
                    if cellArea <= 200
                        newSE = 1;
                    else
                        newSE = 6;
                   end
               elseif (contains(folderName, 'N2DL-HeLa'))
                   if cellArea <= 50
                        newSE = 0;
                    elseif cellArea <= 100
                        newSE = 2;
                    else 
                        newSE = 4;
                   end
               elseif (contains(folderName, 'GOWT1'))
                   if cellArea <= 50
                        newSE = 0;
                    elseif cellArea <= 100
                        newSE = 4;
                    else 
                        newSE = 6;
                   end
               elseif (contains(folderName, 'PSC'))
                   newSE = 2;
               else
                    newSE = diskValue(cellArea);
               end
                
                se = strel('disk',ceil(newSE));
                sCell = imerode(sCell,se);
                mImgAll = mImgAll + sCell;
            end 

            sCell (st == cell) = 0;
        end

        % remove small blobs
        mImgAll = bwareaopen(mImgAll, bwTh);
        % fill blobs with small holes inside
        mImgAll = 1-bwareaopen(1-mImgAll, rbwTH);

        shapeMarker = imbinarize(mImgAll, 0.0);
        shapeMarker = shapeMarker * 255;

        bST = uint8(bST);
        shapeMarker = uint8(shapeMarker);

        fullfile(bSTPath, [fileName, '.png'])
        
        imwrite(bST, fullfile(bSTPath, [fileName, '.png']));
        imwrite(shapeMarker, fullfile(markerPath, [fileName, '.png']));
    end
end

% select erosion values according to the cell area
function dV = diskValue(cellArea)
    if cellArea <= 70
        dV = 0;
    elseif cellArea <= 120
        dV = 1;
    elseif cellArea <= 500
        dV = 4;
    elseif cellArea <= 1000
        dV = 6;
    else 
        dV = 10;
    end
end


 

