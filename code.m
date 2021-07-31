%MSER
%Multi-Scale Morphology segmentation & stroke width transform to filter non-text regions

P = imread('mark.jpg');
I = rgb2gray(P);
disp(size(P));
% Detecting MSER features using detectMSERfeartures function.
[mserRegions, mserConnectedComponents] = detectMSERFeatures(I, 'RegionAreaRange',[250 8000],'ThresholdDelta',3); %80, 0.8
% RegionAreaRange - Size of the region in pixels (regiony w których chcemy by znajdował się tekst) 
% ThresholdDelta - Step size between intensity threshold levels, percentage of the input data type range used in selecting extremal regions while testing for their stability. Decrease this value to return more regions. Typical values range from 0.8 to 4.
figure
imshow(I)
hold on
plot(mserRegions, 'showPixelList', true,'showEllipses',false)
title('MSER regions')
hold off

% Measuring the MSER properties, some properties as string we are intrested
% in
mserStats = regionprops(mserConnectedComponents, 'BoundingBox', 'Eccentricity', 'Solidity', 'Extent', 'Euler', 'Image');
% BoundingBox - Position and size of the smallest box containing the region
% Eccentricity - Eccentricity of the ellipse that has the same second-moments as the region
% Solidity - Proportion of the pixels in the convex hull that are also in the region
% Extent - Ratio of pixels in the region to pixels in the total bounding box
% Euler - Number of objects in the region minus the number of holes in those objects
% Image - Image the same size as the bounding box of the region, returned as a binary (logical) array

% Calculating the aspect ratio using bounding box -> Concatenate arrays vertically
bbox = vertcat(mserStats.BoundingBox);
w = bbox(:,3);
h = bbox(:,4);
aspectRatio = w./h;

%  DeterminIng which regions to remove.
filterregions = aspectRatio' > 3;
filterregions = filterregions | [mserStats.Eccentricity] > .995 ;
filterregions = filterregions | [mserStats.Solidity] < .3;
filterregions = filterregions | [mserStats.Extent] < 0.2 | [mserStats.Extent] > 0.9;
filterregions = filterregions | [mserStats.EulerNumber] < -4;

% Remove regions
mserStats(filterregions) = [];
mserRegions(filterregions) = [];

figure
imshow(I)
hold on
plot(mserRegions, 'showPixelList', true,'showEllipses',false)
title('After Removing Non-Text Regions Based On Geometric Properties')
hold off

% zero padding and binarization
regionImage = mserStats(6).Image;
regionImage = padarray(regionImage, [1 1]);

% Compute the stroke width image 'intelligent' matching
imagedistance = bwdist(~regionImage); % Euclidean distance transform of the binary image  the binary image BW. For each pixel in BW, the distance transform assigns a number that is the distance between that pixel and the nearest nonzero pixel of BW.
skeletonImage = bwmorph(regionImage, 'thin', inf); % Morphological operations on binary image
%applies the operation n times. n can be Inf, in which case the operation is repeated until the image no longer changes.

strokeWidthImage = imagedistance;
strokeWidthImage(~skeletonImage) = 0;

strokeWidthValues = imagedistance(skeletonImage);
strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);
%  stroke width variation metric
strokeWidthThreshold = 0.4;
strokeWidthFilterIdx = strokeWidthMetric > strokeWidthThreshold;
% Processing the remaining regions

for j = 1:numel(mserStats) % = length but more efficient calculated

    regionImage = mserStats(j).Image;
    regionImage = padarray(regionImage, [1 1], 0);

    imagedistance = bwdist(~regionImage);
    skeletonImage = bwmorph(regionImage, 'thin', inf);

    strokeWidthValues = imagedistance(skeletonImage);

    strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);

    strokeWidthFilterIdx(j) = strokeWidthMetric > strokeWidthThreshold;

end

% Remove regions based on the stroke width variation
mserRegions(strokeWidthFilterIdx) = [];
mserStats(strokeWidthFilterIdx) = [];

% Show remaining regions
figure
imshow(I)
hold on
plot(mserRegions, 'showPixelList', true,'showEllipses',false)
title('After Removing Non-Text Regions Based On Stroke Width Variation')
hold off
% Get bounding boxes for all the regions
bboxes = vertcat(mserStats.BoundingBox);


xmin = bboxes(:,1);
ymin = bboxes(:,2);
xmax = xmin + bboxes(:,3) - 1;
ymax = ymin + bboxes(:,4) - 1;

% Expanding  the bounding boxes 
%expansionAmount=0.003;
expansionAmount = 0.017;
xmin = (1-expansionAmount) * xmin;
ymin = (1-expansionAmount) * ymin;
xmax = (1+expansionAmount) * xmax;
ymax = (1+expansionAmount) * ymax;


xmin = max(xmin, 1);
ymin = max(ymin, 1);
xmax = min(xmax, size(I,2));
ymax = min(ymax, size(I,1));

% Showing the expanded bounding boxes
expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
IExpandedBBoxes = insertShape(P,'Rectangle',expandedBBoxes,'LineWidth',3);

figure
imshow(IExpandedBBoxes)
title('Expanded Bounding Boxes Text')
% Compute the overlap ratio
overlapRatio = bboxOverlapRatio(expandedBBoxes, expandedBBoxes);


% simplify the graph representation.
n = size(overlapRatio,1);
overlapRatio(1:n+1:n^2) = 0;
g = graph(overlapRatio);

% Finding the connected text regions 
componentIndices = conncomp(g); % returns the connected components of graph G as bins
% Merging the boxes 
xmin = accumarray(componentIndices', xmin, [], @min);
ymin = accumarray(componentIndices', ymin, [], @min);
xmax = accumarray(componentIndices', xmax, [], @max);
ymax = accumarray(componentIndices', ymax, [], @max);

text = [xmin ymin xmax-xmin+1 ymax-ymin+1];

numRegionsInGroup = histcounts(componentIndices);
text(numRegionsInGroup == 1, :) = [];

% Show the final text detection result.
ITextRegion = insertShape(P, 'Rectangle', text,'LineWidth',3);

figure
imshow(ITextRegion)
title('Detected Text')
ocrtxt = ocr(I, text); % recognizes text in I within one or more rectangular regions. The text input contains an M-by-4 matrix, with M regions of interest.
[ocrtxt.Text] % text from image