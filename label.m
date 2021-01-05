%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Author: Imoleayo Abel								 %
%     Course: Sensing and Estimation Robotics (ECE 276A) %
%    Quarter: Fall 2017									 %
% Instructor: Nikolay Atanasov						     %
%    Project: 01 - Color Segementation                   %
%       File: label.m                                    %
%       Date: Oct-30-2017                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
clc;
close all;

% folder to read images from
imagefiles = dir('trainset/*.png');

% color classes
colors = {'bred','nonbred','brown','blue'};

for color = colors
    fprintf('\n\n Starting new color \n\n');
    data = [];    % array of pixels selected in region of interest
    
    % for each image
    for image = imagefiles'
        imgName = strcat('trainset/',image.name);
        
        % concatenate RGB values for all ROIs selected
        data = [data, getROIs(imgName,color)];
    end
    
    % save data in a mat file in mat_files folder
    if ~isdir('mat_files')
        mkdir mat_files
    end
    save(strcat("mat_files/",color,".mat"),'data');
    close all;
end

% Process single image to collect several ROIs
%
% Return: 3xn array of RGB values of all n-pixels in selected regions of
% interest
function rgbVals = getROIs(img,label)
    disp(strcat('Labeling color:... ',label,'...'));
    I = imread(img);
    
    rgbVals = [];
    
    % collect several ROIs until user decides to stop
    while 1
        mask = roipoly(I);
        
        % save off RGB values in selected ROI
        Ir = I(:,:,1); ImaskR = Ir(mask);
        Ig = I(:,:,2); ImaskG = Ig(mask);
        Ib = I(:,:,3); ImaskB = Ib(mask);
        ImaskRGB = [ImaskR, ImaskG, ImaskB];
        
        % populate array with RGB for all ROIs
        rgbVals = [rgbVals; ImaskRGB];       
        
        % if user wants to collect more ROIs, update pixels of already 
        % selected ROIs to color white and repear ROI collection process
        str = input('Do you want more ROIs? Y/N [N]: ','s');
        if strcmpi(str,'Y') || strcmpi(str,'Yes')
            I = showMask(I,mask);
            continue
        else
            break
        end
    end
    rgbVals = rgbVals';
end

% use 'mask' to set corresponding pixels in img to color white
function imgMask = showMask(img,mask)
    ir = img(:,:,1); ir(mask) = 255;
    ig = img(:,:,2); ig(mask) = 255;
    ib = img(:,:,3); ib(mask) = 255;
    imgMask(:,:,1) = ir;
    imgMask(:,:,2) = ig;
    imgMask(:,:,3) = ib;
end
