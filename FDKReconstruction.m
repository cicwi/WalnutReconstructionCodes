%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This matlab script computes an FDK reconstruction for one of the data sets 
% described in 
% "A Cone-Beam X-Ray CT Data Collection Designed for Machine Learning" by 
% Henri Der Sarkissian, Felix Lucka, Maureen van Eijnatten, 
% Giulia Colacicco, Sophia Bethany Coban, Kees Joost Batenburg
% Please note that the reconstruction shown in the paper were performed
% using the corresponding Python code, not this matlab code!
%
% author: Felix Lucka
% date:        18.03.2019
% last update: 16.09.2020
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all
clc
close all

% add ASTRA toolbox to the matlab path
% Note that for obtaining a comparable scaling of the image intensities 
% between FDK and iterative reconstructions, it is required to use a
% development version of the ASTRA toolbox more recent than 1.9.0dev!
addpath(genpath('~/astra-toolbox/matlab'))

% the script assumes that all zip files were extracted into one folder,
% please enter the path on your system here
data_dir_root  = '~/Walnuts/';

%% set reconstruction parameters

% which walnut data should be reconstructed?
walnut_id = 1; 
% define which source orbit data should be used (2 = middle orbit)
orbit_id  = 2;
% define a sub-sampling factor in angular direction
angluar_sub_sampling = 1;
% number of voxels per mm in one direction (higher = larger res)
voxel_per_mm = 10;  

%% set up scanning and volume geometry

% construct path to data
data_dir  = [data_dir_root 'Walnut' int2str(walnut_id) ...
                '/Projections/tubeV' int2str(orbit_id) '/'];
            
% generate reconstruction geometry
vol_sz      = (50 * voxel_per_mm + 1) * ones(1, 3);
vol_geom    = astra_create_vol_geom(vol_sz);
% By default, ASTRA assumes a voxel size of 1, we need to scale the 
% reconstruction space here by the actual voxel size
vol_geom.option.WindowMaxX = vol_geom.option.WindowMaxX / voxel_per_mm;
vol_geom.option.WindowMinX = vol_geom.option.WindowMinX / voxel_per_mm;
vol_geom.option.WindowMaxY = vol_geom.option.WindowMaxY / voxel_per_mm;
vol_geom.option.WindowMinY = vol_geom.option.WindowMinY / voxel_per_mm;
vol_geom.option.WindowMaxZ = vol_geom.option.WindowMaxZ / voxel_per_mm;
vol_geom.option.WindowMinZ = vol_geom.option.WindowMinZ / voxel_per_mm;

% set up projection geometry
detector_sz = [972, 768]; % detector size

proj_geom                  = [];
proj_geom.type             = 'cone_vec';
proj_geom.DetectorRowCount = detector_sz(1);
proj_geom.DetectorColCount = detector_sz(2);
proj_geom.Vectors          = importdata([data_dir 'scan_geom_corrected.geom']);
% sub-sample in angle, note that the total number of projection is in fact 1201, but the 
% first and last projection come from the same angle and are omitted here
proj_geom.Vectors          = proj_geom.Vectors(1:angluar_sub_sampling:1200, :);

n_pro      = size(proj_geom.Vectors, 1); 

%% read in and normalize all data

% get all projections
pro_files = dir([data_dir, 'scan_*.tif']);
% we need to read in the projection in reverse order due to the portrait
% mode acquision 
pro_files = pro_files(1200:-angluar_sub_sampling:1);

% transformation to apply to each image, we need to get the image from 
% the way the scanner reads it out into to way described in the projection
% geometry
trafo = @(x) flipud(x)';

% read in dark and flat field
dark_field  = trafo(double(imread([data_dir 'di000000.tif'])));
flat_field1 = trafo(double(imread([data_dir 'io000000.tif'])));
flat_field2 = trafo(double(imread([data_dir 'io000001.tif'])));
% we simply average the flat fields here
flat_field = (flat_field1 + flat_field2) / 2;

data      = zeros([n_pro, size(dark_field)]);

% loop over projection data
for i_pro = 1:n_pro
    pro   = trafo(double(imread([data_dir, pro_files(i_pro).name])));
    % flat and dark field correction
    data(i_pro,:, :) = (pro - dark_field)./ (flat_field - dark_field);
end

% reset values smaller or equal to 0
data(data <= 0) = min(data(data > 0));
% values larger than 1 are clipped to 1
data = min(data, 1);
% log data
data = - log(data);

% permute data to ASTRA convention
data = permute(data, [3,1,2]);

%% compute the FDK reconstruction

% Create astra objects for the reconstruction
rec_id  = astra_mex_data3d('create', '-vol', vol_geom, 0);
proj_id = astra_mex_data3d('create', '-proj3d', proj_geom, data);
clear data

% Set up the parameters for a reconstruction algorithm using the GPU
cfg                      = astra_struct('FDK_CUDA');
cfg.ReconstructionDataId = rec_id;
cfg.ProjectionDataId     = proj_id;

% Create the algorithm object from the configuration structure
alg_id = astra_mex_algorithm('create', cfg);

% Run FDK
clock_cmp = tic;
astra_mex_algorithm('run', alg_id);
toc(clock_cmp)

% Get the result
rec = astra_mex_data3d('get', rec_id);

% Clean up. Note that GPU memory is tied up in the algorithm object,
% and main RAM in the data objects.
astra_mex_algorithm('delete', alg_id);
astra_mex_data3d('delete', rec_id, proj_id);

%% simple slice visulization

figure();
sliceX = squeeze(rec(ceil(vol_sz(1)/2), :, :));
sliceY = squeeze(rec(:, ceil(vol_sz(1)/2), :));
sliceZ = squeeze(rec(:, :, ceil(vol_sz(1)/2)));
clim = [0, max([sliceX(:);sliceY(:);sliceZ(:)])];
subplot(1, 3, 1); imagesc(sliceX, clim);
subplot(1, 3, 2); imagesc(sliceY, clim);
subplot(1, 3, 3); imagesc(sliceZ, clim);
