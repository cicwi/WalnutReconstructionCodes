%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This matlab script computes an iterative reconstruction for the full
% data of one of the data sets described in
% "A Cone-Beam X-Ray CT Data Collection Designed for Machine Learning" by
% Henri Der Sarkissian, Felix Lucka, Maureen van Eijnatten,
% Giulia Colacicco, Sophia Bethany Coban, Kees Joost Batenburg
% Please note that the reconstruction shown in the paper were performed
% using the corresponding Python code, not this matlab code!
%
% author: Felix Lucka
% date:        11.04.2019
% last update: 06.11.2020
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

% this script makes use of the Spot Linear-Operator Toolbox,
% see s017_opTomo.m in the matlab samples of the ASTRA toolbox and
% http://www.cs.ubc.ca/labs/scl/spot/
addpath('~/spot/')

% the script assumes that all zip files were extracted into one folder,
% please enter the path on your system here
data_dir_root  = '~/Walnuts/';

%% set reconstruction parameters

% which walnut data should be reconstructed?
walnut_id = 1;
% define a sub-sampling factor in angular direction
angluar_sub_sampling = 10;
% number of voxels per mm in one direction (higher = larger res)
voxel_per_mm = 10;

%% set up scanning and volume geometry

% construct path to data
data_dir  = [data_dir_root 'Walnut' int2str(walnut_id) ...
                '/Projections/'];

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

% get vector describtion from all orbits
proj_geom.Vectors = [];
for orbit=1:3
    vectors_orbit = importdata([data_dir 'tubeV' int2str(orbit) '/scan_geom_corrected.geom']);
    % sub-sample in angle, note that the total number of projection is in fact 1201, but the
    % first and last projection come from the same angle and are omitted here
    vectors_orbit          = vectors_orbit(1:angluar_sub_sampling:1200, :);
    proj_geom.Vectors = [proj_geom.Vectors; vectors_orbit];
end

n_pro      = size(proj_geom.Vectors, 1);


%% read in and normalize all data

% transformation to apply to each image, we need to get the image from
% the way the scanner reads it out into to way described in the projection
% geometry
trafo = @(x) flipud(x)';

data = [];

for orbit=1:3

    % get all projections for this orbit
    pro_files_orbit = dir([data_dir 'tubeV' int2str(orbit) '/scan_*.tif']);
    % we need to read in the projection in reverse order due to the portrait
    % mode acquision
    pro_files_orbit = pro_files_orbit(1201:-angluar_sub_sampling:2);

    % read in dark and flat field
    dark_field  = trafo(double(imread([data_dir 'tubeV' int2str(orbit) '/di000000.tif'])));
    flat_field1 = trafo(double(imread([data_dir 'tubeV' int2str(orbit) '/io000000.tif'])));
    flat_field2 = trafo(double(imread([data_dir 'tubeV' int2str(orbit) '/io000001.tif'])));
    % we simply average the flat fields here
    flat_field = (flat_field1 + flat_field2) / 2;

    data_orbit = zeros([length(pro_files_orbit), size(dark_field)]);

    % loop over projection data
    for i_pro = 1:length(pro_files_orbit)
        pro   = trafo(double(imread([data_dir 'tubeV' int2str(orbit) '/' pro_files_orbit(i_pro).name])));
        % flat and dark field correction
        data_orbit(i_pro,:, :) = (pro - dark_field)./ (flat_field - dark_field);
    end

    % merge with other orbits
    data = cat(1, data, data_orbit);
    clear data_orbit
end

% reset values smaller or equal to 0
data(data <= 0) = min(data(data > 0));
% values larger than 1 are clipped to 1
data = min(data, 1);
% log data
data = - log(data);

% permute data to ASTRA convention
data = permute(data, [3,1,2]);
% after this, we only use the data in vector form;
data = data(:);

%% prepare the iterative reconstruction

% we create a spot operator
A = opTomo('cuda', proj_geom, vol_geom);

% we create functions to translate between volume and vector representation
% of images
im2vec = @(im) im(:);
vec2im = @(vec) reshape(vec, vol_sz);

% to choose a good step size, we approximate the Lipschitzfactor of A^T * A
% by 10 Von Mises power iterations. Note that this needs to be done only
% once for a fixed scanning setup and could also be done in a faster way

% start with random image
x          = im2vec(rand(vol_sz));
x          = x/norm(x);
max_iter   = 10;
lip_approx = zeros(1, max_iter);

% apply A^T A 10 times
fprintf('approximating Lipschitz factor of A^T A')
for iter = 1:max_iter
    fprintf('.')
    x   = A'*(A * x);
    % the norm of x is an approximation of the largest eigenvalue of (A^T A)
    lip_approx(iter) =  norm(x);
    % normalize
    x   = x/norm(x);
end
disp('done.')

figure();
plot(1:max_iter, lip_approx);
drawnow();

%% accelerated gradient descent a la Nesterov for non-negative least squares
% fit to the data y, i.e., we try to solve min_{x >= 0} 1/2 \| A x - y \|_2^2
% we have written it up in a complicated way to compute the residuum norm
% without additional evaluations of A. Note that this requires more memory
% than normally needed!

y            = data; % rename data to "y"
max_iter     = 50;
data_fit     = zeros(max_iter + 1, 1);
data_fit(1)  = 1/2 * norm(y)^2;   % initial data fit for x = 0
nu           = 1/lip_approx(end); % step size

clock_cmp = tic;
fprintf('running  accelerated gradient descent a la Nesterov for non-negative least squares')

% we start with zero image
x            = im2vec(zeros(vol_sz, 'single'));
t_acc         = 1;
x_old         = x; % all zeros
NRMx          = x; % normal operator A'*A applied to x (which is 0)
NRMx_old      = NRMx;
ATy           = A' * y;
gradient      = (NRMx - ATy);

for iter = 1:max_iter
    fprintf('.')

    % update extrapolation variables
    tau = (t_acc-1)/(t_acc+2);
    t_acc = t_acc + 1;

    % compute descent direction
    descent_direction = gradient - tau/nu * (x - x_old) + tau * (NRMx - NRMx_old);

    % update x and apply non-negativity constraints
    x_old  =  x;
    x      =  x - nu * descent_direction;
    x      = max(x, 0);

    % compute all other updates
    Ax            =  A * x;
    NRMx_old      = NRMx;
    NRMx          = A' * Ax;
    gradient      = (NRMx - ATy);

    % compute residuum
    data_fit(iter + 1) = 1/2 * norm(Ax - y)^2;

    if(data_fit(iter+1) > min(data_fit(1:iter)))
        t_acc = 1; % restart acceleration
        fprintf('!')
    end

end
disp('done.')

toc(clock_cmp)


figure();
plot(0:max_iter, data_fit);
drawnow();

% reshape to image
x = vec2im(x);

%% simple slice visulization

figure();
sliceX = squeeze(x(ceil(vol_sz(1)/2), :, :));
sliceY = squeeze(x(:, ceil(vol_sz(1)/2), :));
sliceZ = squeeze(x(:, :, ceil(vol_sz(1)/2)));
clim = [0, max([sliceX(:);sliceY(:);sliceZ(:)])];
subplot(1, 3, 1); imagesc(sliceX, clim);
subplot(1, 3, 2); imagesc(sliceY, clim);
subplot(1, 3, 3); imagesc(sliceZ, clim);
