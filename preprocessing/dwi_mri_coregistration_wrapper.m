%% DWI MRI Pre-processing wrapper script
% This script preprocesses T2 MRI images and associated lesion maps 
% It follows a particular directory structure which must
% be adhered to.
%

%% Clear variables and command window
clear all , clc
%% Specify paths
% Experiment folder
data_path = 'C:\Users\simon\Documents\EPFL\Master\Semestre5\Projet MIPLAB\Data_2016_T2_TRACE_LESION';

if ~(exist(data_path))
    fprintf('Data directory does not exist. Please enter a valid directory.')
end

% Base patient number to take as reference
patient_base = '323197'
patient_base_dir = fullfile(data_path, patient_base);
if ~(exist(patient_base_dir))
    fprintf('The patient used as reference does not exists. Please choose another one.')
else
    mri_dir_base = dir(fullfile(data_path,patient_base, 'Neuro*'));
    if (~ isempty(mri_dir_base))
        mri_dir_base = mri_dir_base.name;
    end
    
    image_patient_base = fullfile(data_path,patient_base,mri_dir_base,...
        strcat('t2_tse_tra_',patient_base,'.nii'))
end


% Subject folders

% Select individual subjects
% subjects = {
% 'patient1'
% };

% Or select subjects based on folders in data_path
d = dir(data_path);
isub = [d(:).isdir]; %# returns logical vector
subjects = {d(isub).name}';
subjects(ismember(subjects,{'.','..'})) = [];


% Base image to co-register to
base_image_dir = data_path;
base_image_prefix = '';
age_ext = '.nii.gz';

addpath(data_path)
%% Initialise SPM defaults
%% Loop to load data from folders and run the job
for i = 1: numel ( subjects )

    mri_dir = dir(fullfile(data_path,subjects{i}, 'Neuro*'));
    if (~ isempty(mri_dir))
        mri_dir = mri_dir.name;
    else
        mri_dir = dir(fullfile(data_path,subjects{i}, 'Irm*'))
        mri_dir = mri_dir.name;
    end
    
%   base_image is the T2
    base_image = fullfile(base_image_dir, subjects{i}, mri_dir, ...
        strcat(base_image_prefix, 'T2W_TSE_tra_', subjects{i}, '.nii'));
    if (~ exist(base_image))
        base_image = fullfile(base_image_dir, subjects{i}, mri_dir, ...
        strcat(base_image_prefix, 't2_tse_tra_', subjects{i}, '.nii'));;
    end
%     if (~ exist(original_base_image))
%         zipped_base_image = strcat(original_base_image, '.gz');
%         gunzip(zipped_base_image);
%     end
  
    % load data for each sequence without a prompt
    dwi_files =  dir(fullfile(data_path, subjects{i}, mri_dir,'*TRACE*'));
    dwi_input = fullfile(data_path, subjects{i}, mri_dir, ...
                 dwi_files.name);

    lesion_map_initial = fullfile(data_path, subjects{i}, ...
                 strcat('VOI_lesion_', subjects{i}, '.nii'));
    lesion_map = fullfile(data_path, subjects{i}, mri_dir, ...
                 strcat('VOI_lesion_', subjects{i}, '.nii'));
    lesion_map_coreg =  fullfile(data_path, subjects{i}, mri_dir, ...
                 strcat('coreg_VOI_lesion_', subjects{i}, '.nii'));
         
    if (exist(lesion_map_initial))
        movefile(lesion_map_initial, lesion_map);
    end
    if (exist(lesion_map))
        copyfile(lesion_map, lesion_map_coreg);
    end

    % display which subject and sequence is being processed
    fprintf('Processing subject "%s" , "%s" \n' ,...
        subjects{i}, dwi_files.name);
    
    %% COREGISTRATION TRACE to T2
    coregistration_to_t2 = coregister_job(base_image, dwi_input, {}, 'coreg_');
    spm('defaults', 'FMRI');
    spm_jobman('run', coregistration_to_t2);

    %% COREGISTRATION T2 to T2 -- to be sure
    coregistration_t2_to_t2 = coregister_job(base_image, base_image, {}, 'coreg_');
    spm('defaults', 'FMRI');
    spm_jobman('run', coregistration_t2_to_t2);
    
    
    %% NORMALISATION TO BASE PATIENT
    dwi_coreg_files =  dir(fullfile(data_path, subjects{i}, mri_dir,'coreg*TRACE*'));
    dwi_coreg_input = fullfile(data_path, subjects{i}, mri_dir, ...
                 dwi_coreg_files.name);
    
    voi_coreg_files =  dir(fullfile(data_path, subjects{i}, mri_dir,'coreg*VOI*'));
    voi_coreg_input = fullfile(data_path, subjects{i}, mri_dir, ...
                 voi_coreg_files.name);
             
    t2_coreg_files =  dir(fullfile(data_path, subjects{i}, mri_dir,'coreg*t2*'));
    t2_coreg_input = fullfile(data_path, subjects{i}, mri_dir, ...
                 t2_coreg_files.name);
    
    other_coreg_images = {dwi_coreg_input;voi_coreg_input};
    patient_coregistration = coregister_job(image_patient_base, t2_coreg_input, other_coreg_images, "w");
    spm('defaults', 'FMRI');
    spm_jobman('run', patient_coregistration);
    
end

