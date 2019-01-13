%% Data prepare
%% pretreatment for data read from "microarray.original.txt"
% Renjie Wu
% 2018-12-25

%path of '.txt' file to be pretreated
filename = '/Users/apple/Desktop/AI_pro/Gene_Chip_Data/microarray.original.txt';

%use importdata to load data
file_read = importdata(filename);
%data store the array we need
all = file_read.data;
%text store the title we don't need
text = file_read.textdata;

%create the output '.txt' file
%file_output = '/Users/apple/Desktop/handled_microarray.original.txt';
%file_id = fopen(file_output,'a+');
%fclose(file_id);

%we use dlmwrite to restore the array
%dlmwrite(file_output, data, 'delimiter', '\t','precision', 16,'newline', 'pc');

clearvars filename file_read text
fprintf('microarray.original.txt patched! \n');
