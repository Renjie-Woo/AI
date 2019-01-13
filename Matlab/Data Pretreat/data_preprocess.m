%% Data pre-Processing 

% Renjie Wu

%% ========Part1:  Importing data====================

clear;close all;clc;
fprintf("Part 1 begin! Loading data and labels... \n")
t1 = clock;
import_data;                     %import all data into variable all
import_label;                    %import all data into variable ETABM185
t2 = clock;
%all = table2array(all);   
%fprintf("all ok!\n")
time_out = etime(t2,t1);

t3 = clock;
ETABM185 = ETABM185(2:5897,8);  %delete the first row,and extract the 8th (CharacteristicsDiseaseState) column
label = table2cell(ETABM185);  % 5896*1 cell
%fprintf("label ok!\n")
t4 = clock;
fprintf('Data has imported!\n');
%label = unique(ETABM185(:,8));

fprintf('Part 2 begin! Filtering \n');
%% ========Part2: Filtering the datasets==============

data = transpose(all);

index_todele = find(cellfun(@isempty,label));
data(index_todele,:) = [];

label = label(~cellfun(@isempty,label)); % deleting the empty label

%%
 
labels = unique(label);  
index_todele = [];
 
% check if specific label data <10
for i =1:size(labels,1)
     index_t = contains(label,labels(i));      %index of label = labels(i)

     if (nnz(index_t)<10)
         index_todele = [index_todele;find(index_t)];  %deleting those index
     end
     
 end
 
label(index_todele) = [];
data(index_todele,:) = [];
 
labels = unique(label);
 
fprintf('Datasets filtering completed!\n');     

%% ========Part3: PCA on training data====================
% coeff-系数矩阵；score-PCA降维结果;latent-所有主成分的影响率（%）。
[coeff, score, latent] = pca(data); % coefficients
recon_deg = cumsum(latent)./sum(latent); 
fprintf('PCA completed!\n');

fprintf(" > 0.85:")
disp(find(recon_deg > 0.85,1)); %89
fprintf("\n > 0.9:")
disp(find(recon_deg > 0.9,1));  %441
fprintf("\n > 0.95:")
disp(find(recon_deg > 0.95,1)); %1063

fprintf("Next continue...\n")

% 95%  
score = zscore(score);
 
score_95 = score(:,1:1063);
score_90 = score(:,1:441);
 
%score_95 = zscore(score_95);
%score_90 = zscore(score_90);
 
%% dlmwrite('/Users/apple/Desktop/AI_pro/Gene_Chip_Data/z-score_data_95.txt',score_95, 'delimiter', '\t','precision', 16,'newline', 'pc');
%% dlmwrite('/Users/apple/Desktop/AI_pro/Gene_Chip_Data/z-score_data_90.txt',score_90, 'delimiter', '\t','precision', 16,'newline', 'pc');
fprintf('z-score completed!');
%resulting a 3k*1k doubel data matrix and a 3k label matrix 92 labels
 
%%
label_processed = zeros(size(label,1),1);
for i =1:size(labels,1)
    index_t = contains(label,labels(i));
    index_t = find(index_t);
    label_processed(index_t,1) = i;  
end
 
 
% rng(1);
% 
% SVMModel = fitcsvm(data_norm,label_01,'KernelFunction','RBF',...
%     'KernelScale','auto');
% 
% CVSVMModel = crossval(SVMModel);
% 
% classLoss = kfoldLoss(CVSVMModel);
 
%% extracting all labels about cancer/tumor
key_word = ["tumor","cancer","adenocarcinoma",...
"AIDS-KS",...
"leukemia",...
"leukaemia",...
"lymphoma",...
"Sarcoma",...
"adenocarcinoma.",...
"alveolar rhabdomyosarcoma",...
"carcinoma",...
"chondroblastoma",...
"chondromyxoid fibroma",...
"chordoma",...
"colon adenocarcinoma",...
"dedifferentiated chondrosarcoma",...
"embryonal rhabdomyosarcoma",...
"fibromatosis",...
"fibroma",...
"follicular thyroid adenoma",...
"follicular thyroid carcinoma",...
"ganglioneuroblastoma",...
"ganglioneuroma",...
"glioblastoma",...
"grade 2, primary hnscc",...
"high-stage neuroblastoma",...
"hlrcc",...
"iatrogenic-KS, KSHV-",...
"leiomyosarcoma",...
"lipoma",...
"low-stage neuroblastoma",...
"lung adenocarcinoma",...
"sarcoma",...
"myxoid liposarcoma",...
"neuroblastoma",...
"neurofibroma",...
"osteosarcoma",...
"adenoma",...
"schwannoma",...
"t4b",...
"tendon xanthomas",...
"uterine fibroid",...
"well-differentiated liposarcoma","aml",...
"Classic-KS, HIV-, nodular (late) stage",...
"Daudi Burkitt's lymphoma"];
 
cancer_index=[];
 
for i =1:size(key_word,2)
    tmp = strfind(label,key_word(i));
    temp = find(~cellfun(@isempty,tmp));
    cancer_index = [cancer_index;temp];
end
 
cancer_index = unique(cancer_index);  %2639*1 double
sort(cancer_index);
 
label_01 = zeros(size(label));
 
for i = 1: size(cancer_index,1)
    tmp = cancer_index(i);
    label_01(tmp) = 1;
end
 
%% crossvalidation
% 3613*1064
% 95%
%score = score(:,1:1063);
final_data_mul = [score_95,label_processed];
final_data_2 = [score_95,label_01];
%% dlmwrite('/Users/apple/Desktop/AI_pro/Gene_Chip_Data/final_data_mul.txt',final_data_mul, 'delimiter', '\t','precision', 16,'newline', 'pc');
%% dlmwrite('/Users/apple/Desktop/AI_pro/Gene_Chip_Data/final_data_2.txt',final_data_2, 'delimiter', '\t','precision', 16,'newline', 'pc');
fprintf('95: final data generating! \n ');

  
%% 90%
data_ninty = score_90;
data_ninty_mul = [data_ninty,final_data_mul(:,1064)];
data_ninty_2 = [data_ninty,final_data_2(:,1064)];
%% dlmwrite('/Users/apple/Desktop/AI_pro/Gene_Chip_Data/data_ninty_mul.txt',data_ninty_mul, 'delimiter', '\t','precision', 16,'newline', 'pc');
%% dlmwrite('/Users/apple/Desktop/AI_pro/Gene_Chip_Data/data_ninty.txt_2',data_ninty_2, 'delimiter', '\t','precision', 16,'newline', 'pc');
fprintf('90: final data generating! \n ');
