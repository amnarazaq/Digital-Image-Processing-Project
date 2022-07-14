clc
close all
%Loading The Image Dataset
rootfolder='D:\AMBER\SEMESTER 7\DIGITAL IMAGE PROCESSING\proj\dataset';
allfoldernames=struct2table(dir(rootfolder));
for i=3:height(allfoldernames)
    new(i-2)=allfoldernames.name(i);
end
categories=new;
imds=imageDatastore(fullfile(rootfolder,categories),'LabelSource','foldernames');
tbl=countEachLabel(imds);
minSetCount= min(tbl{:,2});
%Visualizing The Dataset
numimages=numel(imds.Labels);
idx=randperm(numimages,16);
figure('Name','Random Dataset','NumberTitle','off');
for i=1:16
    subplot(4,4,i);
    I=readimage(imds,idx(i));
    imshow(I);
end
picResize=[640 480];
%Resizing Converting to grayFeatures Extraction
for i=1:size(imds.Files,1)
    %Resizing Converting To Gray
    Image=imresize(imread(char(imds.Files(i))),picResize);
    I=rgb2gray(Image);
    F=medfilt2(I);
   BW=edge(F,'roberts');
    LGBFeatures=extractLBPFeatures(BW,'CellSize',[128 128]);
    HOGFeatures=extractHOGFeatures(BW,'CellSize',[128 128]);
    FusedFeatures(i,:)=horzcat(LGBFeatures,HOGFeatures);
   % FusedFeatures1=horzcat(FusedFeatures,string(Labels(i)));
   
end
%Defining the Labels
Labels=imds.Labels(:,1);
%FusedFeatures1=categorical(FusedFeatures);
FusedFeatures1=table(FusedFeatures,categorical(Labels));
%FusedFeatures1=horzcat(FusedFeatures,string(Labels));
%FusedFeatures2=double(FusedFeatures1);

%Splitting The Dataset into Training(80%) and Testing(20%)
total_examples= size(FusedFeatures,1);
Num_train_examples= round(total_examples * .8);
Num_testing_examples= total_examples - Num_train_examples;
%Getting the random indexes for random data training
indexs=randperm(total_examples,Num_train_examples);
train_examples=FusedFeatures(indexs,:);
%indexs1=randperm(total_examples,total_examples);
%sample=FusedFeatures(indexs1,:);
train_labels=Labels(indexs);
idx=zeros(1,total_examples);
idx(indexs)=1;
idx=logical(1-idx);
validation_example=FusedFeatures(idx,:);
validation_labels=Labels(idx);
%Train the SVM Classifier
%trainedClassifier=fitcecoc(train_examples,train_labels);
%train naive bayes classifier
%trainedClassifier=fitcnb(train_examples,train_labels);
%rng(1); % For reproducibility
%trainedClassifier= fitctree(train_examples,train_labels,'CrossVal','on');
trainedClassifier=fitcknn(train_examples,train_labels,'NumNeighbors',7,'Standardize',1);
%Testing the Classifier
predictrd_labels=predict(trainedClassifier,validation_example);
%Calculating the Accuracy
correctly_classified=0;
incorrectly_classified=0;
for i=1:size(predictrd_labels,1)
    if(validation_labels(i,1) == (predictrd_labels(i,1)))
        correctly_classified=correctly_classified + 1;
    else
        incorrectly_classified=incorrectly_classified+1;
    end
end
accuracy=correctly_classified/size(predictrd_labels,1);
fprintf('Validation Accuracy =%.2f \n',accuracy*100);
%error rate:
error=1-accuracy;
fprintf('Validation error rate =%.2f \n',error*100);

%Validation
idx=randperm(numel(imds.Files),10);
figure('Name','Validation','NumberTitle','off');
for i=1:10
    subplot(5,2,i);
    IRGB=readimage(imds,idx(i));
    I=rgb2gray(imresize(IRGB,picResize));
    F=medfilt2(I);
    BW=edge(F,'roberts');
    %Extracting Features
    ValLBPFeatures=extractLBPFeatures(BW,'CellSize',[128 128]);
    ValHOGFeatures=extractHOGFeatures(BW,'CellSize',[128 128]);
    ValidationFusedFeatures(i,:)=horzcat(ValLBPFeatures,ValHOGFeatures);
    imshow(IRGB);
    ValLable=predict(trainedClassifier,ValidationFusedFeatures);
    title(string(ValLable(i)));
end