%Seting sample size
load 3D_MT_Model_Dataset.mat
num=10000;     %Total data set
vaild_num=num*0.15;%Number of verification sets
test_num=num*0.05; %Number of test sets
w=4;
outputdata=cell(num,w);
inputdata=cell(num,1);
for i =1:num    

    inputdata{i,1}=data_research_domain1{i};

    outputdata{i,1}=((ResXY{i}));
    outputdata{i,2}=((ResYX{i}));
    outputdata{i,3}=(abs(PhaseXY{i}));
    outputdata{i,4}=(abs(PhaseYX{i}));
end
%Scrambled index
rng(2025);
index1=randperm(num,num)';
inputdata1=inputdata;
outputdata1=outputdata;


% data normalization
outputdata_log=cell(num,w);
inputdata_log=cell(num,1);
max_output_temp=zeros(num,w);
min_output_temp=zeros(num,w);
for i=1:num
    inputdata_log{i,1}=log10(inputdata1{i,1});   

    
    outputdata_log{i,1}=(log10(outputdata1{i,1})); 
    outputdata_log{i,2}=(log10(outputdata1{i,2})); 
    outputdata_log{i,3}=(log10(outputdata1{i,3})); 
    outputdata_log{i,4}=(log10(outputdata1{i,4})); 

    for j=1:w
        max_output_temp(i,j)=max(max(max((outputdata_log{i,j})))); 
    end

end
max_input=4;
max_output=max(max(max_output_temp));
for i=1:num

    outputdata_log{i,1}=(outputdata_log{i,1})./(max_output); 
    outputdata_log{i,2}=(outputdata_log{i,2})./(max_output); 
    outputdata_log{i,3}=(outputdata_log{i,3})./(max_output); 
    outputdata_log{i,4}=(outputdata_log{i,4})./(max_output); 

    inputdata_log{i,1}=(inputdata_log{i,1})./(max_input);  

end



%Allocate the training set samples
outputdata_train=outputdata_log(1:num-vaild_num,:);   
inputdata_train=inputdata_log(1:num-vaild_num,:);  

% Allocate the validation set samples
outputdata_Valid=outputdata_log(num-vaild_num+1:num-test_num,:);   
inputdata_Valid=inputdata_log(num-vaild_num+1:num-test_num,:);     

% Allocate the test set samples
outputdata_test =outputdata_log(num-test_num+1:num,:); 
inputdata_test=inputdata_log(num-test_num+1:num,:);      


[n1,m1,z1]=size(inputdata_train{1,1});
[n2,m2,z2]=size(outputdata_train{1,1});

YTrain=zeros(n2,m2,z2,w,num-vaild_num);
XTrain=zeros(n1,m1,z1,1,num-vaild_num);


YValid=zeros(n2,m2,z2,w,vaild_num-test_num);
XValid=zeros(n1,m1,z1,1,vaild_num-test_num); 

YTest=zeros(n2,m2,z2,w,test_num);
XTest=zeros(n1,m1,z1,1,test_num); 

for i=1:num-vaild_num
    
    YTrain(:,:,:,1,i)=outputdata_train{i,1};
    YTrain(:,:,:,2,i)=outputdata_train{i,2};
    YTrain(:,:,:,3,i)=outputdata_train{i,3};
    YTrain(:,:,:,4,i)=outputdata_train{i,4};
    XTrain(:,:,:,1,i)=inputdata_train{i};
end

for i=1:vaild_num-test_num

    YValid(:,:,:,1,i)=outputdata_Valid{i,1};
    YValid(:,:,:,2,i)=outputdata_Valid{i,2};
    YValid(:,:,:,3,i)=outputdata_Valid{i,3};
    YValid(:,:,:,4,i)=outputdata_Valid{i,4};

    XValid(:,:,:,1,i)=inputdata_Valid{i};
end

for i=1:test_num

    YTest(:,:,:,1,i)=outputdata_test{i,1};
    YTest(:,:,:,2,i)=outputdata_test{i,2};
    YTest(:,:,:,3,i)=outputdata_test{i,3};
    YTest(:,:,:,4,i)=outputdata_test{i,4};

    XTest(:,:,:,1,i)=inputdata_test{i};
end


%parameter setting
initialLearningRate = 0.001;
maxEpochs =200;
minibatchSize = 16;
l2reg = 0.0001;
options = trainingOptions("adam",...
    OutputNetwork ='best-validation-loss',...
    ValidationData ={XValid,YValid},...
    ValidationFrequency=50,...
    InitialLearnRate=initialLearningRate, ...   
    MaxEpochs=maxEpochs,...
    MiniBatchSize=minibatchSize,...
    LearnRateSchedule="piecewise",... 
    LearnRateDropPeriod=20,...
    LearnRateDropFactor=0.6,...
    Shuffle="every-epoch",...
    ExecutionEnvironment='auto',...
    VerboseFrequency=50 ...
    );
    

%% train 

MTAGU_Net_layer();  %load MTAGU-Net network structure
[MTAGU_Net_struct,MTAGU_Net_infor] = trainNetwork(XTrain,YTrain,lgraph_MTAGU_Net,options);


