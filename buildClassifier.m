function [model10,Ri10] = buildClassifier(sample_Division1,col,C,L)

classification = 2;
Test = 10;
trainData = cell(1,3);
model10 = cell(10,1);
Ri10 = cell(10,1);

for T = 1:Test
    Data = sample_Division1{T,col};
    trainData{1,1} = full(Data{1,1});%train0
    trainData{1,2} = full(Data{1,2});%train1
    trainData{1,3} = full(Data{1,3});%train2
    [models ,Ri] = DeepSampleCluster_MMD_RotationF2(trainData,C,L,classification);
    model10{T,1} = models;
    Ri10{T,1} = Ri;
end

end