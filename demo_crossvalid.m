
clear all;
close all;
clc;

dataSetname =  ["parkinsons"];
for si = dataSetname
    dataset_s =  char(si)
    for ri = 1:10
        data = [];
        label = [];
        load(['./ExperimentData/',dataset_s,'.mat'])
        Data = [data label];
        % Data = [X y];
        save_mmd_cluster = ['Cluster_MMD_',dataset_s,num2str(ri),'.mat'];
        sample_Division1 = cell(10,3);                        
        Data = Data(randperm(size(Data,1)),:);
        [m_s,n_s]=size(Data);                 
          Y= Data(:,end);                     
        type_num = size(unique(Y),1);    
        
        if min(unique(Y))==0 && max(unique(Y))==1             
            Data(:,end) = Data(:,end)+1;
        end
        
        X = Data(:,1:end-1);
        Y1= Data(:,end);       
        X1 = standardization(X); 
        X2 =    [X1,Y1]; 
         
        class=cell(1,type_num);             
        for i =1 : m_s                               
            for j =1 : type_num                
                if X2(i,end) == j
                    class{1,j} = [class{1,j}; X2(i,:)];
                end
            end
        end
        block = cell(10,type_num);
        for i =1 : type_num                  
            class{1,i} = class{1,i}(randperm(size(class{1,i},1)),:);
            num_class = floor(size(class{1,i},1)/10);
            increase = round((size(class{1,i},1)/10 - num_class)*10);
            if(num_class == 0)
                num_class = 1;
                increase = 0;
            end
            class1 = class{1,i};
            start = 1;
            for j = 1 : 10
                if((j == 10) || ((num_class*j) >= size(class1,1)))
                    block{j,i} = class1(start:end,:);
                    break;
                end
                
                if(increase >= j)
                    increase_i = 1;
                else
                    increase_i = 0;
                end
                
                block{j,i} = class1(start:(start + num_class - 1 + increase_i) , :); 
                start = start + num_class + increase_i;
            end
        
        end
  
        for t= 1:10 
        
            test_blocknum = t;
            if(t == 10)
                valid_blocknum = 1;
            else
                valid_blocknum = t + 1;
            end
            train_blocknum = [1:10];
            train_blocknum(:,find(train_blocknum == test_blocknum)) = [];
            train_blocknum(:,find(train_blocknum == valid_blocknum)) = [];
            
            test0 = [];
            valid0 = [];
            train0 = [];
            for i = 1:type_num
                test0 = [test0;block{test_blocknum,i}];
                valid0 = [valid0;block{valid_blocknum,i}];
                train0 = [train0;cat(1,block{train_blocknum,i})];
            end
            
            dataclass = cell(1,type_num);
            for i = 1:type_num
                dataclass{1,i} = [cat(1,block{train_blocknum,i})];
            end
            
            iter = 3;
            TVdataX = cell(1,iter);
            for j = 1 : type_num                              
                P = 0.8; 
                data1 = dataclass{1,j};
                [m_s1,n_s1] = size(data1); 
                if(m_s1 == 0)
                    continue;
                end                     
                N1 =  round( m_s1 * P); 
                if(N1 == 0)
                    N1 = ceil( m_s1 * P);
                end
                data2 = sampleCluster(data1,N1);
                N2 =  round(N1 * P);
                if(N2 == 0)
                    N1 = ceil(N1 * P);
                end
                data3 = sampleCluster(data2,N2);
                TVdataX{1,1} = [TVdataX{1,1};data1] ;              
                TVdataX{1,2} = [TVdataX{1,2};data2];
                TVdataX{1,3} = [TVdataX{1,3};data3];
            end
            
            order_test1 = searchKNN(TVdataX{1,2},test0);
            test1 = TVdataX{1,2}(order_test1,:);
            order_valid1 = searchKNN(TVdataX{1,2},valid0);
            valid1 = TVdataX{1,2}(order_valid1,:);
            train1 = TVdataX{1,2};
            order_test2 = searchKNN(TVdataX{1,3},test0);
            test2 = TVdataX{1,3}(order_test2,:);
            order_valid2 = searchKNN(TVdataX{1,3},valid0);
            valid2 = TVdataX{1,3}(order_valid2,:);
            train2 = TVdataX{1,3};
            TarData = [test0(:,1:(end-1));valid0(:,1:(end-1));train0(:,1:(end-1))];
            TarLabel = [test0(:,end);valid0(:,end);train0(:,end)];
            SourData = [train1(:,1:(end-1));valid1(:,1:(end-1));test1(:,1:(end-1));...
                        train2(:,1:(end-1));valid2(:,1:(end-1));test2(:,1:(end-1))];
            SourLabel = [train1(:,end);valid1(:,end);test1(:,end);train2(:,end);valid2(:,end);test2(:,end)]; 
            gi = 0;    
            for gammai = [10,100,1000]
                gi = gi+1;
                [data_sou_new,data_tar_new] =  LSPDDA_ch3(SourData , TarData, gammai);
                data_tar_datanew = [ data_tar_new, TarLabel];         
                data_sour_datanew =  [data_sou_new,SourLabel];
                data0_MMD =  data_tar_datanew; 
                data1_MMD =  data_sour_datanew(1:(size(train1,1)+size(valid1,1)+size(test1,1)),:);
                data2_MMD =  data_sour_datanew((size(train1,1)+size(valid1,1)+size(test1,1)+1):end,:);
        
                test_data0_MMD = data0_MMD(1:size(test0,1),:);
                valid_data0_MMD = data0_MMD((size(test0,1)+1):(size(test0,1)+size(valid0,1)),:);
                train_data0_MMD = data0_MMD((size(test0,1)+size(valid0,1)+1):end,:);
        
                train_data1_MMD = data1_MMD(1:size(train1,1),:);
                valid_data1_MMD = data1_MMD((size(train1,1)+1):(size(train1,1)+size(valid1,1)),:);
                test_data1_MMD = data1_MMD((size(train1,1)+size(valid1,1)+1):end,:);
        
                train_data2_MMD = data2_MMD(1:size(train2,1),:);
                valid_data2_MMD = data2_MMD((size(train2,1)+1):(size(train2,1)+size(valid2,1)),:);
                test_data2_MMD = data2_MMD((size(train2,1)+size(valid2,1)+1):end,:);
        
                Data = cell(3,3);
                Data{1,1} =  train_data0_MMD(randperm(size(train_data0_MMD,1)),:);
                Data{1,2} =  train_data1_MMD(randperm(size(train_data1_MMD,1)),:);
                Data{1,3} =  train_data2_MMD(randperm(size(train_data2_MMD,1)),:);
        
                Data{2,1} =  valid_data0_MMD;
                Data{2,2} =  valid_data1_MMD;
                Data{2,3} =  valid_data2_MMD;
        
                Data{3,1} =  test_data0_MMD;
                Data{3,2} =  test_data1_MMD;
                Data{3,3} =  test_data2_MMD;
        
                sample_Division1{t,gi}=Data; 
            end
        end
        save(save_mmd_cluster,'sample_Division1','type_num');
    end

end

%% 
saveAllaac = cell(33,4);
 si = 0;
for  saaci =  dataSetname
    si = si + 1;
    all_acc =[]; 
    all_confidence_acc =[]; 
    dataName =  char(saaci);
    save_str=['ResultMMD_Cluster_',dataName];
    for stri = 1:10
        data_str = ['Cluster_MMD_',dataName,num2str(stri),'.mat']
        load(data_str);
        L = 3;
        C = 10;
        Test = 10;
        classification = 2;
        best_model10 = cell(10,1);
        best_Ri10 = cell(10,1);
        best_Accuracy = 0;
        best_gammai = 0;
        best_flag = 0;
        all_gamma = [10,100,1000];
        for i = 1:3
            [model10,Ri10] = buildClassifier(sample_Division1,i,C,L);
            [Accuracy1] = CalculayteAccCrossValid(sample_Division1,i,model10,Ri10,C,L,1);
            [Accuracy2] = CalculayteAccCrossValid(sample_Division1,i,model10,Ri10,C,L,2);
            if(Accuracy1 > best_Accuracy)
                best_Accuracy = Accuracy1;
                best_gammai = i;
                best_model10 = model10;
                best_Ri10 = Ri10;
                best_flag = 1;
            end
            if(Accuracy2 > best_Accuracy)
                best_Accuracy = Accuracy2;
                best_gammai = i;
                best_model10 = model10;
                best_Ri10 = Ri10;
                best_flag = 2;
            end
        end
        
        [confidence_acc,Std] = CalculayteAccCrossTest(sample_Division1,best_gammai,best_model10,best_Ri10,C,L,best_flag)
        all_confidence_acc = [all_confidence_acc;confidence_acc];
    end
    mean(all_confidence_acc)
    saveAllaac{si,1} = all_confidence_acc;
    saveAllaac{si,2} = dataName;
    saveAllaac{si,5} = mean(all_confidence_acc);
    saveAllaac{si,6} = std(all_confidence_acc');
end





