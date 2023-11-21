function class1 = DividiedClass(X,type_num)
m_s = size(X,1);
class1=cell(1,type_num);              
for i =1 : m_s                               
    for j =1 : type_num                  
        if X(i,end) == j
            class1{1,j} = [class1{1,j}; X(i,:)];
        end
    end
end  
end