%% 基于Tent映射的种群初始化子函数
function Positions=tentInitialization(popsize,dim,ub,lb)
%input：popsize   种群数量
%             dim           变量维度
%             ub             变量上限
%             lb               变量下限
%return：Positions    生成的初始种群位置

%初始化位置0数组
Positions=zeros(popsize,dim);

%对每个个体，混沌映射产生位置
for i = 1:popsize
    value =  Tent(dim);  %混沌映射序列
    Positions(i,:)=value.*(ub-lb)+lb;
    %位置越界限制
    Positions(i,:)=min(Positions(i,:),ub);   %上界调整
    Positions(i,:)=max(Positions(i,:),lb);    %下界调整
end

end



%混沌映射子函数
function sequence=Tent(n)
%input：n    混沌序列长度
%return：value    生成的混沌序列

%初始化数组
sequence=zeros(1,n);    
sequence(1)=rand; %序列起点
beta = 0.7;%参数beta，取值范围(0,1)
for i=1:n-1
    if sequence(i)<beta
        sequence(i+1)=sequence(i)/beta;
    end
    if sequence(i)>=beta
        sequence(i+1)=(1-sequence(i))/(1-beta);
    end
end

end




