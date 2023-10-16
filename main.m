%% 初始化
clear
close all
clc
warning off
format short %精确到小数点后4位
%% 数据读取
data=xlsread('数据.xlsx'); %%使用xlsread函数读取EXCEL中对应范围的数据即可

%输入输出数据
input=data(:,1:end-1);    %data的第一列-倒数第二列为特征指标
output=data(:,end);  %data的最后面一列为输出的指标值

N=length(output);   %全部样本数目
testNum=40;   %设定测试样本数目
validNum=48;  %设定验证集数目
trainNum=N-testNum-validNum;    %计算训练样本数目

%% 划分训练集、测试集
input_train = input(1:trainNum,:)'; %训练集输入
output_train =output(1:trainNum)';  %训练集输出
input_test =input(trainNum+1:trainNum+testNum,:)'; %测试集输入
output_test =output(trainNum+1:trainNum+testNum)'; %测试集输出
input_valid =input(trainNum+testNum+1:trainNum+testNum+validNum,:)'; %验证集输入
output_valid =output(trainNum+testNum+1:trainNum+testNum+validNum)'; %验证集输出

%% 数据归一化
[inputn,inputps]=mapminmax(input_train,0,1);  % 训练集输入归一化到[0,1]之间,inoutn为按行归一化的数组，inputps包含最值
[outputn,outputps]=mapminmax(output_train);   % 训练集输出归一化到默认区间[-1, 1]
inputn_test=mapminmax('apply',input_test,inputps); % 测试集输入采用和训练集输入相同的归一化方式
inputn_valid=mapminmax('apply',input_valid,inputps); % 验证集输入采用和训练集输入相同的归一化方式

%% 获取输入层节点、输出层节点个数
inputnum=size(input,2); %size用来求取矩阵的行数和列数，1代表行数，2代表列数
outputnum=size(output,2);
disp('/////////////////////////////////')
disp('神经网络结构...')
disp(['输入层的节点数为：',num2str(inputnum)])
disp(['输出层的节点数为：',num2str(outputnum)])
disp(' ')
disp('隐含层节点的确定过程...')

%确定隐含层节点个数
%采用经验公式hiddennum=sqrt(m+n)+a，m为输入层节点个数，n为输出层节点个数，a一般取为1-10之间的整数
MSE=1e+5; %初始化最小误差
transform_func={'tansig','purelin'};  %激活函数采用tan-sigmoid和purelin
train_func='trainlm';                 %训练算法
for hiddennum=fix(sqrt(inputnum+outputnum))+1:fix(sqrt(inputnum+outputnum))+10
    
    %构建网络
    net=newff(inputn,outputn,hiddennum,transform_func,train_func);%构建BP网络
    % 网络参数
    net.trainParam.epochs=10000;         % 训练次数
    net.trainParam.lr=0.001;                   % 学习速率0.01
    net.trainParam.goal=0.000001;        % 训练目标最小误差，新增动量和梯度
    net.trainParam.show=25;                % 显示频率，这里设置为每训练25次显示一次
    net.trainParam.mc=0.01;                 % 动量因子0.01
    net.trainParam.min_grad=1e-6;       % 最小性能梯度
    net.trainParam.max_fail=6;               % 最高失败次数
    % 网络训练
    net=train(net,inputn,outputn);
    an0=sim(net,inputn);  %仿真结果
    mse0=mse(outputn,an0);  %仿真的均方误差
    disp(['隐含层节点数为',num2str(hiddennum),'时，训练集的均方误差为：',num2str(mse0)])
    
    %更新最佳的隐含层节点
    if mse0<MSE
        MSE=mse0;
        hiddennum_best=hiddennum;
    end
end
disp(['最佳的隐含层节点数为：',num2str(hiddennum_best),'，相应的均方误差为：',num2str(MSE)])
 


%% 构建最佳隐含层节点的BP神经网络
disp(' ')
disp('标准的BP神经网络：')
net0=newff(inputn,outputn,hiddennum_best,{'tansig','purelin'},'trainlm');% 建立模型

%网络参数配置
net0.trainParam.epochs=10000;         % 训练次数，这里设置为1000次
net0.trainParam.lr=0.001;                   % 学习速率，这里设置为0.01
net0.trainParam.goal=0.00001;                    % 训练目标最小误差，这里设置为0.0001
net0.trainParam.show=25;                % 显示频率，这里设置为每训练25次显示一次
net0.trainParam.mc=0.01;                 % 动量因子
net0.trainParam.min_grad=1e-6;       % 最小性能梯度
net0.trainParam.max_fail=6;               % 最高失败次数

%开始训练
net0=train(net0,inputn,outputn);
%保存神经网络
save('net0.mat','net0'); 
%预测
an0=sim(net0,inputn_test); %用训练好的模型进行仿真
an2=sim(net0,inputn_valid); %用训练好的模型进行仿真

%预测结果反归一化与误差计算
test_simu0=mapminmax('reverse',an0,outputps); %把仿真得到的数据还原为原始的数量级////////
test_simu2=mapminmax('reverse',an2,outputps); %把仿真得到的数据还原为原始的数量级
%误差指标
[mae0,mse0,rmse0,mape0,error0,errorPercent0]=calc_error(output_test,test_simu0);
[mae2,mse2,rmse2,mape2,error2,errorPercent2]=calc_error(output_valid,test_simu2);

 figure
 plot(output_test,'b-*','linewidth',1)
 hold on
 plot(test_simu0,'r-v','linewidth',1,'markerfacecolor','r')
 legend('真实值','BP预测值')
 xlabel('测试样本编号')
 ylabel('指标值')
 title('BP神经网络预测值和真实值对比图')
    
 figure
 bar(error2)
 xlabel('测试样本编号'),ylabel('预测残差值')
 title('BP神经网络测试集的预测残差图')
 set(gca,'fontsize',12)

 figure
 plot(output_valid,'b-*','linewidth',1)
 hold on
 plot(test_simu2,'r-v','linewidth',1,'markerfacecolor','r')
 legend('真实值','BP预测值')
 xlabel('测试样本编号')
 ylabel('指标值')
 title('BP神经网络预测值和真实值对比图')
    
 figure
 bar(error2)
 xlabel('测试样本编号'),ylabel('预测残差值')
 title('BP神经网络验证集的预测残差图')
 set(gca,'fontsize',12)


%% Tent混沌映射改进的麻雀搜索算法寻最优权值阈值
disp(' ')
disp('Tent-SSA优化BP神经网络：')
net=newff(inputn,outputn,hiddennum_best,{'tansig','purelin'},'trainlm');% 建立模型

%网络参数配置
net.trainParam.epochs=10000;         % 训练次数，这里设置为1000次
net.trainParam.lr=0.01;                   % 学习速率，这里设置为0.01
net.trainParam.goal=0.00001;                    % 训练目标最小误差，这里设置为0.0001
net.trainParam.show=25;                % 显示频率，这里设置为每训练25次显示一次
net.trainParam.mc=0.01;                 % 动量因子
net.trainParam.min_grad=1e-6;       % 最小性能梯度
net.trainParam.max_fail=6;               % 最高失败次数

%初始化SSA参数
popsize=50;   %初始种群规模
maxgen=50;   %最大进化代数
dim=inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum+outputnum;    %自变量个数
lb=repmat(-3,1,dim);    %自变量下限
ub=repmat(3,1,dim);   %自变量上限
ST = 0.6;%安全值
PD = 0.7;%发现者的比列，剩下的是加入者
SD = 0.2;%意识到有危险麻雀的比重
PDNumber = popsize*PD; %发现者数量
SDNumber = popsize - popsize*PD;%意识到有危险麻雀数量

%% Tent混沌映射初始化种群位置
k=3;   %k为1到n的正整数，Tent映射初始化k*popsize个种群，从中选出适应度最好的popsize个个体作为初始种群
X0 = tentInitialization(popsize*k,dim,ub,lb);
X=X0;

% 计算初始适应度值
fit = zeros(1,popsize*k);
for i = 1:popsize*k
    fit(i) =  fitness(X(i,:),inputnum,hiddennum_best,outputnum,net,inputn,outputn,output_train,inputn_test,outputps,output_test);
end
[fit, index]= sort(fit);%排序
%选出初始的popsize个最佳个体
fit=fit(1:popsize);
index=index(1:popsize);
%计算初始最优与最差个体
BestF = fit(1);
WorstF = fit(end);
GBestF = fit(1);%全局最优适应度值
for i = 1:popsize
    X(i,:) = X0(index(i),:);
end
curve=zeros(1,maxgen);
GBestX = X(1,:);%全局最优位置
X_new = X;
%% 开始优化
h0=waitbar(0,'Tent-SSA optimization...');
for i = 1: maxgen
    
    BestF = fit(1);
    WorstF = fit(end);
    R2 = rand(1);  %预警值
    %更新发现者位置
    for j = 1:PDNumber
        if(R2<ST)
            X_new(j,:) = X(j,:).*exp(-j/(rand(1)*maxgen));
        else
            X_new(j,:) = X(j,:) + randn()*ones(1,dim);
        end
    end
    %更新加入者位置
    for j = PDNumber+1:popsize
        if(j>(popsize - PDNumber)/2 + PDNumber)
            X_new(j,:)= randn().*exp((X(end,:) - X(j,:))/j^2);
        else
            %产生-1，1的随机数
            A = ones(1,dim);
            for a = 1:dim
                if(rand()>0.5)
                    A(a) = -1;
                end
            end
            AA = A'*inv(A*A');
            X_new(j,:)= X(1,:) + abs(X(j,:) - X(1,:)).*AA';
        end
    end
    %反捕食行为更新麻雀位置
    Temp = randperm(popsize);
    SDchooseIndex = Temp(1:SDNumber);
    for j = 1:SDNumber
        if(fit(SDchooseIndex(j))>BestF)
            X_new(SDchooseIndex(j),:) = X(1,:) + randn().*abs(X(SDchooseIndex(j),:) - X(1,:));
        elseif(fit(SDchooseIndex(j))== BestF)
            K = 2*rand() -1;
            X_new(SDchooseIndex(j),:) = X(SDchooseIndex(j),:) + K.*(abs( X(SDchooseIndex(j),:) - X(end,:))./(fit(SDchooseIndex(j)) - fit(end) + 10^-8));
        end
    end
    %边界控制
    for j = 1:popsize
        for a = 1: dim
            if(X_new(j,a)>ub)
                X_new(j,a) =ub(a);
            end
            if(X_new(j,a)<lb)
                X_new(j,a) =lb(a);
            end
        end
    end
    %更新位置
    for j=1:popsize
        fitness_new(j) = fitness(X_new(j,:),inputnum,hiddennum_best,outputnum,net,inputn,outputn,output_train,inputn_test,outputps,output_test);
    end
    for j = 1:popsize
        if(fitness_new(j) < GBestF)
            GBestF = fitness_new(j);
            GBestX = X_new(j,:);
        end
    end
    X = X_new;
    fit = fitness_new;
    %排序更新
    [fit, index]= sort(fit);%排序
    BestF = fit(1);
    WorstF = fit(end);
    for j = 1:popsize
        X(j,:) = X(index(j),:);
    end
    curve(i) = GBestF;
    waitbar(i/maxgen,h0)
end
close(h0)
Best_pos =GBestX;
Best_score = curve(end);
setdemorandstream(pi);  

%% 绘制进化曲线
figure
plot(curve,'r-','linewidth',2)
xlabel('进化代数')
ylabel('均方误差')
legend('最佳适应度')
title('Tent-SSA的进化收敛曲线')
w1=Best_pos(1:inputnum*hiddennum_best);         %输入层到中间层的权值
B1=Best_pos(inputnum*hiddennum_best+1:inputnum*hiddennum_best+hiddennum_best);   %中间各层神经元阈值
w2=Best_pos(inputnum*hiddennum_best+hiddennum_best+1:inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum);   %中间层到输出层的权值
B2=Best_pos(inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum+1:inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum+outputnum);   %输出层各神经元阈值
%矩阵重构
net.iw{1,1}=reshape(w1,hiddennum_best,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum_best);
net.b{1}=reshape(B1,hiddennum_best,1);
net.b{2}=reshape(B2,outputnum,1);

%% 优化后的神经网络训练
net=train(net,inputn,outputn);%开始训练，其中inputn,outputn分别为输入输出样本
%保存神经网络
save('net.mat','net'); 
%% 优化后的神经网络测试
an1=sim(net,inputn_test);
an3=sim(net,inputn_valid);
test_simu1=mapminmax('reverse',an1,outputps); %把仿真得到的数据还原为原始的数量级
test_simu3=mapminmax('reverse',an3,outputps);
%误差指标
[mae1,mse1,rmse1,mape1,error1,errorPercent1]=calc_error(output_test,test_simu1);
[mae3,mse3,rmse3,mape3,error3,errorPercent3]=calc_error(output_valid,test_simu3);


%% 作图
 figure
 plot(output_test,'b-*','linewidth',1)
 hold on
 plot(test_simu1,'r-v','linewidth',1,'markerfacecolor','r')
 legend('真实值','Tent-SSA优化BP预测值')
 xlabel('测试样本编号')
 ylabel('指标值')
 title('Tent-SSA优化BP神经网络预测值和真实值对比图')
    
 figure
 bar(error1)
 xlabel('测试样本编号'),ylabel('预测残差值')
 title('Tent-SSA优化BP神经网络测试集的预测残差图')
 set(gca,'fontsize',12)




figure
plot(output_test,'b-*','linewidth',1)
hold on
plot(test_simu0,'r-v','linewidth',1,'markerfacecolor','r')
hold on
plot(test_simu1,'k-o','linewidth',1,'markerfacecolor','k')
legend('真实值','BP预测值','Tent-SSA BP预测值')
xlabel('测试样本编号')
ylabel('指标值')
title('Tent-SSA优化前后的BP神经网络预测值和真实值对比图')

figure
plot(error0,'rv-','markerfacecolor','r')
hold on
plot(error1,'ko-','markerfacecolor','k')
legend('BP预测误差','Tent-SSA BP预测误差')
xlabel('测试样本编号')
ylabel('预测偏差')
title('Tent-SSA优化前后的BP神经网络预测值和真实值误差对比图')

figure
 plot(output_valid,'b-*','linewidth',1)
 hold on
 plot(test_simu3,'r-v','linewidth',1,'markerfacecolor','r')
 legend('真实值','Tent-SSA优化BP预测值')
 xlabel('测试样本编号')
 ylabel('指标值')
 title('Tent-SSA优化BP神经网络预测值和真实值对比图')
    
 figure
 bar(error3)
 xlabel('测试样本编号'),ylabel('预测残差值')
 title('Tent-SSA优化BP神经网络验证集的预测残差图')
 set(gca,'fontsize',12)




figure
plot(output_valid,'b-*','linewidth',1)
hold on
plot(test_simu2,'r-v','linewidth',1,'markerfacecolor','r')
hold on
plot(test_simu3,'k-o','linewidth',1,'markerfacecolor','k')
legend('真实值','BP预测值','Tent-SSA BP预测值')
xlabel('测试样本编号')
ylabel('指标值')
title('Tent-SSA优化前后的BP神经网络预测值和真实值对比图')

figure
plot(error2,'rv-','markerfacecolor','r')
hold on
plot(error3,'ko-','markerfacecolor','k')
legend('BP预测误差','Tent-SSA BP预测误差')
xlabel('测试样本编号')
ylabel('预测偏差')
title('Tent-SSA优化前后的BP神经网络预测值和真实值误差对比图')
% disp(' ')
% disp('/////////////////////////////////')
% disp('打印结果表格')
% disp('样本序号     实测值      BP预测值  Tent-SSA-BP值   BP误差   Tent-SSA-BP误差')
% for i=1:testNum
%     disp([i output_test(i),test_simu0(i),test_simu1(i),error0(i),error1(i)])
% end
