%% ��ʼ��
clear
close all
clc
warning off
format short %��ȷ��С�����4λ
%% ���ݶ�ȡ
data=xlsread('����.xlsx'); %%ʹ��xlsread������ȡEXCEL�ж�Ӧ��Χ�����ݼ���

%�����������
input=data(:,1:end-1);    %data�ĵ�һ��-�����ڶ���Ϊ����ָ��
output=data(:,end);  %data�������һ��Ϊ�����ָ��ֵ

N=length(output);   %ȫ��������Ŀ
testNum=40;   %�趨����������Ŀ
validNum=48;  %�趨��֤����Ŀ
trainNum=N-testNum-validNum;    %����ѵ��������Ŀ

%% ����ѵ���������Լ�
input_train = input(1:trainNum,:)'; %ѵ��������
output_train =output(1:trainNum)';  %ѵ�������
input_test =input(trainNum+1:trainNum+testNum,:)'; %���Լ�����
output_test =output(trainNum+1:trainNum+testNum)'; %���Լ����
input_valid =input(trainNum+testNum+1:trainNum+testNum+validNum,:)'; %��֤������
output_valid =output(trainNum+testNum+1:trainNum+testNum+validNum)'; %��֤�����

%% ���ݹ�һ��
[inputn,inputps]=mapminmax(input_train,0,1);  % ѵ���������һ����[0,1]֮��,inoutnΪ���й�һ�������飬inputps������ֵ
[outputn,outputps]=mapminmax(output_train);   % ѵ���������һ����Ĭ������[-1, 1]
inputn_test=mapminmax('apply',input_test,inputps); % ���Լ�������ú�ѵ����������ͬ�Ĺ�һ����ʽ
inputn_valid=mapminmax('apply',input_valid,inputps); % ��֤��������ú�ѵ����������ͬ�Ĺ�һ����ʽ

%% ��ȡ�����ڵ㡢�����ڵ����
inputnum=size(input,2); %size������ȡ�����������������1����������2��������
outputnum=size(output,2);
disp('/////////////////////////////////')
disp('������ṹ...')
disp(['�����Ľڵ���Ϊ��',num2str(inputnum)])
disp(['�����Ľڵ���Ϊ��',num2str(outputnum)])
disp(' ')
disp('������ڵ��ȷ������...')

%ȷ��������ڵ����
%���þ��鹫ʽhiddennum=sqrt(m+n)+a��mΪ�����ڵ������nΪ�����ڵ������aһ��ȡΪ1-10֮�������
MSE=1e+5; %��ʼ����С���
transform_func={'tansig','purelin'};  %���������tan-sigmoid��purelin
train_func='trainlm';                 %ѵ���㷨
for hiddennum=fix(sqrt(inputnum+outputnum))+1:fix(sqrt(inputnum+outputnum))+10
    
    %��������
    net=newff(inputn,outputn,hiddennum,transform_func,train_func);%����BP����
    % �������
    net.trainParam.epochs=10000;         % ѵ������
    net.trainParam.lr=0.001;                   % ѧϰ����0.01
    net.trainParam.goal=0.000001;        % ѵ��Ŀ����С�������������ݶ�
    net.trainParam.show=25;                % ��ʾƵ�ʣ���������Ϊÿѵ��25����ʾһ��
    net.trainParam.mc=0.01;                 % ��������0.01
    net.trainParam.min_grad=1e-6;       % ��С�����ݶ�
    net.trainParam.max_fail=6;               % ���ʧ�ܴ���
    % ����ѵ��
    net=train(net,inputn,outputn);
    an0=sim(net,inputn);  %������
    mse0=mse(outputn,an0);  %����ľ������
    disp(['������ڵ���Ϊ',num2str(hiddennum),'ʱ��ѵ�����ľ������Ϊ��',num2str(mse0)])
    
    %������ѵ�������ڵ�
    if mse0<MSE
        MSE=mse0;
        hiddennum_best=hiddennum;
    end
end
disp(['��ѵ�������ڵ���Ϊ��',num2str(hiddennum_best),'����Ӧ�ľ������Ϊ��',num2str(MSE)])
 


%% �������������ڵ��BP������
disp(' ')
disp('��׼��BP�����磺')
net0=newff(inputn,outputn,hiddennum_best,{'tansig','purelin'},'trainlm');% ����ģ��

%�����������
net0.trainParam.epochs=10000;         % ѵ����������������Ϊ1000��
net0.trainParam.lr=0.001;                   % ѧϰ���ʣ���������Ϊ0.01
net0.trainParam.goal=0.00001;                    % ѵ��Ŀ����С����������Ϊ0.0001
net0.trainParam.show=25;                % ��ʾƵ�ʣ���������Ϊÿѵ��25����ʾһ��
net0.trainParam.mc=0.01;                 % ��������
net0.trainParam.min_grad=1e-6;       % ��С�����ݶ�
net0.trainParam.max_fail=6;               % ���ʧ�ܴ���

%��ʼѵ��
net0=train(net0,inputn,outputn);
%����������
save('net0.mat','net0'); 
%Ԥ��
an0=sim(net0,inputn_test); %��ѵ���õ�ģ�ͽ��з���
an2=sim(net0,inputn_valid); %��ѵ���õ�ģ�ͽ��з���

%Ԥ��������һ����������
test_simu0=mapminmax('reverse',an0,outputps); %�ѷ���õ������ݻ�ԭΪԭʼ��������////////
test_simu2=mapminmax('reverse',an2,outputps); %�ѷ���õ������ݻ�ԭΪԭʼ��������
%���ָ��
[mae0,mse0,rmse0,mape0,error0,errorPercent0]=calc_error(output_test,test_simu0);
[mae2,mse2,rmse2,mape2,error2,errorPercent2]=calc_error(output_valid,test_simu2);

 figure
 plot(output_test,'b-*','linewidth',1)
 hold on
 plot(test_simu0,'r-v','linewidth',1,'markerfacecolor','r')
 legend('��ʵֵ','BPԤ��ֵ')
 xlabel('�����������')
 ylabel('ָ��ֵ')
 title('BP������Ԥ��ֵ����ʵֵ�Ա�ͼ')
    
 figure
 bar(error2)
 xlabel('�����������'),ylabel('Ԥ��в�ֵ')
 title('BP��������Լ���Ԥ��в�ͼ')
 set(gca,'fontsize',12)

 figure
 plot(output_valid,'b-*','linewidth',1)
 hold on
 plot(test_simu2,'r-v','linewidth',1,'markerfacecolor','r')
 legend('��ʵֵ','BPԤ��ֵ')
 xlabel('�����������')
 ylabel('ָ��ֵ')
 title('BP������Ԥ��ֵ����ʵֵ�Ա�ͼ')
    
 figure
 bar(error2)
 xlabel('�����������'),ylabel('Ԥ��в�ֵ')
 title('BP��������֤����Ԥ��в�ͼ')
 set(gca,'fontsize',12)


%% Tent����ӳ��Ľ�����ȸ�����㷨Ѱ����Ȩֵ��ֵ
disp(' ')
disp('Tent-SSA�Ż�BP�����磺')
net=newff(inputn,outputn,hiddennum_best,{'tansig','purelin'},'trainlm');% ����ģ��

%�����������
net.trainParam.epochs=10000;         % ѵ����������������Ϊ1000��
net.trainParam.lr=0.01;                   % ѧϰ���ʣ���������Ϊ0.01
net.trainParam.goal=0.00001;                    % ѵ��Ŀ����С����������Ϊ0.0001
net.trainParam.show=25;                % ��ʾƵ�ʣ���������Ϊÿѵ��25����ʾһ��
net.trainParam.mc=0.01;                 % ��������
net.trainParam.min_grad=1e-6;       % ��С�����ݶ�
net.trainParam.max_fail=6;               % ���ʧ�ܴ���

%��ʼ��SSA����
popsize=50;   %��ʼ��Ⱥ��ģ
maxgen=50;   %����������
dim=inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum+outputnum;    %�Ա�������
lb=repmat(-3,1,dim);    %�Ա�������
ub=repmat(3,1,dim);   %�Ա�������
ST = 0.6;%��ȫֵ
PD = 0.7;%�����ߵı��У�ʣ�µ��Ǽ�����
SD = 0.2;%��ʶ����Σ����ȸ�ı���
PDNumber = popsize*PD; %����������
SDNumber = popsize - popsize*PD;%��ʶ����Σ����ȸ����

%% Tent����ӳ���ʼ����Ⱥλ��
k=3;   %kΪ1��n����������Tentӳ���ʼ��k*popsize����Ⱥ������ѡ����Ӧ����õ�popsize��������Ϊ��ʼ��Ⱥ
X0 = tentInitialization(popsize*k,dim,ub,lb);
X=X0;

% �����ʼ��Ӧ��ֵ
fit = zeros(1,popsize*k);
for i = 1:popsize*k
    fit(i) =  fitness(X(i,:),inputnum,hiddennum_best,outputnum,net,inputn,outputn,output_train,inputn_test,outputps,output_test);
end
[fit, index]= sort(fit);%����
%ѡ����ʼ��popsize����Ѹ���
fit=fit(1:popsize);
index=index(1:popsize);
%�����ʼ������������
BestF = fit(1);
WorstF = fit(end);
GBestF = fit(1);%ȫ��������Ӧ��ֵ
for i = 1:popsize
    X(i,:) = X0(index(i),:);
end
curve=zeros(1,maxgen);
GBestX = X(1,:);%ȫ������λ��
X_new = X;
%% ��ʼ�Ż�
h0=waitbar(0,'Tent-SSA optimization...');
for i = 1: maxgen
    
    BestF = fit(1);
    WorstF = fit(end);
    R2 = rand(1);  %Ԥ��ֵ
    %���·�����λ��
    for j = 1:PDNumber
        if(R2<ST)
            X_new(j,:) = X(j,:).*exp(-j/(rand(1)*maxgen));
        else
            X_new(j,:) = X(j,:) + randn()*ones(1,dim);
        end
    end
    %���¼�����λ��
    for j = PDNumber+1:popsize
        if(j>(popsize - PDNumber)/2 + PDNumber)
            X_new(j,:)= randn().*exp((X(end,:) - X(j,:))/j^2);
        else
            %����-1��1�������
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
    %����ʳ��Ϊ������ȸλ��
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
    %�߽����
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
    %����λ��
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
    %�������
    [fit, index]= sort(fit);%����
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

%% ���ƽ�������
figure
plot(curve,'r-','linewidth',2)
xlabel('��������')
ylabel('�������')
legend('�����Ӧ��')
title('Tent-SSA�Ľ�����������')
w1=Best_pos(1:inputnum*hiddennum_best);         %����㵽�м���Ȩֵ
B1=Best_pos(inputnum*hiddennum_best+1:inputnum*hiddennum_best+hiddennum_best);   %�м������Ԫ��ֵ
w2=Best_pos(inputnum*hiddennum_best+hiddennum_best+1:inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum);   %�м�㵽������Ȩֵ
B2=Best_pos(inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum+1:inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum+outputnum);   %��������Ԫ��ֵ
%�����ع�
net.iw{1,1}=reshape(w1,hiddennum_best,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum_best);
net.b{1}=reshape(B1,hiddennum_best,1);
net.b{2}=reshape(B2,outputnum,1);

%% �Ż����������ѵ��
net=train(net,inputn,outputn);%��ʼѵ��������inputn,outputn�ֱ�Ϊ�����������
%����������
save('net.mat','net'); 
%% �Ż�������������
an1=sim(net,inputn_test);
an3=sim(net,inputn_valid);
test_simu1=mapminmax('reverse',an1,outputps); %�ѷ���õ������ݻ�ԭΪԭʼ��������
test_simu3=mapminmax('reverse',an3,outputps);
%���ָ��
[mae1,mse1,rmse1,mape1,error1,errorPercent1]=calc_error(output_test,test_simu1);
[mae3,mse3,rmse3,mape3,error3,errorPercent3]=calc_error(output_valid,test_simu3);


%% ��ͼ
 figure
 plot(output_test,'b-*','linewidth',1)
 hold on
 plot(test_simu1,'r-v','linewidth',1,'markerfacecolor','r')
 legend('��ʵֵ','Tent-SSA�Ż�BPԤ��ֵ')
 xlabel('�����������')
 ylabel('ָ��ֵ')
 title('Tent-SSA�Ż�BP������Ԥ��ֵ����ʵֵ�Ա�ͼ')
    
 figure
 bar(error1)
 xlabel('�����������'),ylabel('Ԥ��в�ֵ')
 title('Tent-SSA�Ż�BP��������Լ���Ԥ��в�ͼ')
 set(gca,'fontsize',12)




figure
plot(output_test,'b-*','linewidth',1)
hold on
plot(test_simu0,'r-v','linewidth',1,'markerfacecolor','r')
hold on
plot(test_simu1,'k-o','linewidth',1,'markerfacecolor','k')
legend('��ʵֵ','BPԤ��ֵ','Tent-SSA BPԤ��ֵ')
xlabel('�����������')
ylabel('ָ��ֵ')
title('Tent-SSA�Ż�ǰ���BP������Ԥ��ֵ����ʵֵ�Ա�ͼ')

figure
plot(error0,'rv-','markerfacecolor','r')
hold on
plot(error1,'ko-','markerfacecolor','k')
legend('BPԤ�����','Tent-SSA BPԤ�����')
xlabel('�����������')
ylabel('Ԥ��ƫ��')
title('Tent-SSA�Ż�ǰ���BP������Ԥ��ֵ����ʵֵ���Ա�ͼ')

figure
 plot(output_valid,'b-*','linewidth',1)
 hold on
 plot(test_simu3,'r-v','linewidth',1,'markerfacecolor','r')
 legend('��ʵֵ','Tent-SSA�Ż�BPԤ��ֵ')
 xlabel('�����������')
 ylabel('ָ��ֵ')
 title('Tent-SSA�Ż�BP������Ԥ��ֵ����ʵֵ�Ա�ͼ')
    
 figure
 bar(error3)
 xlabel('�����������'),ylabel('Ԥ��в�ֵ')
 title('Tent-SSA�Ż�BP��������֤����Ԥ��в�ͼ')
 set(gca,'fontsize',12)




figure
plot(output_valid,'b-*','linewidth',1)
hold on
plot(test_simu2,'r-v','linewidth',1,'markerfacecolor','r')
hold on
plot(test_simu3,'k-o','linewidth',1,'markerfacecolor','k')
legend('��ʵֵ','BPԤ��ֵ','Tent-SSA BPԤ��ֵ')
xlabel('�����������')
ylabel('ָ��ֵ')
title('Tent-SSA�Ż�ǰ���BP������Ԥ��ֵ����ʵֵ�Ա�ͼ')

figure
plot(error2,'rv-','markerfacecolor','r')
hold on
plot(error3,'ko-','markerfacecolor','k')
legend('BPԤ�����','Tent-SSA BPԤ�����')
xlabel('�����������')
ylabel('Ԥ��ƫ��')
title('Tent-SSA�Ż�ǰ���BP������Ԥ��ֵ����ʵֵ���Ա�ͼ')
% disp(' ')
% disp('/////////////////////////////////')
% disp('��ӡ������')
% disp('�������     ʵ��ֵ      BPԤ��ֵ  Tent-SSA-BPֵ   BP���   Tent-SSA-BP���')
% for i=1:testNum
%     disp([i output_test(i),test_simu0(i),test_simu1(i),error0(i),error1(i)])
% end
