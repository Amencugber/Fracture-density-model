%% 初始化
clear
close all
clc
warning off
format short %精确到小数点后4位
%%  导入数据
res = xlsread('数据.xlsx');%导入数据库

%%  划分训练集和测试集
temp= 1:1:215;%不打乱顺序；

P_train = res(temp(1: 127), 1: 3)';%划分训练集输入
T_train = res(temp(1: 127), 4)';%划分训练集输出
M = size(P_train,2);

P_test = res(temp(128: 167), 1: 3)';%划分测试集输入
T_test = res(temp(128: 167), 4)';%划分测试集输出
N = size(P_test,2);

P_valid = res(temp(168: end), 1: 3)';
T_valid = res(temp(168: end), 4)';
V = size(P_valid, 2);
%%  数据归一化及转置
[p_train, ps_input] = mapminmax(P_train, 0, 1);%归一化到（0，1）
p_test = mapminmax('apply', P_test, ps_input);
p_valid = mapminmax('apply', P_valid, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);
t_valid = mapminmax('apply', T_valid, ps_output);

p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';
p_valid = p_valid'; t_valid = t_valid';

%%  模型参数设置及训练模型
trees = 200; % 决策树数目
leaf  = 2; % 最小叶子数
OOBPrediction = 'on';  % 打开误差图
OOBPredictorImportance = 'on'; % 计算特征重要性
Method = 'regression';  % 选择回归或分类
net = TreeBagger(trees, p_train, t_train, 'OOBPredictorImportance', OOBPredictorImportance,...
      'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
importance = net.OOBPermutedPredictorDeltaError;  % 重要性
%保存训练出的模型
save('model.mat', 'net');
%%  仿真测试
t_sim1 = predict(net, p_train);
t_sim2 = predict(net, p_test);
[t_sim3, error_3] = predict(net, p_valid);
%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_sim3 = mapminmax('reverse', t_sim3, ps_output);
%%  均方根误差RMSE
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);
error3 = sqrt(sum((T_sim3' - T_valid ).^2) ./ V);

disp(['训练集数据的RMSE为：', num2str(error1)])
disp(['测试集数据的RMSE为：', num2str(error2)])
disp(['验证集数据的RMSE为：', num2str(error3)])
%%  相关指标计算
% R2
R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2')^2 / norm(T_test  - mean(T_test ))^2;
R3 = 1 - norm(T_valid  - T_sim3')^2 / norm(T_valid  - mean(T_valid ))^2;

disp(['训练集数据的R2为：', num2str(R1)])
disp(['测试集数据的R2为：', num2str(R2)])
disp(['验证集数据的R2为：', num2str(R3)])

% MAE
mae1 = sum(abs(T_sim1' - T_train)) ./ M;
mae2 = sum(abs(T_sim2' - T_test )) ./ N;
mae3 = sum(abs(T_sim3' - T_valid )) ./ V;

disp(['训练集数据的MAE为：', num2str(mae1)])
disp(['测试集数据的MAE为：', num2str(mae2)])
disp(['验证集数据的MAE为：', num2str(mae3)])

% MBE
mbe1 = sum(T_sim1' - T_train) ./ M ;
mbe2 = sum(T_sim2' - T_test ) ./ N ;
mbe3 = sum(T_sim3' - T_valid ) ./ V ;

disp(['训练集数据的MBE为：', num2str(mbe1)])
disp(['测试集数据的MBE为：', num2str(mbe2)])
disp(['验证集数据的MBE为：', num2str(mbe3)])

% MAPE
mape1 = (mean(abs((T_train - T_sim1') ./ T_train)))*100;
mape2 = (mean(abs(T_test - T_sim2' ) ./ T_test))*100;
mape3 = (mean(abs((T_valid - T_sim3' ) ./ T_valid)))*100;

disp(['训练集数据的MAPE为：', num2str(mape1),'%'])
disp(['测试集数据的MAPE为：', num2str(mape2),'%'])
disp(['验证集数据的MAPE为：', num2str(mape3),'%'])

% MAPE准确率
disp(['训练集数据的预测准确率为：', num2str((1-mape1/100)*100),'%'])
disp(['测试集数据的预测准确率为：', num2str((1-mape2/100)*100),'%'])
disp(['验证集数据的预测准确率为：', num2str((1-mape3/100)*100),'%'])

% MSE
mse1 = sum((T_sim1' - T_train).^2) ./ M;
mse2 = sum((T_sim2' - T_test ).^2) ./ N;
mse3 = sum((T_sim3' - T_valid ).^2) ./ V;

disp(['训练集数据的MSE为：', num2str(mse1)])
disp(['测试集数据的MSE为：', num2str(mse2)])
disp(['验证集数据的MSE为：', num2str(mse3)])

[mae10,mse10,rmse10,mape10,error10,errorPercent10]=calc_error(T_test,T_sim2);
[mae11,mse11,rmse11,mape11,error11,errorPercent11]=calc_error(T_valid,T_sim3);
%%  绘图
figure %画图真实值与预测值对比图
plot(1:M, T_train, 'r-*', 1:M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值','预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1,M])
grid 

figure %画图真实值与预测值对比图
plot(1:N, T_test, 'r-*', 1:N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值','预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1,N])
grid 

figure
plot(1: V, T_valid, 'r-*', 1: V, T_sim3, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'验证集预测结果对比'; ['RMSE=' num2str(error3)]};
title(string)
xlim([1, V])
grid

%%绘制误差曲线
figure
plot(1:trees, oobError(net), 'b-', 'LineWidth', 1)
legend('误差曲线')
xlabel('决策树数目')
ylabel('误差')
xlim([1, trees])
grid

figure % 绘制特征重要性图
bar(importance)
legend('各因素重要性')
xlabel('特征')
ylabel('重要性')


