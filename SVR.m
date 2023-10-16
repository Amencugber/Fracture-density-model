%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  ��������
res = xlsread('����.xlsx');

%%  ����ѵ��������֤���Ͳ��Լ�
temp= 1:1:215;

P_train = res(temp(1: 127), 1: 3)';
T_train = res(temp(1: 127), 4)';
M = size(P_train, 2);

P_test = res(temp(128: 167), 1: 3)';
T_test = res(temp(128: 167), 4)';
N = size(P_test, 2);

P_valid = res(temp(168: end), 1: 3)';
T_valid = res(temp(168: end), 4)';
V = size(P_valid, 2);
%%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);
p_valid = mapminmax('apply', P_valid, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);
t_valid = mapminmax('apply', T_valid, ps_output);


%%  ת������Ӧģ��
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';
p_valid = p_valid'; t_valid = t_valid';

%%  ����ģ��
c = 4;    % �ͷ�����
g = 10;    % �������������
cmd = [' -t 2',' -c ',num2str(c),' -g ',num2str(g),' -s 3 -p 0.01'];
model = svmtrain(t_train, p_train, cmd);
%����ѵ������ģ��
save('model.mat', 'model');
%%  ����Ԥ��
[t_sim1, error_1] = svmpredict(t_train, p_train, model);
[t_sim2, error_2] = svmpredict(t_test , p_test , model);
[t_sim3, error_3] = svmpredict(t_valid , p_valid , model);

%%  ���ݷ���һ��
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_sim3 = mapminmax('reverse', t_sim3, ps_output);

%%  ���������
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);
error3 = sqrt(sum((T_sim3' - T_valid ).^2) ./ V);

disp(['ѵ�������ݵ�RMSEΪ��', num2str(error1)])
disp(['���Լ����ݵ�RMSEΪ��', num2str(error2)])
disp(['��֤�����ݵ�RMSEΪ��', num2str(error3)])

%%  ���ָ�����
% R2
R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2')^2 / norm(T_test  - mean(T_test ))^2;
R3 = 1 - norm(T_valid  - T_sim3')^2 / norm(T_valid  - mean(T_valid ))^2;

disp(['ѵ�������ݵ�R2Ϊ��', num2str(R1)])
disp(['���Լ����ݵ�R2Ϊ��', num2str(R2)])
disp(['��֤�����ݵ�R2Ϊ��', num2str(R3)])

% MAE
mae1 = sum(abs(T_sim1' - T_train)) ./ M;
mae2 = sum(abs(T_sim2' - T_test )) ./ N;
mae3 = sum(abs(T_sim3' - T_valid )) ./ V;

disp(['ѵ�������ݵ�MAEΪ��', num2str(mae1)])
disp(['���Լ����ݵ�MAEΪ��', num2str(mae2)])
disp(['��֤�����ݵ�MAEΪ��', num2str(mae3)])

% MBE
mbe1 = sum(T_sim1' - T_train) ./ M ;
mbe2 = sum(T_sim2' - T_test ) ./ N ;
mbe3 = sum(T_sim3' - T_valid ) ./ V ;

disp(['ѵ�������ݵ�MBEΪ��', num2str(mbe1)])
disp(['���Լ����ݵ�MBEΪ��', num2str(mbe2)])
disp(['��֤�����ݵ�MBEΪ��', num2str(mbe3)])

% MAPE
mape1 = (mean(abs((T_train - T_sim1') ./ T_train)))*100;
mape2 = (mean(abs(T_test - T_sim2' ) ./ T_test))*100;
mape3 = (mean(abs((T_valid - T_sim3' ) ./ T_valid)))*100;

disp(['ѵ�������ݵ�MAPEΪ��', num2str(mape1),'%'])
disp(['���Լ����ݵ�MAPEΪ��', num2str(mape2),'%'])
disp(['��֤�����ݵ�MAPEΪ��', num2str(mape3),'%'])

% MAPE׼ȷ��
disp(['ѵ�������ݵ�Ԥ��׼ȷ��Ϊ��', num2str((1-mape1/100)*100),'%'])
disp(['���Լ����ݵ�Ԥ��׼ȷ��Ϊ��', num2str((1-mape2/100)*100),'%'])
disp(['��֤�����ݵ�Ԥ��׼ȷ��Ϊ��', num2str((1-mape3/100)*100),'%'])


% MSE
mse1 = sum((T_sim1' - T_train).^2) ./ M;
mse2 = sum((T_sim2' - T_test ).^2) ./ N;
mse3 = sum((T_sim3' - T_valid ).^2) ./ V;

disp(['ѵ�������ݵ�MSEΪ��', num2str(mse1)])
disp(['���Լ����ݵ�MSEΪ��', num2str(mse2)])
disp(['��֤�����ݵ�MSEΪ��', num2str(mse3)])

[mae10,mse10,rmse10,mape10,error10,errorPercent10]=calc_error(T_test,T_sim2);
[mae11,mse11,rmse11,mape11,error11,errorPercent11]=calc_error(T_valid,T_sim3);
%%  ��ͼ
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'ѵ����Ԥ�����Ա�'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'���Լ�Ԥ�����Ա�'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

figure
plot(1: V, T_valid, 'r-*', 1: V, T_sim3, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'��֤��Ԥ�����Ա�'; ['RMSE=' num2str(error3)]};
title(string)
xlim([1, V])
grid


%%  ����ɢ��ͼ
sz = 25;
c = 'b';

figure
scatter(T_train, T_sim1, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('ѵ������ʵֵ');
ylabel('ѵ����Ԥ��ֵ');
xlim([min(T_train) max(T_train)])
ylim([min(T_sim1) max(T_sim1)])
title('ѵ����Ԥ��ֵ vs. ѵ������ʵֵ')

figure
scatter(T_test, T_sim2, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('���Լ���ʵֵ');
ylabel('���Լ�Ԥ��ֵ');
xlim([min(T_test) max(T_test)])
ylim([min(T_sim2) max(T_sim2)])
title('���Լ�Ԥ��ֵ vs. ���Լ���ʵֵ')

figure
scatter(T_valid, T_sim3, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('��֤����ʵֵ');
ylabel('��֤��Ԥ��ֵ');
xlim([min(T_valid) max(T_valid)])
ylim([min(T_sim3) max(T_sim3)])
title('��֤��Ԥ��ֵ vs. ��֤����ʵֵ')

