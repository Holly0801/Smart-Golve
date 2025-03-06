% 初始化串口
s = serialport('COM5', 9600);
fopen(s);

% 初始化图形
figure;
h = plot(zeros(5, 100));
ylim([0 1023]);
title('Real Time Sensor Data');
xlabel('Time');
ylabel('Sensor Data');
legend('Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4', 'Sensor 5');

% 数据缓冲区
dataBuffer = zeros(5, 100);

% 实时读取和绘图
while ishandle(h(1))
    data = fscanf(s, '%d,%d,%d,%d,%d'); % 读取数据
    dataBuffer = [dataBuffer(:, 2:end), data]; % 更新缓冲区
    for i = 1:5
        set(h(i), 'YData', dataBuffer(i, :)); % 更新图形
    end
    drawnow;
end

% 关闭串口
fclose(s);
delete(s);
clear s;