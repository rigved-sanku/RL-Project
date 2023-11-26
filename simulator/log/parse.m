clear all
load states.mat
load control.mat
close all
xyz = state(:, 1:3);
vxyz = state(:, 4:6);
quat = state(:, 7:10);
pqr = state(:, 11:13);

plot(time, xyz);
title('xyz')

figure
plot(time, vxyz)
title('vxyz')

figure
plot(time, 180/pi*quat2eul(quat))
title('ypr')
legend

figure
plot(time, control)
title('control array')

figure
plot(time, pqr)
title('turn rate')

figure
plot(control_time, control_premix)
title('control before mixing')
legend