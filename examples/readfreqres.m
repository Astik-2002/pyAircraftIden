function [sys] = readfreqres(name)
%READFREQRES �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
d = csvread(name);
freq = d(:,1);
resp = d(:,2) + d(:,3)*1i;
sys = frd(resp, freq);
end

