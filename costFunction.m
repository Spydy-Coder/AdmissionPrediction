function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION 

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));
J=(1/m)*(((-1*y)'*log(sigmoid(X*theta)))-((1-y)'*log(1-sigmoid(X*theta))));
error=sigmoid(X*theta)-y;
grad=(1/m)*((X')*(error));

end
