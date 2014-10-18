function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
theta_squared = 0
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
q = size(theta,1)
r = size(theta,2)

for i=1:m,
	g = sigmoid(theta' * X(i,:)');
	J = J + (-1 * y(i, :) * log(g)) - ((1 - y(i, :)) * log(1 - g));
	
	for j=1:q,
		grad(j) = grad(j) + (g - y(i, :)) * X(i, j);
	end;
end;


for i=1:q,
	r = ((lambda * theta(i)) / m)
	if i == 1,
		grad(i) = grad(i)/m
	else
		theta_squared = theta_squared + theta(i)^2;
		grad(i) = (grad(i)/m) + r
	end;
end;
r = ((theta_squared * lambda) / (2 * m));
J = (1/m * J) + r




% =============================================================

end
