function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);% vector length
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

y_matrix = eye(num_labels)(y,:); % Expands y= 1 2 ... 10 into matrices or 0s and 1s

a1= [ones(m, 1) X];
z2=a1*Theta1';
s=size(z2, 1);
a2= [ones(s,1) sigmoid(z2)];
a3=sigmoid(a2*Theta2');

q = ones(size(y_matrix'));
t= ones(size(a3));

%J=1/m * sum( sum( (-y_matrix.*log(a3))-(1-y_matrix).*log(1-a3)))%elementwise version
J=1/m * trace ((-y_matrix'*log(a3))-(q-y_matrix')*log(t-a3));% vectorised version
%Theta(:,2:end)%excludes the bias colum for theta
%reg_cost=lambda/(2*m) *(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
reg_cost=lambda/(2*m) *(trace(Theta1(:,2:end)*Theta1(:,2:end)')+trace(Theta2(:,2:end)*Theta2(:,2:end)'));
J=J+reg_cost;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Back Propagation
d3=a3-y_matrix;%error on hypothesis nodes
z2=a1*Theta1';
disp(size(Theta1(1,:)))
d2=d3*Theta2(:,2:end).*sigmoidGradient(z2);% error on nodes in second layer (hidden)

Delta1=d2'*a1;
Delta2=d3'*a2;

Theta1_grad=1/m *Delta1;
Theta2_grad=1/m *Delta2;

Theta1_unbias=Theta1;
Theta2_unbias=Theta2;

Theta1_unbias(:,1) = 0  ;  
Theta2_unbias(:,1) = 0 ;

Theta11=Theta1_unbias*lambda/m;
Theta22=Theta2_unbias*lambda/m;

Theta1_grad=1/m *Delta1 + Theta11;
Theta2_grad=1/m *Delta2 + Theta22;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
