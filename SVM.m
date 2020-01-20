function SVM()

    % Solve the quadratic optimisation problem. Estimate the labels for 
    % each of the test samples and report the accuracy of your trained SVM 
    % utilising the ground truth labels for the test data.

    
    load('X.mat');
    load('l.mat');
    load('X_test.mat');
    load('l_test.mat');
    
    N = size(X,1);
    n = size(X,2);
    % Get inverse of S_t
    invSt = inv(X*X'/N);
    
    % Compute Hessian
    H = (l.*l').*(X'*(invSt)*X);
    
    % Initialise f, A, c, A_e, c_e, g_l, g_u
    f = -ones(1,n);
    A = zeros(1,n);
    c = 0;
    A_e = l';
    c_e = 0;
    g_l = zeros(n,1);
    g_u = ones(n,1);
    
    % Get alpha
    alpha = quadprog(H, f, A, c, A_e, c_e, g_l, g_u);
    
    svm.sv = find(alpha>0.0);
    svm.ns = size(svm.sv,1);
    % Find w 
    svm.w = invSt*(X*(alpha.*l));
    X_t = X';
    % Find b
    svm.b = 1/svm.ns * sum(l(svm.sv) - X_t(svm.sv,:)*svm.w);
    
    % Find estimated label  
    labelEst = sign(svm.w'*X_test + svm.b);
    % Compare estimated label with l_test and get accuracy
    accuracy = (sum(labelEst(:)'== l_test(:)')/size(l_test,1))*100;
    fprintf('Accuracy on the test set is %3.2f\n', accuracy);
    

end