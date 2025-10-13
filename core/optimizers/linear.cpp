#include <pybind11/pybind11.h>
#include <utility>
#include <vector>
#include <Eigen/Dense>

namespace py = pybind11;
using namespace std;

double compute_prediction(double point, double weights, double bias) {
    return (point * weights) + bias;
}

pair<double, double> gradient_descent(vector<double> X, vector<double> Y, double weights, double bias, double learning_rate) {
    int size = X.size();
    double dw = 0.0, db = 0.0;
    for (int i = 0; i < size; ++i) {
        double y_pred = compute_prediction(X[i], weights, bias);
        double error = y_pred - Y[i];
        dw += X[i] * error;
        db += error;
    }
    dw /= size;
    db /= size;
    weights -= learning_rate * dw;
    bias -= learning_rate * db;
    return {weights, bias};
}

pair<double,double> stochastic_gradient_descent(vector<double> X, vector<double> Y, double weights, double bias, double learning_rate) {
    int size = X.size();
    int pt = rand() % size; 
    double y_pred = compute_prediction(X[pt],weights,bias);
    double error = y_pred - Y[pt];
    double dw = X[pt] * error;
    double db = error;
    weights -= learning_rate * dw;
    bias -= learning_rate * db;
    return {weights,bias};
}

Eigen::MatrixXd convert_vvd_to_matrix(const vector<vector<double>>& vvd) {
    if (vvd.empty() || vvd[0].empty()) {
        return Eigen::MatrixXd();
    }
    size_t rows = vvd.size();
    size_t cols = vvd[0].size();
    Eigen::MatrixXd mat(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mat(i, j) = vvd[i][j];
        }
    }
    return mat;
}

vector<vector<double>> eigen_to_vvd(const Eigen::MatrixXd& mat) {
    vector<vector<double>> vvd(mat.rows(), vector<double>(mat.cols()));
    for (int i = 0; i < mat.rows(); ++i)
        for (int j = 0; j < mat.cols(); ++j)
            vvd[i][j] = mat(i, j);
    return vvd;
}

vector<vector<double>> transpose(vector<vector<double>>& grid) {
    int rows = grid.size();
    int cols = grid[0].size();
    vector<vector<double>> result(cols,vector<double>(rows));
    for(int i = 0; i < rows; i++) 
        for(int j = 0; j < cols; j++) 
            result[j][i] = grid[i][j];
    return result;
}

vector<vector<double>> invert_matrix(vector<vector<double>>& grid) {
    Eigen::MatrixXd inv = convert_vvd_to_matrix(grid).inverse();
    return eigen_to_vvd(inv);
}

vector<vector<double>> matrix_multi(vector<vector<double>>& A,vector<vector<double>>& B) {
    int rows = A.size();
    int cols = B[0].size();
    int common = A[0].size();
    vector<vector<double>> result(rows,vector<double>(cols,0.0));
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            double sum = 0.0;
            for(int k = 0; k < common; k++) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

pair<double,double> closed_form(vector<vector<double>> X,
        vector<vector<double>> Y, double weights, double bias){
    Eigen::MatrixXd Xmat = convert_vvd_to_matrix(X);
    Eigen::MatrixXd ymat = convert_vvd_to_matrix(Y);
    Eigen::VectorXd theta = (Xmat.transpose() * Xmat).ldlt().solve(Xmat.transpose() * ymat);
    weights = theta(0);
    if(theta.size() > 1) bias = theta(1);
    return {weights,bias};
}

PYBIND11_MODULE(optimizers, m) {
    m.def("gradient_descent", &gradient_descent, "Gradient Descent optimizer");
    m.def("stochastic_gradient_descent", &stochastic_gradient_descent, "Stochastic Gradient Descent optimizer");
    m.def("closed_form", &closed_form, "Closed Form optimizer");
}
