#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;
    // checking dimensions of estimations and ground_truth vectors
    if (estimations.size() == 0 || (estimations.size() != ground_truth.size())) {
        cout << "Invalid estimation or ground truth data " << endl;
        return rmse;
    }

    // calculate squared difference between estimations and ground_truth vectors
    for (int i = 0; i < estimations.size(); i++) {
        VectorXd result = estimations[i] - ground_truth[i];
        result = result.array() * result.array();
        rmse += result;
    }

    // calculate mean
    rmse = rmse / estimations.size();
    // calculate rmse
    rmse = rmse.array().sqrt();
    return rmse;
}
