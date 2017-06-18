#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

/*
Utility function which calculates RMSE 
between estimations and ground truth
*/

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
        cout << "Invalid estimation or ground_truth data" << endl;
        return rmse;
    }
    for (int i = 0; i < estimations.size(); ++i) {
        VectorXd res = estimations[i] - ground_truth[i];
        res = res.array() * res.array();
        rmse += res;
    }
    rmse /= estimations.size();
    rmse = rmse.array().sqrt();

    return rmse;


}
/*
Calculate the Jacobian for the EKF
*/
MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    // initialize Jacobian matrix
    MatrixXd Hj(3, 4);
    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);

    //pre-compute a set of terms to avoid repeated calculation
    double c1 = px*px+py*py;
    double c2 = sqrt(c1);
    double c3 = (c1*c2);

    //check division by zero
    if(fabs(c1) < 0.0001){
        cout << "CalculateJacobian () - Error - Division by Zero" << endl;
        return Hj;
    }

    //compute the Jacobian matrix
    Hj << (px/c2), (py/c2), 0, 0,
            -(py/c1), (px/c1), 0, 0,
            py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

    return Hj;
}
