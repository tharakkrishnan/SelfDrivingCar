#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
    // Prediction part of the Kalman Filter algorithm
    // This is common for both standard and extended Kalman Filter algorithms
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;

}

void KalmanFilter::Update(const VectorXd &z) {
    // Update step of the standard Kalman Filter algorithm
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    // new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ -= K * H_ * P_;
}

void KalmanFilter::NormalizeAngle(double& phi)
{
  phi = atan2(sin(phi), cos(phi));
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    // Update state of the extended Kalman Filter
    double px = x_[0];
    double py = x_[1];
    double vx = x_[2];
    double vy = x_[3];

    if (fabs(px) < 0.0001) {
        px = 0.0001;
    }

    double rho = sqrt(px * px + py * py);
    if (fabs(rho) < 0.0001) {
        rho = 0.0001;
    }

    double theta = atan2(py, px);
    double rho_dot = (px * vx + py * vy) / rho;

    VectorXd z_pred(3);
    z_pred << rho, theta, rho_dot;

    VectorXd y = z - z_pred;
    NormalizeAngle(y(1));
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    // new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}
