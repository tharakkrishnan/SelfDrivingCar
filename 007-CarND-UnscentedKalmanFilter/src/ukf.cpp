#include <iostream>
#include "ukf.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
    is_initialized = false;
    time_us = 0;
    // if this is false, laser measurements will be ignored (except during init)
    use_laser = true;
    // if this is false, radar measurements will be ignored (except during init)
    use_radar = true;
    // State dimension
    Nx = 5;
    // Augmented state dimension
    Naug = 7;
    // initial state vector
    x_ = VectorXd(Nx);
    // initial covariance matrix
    P_ = MatrixXd(Nx, Nx);
    // predicted sigma points matrix
    Xsig_pred = MatrixXd(Nx, 2 * Naug + 1);
    // Sigma point spreading parameter
    lambda = 3 - Naug;
    // Weights of sigma points
    weights = VectorXd(2 * Naug + 1);
    weights.segment(1, 2 * Naug).fill(0.5 / (Naug + lambda));
    weights(0) = lambda / (lambda + Naug);
    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a = 0.8;
    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd = 0.6;
    // Laser measurement noise standard deviation position1 in m
    std_laspx = 0.15;
    // Laser measurement noise standard deviation position2 in m
    std_laspy = 0.15;
    // Radar measurement noise standard deviation radius in m
    std_radr = 0.3;
    // Radar measurement noise standard deviation angle in rad
    std_radphi = 0.03;
    // Radar measurement noise standard deviation radius change in m/s
    std_radrd = 0.3;
    // the current NIS for radar
    NIS_radar = 0.0;
    // the current NIS for laser
    NIS_laser = 0.0;
}

UKF::~UKF()
{
}

MatrixXd UKF::PredictedSigmaPoints(double timdediff)
{

    //create augmented mean vector
    VectorXd x_aug = VectorXd(Naug);
    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(Naug, Naug);
    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(Naug, 2 * Naug + 1);
    //create augmented mean state
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5, 5) = P_;
    P_aug(5, 5) = std_a * std_a;
    P_aug(6, 6) = std_yawdd * std_yawdd;

    //create square root of P
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < Naug; i++)
    {
        Xsig_aug.col(i + 1) = x_aug + sqrt(lambda + Naug) * L.col(i);
        Xsig_aug.col(i + 1 + Naug) = x_aug - sqrt(lambda + Naug) * L.col(i);
    }

    // sigma points prediction
    for (int i = 0; i < 2 * Naug + 1; i++)
    {
        //extract values for better readability
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        // improve safety
        if (fabs(p_x) < 0.001 && fabs(p_y) < 0.001)
        {
            p_x = 0.1;
            p_y = 0.1;
        }

        //predicted state values
        double px_p, py_p;
        //avoid division by zero
        if (fabs(yawd) > 0.001)
        {
            px_p = p_x + v / yawd * (sin(yaw + yawd * timdediff) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * timdediff));
        }
        else
        {
            px_p = p_x + v * timdediff * cos(yaw);
            py_p = p_y + v * timdediff * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd * timdediff;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5 * nu_a * timdediff * timdediff * cos(yaw);
        py_p = py_p + 0.5 * nu_a * timdediff * timdediff * sin(yaw);
        v_p = v_p + nu_a * timdediff;

        yaw_p = yaw_p + 0.5 * nu_yawdd * timdediff * timdediff;
        yawd_p = yawd_p + nu_yawdd * timdediff;

        //write predicted sigma point into right column
        Xsig_pred(0, i) = px_p;
        Xsig_pred(1, i) = py_p;
        Xsig_pred(2, i) = v_p;
        Xsig_pred(3, i) = yaw_p;
        Xsig_pred(4, i) = yawd_p;
    }

    return Xsig_pred;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
   
    if (!is_initialized)
    {
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
        {
            /**
            Convert radar from polar to cartesian coordinates and initialize state.
            */
            float rho = meas_package.raw_measurements_[0];
            float phi = meas_package.raw_measurements_[1];
            float rho_dot = meas_package.raw_measurements_[2];
            x_ << rho * cos(phi), rho * sin(phi), rho_dot, 0.0, 0.0;
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
        {
            /**
            Initialize state.
            */
            // Initialize the state ekf_.x_ with the first measurement.
            // set the state with the initial location and zero velocity
            float px = meas_package.raw_measurements_[0];
            float py = meas_package.raw_measurements_[1];
            x_ << px, py, 0.0, 0.0, 0.0;
        }

        P_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;

        time_us = meas_package.timestamp_;
        is_initialized = true;
        return;
    }

    // after initialization step prediction and update starts

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    //compute the time elapsed between the current and previous measurements
    double time_delta = (meas_package.timestamp_ - time_us) / 1000000.0;
    time_us = meas_package.timestamp_;

    // Updates x_ and P_
    Prediction(time_delta);

    /*****************************************************************************
     *  Update
     ****************************************************************************/

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
        // Radar updates
        UpdateRadar(meas_package);
    }
    else
    {
        // Lidar updates
        UpdateLidar(meas_package);
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t)
{
     MatrixXd Xsig_pred = PredictedSigmaPoints(delta_t);
    //create vector for predicted state (5 x 1)
    VectorXd x = VectorXd(Nx);

    //create covariance matrix for prediction (5 x 5)
    MatrixXd P = MatrixXd(Nx, Nx);

    //predicted state mean
    x.fill(0.0);
    for (int i = 0; i < 2 * Naug + 1; i++)
    { //iterate over sigma points
        x = x + weights(i) * Xsig_pred.col(i);
    }

    //predicted state covariance matrix
    P.fill(0.0);
    for (int i = 0; i < 2 * Naug + 1; i++)
    { //iterate over sigma points

        // state difference
        VectorXd x_diff = Xsig_pred.col(i) - x;
        //angle normalization
        while (x_diff(3) > M_PI)
            x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI)
            x_diff(3) += 2. * M_PI;

        P = P + weights(i) * x_diff * x_diff.transpose();
    }

    // Update predicted state and covariance Matrix
    x_ = x;
    P_ = P;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package)
{
 
    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    //set measurement dimension, lidar can measure px, py
    int n_z = 2;
    // 2 x 15
    MatrixXd Zsig = MatrixXd(n_z, 2 * Naug + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * Naug + 1; i++)
    {

        // extract values for better readability
        double p_x = Xsig_pred(0, i);
        double p_y = Xsig_pred(1, i);

        // measurement model
        Zsig(0, i) = p_x;
        Zsig(1, i) = p_y;
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * Naug + 1; i++)
    {
        z_pred = z_pred + weights(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * Naug + 1; i++)
    {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1) > M_PI)
            z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI)
            z_diff(1) += 2. * M_PI;

        S = S + weights(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z, n_z);
    R << std_laspx * std_laspx, 0, 0, std_laspy * std_laspy;
    S = S + R;

    /*****************************************************************************
      *  Update
      ****************************************************************************/

    VectorXd z = VectorXd(n_z);
    // measurement for laser
    z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];

    // 5 x 2
    MatrixXd Tc = MatrixXd(Nx, n_z);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * Naug + 1; i++)
    {

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        while (z_diff(1) > M_PI)
            z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI)
            z_diff(1) += 2. * M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred.col(i) - x_;
        //angle normalization
        while (x_diff(3) > M_PI)
            x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI)
            x_diff(3) += 2. * M_PI;

        Tc = Tc + weights(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = z - z_pred;

    //angle normalization
    while (z_diff(1) > M_PI)
        z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
        z_diff(1) += 2. * M_PI;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    NIS_laser = (meas_package.raw_measurements_ - z_pred).transpose() * S.inverse() *
                 (meas_package.raw_measurements_ - z_pred);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
 
    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 3;
    MatrixXd Zsig = MatrixXd(n_z, 2 * Naug + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * Naug + 1; i++)
    {

        // extract values for better readability
        double p_x = Xsig_pred(0, i);
        double p_y = Xsig_pred(1, i);
        double v = Xsig_pred(2, i);
        double yaw = Xsig_pred(3, i);

        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;

        // measurement model
        Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y); //r
        if (fabs(p_y) > 0.001 && fabs(p_x) > 0.001)
        {
            Zsig(1, i) = atan2(p_y, p_x);
        }
        else
        {
            Zsig(1, i) = 0.0;
        } //phi
        if (fabs(sqrt(p_x * p_x + p_y * p_y)) > 0.001)
        { //r_dot
            Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);
        }
        else
        {
            Zsig(2, i) = 0.0;
        }
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * Naug + 1; i++)
    {
        z_pred = z_pred + weights(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * Naug + 1; i++)
    {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1) > M_PI)
            z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI)
            z_diff(1) += 2. * M_PI;

        S = S + weights(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z, n_z);
    R << std_radr * std_radr, 0, 0,
        0, std_radphi * std_radphi, 0,
        0, 0, std_radrd * std_radrd;
    S = S + R;

    /*****************************************************************************
    *  Update
    ****************************************************************************/

    VectorXd z = VectorXd(n_z);
    // measurement for radar
    z << meas_package.raw_measurements_[0],
        meas_package.raw_measurements_[1],
        meas_package.raw_measurements_[2];

    // 5 x 3
    MatrixXd Tc = MatrixXd(Nx, n_z);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * Naug + 1; i++)
    { //2n+1 simga points

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        while (z_diff(1) > M_PI)
            z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI)
            z_diff(1) += 2. * M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred.col(i) - x_;
        //angle normalization
        while (x_diff(3) > M_PI)
            x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI)
            x_diff(3) += 2. * M_PI;

        Tc = Tc + weights(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = z - z_pred;

    //angle normalization
    while (z_diff(1) > M_PI)
        z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
        z_diff(1) += 2. * M_PI;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    // Update radar
    NIS_radar = (meas_package.raw_measurements_ - z_pred).transpose() * S.inverse() *
                 (meas_package.raw_measurements_ - z_pred);
}
