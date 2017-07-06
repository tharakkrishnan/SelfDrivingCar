#ifndef UKF_H
#define UKF_H
#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred;

  ///* time when the state is true, in us
  long time_us;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd;

  ///* Weights of sigma points
  VectorXd weights;

  ///* State dimension
  int Nx;

  ///* Augmented state dimension
  int Naug;

  ///* Sigma point spreading parameter
  double lambda;

  ///* the current NIS for radar
  double NIS_radar;

  ///* the current NIS for laser
  double NIS_laser;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

    /**
     * Init Initializes Unscented Kalman filter
     */
    void Init();

    /**
     * Helper functions
     */
    MatrixXd AugmentedSigmaPoints();

    MatrixXd PredictedSigmaPoints(double timdediff);

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);
};

#endif /* UKF_H */
