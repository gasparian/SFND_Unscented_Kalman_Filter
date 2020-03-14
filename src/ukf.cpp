#include <iostream>
#include <stdexcept>
#include "ukf.h"
#include "Eigen/Dense"
#include "Eigen/QR"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  is_initialized_ = false;
  // state vector and augmented state vector lengths
  n_x_ = 5;
  n_aug_ = 7;

  // define spreading parameter
  lambda_ = 3 - n_aug_;
  lambda_sq_ = sqrt(lambda_ + n_aug_);

  weights_ = VectorXd(2 * n_aug_ + 1);
  // set weights
  weights_(0) = 1.0 * lambda_ / (lambda_ + n_aug_);
  for ( int i=1; i < 2 * n_aug_ + 1; i++ ) {
      weights_(i) = 1.0 * 1 / (2 * ( lambda_ + n_aug_));
  }

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // matrix for predicted sigma points in state space
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.2; // was 30

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5; // was 30
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */

  // lidar and radar measurement noise covariances matrices
  lidar_R_ = MatrixXd(2, 2);
  lidar_R_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;
  radar_R_ = MatrixXd(3, 3);
  radar_R_ << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  
  // std::cout << "Kalman Filter Initialization " << std::endl;

  NIS_radar_ = 0.0;
  NIS_lidar_ = 0.0;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  double px = 1.0, py = 1.0, v = 1.0, phi = 0.0, phi_ = 0.0, init_var = 1.0;

  // std::cout << "Process measurements " << std::endl;

  if (!this->is_initialized_) {
    // set the state with the initial location and zero velocity (CTRV model -> 5 variables)
    if ( meas_package.sensor_type_ == MeasurementPackage::LASER ) {
      px = meas_package.raw_measurements_[0];
      py = meas_package.raw_measurements_[1];

      std::cout << "Filter initialized with laser: " << px << ',' << py << ',' << std::endl;

    } else if ( meas_package.sensor_type_ == MeasurementPackage::RADAR ) {
      px = meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]);
      py = meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]);
      v = sqrt(meas_package.raw_measurements_[2] * meas_package.raw_measurements_[2] *
               cos(meas_package.raw_measurements_[1]) * cos(meas_package.raw_measurements_[1]) +
               meas_package.raw_measurements_[2] * meas_package.raw_measurements_[2] *
               sin(meas_package.raw_measurements_[1]) * sin(meas_package.raw_measurements_[1]));

      std::cout << "Filter initialized with radar: " << px << ',' << py << ',' << v << ',' << std::endl;
    }

    this->x_ << px, py, v, phi, phi_;
    this->P_ << 1.0, 0, 0, 0, 0,
                0, 1.0, 0, 0, 0,
                0, 0, 1.0, 0, 0,
                0, 0, 0, std_radphi_, 0,
                0, 0, 0, 0, std_radphi_;

    this->time_us_ = meas_package.timestamp_;
    this->is_initialized_ = true;
    return;
  }

  // compute the time elapsed between the current and previous measurements
  // dt - expressed in seconds
  double dt = (meas_package.timestamp_ - this->time_us_) / 1000000.0;
  this->time_us_ = meas_package.timestamp_;

  // make the prediction and update state
  this->Prediction(dt);
  if ( meas_package.sensor_type_ == MeasurementPackage::LASER & this->use_laser_) {
    this->UpdateLidar(meas_package);
  } else if ( meas_package.sensor_type_ == MeasurementPackage::RADAR & this->use_radar_ ) {
    this->UpdateRadar(meas_package);
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  // std::cout << "Prediction step " << std::endl;

  // create augmented mean state
  VectorXd x_aug = VectorXd(this->n_aug_);
  x_aug.fill(0.0);
  x_aug.head(this->n_x_) = this->x_;

  // create augmented covariance matrix
  MatrixXd P_aug = MatrixXd(this->n_aug_, this->n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(this->n_x_, this->n_x_) = this->P_;
  // add Q matrix
  P_aug(5, 5) = this->std_a_ * this->std_a_;
  P_aug(6, 6) = this->std_yawdd_ * this->std_yawdd_;

  // create square root matrix
  MatrixXd A_aug = P_aug.llt().matrixL();

  // create augmented sigma points
  this->Xsig_aug_.col(0) = x_aug;
  for ( int i=0; i < this->n_aug_; i++ ) {
      this->Xsig_aug_.col(i+1) = x_aug + this->lambda_sq_ * A_aug.col(i);
      this->Xsig_aug_.col(i+1+this->n_aug_) = x_aug - this->lambda_sq_ * A_aug.col(i);
  }

  // predict sigma points
  double delta_sq = 0.5 * delta_t * delta_t;
  VectorXd x_sigma(this->n_x_), noise(this->n_x_);
  // p_x -> 0; p_y -> 1; v -> 2; phi -> 3; phi^ -> 4
  // nu_a -> 5; nu_yaw_rate -> 6
  for ( int i=0; i < 2 * this->n_aug_ + 1; i++ ) {
    noise << delta_sq * cos(this->Xsig_aug_(3, i)) * this->Xsig_aug_(5, i),
             delta_sq * sin(this->Xsig_aug_(3, i)) * this->Xsig_aug_(5, i),
             delta_t * this->Xsig_aug_(5, i),
             delta_sq * this->Xsig_aug_(6, i),
             delta_t * this->Xsig_aug_(6, i);

    // avoid division by zero
    if ( fabs(this->Xsig_aug_(4, i)) > 0.001 ) {
        x_sigma << (this->Xsig_aug_(2, i) / this->Xsig_aug_(4, i)) * (sin(this->Xsig_aug_(3, i) + this->Xsig_aug_(4, i) * delta_t) - sin(this->Xsig_aug_(3, i))), 
                   (this->Xsig_aug_(2, i) / this->Xsig_aug_(4, i)) * (cos(this->Xsig_aug_(3, i)) - cos(this->Xsig_aug_(3, i) + this->Xsig_aug_(4, i) * delta_t)), 
                   0, 
                   this->Xsig_aug_(4, i) * delta_t, 
                   0;
    } else {
        x_sigma << this->Xsig_aug_(2, i) * cos(this->Xsig_aug_(3, i)) * delta_t, 
                   this->Xsig_aug_(2, i) * sin(this->Xsig_aug_(3, i)) * delta_t, 
                   0, 0, 0;
    }
    this->Xsig_pred_.col(i) = this->Xsig_aug_.col(i).head(this->n_x_) + x_sigma + noise;
  }

  // final step - predict (mean) state and state covariance matrix
  // predict state mean
  this->x_.fill(0.0);
  for ( int i=0; i < 2*this->n_aug_+1; i++ ) {
      this->x_ += this->weights_(i) * this->Xsig_pred_.col(i);
  }
  // angle normalization
  while (this->x_(3)> M_PI) this->x_(3)-=2.*M_PI;
  while (this->x_(3)<-M_PI) this->x_(3)+=2.*M_PI;

  // predicted state covariance matrix
  this->P_.fill(0.0);
  for (int i = 0; i < 2 * this->n_aug_ + 1; ++i) {  // iterate over sigma points
    // state difference
    VectorXd x_diff = this->Xsig_pred_.col(i) - this->x_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    this->P_ = this->P_ + this->weights_(i) * x_diff * x_diff.transpose();
  }

  // std::cout << "X: " << this->x_ << std::endl;

}

void UKF::UpdateLidar(MeasurementPackage& meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  // std::cout << "Update lidar " << std::endl;

  int n_z = 2;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * this->n_aug_ + 1);
  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // transform sigma points into measurement space
  for ( int i=0; i < 2 * this->n_aug_ + 1; i++ ) {
      Zsig(0, i) = this->Xsig_pred_(0, i);
      Zsig(1, i) = this->Xsig_pred_(1, i);
  }
  // calculate mean predicted measurement
  for ( int i=0; i < 2*this->n_aug_+1; i++ ) {
      z_pred += this->weights_(i) * Zsig.col(i);
  }
  // calculate innovation covariance matrix S
  for ( int i=0; i < 2*this->n_aug_+1; i++ ) {
      S += this->weights_(i) * (Zsig.col(i) - z_pred) * (Zsig.col(i) - z_pred).transpose();
  }
  S += this->lidar_R_;

  // std::cout << "Lidar pred: " << z_pred << std::endl;

  // final update step
  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(this->n_x_, n_z);
  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * this->n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = this->Xsig_pred_.col(i) - this->x_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + this->weights_(i) * x_diff * z_diff.transpose();
  }

  // MatrixXd S_inv = S.inverse();
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  // update state mean and covariance matrix
  this->x_ = this->x_ + K * z_diff;
  this->P_ = this->P_ - K * S * K.transpose();

  // calculate NIS
  // this->NIS_lidar_ = z_diff.transpose() * S_inv * z_diff;
}

void UKF::UpdateRadar(MeasurementPackage& meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */ 

  // std::cout << "Update radar " << std::endl;

  int n_z = 3;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * this->n_aug_ + 1);
  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // transform sigma points into measurement space
  for ( int i=0; i < 2 * this->n_aug_ + 1; i++ ) {
      Zsig(0, i) = sqrt(this->Xsig_pred_(0, i) * this->Xsig_pred_(0, i) + this->Xsig_pred_(1, i) * this->Xsig_pred_(1, i));
      Zsig(1, i) = atan2(this->Xsig_pred_(1, i), this->Xsig_pred_(0, i));
      Zsig(2, i) = (this->Xsig_pred_(0, i) * cos(this->Xsig_pred_(3, i)) * this->Xsig_pred_(2, i) + this->Xsig_pred_(1, i) * sin(this->Xsig_pred_(3, i)) * this->Xsig_pred_(2, i))
                    / std::max(0.001, Zsig(0, i));
  }
  // calculate mean predicted measurement
  for ( int i=0; i < 2*this->n_aug_+1; i++ ) {
      z_pred += this->weights_(i) * Zsig.col(i);
  }
  // innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * this->n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S += this->weights_(i) * z_diff * z_diff.transpose();
  }
  S += this->radar_R_;

  // std::cout << "Radar pred: " << z_pred << std::endl;

  // final update step
  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(this->n_x_, n_z);
  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * this->n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = this->Xsig_pred_.col(i) - this->x_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + this->weights_(i) * x_diff * z_diff.transpose();
  }

  // MatrixXd S_inv = S.inverse();
  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  // residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  // angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  // update state mean and covariance matrix
  this->x_ = this->x_ + K * z_diff;
  this->P_ = this->P_ - K * S * K.transpose();

  // calculate NIS
  // this->NIS_radar_ = z_diff.transpose() * S_inv * z_diff;
}
