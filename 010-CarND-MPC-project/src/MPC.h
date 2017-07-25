#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

using namespace std;
const int latency = 2;
class MPC {
 public:
  MPC();

  virtual ~MPC();

  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuatotions.

    // These two vectors hold MPC X and Y coordinates

    vector<double> mpc_x_vals;
    vector<double> mpc_y_vals;
    vector<double> delta;
    vector<double> a;
    
    vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);
    double delta_prev {0};
    double a_prev {0.1};
};

#endif /* MPC_H */
