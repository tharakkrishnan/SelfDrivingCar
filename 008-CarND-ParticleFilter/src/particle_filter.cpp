#include <random>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>

#include "particle_filter.h"

using namespace std;

#define NUM_PARTICLES 100;

/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */

void ParticleFilter::init(double x, double y, double theta, double std[])
{

    // create normal distributions for x, y, and theta
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);
    std::default_random_engine gen;

    // resize the vectors of particles and weights
    num_particles = NUM_PARTICLES;
    particles.resize(num_particles);

    // generate the particles
    for (auto &p : particles)
    {
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1;
    }

    is_initialized = true;
}

/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{

    std::default_random_engine gen;

    // generate random Gaussian noise
    std::normal_distribution<double> dist_x(0, std_pos[0]);
    std::normal_distribution<double> dist_y(0, std_pos[1]);
    std::normal_distribution<double> dist_theta(0, std_pos[2]);

    for (auto &p : particles)
    {

        // add measurements to each particle
        if (fabs(yaw_rate) < 0.0001)
        { // constant velocity
            p.x += velocity * delta_t * cos(p.theta);
            p.y += velocity * delta_t * sin(p.theta);
        }
        else
        {
            p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
            p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
            p.theta += yaw_rate * delta_t;
        }

        // predicted particles with added sensor noise
        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);
    }
}
/**
	 * dataAssociation Finds which observations correspond to which landmarks (likely by using
	 *   a nearest-neighbors data association).
	 * @param predicted Vector of predicted landmark observations
	 * @param observations Vector of landmark observations
	 */
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{

    for (auto &obs : observations)
    {
        double minD = std::numeric_limits<float>::max();

        for (const auto &pred : predicted)
        {
            double distance = dist(obs.x, obs.y, pred.x, pred.y);
            if (minD > distance)
            {
                minD = distance;
                obs.id = pred.id;
            }
        }
    }
}
/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the 
	 *   observed measurements. 
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [standard deviation of range [m],
	 *   standard deviation of bearing [rad]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks)
{

    for (auto &p : particles)
    {
        p.weight = 1.0;

        // step 1: collect valid landmarks
        vector<LandmarkObs> predictions;
        for (const auto &lm : map_landmarks.landmark_list)
        {
            double distance = dist(p.x, p.y, lm.x_f, lm.y_f);
            if (distance < sensor_range)
            { // if the landmark is within the sensor range, save it to predictions
                predictions.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
            }
        }

        // step 2: convert observations coordinates from vehicle to map
        vector<LandmarkObs> observations_map;
        for (const auto &obs : observations)
        {
            LandmarkObs tmp;
            tmp.x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
            tmp.y = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;
            observations_map.push_back(tmp);
        }

        // step 3: find landmark index for each observation
        dataAssociation(predictions, observations_map);

        // step 4: compute the particle's weight:
        // see equation this link:
        for (const auto &obs_m : observations_map)
        {

            Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs_m.id - 1);
            double x_term = pow(obs_m.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
            double y_term = pow(obs_m.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
            double w = exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
            p.weight *= w;
        }

        weights.push_back(p.weight);
    }
}

/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
void ParticleFilter::resample()
{

    // generate distribution according to weights
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(weights.begin(), weights.end());

    // create resampled particles
    vector<Particle> resampled_particles;
    resampled_particles.resize(num_particles);

    // resample the particles according to weights
    for (int i = 0; i < num_particles; i++)
    {
        int idx = dist(gen);
        resampled_particles[i] = particles[idx];
    }

    // assign the resampled_particles to the previous particles
    particles = resampled_particles;

    // clear the weight vector for the next round
    weights.clear();
}

/* particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
 * associations: The landmark id that goes along with each listed association
 * sense_x: the associations x mapping already converted to world coordinates
 * sense_y: the associations y mapping already converted to world coordinates
 */
Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}
