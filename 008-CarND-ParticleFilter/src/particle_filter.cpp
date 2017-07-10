/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"
#include "helper_functions.h"

void ParticleFilter::init(double x, double y, double theta, double std[])
{
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // specify number of particles
    num_particles = 50;

    // clear particles and weights vectors
    particles.clear();
    weights.clear();

    // initialize normal distribution
    std::default_random_engine gen;
    std::normal_distribution<double> x_random(x, std[0]);
    std::normal_distribution<double> y_random(y, std[1]);
    std::normal_distribution<double> theta_random(theta, std[2]);

    // creating particles
    for (int i = 0; i < num_particles; i++)
    {
        Particle temp_particle;
        temp_particle.id = i;
        temp_particle.x = x_random(gen);
        temp_particle.y = y_random(gen);
        temp_particle.theta = theta_random(gen);

        particles.push_back(temp_particle);
        weights.push_back(temp_particle.weight);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    // initialize normal distribution
    std::default_random_engine gen;
    std::normal_distribution<double> x_random(0, std_pos[0]);
    std::normal_distribution<double> y_random(0, std_pos[1]);
    std::normal_distribution<double> theta_random(0, std_pos[2]);

    // handling division by zero error
    if (fabs(0.0001) > yaw_rate)
    {
        yaw_rate = 0.0001;
    }
    for (int i = 0; i < num_particles; i++)
    {
        particles[i].x +=
            (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) +
            x_random(gen);
        particles[i].y +=
            (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) +
            y_random(gen);
        particles[i].theta += yaw_rate * delta_t + y_random(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks)
{
    // TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
    //   for the fact that the map's y-axis actually points downwards.)
    //   http://planning.cs.uiuc.edu/node99.html

    for (int i = 0; i < num_particles; i++)
    {

        double weight = 1;

        //transforming the observations to map space
        for (int j = 0; j < observations.size(); j++)
        {

            double transformed_x =
                observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) +
                particles[i].x;
            double transformed_y =
                observations[j].y * cos(particles[i].theta) + observations[j].x * sin(particles[i].theta) +
                particles[i].y;

            Map::single_landmark_s nearest_landmark;
            double min_sensor_distance = sensor_range;

            //calculating the distance between landmarks and transformed observations
            for (int k = 0; k < map_landmarks.landmark_list.size(); k++)
            {

                Map::single_landmark_s current_landmark = map_landmarks.landmark_list[k];
                double distance =
                    fabs(transformed_x - current_landmark.x_f) + fabs(transformed_y - current_landmark.y_f);

                // looking at neatest landmark which matches with the observations
                if (distance < min_sensor_distance)
                {
                    min_sensor_distance = distance;
                    nearest_landmark = current_landmark;
                }
            }

            // Next, we calculate weight using Normal Distribution
            long double prob = exp(-0.5 *
                                   (((nearest_landmark.x_f - transformed_x) * (nearest_landmark.x_f - transformed_x)) /
                                        (std_landmark[0] * std_landmark[0]) +
                                    ((nearest_landmark.y_f - transformed_y) * (nearest_landmark.y_f - transformed_y)) /
                                        (std_landmark[1] * std_landmark[1])));
            long double norm_const = 2 * M_PI * std_landmark[0] * std_landmark[1];
            weight *= prob / norm_const;
        }
        particles[i].weight = weight;
        weights[i] = weight;
    }
}

void ParticleFilter::resample()
{
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::vector<Particle> new_particles;
    std::default_random_engine gen;
    std::discrete_distribution<> distribution(weights.begin(), weights.end());

    for (int i = 0; i < num_particles; i++)
    {

        int number = distribution(gen);
        new_particles.push_back(particles[number]);
    }
    particles = new_particles;
}

std::string getAssociations(Particle best) {}
std::string getSenseX(Particle best) {}
std::string getSenseY(Particle best) {}

void ParticleFilter::write(std::string filename)
{
    // You don't need to modify this file.
    std::ofstream dataFile;
    dataFile.open(filename, std::ios::app);
    for (int i = 0; i < num_particles; ++i)
    {
        dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
    }
    dataFile.close();
}
