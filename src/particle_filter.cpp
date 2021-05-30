
#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;
using std::string;
using std::vector;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  if(is_initialized)
  {
    return;
  }
  num_particles = 100;  // TODO: Set the number of particles
  //Set standard deviations for x, y, and theta
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  // This line creates a normal (Gaussian) distribution for x
  normal_distribution<double> dist_x(x, std_x);

  // Create normal distributions for y and theta
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);
  
  for (int i=0; i<num_particles; i++)
  {
    Particle particle;
    
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;

    particles.push_back(particle);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  //Set standard deviations for x, y, and theta
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  // Normal distribution for sensor noise
  normal_distribution<double> dist_x(0, std_x);
  normal_distribution<double> dist_y(0, std_y);
  normal_distribution<double> dist_theta(0, std_theta);
  
  for (int i = 0; i < num_particles; i++)
  {
    if (fabs(yaw_rate) < 0.00001) // When yaw is not changing. ie; yaw rate = 0
    {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else // When yaw is not zero
    {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
  
  //Adding noise  
  particles[i].x += dist_x(gen);
  particles[i].y += dist_y(gen);
  particles[i].theta += dist_theta(gen);
}

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (unsigned int i=0; i< observations.size(); i++)
  {
    //initialize minimum distance to max number.
    double minimum_distance = numeric_limits<double>::max();
    
    //initialize nearest map landmark id
    int map_id = -1;
    for (unsigned int j = 0; j < predicted.size(); j++)
    {
      
      double current_distance = dist(observations[i].x,observations[i].y, predicted[j].x, predicted[j].y);
      // find the predicted landmark nearest the current observed landmark
      if ( current_distance < minimum_distance ) {
        minimum_distance = current_distance;
        map_id = predicted[j].id;
      }
    }
    // set the observation's id to the nearest predicted landmark's id
     observations[i].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
   for (int i = 0; i < num_particles; i++) {

    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;
     
     //only consider landmarks within sensor range of the particle 
     vector<LandmarkObs> predicted_landmark;
     for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++)
     {
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;
       
      double landmark_dist = dist(x, y, landmark_x, landmark_y);
       
        
      //only consider landmarks within sensor range of the particle 
      if (landmark_dist < sensor_range) {
        // adding prediction to vector
        predicted_landmark.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
      }
     }
     
     // Transform observations from vehicle co-ordinates to map co-ordinates
     vector<LandmarkObs> transformed_observations;
     for (unsigned int j=0; j<observations.size(); j++)
     {
       double x_t = observations[j].x * cos(theta) - observations[j].y * sin(theta) + x;
       double y_t = observations[j].x * sin(theta) + observations[j].y * cos(theta) + y;
       transformed_observations.push_back(LandmarkObs{ observations[j].id, x_t, y_t });
     }
     
     //Associate observations to predicted landmarks
     dataAssociation(predicted_landmark, transformed_observations);
 
     //reinitialize weights
     particles[i].weight = 1.0;
     for (unsigned int j = 0; j < transformed_observations.size(); j++) 
     {
       double obs_x = transformed_observations[j].x;
       double obs_y = transformed_observations[j].y;
       
       int landmark_id = transformed_observations[j].id;
       double pred_x, pred_y;
       //Calculate the weight of particle based on the multivariate Gaussian probability function
       for (unsigned int k = 0; k < predicted_landmark.size(); k++) 
       {
        if (predicted_landmark[k].id == landmark_id) 
        {
          pred_x = predicted_landmark[k].x;
          pred_y = predicted_landmark[k].y;
        }
      }
       
       double s_x = std_landmark[0];
       double s_y = std_landmark[1];
       double obs_w = ( 1/(2*M_PI*s_x*s_y)) * exp( -( pow(pred_x-obs_x,2)/(2*pow(s_x, 2)) + (pow(pred_y-obs_y,2)/(2*pow(s_y, 2))) ));
         
          //total observation weight
       particles[i].weight *= obs_w;

       
     }
   }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // Get weights
  vector<double> weights;
  double max_weight = numeric_limits<double>::min();
  
   vector<Particle> resampled_particles;
  
  
  for(int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
    if ( particles[i].weight > max_weight ) {
      max_weight = particles[i].weight;
    }
  }
  
   //Generate random particle index
   uniform_int_distribution<int> particle_index(0, num_particles - 1);
  
   int current_index = particle_index(gen);
   double beta = 0.0;
   
  
   for (int i = 0; i < num_particles; i++)
   {
     uniform_real_distribution<double> random_weight(0.0, max_weight);
     beta += random_weight(gen);
     
     while (beta > weights[current_index])
     {
       beta -= weights[current_index];
       current_index = (current_index + 1) % num_particles;
     }
     
     resampled_particles.push_back(particles[current_index]);
   }
  particles = resampled_particles;
  
}
Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
// void ParticleFilter::SetAssociations(Particle& particle, 
//                                      const vector<int>& associations, 
//                                      const vector<double>& sense_x, 
//                                      const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
