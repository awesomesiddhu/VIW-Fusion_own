#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../estimator/parameters.h"
#include "wheel_integration_base.h"
#include "../utility/sophus_utils.hpp"
#include <ceres/ceres.h>
class VanishingPointFactor : public ceres::SizedCostFunction<2, 7, 7>
{
  public:
    // Constructor now takes u, v, fx, fy, cx, cy to compute vp_ in 3D space
    VanishingPointFactor(double u, double v, double fx, double fy, double cx, double cy)
    {
        // Compute the camera space coordinates
        double x_cam = (u - cx) / fx;
        double y_cam = (v - cy) / fy;
        double z_cam = 1.0;

        // Create and normalize the direction vector in the camera frame
        Eigen::Vector3d direction(x_cam, y_cam, z_cam);
        vp_ = direction.normalized(); // Set the vanishing point direction as a normalized vector
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        // Extract the camera poses
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
        //printf("Begin qic: %f %f %f %f \n",parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Quaterniond qic(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
        //printf("End qic: %f %f %f %f \n",parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
        // Transform the vanishing point to the camera frame
        Eigen::Vector3d vp_camera = Qi.inverse() * vp_;

        // Residual: we want the vanishing point to align with the z-axis (forward direction of the camera)
        residuals[0] = vp_camera[0]; // Misalignment in x
        residuals[1] = vp_camera[1]; // Misalignment in y

        // Information matrix (optional)
        Eigen::Matrix2d sqrt_info = Eigen::Matrix2d::Identity();
        Eigen::Map<Eigen::Vector2d> residual(residuals);
        residual = sqrt_info * residual;

        // Compute Jacobians if required
        if (jacobians)
        {
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                // Derivative of residual wrt camera orientation
                jacobian_pose_i.block<2, 3>(0, O_R) = sqrt_info * Utility::skewSymmetric(Qi.inverse() * vp_).block<2, 3>(0, 0);

                // We can ignore position Jacobians since the residuals depend only on orientation
            }

            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[1]);
                jacobian_ex_pose.setZero();

                // The vanishing point constraint affects the relative pose between camera and inertial frame
                jacobian_ex_pose.block<2, 3>(0, O_R) = sqrt_info * Utility::skewSymmetric(qic.inverse() * vp_camera).block<2, 3>(0, 0);
            }
        }

        return true;
    }

  private:
    Eigen::Vector3d vp_; // Vanishing point direction in camera frame
};
