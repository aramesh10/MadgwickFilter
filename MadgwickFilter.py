# Reference paper: https://x-io.co.uk/downloads/madgwick_internal_report.pdf 
# Angles in radians unless noted otherwise

import math
import Quaternion
import numpy as np

class MadgwickFilter:
  # 
  # MadgwickFilter Object
  # Creates a MadgwickFilter object and initializes quaternions. Magnetic field will need to be updated per location 
  #
  # INPUTS: N/A
  # OUTPUTS: MadgwickFilter object
  #
  # Add parameters
  def __init__(self, param = {}):
    self.w_s = Quaternion() # Angular rate
    self.a_s = Quaternion() # Acceleration
    self.m_s = Quaternion() # Magnetic field
    
    self.g_e = Quaternion(0, 0, 0, 1)
    self.b_e = Quaternion() # FIND VALUE FOR MAGNETIC FIELD
    
    self.orientation = Quaternion()

  def update(self, wx_raw, wy_raw, wz_raw, ax_raw, ay_raw, az_raw, mx_raw, my_raw, mz_raw, alpha=1000, delta_t = .01):
    # Update Quaternion variables with raw values
    self.w_s.updateQ(0, wx_raw, wy_raw, wz_raw)
    self.a_s.updateQ(0, ax_raw, ay_raw, az_raw)
    self.m_s.updateQ(0, mx_raw, my_raw, mz_raw)

    gyro_derivative = .5 * self.orientation * self.w_s
    gyro_measurement_error_mag = 2 # UPDATE VALUE
    dir_of_error = self.calcDirOfError(alpha)

    ROC_of_orientation = gyro_derivative - gyro_measurement_error_mag * dir_of_error

    self.orientation += ROC_of_orientation * delta_t
    self.orientation.normalize()

  # 
  # calcDirOfError
  # Calculates direction of error from vector observations (gravitational field + magnetic field)
  # 
  # INPUT: delta_t = sampling period
  # OUTPUT: quaternion representing orientation calculated from vector observation
  #
  def calcDirOfError(self, alpha = 1000):
    if (alpha <= 1):
      raise ValueError(f"Alpha should be larger than 1 (Current: {alpha}")

    # Calculate objective function and its Jacobian for gravity orientation
    obj_f_grav0 = 2 * (self.orientation[1] * self.orientation[3] - self.orientation[0] * self.orientation[2]) - self.a_s[1]
    obj_f_grav1 = 2 * (self.orientation[0] * self.orientation[1] + self.orientation[2] * self.orientation[3]) - self.a_s[2]
    obj_f_grav2 = 2 * ((1/2) - self.orientation[1] * self.orientation[1] - self.orientation[2] * self.orientation[2]) - self.a_s[3]
    obj_f_grav = np.array(obj_f_grav0, obj_f_grav1, obj_f_grav2).T

    jacobian_f_grav0 = np.array([-2 * self.orientation[2], 2 * self.orientation[3], -2 * self.orientation[0], 2 * self.orientation[1]])
    jacobian_f_grav1 = np.array([2 * self.orientation[1], 2 * self.orientation[0], 2 * self.orientation[3], 2 * self.orientation[2]])
    jacobian_f_grav2 = np.array([0, -4 * self.orientation[1], -4 * self.orientation[2], 0])
    jacobian_f_grav = np.array(jacobian_f_grav0, jacobian_f_grav1, jacobian_f_grav2)

    # Calculate objective function and its Jacoobian for magnetic orientation
    obj_f_mag0 = 2 * self.b_e[1] * (0.5 - self.orientation[2] * self.orientation[2] - self.orientation[3] * self.orientation[3]) + 2 * self.b_e[3] * (self.orientation[1] * self.orientation[3] - self.orientation[0] * self.orientation[2]) - self.m_s[1]
    obj_f_mag1 = 2 * self.b_e[1] * (self.orientation[1] * self.orientation[2] - self.orientation[0] * self.orientation[3]) + 2 * self.b_e[3] * (self.orientation[0] * self.orientation[1] + self.orientation[2] * self.orientation[3]) - self.m_s[2]
    obj_f_mag2 = 2 * self.b_e[1] * (self.orientation[0] * self.orientation[2] + self.orientation[1] * self.orientation[3]) + 2 * self.b_e[3] * (0.5 - self.orientation[1] * self.orientation[1] - self.orientation[2] * self.orientation[2]) - self.m_s[3]
    obj_f_mag = np.array(obj_f_mag0, obj_f_mag1, obj_f_mag2).T

    jacobian_f_mag0 = np.array([-2 * self.b_e[3] * self.orientation[2], 2 * self.b_e[3] * self.orientation[3], -4 * self.b_e[1] * self.orientation[2] - 2 * self.b_e[3] * self.orientation[0], -4 * self.b_e[1] * self.orientation[3] + 2 * self.b_e[3] * self.orientation[1]])
    jacobian_f_mag1 = np.array([-2 * self.b_e[1] * self.orientation[3] + 2 * self.b_e[3] * self.orientation[1], 2 * self.b_e[1] * self.orientation[2] + 2 * self.b_e[3] * self.orientation[0], 2 * self.b_e[1] * self.orientation[1] - 4 * self.b_e[3] * self.orientation[3], -2 * self.b_e[1] * self.orientation[0] + 2 * self.b_e[3] * self.orientation[2]])
    jacobian_f_mag2 = np.array([2 * self.b_e[1] * self.orientation[2], 2 * self.b_e[1] * self.orientation[3] - 4 * self.b_e[3] * self.orientation[1], 2 * self.b_e[1] * self.orientation[0] - 4 * self.b_e[3] * self.orientation[2], 2 * self.b_e[1] * self.orientation[1]])
    jacobian_f_mag = np.array(jacobian_f_mag0, jacobian_f_mag1, jacobian_f_mag2)

    obj_f = np.array(obj_f_grav, obj_f_mag).T
    jacobian_f = np.array(jacobian_f_grav, jacobian_f_mag).T

    obj_f_grad = np.matmul(jacobian_f.T, obj_f)
    obj_f_grad_mag = np.linalg.norm(obj_f_grad)

    return obj_f_grad / obj_f_grad_mag

def main():
  filter = MadgwickFilter()

if __name__ == "__main__":
  main()
