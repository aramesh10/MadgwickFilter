# Reference paper: https://x-io.co.uk/downloads/madgwick_internal_report.pdf 
# Angles in radians unless noted otherwise

import math
import numpy as np

class Quaternion:
# **************************************************
# Quaternion object
# q[0] + q[1]*i + q[2]*j + q[3]*k
# Settings:
#   default:
#     Inputs: q0, q1, q2, q3
#     Output: Quaternion object w/
#             q[0] = q0
#             q[1] = q1
#             q[2] = q2
#             q[3] = q3
#   vector:
#     Inputs: rx, ry, rz, theta (radians)
#     Output: Quaternion object w/
#             q[0] = cos(theta)
#             q[1] = -rx * sin(theta / 2)
#             q[2] = -ry * sin(theta / 2)
#             q[3] = -rz * sin(theta / 2)
# **************************************************
  def __init__(self, q0 = 0, q1 = 0, q2 = 0, q3 = 0, setting = "default"):
    self.q = np.zeros(4) # [q0, q1, q2, q3]
    if setting == "default":
      self.q[0] = q0
      self.q[1] = q1
      self.q[2] = q2
      self.q[3] = q3
    elif setting == "vector": # q0 = rx, q1 = ry, q2 = rz, q3 = theta
      self.q[0] = math.cos(q3 / 2)
      self.q[1] = -q0 * math.sin(q3 / 2)
      self.q[2] = -q1 * math.sin(q3 / 2)
      self.q[3] = -q2 * math.sin(q3 / 2)
    else:
      raise ValueError("Incorrect setting used. Currently using {} (acceptable: default | vector)".format(setting))

  # returns string of numpy array q
  def __str__(self):
    q_str = "{}".format(self.q)
    return q_str

  # mult operator overload
  def __mul__(self, other):
    # Quaternion product
    if isinstance(other, Quaternion):
      q0 = self.q[0] * other.q[0] - self.q[1] * other.q[1] - self.q[2] * other.q[2] - self.q[3] * other.q[3]
      q1 = self.q[0] * other.q[1] + self.q[1] * other.q[0] + self.q[2] * other.q[3] - self.q[3] * other.q[2]
      q2 = self.q[0] * other.q[2] - self.q[1] * other.q[3] + self.q[2] * other.q[0] + self.q[3] * other.q[1]
      q3 = self.q[0] * other.q[3] + self.q[1] * other.q[2] - self.q[2] * other.q[1] + self.q[3] * other.q[0]
      return Quaternion(q0, q1, q2, q3)

    # 3D-Vector product
    elif isinstance(other, np.ndarray):
      if (other.shape != (3,)):
        raise ValueError("Invalid shape of vector (current shape: {})".format(other.shape))
      vector = Quaternion(0, other[0], other[1], other[2])
      q0 = self.q[0] * vector.q[0] - self.q[1] * vector.q[1] - self.q[2] * vector.q[2] - self.q[3] * vector.q[3]
      q1 = self.q[0] * vector.q[1] + self.q[1] * vector.q[0] + self.q[2] * vector.q[3] - self.q[3] * vector.q[2]
      q2 = self.q[0] * vector.q[2] - self.q[1] * vector.q[3] + self.q[2] * vector.q[0] + self.q[3] * vector.q[1]
      q3 = self.q[0] * vector.q[3] + self.q[1] * vector.q[2] - self.q[2] * vector.q[1] + self.q[3] * vector.q[0]
      return Quaternion(q0, q1, q2, q3)

    # Scalar and Quaternion product
    elif isinstance(other, (int, float)):
      q0 = self.q[0] * other
      q1 = self.q[1] * other
      q2 = self.q[2] * other
      q3 = self.q[3] * other
      return Quaternion(q0, q1, q2, q3)
    
    raise TypeError("Can not multiply quaternion with type {}".format(type(other)))

  def __rmul__(self, other):
    # Quaternion product
    if isinstance(other, Quaternion):
      q0 = other.q[0] * self.q[0] - other.q[1] * self.q[1] - other.q[2] * self.q[2] - other.q[3] * self.q[3]
      q1 = other.q[0] * self.q[1] + other.q[1] * self.q[0] + other.q[2] * self.q[3] - other.q[3] * self.q[2]
      q2 = other.q[0] * self.q[2] - other.q[1] * self.q[3] + other.q[2] * self.q[0] + other.q[3] * self.q[1]
      q3 = other.q[0] * self.q[3] + other.q[1] * self.q[2] - other.q[2] * self.q[1] + other.q[3] * self.q[0]
      return Quaternion(q0, q1, q2, q3)

    # 3D-Vector product
    elif isinstance(other, np.ndarray):
      if (other.shape != (3,)):
        raise ValueError("Invalid shape of vector (current shape: {})".format(other.shape))
      vector = Quaternion(0, other[0], other[1], other[2])
      q0 = vector.q[0] * self.q[0] - vector.q[1] * self.q[1] - vector.q[2] * self.q[2] - vector.q[3] * self.q[3]
      q1 = vector.q[0] * self.q[1] + vector.q[1] * self.q[0] + vector.q[2] * self.q[3] - vector.q[3] * self.q[2]
      q2 = vector.q[0] * self.q[2] - vector.q[1] * self.q[3] + vector.q[2] * self.q[0] + vector.q[3] * self.q[1]
      q3 = vector.q[0] * self.q[3] + vector.q[1] * self.q[2] - vector.q[2] * self.q[1] + vector.q[3] * self.q[0]
      return Quaternion(q0, q1, q2, q3)

    # Scalar and Quaternion product
    elif isinstance(other, (int, float)):
      q0 = self.q[0] * other
      q1 = self.q[1] * other
      q2 = self.q[2] * other
      q3 = self.q[3] * other
      return Quaternion(q0, q1, q2, q3)
    
    raise TypeError("Can not multiply quaternion with type {}".format(type(other)))

  # Update Quaternion
  def updateQ(self, q0 = 0, q1 = 0, q2 = 0, q3 = 0, setting = "default"):
    if setting == "default":
      self.q[0] = q0
      self.q[1] = q1
      self.q[2] = q2
      self.q[3] = q3
    elif setting == "normalized_vector": # q0 = rx, q1 = ry, q2 = rz, q3 = theta
      self.q[0] = math.cos(q3 / 2)
      self.q[1] = -q0 * math.sin(q3 / 2)
      self.q[2] = -q1 * math.sin(q3 / 2)
      self.q[3] = -q2 * math.sin(q3 / 2)
    else:
      raise ValueError("Incorrect setting used. Currently using {} (acceptable: default | normalized_vector)".format(setting))

  # Normalizes Quaternion
  def normalize(self):
    self.q /= self.getMagnitude()

  # Returns normalized quaternion while preserving original Quaternion
  def getNormalize(self):
    normal_q = self.q / self.getMagnitude()
    return Quaternion(normal_q[0], normal_q[1], normal_q[2], normal_q[3])

  # Returns magnitude of Quaternion vector
  def getMagnitude(self):
    mag = math.sqrt(self.q[0] * self.q[0] + self.q[1] * self.q[1] + self.q[2] * self.q[2] + self.q[3] * self.q[3])
    return mag

  # Calculate and return Euler Angles psi, theta, phi
  def getEulerAngles(self):
    normalized_self = self.getNormalize()
    psi = math.atan2(2 * normalized_self.q[1] * normalized_self.q[2] - 2 * normalized_self.q[0] * normalized_self.q[3], 2 * normalized_self.q[0] * normalized_self.q[0] + 2 * normalized_self.q[1] * normalized_self.q[1] - 1)
    theta = -math.asin(2 * normalized_self.q[1] * normalized_self.q[3] + 2 * normalized_self.q[0] * normalized_self.q[2])
    phi = math.atan2(2 * normalized_self.q[2] * normalized_self.q[3] - 2 * normalized_self.q[0] * normalized_self.q[1], 2 * normalized_self.q[0] * normalized_self.q[0] + 2 * normalized_self.q[3] * normalized_self.q[3] - 1)
    return psi, theta, phi
  
  # Calculate and return rotation matrix
  def getRotationMatrix(self):
    normalized_self = self.getNormalize()
    rot = np.zeros((3,3))
    rot[0] = np.array([2 * normalized_self.q[0] * normalized_self.q[0] - 1 + 2 * normalized_self.q[1] * normalized_self.q[1], 2 * (normalized_self.q[1] * normalized_self.q[2] + normalized_self.q[0] * normalized_self.q[3]), 2 * (normalized_self.q[1] * normalized_self.q[3] - normalized_self.q[0] * normalized_self.q[2])])
    rot[1] = np.array([2 * (normalized_self.q[1] * normalized_self.q[2] - normalized_self.q[0] * normalized_self.q[3]), 2 * normalized_self.q[0] * normalized_self.q[0] - 1 + 2 * normalized_self.q[2] * normalized_self.q[2], 2 * (normalized_self.q[2] * normalized_self.q[3] + normalized_self.q[0] * normalized_self.q[1])])
    rot[2] = np.array([2 * (normalized_self.q[1] * normalized_self.q[3] + normalized_self.q[0] * normalized_self.q[2]), 2 * (normalized_self.q[2] * normalized_self.q[3] - normalized_self.q[0] * normalized_self.q[1]), 2 * normalized_self.q[0] * normalized_self.q[0] - 1 + 2 * normalized_self.q[3] * normalized_self.q[3]])
    return rot.T
  
  # Returns conjugate of Quaternion
  def getConjugate(self):
    q_conj0 = self.q[0]
    q_conj1 = -self.q[1]
    q_conj2 = -self.q[2]
    q_conj3 = -self.q[3]
    return Quaternion(q_conj0, q_conj1, q_conj2, q_conj3)

  # Rotates vector based on conjugate
  def rotateVector(self, vector):
    if not isinstance(vector, np.ndarray):
      raise TypeError("Vector not type numpy.ndarray")
    if vector.shape != (3,):
      raise ValueError("Invalid shape of vector (current shape: {})".format(vector.shape))
      
    normalized_self = self.getNormalize()
    quant_rotVector = normalized_self * vector * normalized_self.getConjugate()
    rotVector = np.array(quant_rotVector.q[1:4])
    return rotVector

# demo main function
if __name__ == "__main__":
  print("*" * 50)
  
  vec = np.array([1, 2, 3])
  
  quant1 = Quaternion(3,1,-2,1)
  quant2 = Quaternion(2,-1,2,3)
  quant3 = quant1 * quant2
  
  print("Q1 = {}".format(quant1))
  print("Q2 = {}".format(quant2))
  print("Q3 = Q1 * Q2 = {}".format(quant3))

  print("*" * 50)

  psi, theta, phi = quant3.getEulerAngles()
  print("Q3 Angles:")
  print("psi = {}".format(psi))
  print("theta = {}".format(theta))
  print("phi = {}".format(phi))

  print("*" * 50)

  rotationMatrix = quant3.getRotationMatrix()
  print("Q3 Rotation Matrix:")
  print(rotationMatrix)
  
  print("*" * 50)
  
  rotVec = quant3.rotateVector(vec)
  rotVec_Matrix = np.matmul(rotationMatrix, vec)
  print("Using Q3 to rotate vector: ")
  print("Vector = {}".format(vec))
  print("Rotated Vector = {}".format(rotVec))
  print("Rotated Vector (using Rotation Matrix) = {}".format(rotVec_Matrix))
  
  print("*" * 50)