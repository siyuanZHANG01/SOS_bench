from __future__ import annotations
import numpy as np

class OptimalPiecewiseLinearModel:
    def __init__(self, epsilon):
        self.epsilon = int(epsilon)  # Ensure epsilon is integer
        self.lower = []  # Corresponds to C++ lower vector
        self.upper = []  # Corresponds to C++ upper vector
        self.points = [] # For logic consistency
        self.rect = [None] * 4 # Corresponds to C++ rectangle[4]
        self.lower_start = 0
        self.upper_start = 0
        self.points_in_hull = 0
        self.first_x = 0
        self.last_x = 0

    def reset(self):
        self.points_in_hull = 0
        self.lower.clear()
        self.upper.clear()
        self.points.clear()
        self.lower_start = 0
        self.upper_start = 0
        self.rect = [None] * 4

    def cross(self, O, A, B):
        OA_x = A[0] - O[0]
        OA_y = A[1] - O[1]
        OB_x = B[0] - O[0]
        OB_y = B[1] - O[1]
        return OA_x * OB_y - OA_y * OB_x

    def add_point(self, x, y):
        # Corresponds to C++: if (points_in_hull > 0 && x <= last_x) check
        if self.points_in_hull > 0 and x < self.last_x:
             pass 

        self.last_x = x
        
        # Corresponds to C++: Point p1{x, y + epsilon}, p2{x, y - epsilon}
        p1 = (x, y + self.epsilon) # Upper bound point
        p2 = (x, y - self.epsilon) # Lower bound point

        if self.points_in_hull == 0:
            self.first_x = x
            self.rect[0] = p1
            self.rect[1] = p2
            self.upper.append(p1)
            self.lower.append(p2)
            self.upper_start = 0
            self.lower_start = 0
            self.points_in_hull += 1
            return True

        if self.points_in_hull == 1:
            self.rect[2] = p2
            self.rect[3] = p1
            self.upper.append(p1)
            self.lower.append(p2)
            self.points_in_hull += 1
            return True

        # Slope comparison helper functions
        def slope_lt(A, B, C, D): 
            # Check if slope(B-A) < slope(D-C)
            dy1, dx1 = B[1]-A[1], B[0]-A[0]
            dy2, dx2 = D[1]-C[1], D[0]-C[0]
            return dy1 * dx2 < dy2 * dx1

        def slope_gt(A, B, C, D):
            dy1, dx1 = B[1]-A[1], B[0]-A[0]
            dy2, dx2 = D[1]-C[1], D[0]-C[0]
            return dy1 * dx2 > dy2 * dx1

        # C++: bool outside_line1 = p1 - rectangle[2] < slope1;
        outside_line1 = slope_lt(self.rect[2], p1, self.rect[0], self.rect[2])

        # C++: bool outside_line2 = p2 - rectangle[3] > slope2;
        outside_line2 = slope_gt(self.rect[3], p2, self.rect[1], self.rect[3])

        if outside_line1 or outside_line2:
            self.points_in_hull = 0 # Seg fail
            return False

        # C++: if (p1 - rectangle[1] < slope2)
        if slope_lt(self.rect[1], p1, self.rect[1], self.rect[3]):
            min_slope_pt = self.lower[self.lower_start]
            min_i = self.lower_start
            
            for i in range(self.lower_start + 1, len(self.lower)):
                if slope_gt(p1, self.lower[i], p1, min_slope_pt):
                    break
                min_slope_pt = self.lower[i]
                min_i = i
            
            self.rect[1] = self.lower[min_i]
            self.rect[3] = p1
            self.lower_start = min_i

            # Hull update (Upper)
            while len(self.upper) >= self.upper_start + 2:
                if self.cross(self.upper[-2], self.upper[-1], p1) <= 0:
                    self.upper.pop()
                else:
                    break
            self.upper.append(p1)

        # C++: if (p2 - rectangle[0] > slope1)
        if slope_gt(self.rect[0], p2, self.rect[0], self.rect[2]):
            max_slope_pt = self.upper[self.upper_start]
            max_i = self.upper_start
            
            for i in range(self.upper_start + 1, len(self.upper)):
                if slope_lt(p2, self.upper[i], p2, max_slope_pt):
                    break
                max_slope_pt = self.upper[i]
                max_i = i

            self.rect[0] = self.upper[max_i]
            self.rect[2] = p2
            self.upper_start = max_i

            # Hull update (Lower)
            while len(self.lower) >= self.lower_start + 2:
                if self.cross(self.lower[-2], self.lower[-1], p2) >= 0:
                    self.lower.pop()
                else:
                    break
            self.lower.append(p2)

        self.points_in_hull += 1
        return True

def pgm_segments_and_starts(keys: np.ndarray, eps: int) -> tuple[int, list]:
    n = keys.size
    if n == 0:
        return 0, []
    if n == 1:
        return 1, [keys[0]]

    # Initialize convex hull model
    model = OptimalPiecewiseLinearModel(eps)
    
    segs = 1
    starts = [keys[0]]
    
    model.reset()
    
    for i in range(n):
        k = keys[i]
        # Note: y passed here is original rank i (0, 1, 2...)
        if not model.add_point(k, i):
            segs += 1
            starts.append(k)
            model.reset()
            # Reset and add current point (becomes start of new segment)
            model.add_point(k, i)
            
    return segs, starts
