import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from time import time

np.set_printoptions(suppress=True, precision=3)
# This is the mathemathical simulation.
class RobotModel:
    def __init__(self,
                 initial_pose=np.array([0.0, 0.0, 0.0]),
                 cam_position_rel_robot=np.array([0.0, 0.0, 0.5]),
                 cam_rpy_rel_robot=np.array([0.0, 0.0, 0.0]),
                 image_width=640, image_height=480, fov_deg=60,
                 a_max=0.5, alpha_max=0.5, dt=0.1, pixel_block_size=2):

        self.cur_pose = initial_pose # [x, y, theta]
        self.cur_v = 0.0
        self.cur_w = 0.0
        self.des_v = 0.0
        self.des_w = 0.0
        self.a_max = a_max
        self.alpha_max = alpha_max
        self.dt = dt

        self.cam_position_rel_robot = cam_position_rel_robot # [x, y, z]
        self.cam_rpy_rel_robot      = cam_rpy_rel_robot      # [roll, pitch, yaw]

        self.image_width = image_width
        self.image_height = image_height
        self.fov_deg = fov_deg

        self.focal_length = (self.image_width / 2) / np.tan(np.radians(self.fov_deg) / 2)

        self.K = np.array([[self.focal_length,                 0,  self.image_width / 2],
                           [0                , self.focal_length, self.image_height / 2],
                           [0                ,                 0,                     1]])

        self.pixel_block_size = pixel_block_size

        self.T_cam_original_to_cam_standard = self.get_camera_standard_transform(
            standard_x_axis='-Y_current',
            standard_y_axis='-Z_current',
            standard_z_axis='X_current'
        )

        self.T_wr = self._pose_to_transform(self.cur_pose[0], self.cur_pose[1], self.cur_pose[2])
        self.T_rc = self._create_transform(self.cam_position_rel_robot, self.cam_rpy_rel_robot)

    def _update_pose(self):
        x, y, theta = self.cur_pose
        v, w = self.cur_v, self.cur_w

        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        theta_new = theta + w * self.dt

        self.cur_pose = np.array([x_new, y_new, theta_new])
        return self.cur_pose

    def _update_speed(self):
        dv = np.clip(self.des_v - self.cur_v, -self.a_max * self.dt, self.a_max * self.dt)
        dw = np.clip(self.des_w - self.cur_w, -self.alpha_max * self.dt, self.alpha_max * self.dt)

        self.cur_v = self.cur_v + dv
        self.cur_w = self.cur_w + dw

        return self.cur_v, self.cur_w

    def _pose_to_transform(self, x, y, theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0, x],
                         [np.sin(theta),  np.cos(theta), 0, y],
                         [0,              0,             1, 0],
                         [0,              0,             0, 1]])

    def _create_transform(self, xyz, rpy):
        r = R.from_euler('xyz', rpy)
        T = np.eye(4)
        T[:3, :3] = r.as_matrix()
        T[:3,  3] = xyz
        return T

    def _transform_point(self, T, p):
        p_h = np.array([p[0], p[1], p[2], 1])
        p_t = T @ p_h
        return p_t[:3]

    def _project_point(self, point_cam_standard):
        x, y, z = point_cam_standard
        if z <= 0.0001:
            return None
        p_img_proj = self.K @ (np.array([x, y, z]) / z)
        return p_img_proj[0], p_img_proj[1]

    def get_camera_standard_transform(self, standard_x_axis, standard_y_axis, standard_z_axis):
        axis_map = {
            'X_current': np.array([1, 0, 0]),
            '-X_current': np.array([-1, 0, 0]),
            'Y_current': np.array([0, 1, 0]),
            '-Y_current': np.array([0, -1, 0]),
            'Z_current': np.array([0, 0, 1]),
            '-Z_current': np.array([0, 0, -1]),
        }

        if not all(axis in axis_map for axis in [standard_x_axis, standard_y_axis, standard_z_axis]):
            raise ValueError("Eixos de câmera inválidos. Use 'X_current', '-X_current', etc.")

        R_cam_standard_original = np.array([
            axis_map[standard_x_axis],
            axis_map[standard_y_axis],
            axis_map[standard_z_axis]
        ])

        T = np.eye(4)
        T[:3, :3] = R_cam_standard_original
        return T

    def render_image(self, objects_uv_color):
        rendered_width  = self.image_width * self.pixel_block_size
        rendered_height = self.image_height * self.pixel_block_size
        image = np.zeros((rendered_height, rendered_width, 3), dtype=np.uint8)

        for obj in objects_uv_color:
            for i in range(len(obj) - 1):
                (u1, v1), color1 = obj[i]
                (u2, v2), _      = obj[i + 1]

                pt1 = (int(round(u1 * self.pixel_block_size)), int(round(v1 * self.pixel_block_size)))
                pt2 = (int(round(u2 * self.pixel_block_size)), int(round(v2 * self.pixel_block_size)))

                cv2.line(image, pt1, pt2, color1, thickness=self.pixel_block_size)

        return image

    def step(self, des_vel = [0.0, 0.0], world_obj=None):
        self.des_v, self.des_w = des_vel

        self._update_speed()
        self._update_pose()

        self.T_wr = self._pose_to_transform(self.cur_pose[0], self.cur_pose[1], self.cur_pose[2])
        self.T_rc = self._create_transform(self.cam_position_rel_robot, self.cam_rpy_rel_robot)
        
        T_wc_original = self.T_wr @ self.T_rc
        T_original_w = np.linalg.inv(T_wc_original)

        projected_objects = []
        
        debug_info = {
            'p_rob_list': [],
            'p_cam_original_list': [],
            'p_cam_standard_list': [],
            'uv_list': []
        }

        if world_obj is not None:
            for i,obj in enumerate(world_obj):
                projected_points_data = []

                for p_world in obj:
                    T_rw = np.linalg.inv(self.T_wr)
                    p_rob = self._transform_point(T_rw, p_world)

                    p_cam_original = self._transform_point(T_original_w, p_world)

                    p_cam_standard = self._transform_point(self.T_cam_original_to_cam_standard, p_cam_original)
                    
                    uv = self._project_point(p_cam_standard)

                    dist_xy = np.linalg.norm(p_world[:2] - self.cur_pose[:2])
                    norm = int(np.clip((1-(dist_xy / 6)) * 255, 0, 255))

                    #if i == 0: color = (0, norm, norm)
                    if i == (len(world_obj)-1) : color = (norm, norm, 0)
                    else: color = (norm, norm, norm)

                    if uv is not None:
                        projected_points_data.append((uv, color))
                    
                    debug_info['p_rob_list'].append(p_rob)
                    debug_info['p_cam_original_list'].append(p_cam_original)
                    debug_info['p_cam_standard_list'].append(p_cam_standard)
                    debug_info['uv_list'].append(uv)

                projected_objects.append(projected_points_data)

        return self.cur_pose, projected_objects, debug_info

def regress_line(obj):
    uv = np.array([pt[0] for pt in obj])
    u = uv[:, 0]
    v = uv[:, 1]

    A = np.vstack([u, np.ones_like(u)]).T
    m, b = np.linalg.lstsq(A, v, rcond=None)[0]

    u_min = u.min()
    u_max = u.max()
    u1 = u_min + (u_max - u_min) / 4
    u2 = u_min + 5 * (u_max - u_min) / 6

    v1 = m * u1 + b
    v2 = m * u2 + b

    pt1 = (int(round(u1)), int(round(v1)))
    pt2 = (int(round(u2)), int(round(v2)))

    if pt1[1] > pt2[1]:
        pt1, pt2 = pt2, pt1

    return pt1, pt2

def get_border_intersection(xA, yA, theta_rad, W, H):
    # Direction vector: theta is with respect to vertical (downward)
    dx = -np.sin(theta_rad)
    dy = -np.cos(theta_rad)

    intersections = []

    # Check both directions
    for sign in [+1, -1]:
        dx_dir = dx * sign
        dy_dir = dy * sign

        # Top edge (y = 0)
        if not np.isclose(dy_dir, 0):
            t = (0 - yA) / dy_dir
            x = xA + t * dx_dir
            if 0 <= x <= W:
                intersections.append(((x, 0), abs(t)))

        # Bottom edge (y = H)
        if not np.isclose(dy_dir, 0):
            t = (H - yA) / dy_dir
            x = xA + t * dx_dir
            if 0 <= x <= W:
                intersections.append(((x, H), abs(t)))

        # Left edge (x = 0)
        if not np.isclose(dx_dir, 0):
            t = (0 - xA) / dx_dir
            y = yA + t * dy_dir
            if 0 <= y <= H:
                intersections.append(((0, y), abs(t)))

        # Right edge (x = W)
        if not np.isclose(dx_dir, 0):
            t = (W - xA) / dx_dir
            y = yA + t * dy_dir
            if 0 <= y <= H:
                intersections.append(((W, y), abs(t)))

    if not intersections:
        return None

    # Return the intersection with the smallest distance from A
    (x_int, y_int), _ = min(intersections, key=lambda tup: tup[1])
    return int(x_int), int(y_int)

def draw_arrow(image, pt_A, pt_B, theta_rad):
    xA_p, yA_p = pt_A
    xB_p, yB_p = pt_B

    bp = get_border_intersection(xB_p, yB_p, theta_rad, image.shape[1], image.shape[0])

    if bp is not None:

        final_angle = 180-np.degrees(theta_rad)
        if final_angle > 180:
            final_angle -= 360

        ep = (int(bp[0] + 60 * np.sin(theta_rad)), int(bp[1] + 60 * np.cos(theta_rad)))
        cv2.arrowedLine(image, bp, ep, (0, 0, 0), 5, tipLength=0.15)
        cv2.arrowedLine(image, bp, ep, (0, 255, 0), 3, tipLength=0.15)
        cv2.circle(image, bp, 7, (0, 255, 0), -1)
        cv2.circle(image, bp, 3, (0, 0, 0), -1)
    else:

        return -1000, -1000, -1000, image

    return bp[0], bp[1], final_angle, image

def signi(x, gamma=0.5):
    y = abs(x)**gamma * np.sign(x)
    return y

def row_Controller(point_x, theta, Y = 1, lambdax = 0.4, lambdatheta = 0.8, Vconst = 0.2, Wmax = 0.2, ro = 0.8, tz = 0.5, ty = 0.0, robust = False):
    X = point_x

    Lx = np.array([(-np.sin(ro) - Y * np.cos(ro))/tz, 0, (X * (np.sin(ro)+Y*np.cos(ro)))/tz, X*Y, -1-X**2, Y])

    Ltheta = np.array([(np.cos(ro) * np.cos(theta)**2)/tz, 
                     (np.cos(ro) * np.cos(theta) * np.sin(theta))/tz, 
                    -(np.cos(ro)*np.cos(theta) * (Y*np.sin(theta) + X * np.cos(theta)))/tz, 
                    -(Y*np.sin(theta) + X*np.cos(theta)) *np.cos(theta), 
                    -(Y*np.sin(theta) + X*np.cos(theta))*np.sin(theta), 
                    -1])

    Ls = np.vstack((Lx, Ltheta))

    Tv = np.array([ 0, -np.sin(ro),  np.cos(ro), 0, 0, 0 ]).transpose()[:, None]
    Tw = np.array([-ty, 0, 0, 0, -np.cos(ro), -np.sin(ro)]).transpose()[:, None]

    Ar = np.matmul(Ls, Tv)
    Br = np.matmul(Ls, Tw)

    Brp = np.linalg.pinv(Br)

    ex = point_x
    etheta = theta

    if robust:
        e_r = np.array([[ex], [etheta]])

        sigma_r = e_r + GAMMA_r @ e_r * dt

        zeta_r = -mu_r @ np.sign(sigma_r) * dt

        tau_r = K_r @ signi(sigma_r, gamma=0.5) + zeta_r

        v_r = LAMBDA_r @ e_r + tau_r

        w_r = - Brp @ v_r

        return w_r
    else:
        matriz_ganho_erro = np.array([lambdax * ex, lambdatheta * etheta]).transpose()[:,None]

        w = - np.matmul(Brp,(matriz_ganho_erro + Ar * Vconst))

        # Ensure that the calculated angular velocity does not exceed the maximum allowed value (Wmax)
        if(abs(w) > Wmax):
            w = Wmax * np.sign(w)

    return w

# Robust Gains -----------------------------------------
GAMMA_r  = np.array([[1.0, 0.0],
                     [0.0, 1.0]])

K_r      = np.array([[0.02, 0.00],
                     [0.00, 0.02]])

mu_r     = np.array([[0.5, 0.0],
                     [0.0, 0.5]])

LAMBDA_r = np.array([[1.0, 0.0],
                     [0.0, 2.0]])*0.5

# Proportional Gains -----------------------------------

lambdax     = 0.5
lambdatheta = 1.0

# Simulation Step --------------------------------------
dt = 0.1

# Noise seed -------------------------------------------
np.random.seed(2025)

if __name__ == "__main__":
    start = time()

    robot = RobotModel(
        initial_pose=np.array([0.5, 0.0, -0.3]),
        cam_position_rel_robot=np.array([0.0, 0.0, 0.5]),
        cam_rpy_rel_robot=np.array([0.0, 0.7, 0.0]),
        image_width=640, image_height=480, fov_deg=60,
        dt=dt)
    
    line_mode = False
    
    robot.pixel_block_size = 1
    robot.a_max = 0.1
    robot.alpha_max = 0.1

    square_y = 0.5
    square_x = 1.0
    spacing = 0.25
    rows = 4
    cols = 20
    z = 0.0

    boxes = []
    
    for row in range(rows):
        for col in range(cols):
            x0 = 1.0 + col * (square_x + spacing)
            x1 = x0 + square_x
            y0 = 0.625 - row * (square_y + spacing)
            y1 = y0 - square_y
            box = [
                [x0, y0, z],
                [x1, y0, z],
                [x1, y1, z],
                [x0, y1, z],
                [x0, y0, z]
            ]
            boxes.append(box)

    world_obj = boxes + [[[0,0,0], [0,0,0]]]

    des_v = 0.0
    des_w = 0.0

    finish = time()
    while True:
        
        pose = robot.cur_pose
        A = np.array([0.2 + pose[0]*np.cos(0), -0.375 + pose[1]*np.sin(0), 0])
        B = np.array([1.7 + pose[0]*np.cos(0), -0.375 + pose[1]*np.sin(0), 0])

        line2 = [A, B]
        world_obj[-1] = line2

        current_pose, projected_objects, debug_info = robot.step(des_vel=[des_v, des_w], world_obj=world_obj)
        rendered_img = robot.render_image(projected_objects)
        image_bgr = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)

        pt1, pt2 = regress_line(projected_objects[-1])
        du = pt1[0] - pt2[0]
        dv = pt1[1] - pt2[1]
        angle_rad = np.arctan2(du, dv)

        #cv2.line(image_bgr, pt1, pt2, (255, 0, 255), 3)
        #cv2.circle(image_bgr, pt1, 5, (255, 0, 0), -1)  # Start point in blue
        #cv2.circle(image_bgr, pt2, 5, (0, 0, 255), -1)  # end point in red

        X, Y, T, image_bgr = draw_arrow(image_bgr, pt1, pt2, angle_rad)

        finish = time()

        elapsed_time = finish - start

        start = time()
        cv2.putText(image_bgr, f"Loop Time: {elapsed_time:.3f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Rendered Image - Proportional', image_bgr)
        key = cv2.waitKey(int(1/robot.dt)*2)

        if key == ord('w'): des_v += 0.05
        elif key == ord('s'): des_v -= 0.05
        elif key == ord('a'): des_w += 0.05
        elif key == ord('d'): des_w -= 0.05
        elif key == ord('r'): des_v = des_w = 0.0; robot.cur_pose = np.array([1.0, 0.0, 0.0])
        elif key == ord('i'): line_mode = True
        elif key == ord('o'): line_mode = False; des_v = des_w = 0.0

        if key == ord('q'): break

        if line_mode:
            des_v = 0.1

            noise = 0# np.random.normal(loc=0.0, scale=des_v/3)
            if X != -1000: w = np.squeeze(row_Controller((X/(robot.image_width/2))-1, np.deg2rad(T), Y=1, lambdax=lambdax, lambdatheta=lambdatheta, Vconst=des_v, Wmax=0.5, ro=0.7, tz=0.5, ty=0.0, robust=True)) + noise
            else: line_mode = False; des_v = 0; w = 0.0
            des_w = w
        
        #print(f"Control Input: ({(X/(robot.image_width/2))-1:.3f}, {np.deg2rad(T):.3f}), Control Output: {des_w:.3f}")

    cv2.destroyAllWindows()