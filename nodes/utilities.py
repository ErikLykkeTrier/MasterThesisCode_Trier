#!/usr/bin/env python3  

import numpy as np
from math import cos, sin
import matplotlib.pyplot as plt
import os

import os

#GPU id to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, json, cv2

from pathlib import Path

# Useful notation #1
def signi(vector, gamma):
    return np.sign(vector) * np.abs(vector) ** gamma
def norm(x, max):
    return 2 * ((x) / (max)) - 1

def shift(x,max):
    if x > max:
        x = np.sign(x) * max
    elif x < 0:
        x = 0
    return (2*x - max)/2

def lim_w(w, max):
    if np.abs(w) > max:
        return np.sign(w)*max
    else:
        return w
    
def T(t_cr, ro):
            # Compute skew-symmetric matrix Q(t_cr)
        Q_tcr = np.array([
            [0, t_cr[0,0], t_cr[1,0]],
        [-t_cr[0,0], 0, t_cr[2,0]],
        [-t_cr[1,0], -t_cr[2,0], 0]
        ])
        # Rotation matrix R_cr = Rx(ψ) * Ry(π/2) * Rx(−π/2)

        Rx_psi = np.array([
            [1, 0, 0],
            [0, np.cos(ro), -np.sin(ro)],
            [0, np.sin(ro), np.cos(ro)]
        ])
        Ry_90 = np.array([
            [np.cos(np.pi/2), 0, np.sin(np.pi/2)],
            [0, 1, 0],
            [-np.sin(np.pi/2), 0, np.cos(np.pi/2)]
        ])
        Rx_minus_90 = np.array([
            [1, 0, 0],
            [0, np.cos(-np.pi/2), -np.sin(-np.pi/2)],
            [0, np.sin(-np.pi/2), np.cos(-np.pi/2)]
        ])


        R_cr = Rx_psi @ Ry_90 @ Rx_minus_90  # Shape: (3, 3)

        top = np.hstack((R_cr, Q_tcr @ R_cr))
        bottom = np.hstack((np.zeros((3, 3)), R_cr))
        A_cr = np.vstack((top, bottom))  # Shape: (6, 6)

        S = np.array([[1,0,0,0,0,0],[0,0,0,0,0,1]]).transpose()

        T_cr = A_cr @ S  # (6, 2)
        return T_cr

class BoundedList:
    def __init__(self, capacity, name="list"):
        self.capacity = capacity
        self.data = []
        self.name = name

    def append(self, value):
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(value)

    def __repr__(self):
        return repr(self.data)

    def sum(self):
        if not self.data:
            return 0

        first = self.data[0]

        if isinstance(first, np.ndarray):
            try:
                stacked = np.stack(self.data, axis=0)
                return np.sum(stacked, axis=0)
            except Exception as e:
                raise TypeError(f"Feil under summering av NumPy-arrays: {e}")

        return self._recursive_sum(self.data)

    def _recursive_sum(self, elements):
        if isinstance(elements[0], (int, float, np.integer, np.floating)):
            return sum(elements)
        return [self._recursive_sum([el[i] for el in elements]) for i in range(len(elements[0]))]

    def plotting(self):
        if not self.data:
            print("Listen er tom – ingenting å plotte.")
            return

        try:
            arr = np.stack(self.data, axis=0)
        except Exception as e:
            print(f"Kan ikke stacke data: {e}")
            return

        # Fjern singleton-dimensjon hvis shape er (N, D, 1)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]

        # Hvis array er 1D (shape (N,)), gjør det til 2D for å kunne iterere over "kolonner"
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]

        rows = arr.shape[1]
        os.makedirs("./plots", exist_ok=True)

        for i in range(rows):
            values = arr[:, i]
            plt.figure()
            plt.plot(values)
            plt.title(f"{self.name} – posisjon {i}")
            plt.xlabel("Tid / indeks")
            plt.ylabel(f"Verdi på posisjon {i}")
            filename = f"./Shifted_Plots/{self.name}_p{i}.png"
            plt.savefig(filename)
            plt.close()
            print(f"Lagret: {filename}")

    
class RobustController:
    def __init__(self, lim, vd,dt, robust):
        """
            Initialize everything that is to be static is defined as part of the instance of the controller.
        """
        # Might change to variable desired velocity over time?
        self.v_d = vd
        self.dt = dt
        self.robust = robust
        # Camera pos, z offset, x offset, pitch
        # Ground truth :D 
        self.t_z = 1.2
        self.t_y = 0.045
        self.t_x = -0.27 
        self.ro = 0.8

        # "Uncertain cam pose"
        # self.t_z = 0.5
        # self.t_y = 0
        # self.t_x = 0
        # self.ro = 0.2

        self.t_cr = np.array([[0], [self.t_z], [-self.t_x]])
        
        self.y_star = 1 # Bootom of image
        self.x_star = 0 # Center of image
        self.theta_star = 0 # Straight

        # Needed for controller
        self.e_c_list = BoundedList(lim)
        self.e_r_list = BoundedList(lim)
        self.zeta_c_list = BoundedList(lim)
        self.zeta_r_list = BoundedList(lim)
        
        # Lists to store robot pose and image feature poses
        self.robot_pos_list = []
        self.image_feature_pos_list = []
        self.control_ouput_list = []

        # Gains

        self.Gamma_r = np.diag([0.002,0.002])
        # self.Gamma_r = np.diag([2.5, 5])
        self.mu_r = np.diag([0.002, 0.002])

        if self.robust:
            # self.Lambda_r = np.array([[0.4, 0], [0,1.2]]) # Robust
            self.Lambda_r = np.diag([2.5, 5]) #Paper
            self.Lambda_c = np.diag([3, 6])
            # self.Lambda_r = np.diag([1.5, 1.5])
            # self.Lambda_c = np.diag([0.75, 1.5]) 

        else:
            self.Lambda_r = np.diag([2.8, 5.5])
            self.Lambda_c = np.diag([3, 6]) 
            
            # self.Lambda_r = np.diag([0.5, 1])
            # self.Lambda_c = np.diag([0.75, 1.5]) 
        # self.K_r = np.array([[0.005, 0], [0, 0.01]])
        
        self.K_r = np.array([[0.02, 0], [0, 0.02]])
        # self.K_r = np.array([[0.2, 0], [0, 0.2]])

        self.Gamma_c = np.diag([0.2,0.2])

        self.K_c = np.diag([0.02, 0.02])

        self.mu_c = np.diag([0.2,0.2])

        # Gain scalar, between 0 and 1
        self.gamma = 0.5
        T_cr = T(self.t_cr, self.ro)
        self.T_v = T_cr[:,0]
        self.T_w = T_cr[:,1]

        self.omega_prime = 0
    
    def restart_controller(self, which):
        """
            Restarts the error lists as not to keep the previous errors.

            low priotity

            args:
            which(string): Wether to reset column or row controller.
        """
        if which == 'row':
            self.e_r_list.reset()
            self.zeta_r_list.reset()
        elif which == 'col':
            self.e_c_list.reset()
            self.zeta_c_list.reset()
        else:
            print('This is not an option: ', which)
            
    def step_row(self, x, y, theta):
        """
            x,y,theta is not normalized or shifted.
        """

        # Tried with normalized x and y, but it did not work
        # x_norm = norm(x, 640)
        # y_norm = norm(y, 480)
        # x_norm = x
        # y_norm = y
        # Lx and Ltheta is from Cherubini paper, Eq. 2
        # L_x = np.array([(-sin(self.ro) - y_norm * cos(self.ro))/self.t_z, 0, (x_norm * (sin(self.ro)+y_norm*cos(self.ro)))/self.t_z, x_norm*y_norm, -1-x_norm**2, y_norm])
        # # display(L_x)
        # L_theta = np.array([(cos(self.ro) * cos(theta)**2)/self.t_z, 
        #     (cos(self.ro) * cos(theta) * sin(theta))/self.t_z, 
        #     -(cos(self.ro)*cos(theta) * (y_norm*sin(theta) + x_norm * cos(theta)))/self.t_z, 
        #     -(y_norm*sin(theta) + x_norm*cos(theta)) *cos(theta), 
        #     -(y_norm*sin(theta) + x_norm*cos(theta))*sin(theta), 
        #     -1])


        L_x = np.array([(-sin(self.ro) - y * cos(self.ro))/self.t_z, 0, (x * (sin(self.ro)+y*cos(self.ro)))/self.t_z, x*y, -1-x**2, y])
        L_theta = np.array([(cos(self.ro) * cos(theta)**2)/self.t_z, 
            (cos(self.ro) * cos(theta) * sin(theta))/self.t_z, 
            -(cos(self.ro)*cos(theta) * (y*sin(theta) + x * cos(theta)))/self.t_z, 
            -(y*sin(theta) + x*cos(theta)) *cos(theta), 
            -(y*sin(theta) + x*cos(theta))*sin(theta), 
            -1])
        
        J_wr = [[L_x], [L_theta]] @ self.T_w 
        
        J_wrp = np.linalg.pinv(J_wr)
        

        e_x = x - self.x_star
        e_theta = theta - self.theta_star
        e_r = np.array([[e_x], [e_theta]]) # 2x1

        self.e_r_list.append(e_r*self.dt) # 2x1 in each element
        J_vr = [[L_x], [L_theta]] @ self.T_v 
        if self.robust:
            
            sigma_r = e_r + self.Gamma_r @ self.e_r_list.sum()
            zeta_dot_r = -self.mu_r @ np.sign(sigma_r) 
            try:
                zeta_r = self.zeta_r_list.data[-1] + zeta_dot_r*self.dt
            except:
                zeta_r=0

            self.zeta_r_list.append(zeta_r) 
            tau_r = self.K_r @ signi(sigma_r, self.gamma) + self.zeta_r_list.sum() # Making the error smaller by the signi function(since gamma is 1 it does not) adds the error in the other direction
            v_r = self.Lambda_r @ e_r + tau_r 
            w_r = J_wrp @ v_r
            w_r = lim_w(w_r.item(), np.pi/8)
            # print error, v_r and w_r
            print(f"sigma={sigma_r}, e_x={e_r[0].item()}, e_theta={e_r[1].item()},\n\n w_r={w_r}")
        else:
            # print(f"e_r={e_r}")
            # if abs(e_r[0]) < 0.3 and abs(e_r[1]) < 0.3:
            #     Lambda_r = self.Lambda_r @ np.array([[0.6, 0], [0, 0.6]])
            #     w_r = (-J_wrp @ (Lambda_r @ e_r + J_vr * self.v_d))
            # else:
            #     w_r = (-J_wrp @ (self.Lambda_r @ e_r + J_vr * self.v_d))
            w_r = (-J_wrp @ (self.Lambda_r @ e_r + J_vr * self.v_d))
            w_r = lim_w(w_r.item(), np.pi/8)
            w_r = -w_r
            # if np.abs(w_r) > np.pi/3:
            #     w_r = np.sign(w_r) * np.pi/3
            # Print error and w_r 
            print(f"e_x={e_r[0].item()}, e_theta={e_r[1].item()}, w_r={-w_r}")
        
        self.image_feature_pos_list.append([x, y, theta])
        self.control_ouput_list.append(self.v_d, w_r)
        return  w_r
        
    
    def row_Controller(self, point_x, point_y, theta, Wmax=0.2):
        x = point_x
        y = point_y
        
        Lx = np.array([(-np.sin(self.ro) - y * np.cos(self.ro))/self.t_z, 0, (x * (np.sin(self.ro)+y*np.cos(self.ro)))/self.t_z, x*y, -1-x**2, y])

        Ltheta = np.array([(np.cos(self.ro) * np.cos(theta)**2)/self.t_z, 
                        (np.cos(self.ro) * np.cos(theta) * np.sin(theta))/self.t_z, 
                        -(np.cos(self.ro)*np.cos(theta) * (y*np.sin(theta) + x * np.cos(theta)))/self.t_z, 
                        -(y*np.sin(theta) + x*np.cos(theta)) *np.cos(theta), 
                        -(y*np.sin(theta) + x*np.cos(theta))*np.sin(theta), 
                        -1])

        Ls = np.vstack((Lx, Ltheta))

        Tv = np.array([ 0, -np.sin(self.ro),  np.cos(self.ro), 0, 0, 0 ]).transpose()[:, None]
        Tw = np.array([-self.t_y, 0, 0, 0, -np.cos(self.ro), -np.sin(self.ro)]).transpose()[:, None]

        Ar = np.matmul(Ls, Tv)
        Br = np.matmul(Ls, Tw)

        Brp = np.linalg.pinv(Br)

        ex = point_x
        etheta = theta
        # print(f"ex={ex}, etheta={etheta}")

        if self.robust:
            e_r = np.array([[ex], [etheta]])

            sigma_r = e_r + self.Gamma_r @ e_r * self.dt

            zeta_r = -self.mu_r @ np.sign(sigma_r) * self.dt

            tau_r = self.K_r @ signi(sigma_r, gamma=0.5) + zeta_r

            v_r = self.Lambda_r @ e_r + tau_r

            w_r = - Brp @ v_r

            # return w_r
        else:
            lambdax = self.Lambda_r[0, 0]
            lambdatheta = self.Lambda_r[1, 1]
            matriz_ganho_erro = np.array([lambdax * ex, lambdatheta * etheta]).transpose()[:,None]
            w_r = - np.matmul(Brp,(matriz_ganho_erro + Ar * self.v_d))

            # Ensure that the calculated angular velocity does not exceed the maximum allowed value (Wmax)
            # if(abs(w) > Wmax):
            #     w = Wmax * np.sign(w)

        # Prime time
        # lb = 0.3
        # self.omega_prime = (lb*self.omega_prime)/(lb+self.dt) + (w_r * self.dt)/(lb + self.dt)

        # return self.omega_prime
        return w_r

    def col_Controller(self, point_x, point_y, theta, Wmax=0.2):
        
        y = point_y
        x = point_x

        Ly = np.array([0, (-np.sin(self.ro)-y * np.cos(self.ro)/self.t_z), y*(np.sin(self.ro)+y*np.cos(self.ro))/self.t_z, 1+y**2, -x*y, -x])

        Ltheta = np.array([(np.cos(self.ro) * np.cos(theta)**2)/self.t_z, 
                        (np.cos(self.ro) * np.cos(theta) * np.sin(theta))/self.t_z, 
                        -(np.cos(self.ro)*np.cos(theta) * (y*np.sin(theta) + x * np.cos(theta)))/self.t_z, 
                        -(y*np.sin(theta) + x*np.cos(theta)) *np.cos(theta), 
                        -(y*np.sin(theta) + x*np.cos(theta))*np.sin(theta), 
                        -1])
        Ls = np.vstack((Ly, Ltheta))

        Tv = np.array([ 0, -np.sin(self.ro),  np.cos(self.ro), 0, 0, 0 ]).transpose()[:, None]
        Tw = np.array([-self.t_y, 0, 0, 0, -np.cos(self.ro), -np.sin(self.ro)]).transpose()[:, None]

        Ar = np.matmul(Ls, Tv)
        Br = np.matmul(Ls, Tw)

        Brp = np.linalg.pinv(Br)

        ey = point_y - self.y_star
        etheta = theta - self.theta_star
        # print(f"ey={ey}, etheta={etheta}")

        if self.robust:
            e_c = np.array([[ey], [etheta]])

            sigma_c = e_c + self.Gamma_c @ e_c * self.dt

            zeta_c = -self.mu_c @ np.sign(sigma_c) * self.dt

            tau_c = self.K_c @ signi(sigma_c, self.gamma) + zeta_c

            v_c = self.Lambda_c @ e_c + tau_c

            w_c = - Brp @ v_c

            return w_c
        else:
            lambday = self.Lambda_c[0, 0]
            lambdatheta = self.Lambda_c[1, 1]
            matrix_ganho_erro = np.array([lambday * ey, lambdatheta * etheta]).transpose()[:,None]
            
            w = - np.matmul(Brp, (matrix_ganho_erro + Ar * self.v_d))

            # if abs(w) > np.pi/8:
            #     w = - np.sign(w) * Wmax

        return w

    def control_step(self, point_x, point_y, theta):

        if 0.98 < point_y <= 1:
            # print("row")
            return self.row_Controller(point_x, -1, theta)
        else:
            # print("col")
            return self.col_Controller(point_x, point_y, theta)

    def step_col(self, x, y, theta, robust=True):
    
        L_y = np.array([0, (-sin(self.ro) - y * cos(self.ro))/self.t_z, (y*(sin(self.ro)+y*cos(self.ro)))/self.t_z, 1+x**2, -x*y, -x])

        L_theta = np.array([(cos(self.ro) * cos(theta)**2)/self.t_z, 
        (cos(self.ro) * cos(theta) * sin(theta))/self.t_z, 
        -(cos(self.ro)*cos(theta) * (y*sin(theta) + x * cos(theta)))/self.t_z, 
        -(y*sin(theta) + x*cos(theta)) *cos(theta), 
        -(y*sin(theta) + x*cos(theta))*sin(theta), 
        -1])
         # Compute skew-symmetric matrix Q(t_cr)
        Q_tcr = np.array([
            [0, -self.t_cr[2, 0], self.t_cr[1, 0]],
            [self.t_cr[2, 0], 0, -self.t_cr[0, 0]],
            [-self.t_cr[1, 0], self.t_cr[0, 0], 0]
        ])

        # Rotation matrix R_cr = Rx(ψ) * Ry(π/2) * Rx(−π/2)
        psi = 0.75  # for example

        Rx_psi = np.array([
            [1, 0, 0],
            [0, np.cos(psi), -np.sin(psi)],
            [0, np.sin(psi), np.cos(psi)]
        ])

        Ry_90 = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])

        Rx_minus_90 = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])

        R_cr = Rx_psi @ Ry_90 @ Rx_minus_90  # Shape: (3, 3)

        top = np.hstack((R_cr, Q_tcr @ R_cr))
        bottom = np.hstack((np.zeros((3, 3)), R_cr))
        A_cr = np.vstack((top, bottom))  # Shape: (6, 6)

        S = np.array([[1,0,0,0,0,0],[0,0,0,0,0,1]]).transpose()

        T_cr = A_cr @ S  # (6, 2)

        T_v = T_cr[:,0]
        T_w = T_cr[:,1]

        top = np.hstack((R_cr, Q_tcr @ R_cr))
        bottom = np.hstack((np.zeros((3, 3)), R_cr))
        A_cr = np.vstack((top, bottom))  # Shape: (6, 6)

        S = np.array([[1,0,0,0,0,0],[0,0,0,0,0,1]]).transpose()

        T_cr = A_cr @ S  # (6, 2)

        T_v = T_cr[:,0]
        T_w = T_cr[:,1]
        
        # J_wr, called B_r in Cherubini paper, is described in Cherubini Eq. 3, Toni Eq. 11
        J_wc = [[L_y], [L_theta]] @ T_w 
        J_wcp = np.linalg.pinv(J_wc) 

        # Error for row control
        e_y = y - self.y_star
        e_theta = theta - self.theta_star
        e_c = np.array([[e_y], [e_theta]]) # 2x1
        self.e_c_list.append(e_c) # 2x1 in each element
        if robust:
            sigma_c = e_c + self.Gamma_c @ self.e_c_list.sum()  # 2x1
            zeta_dot_c = -self.mu_c @ np.sign(sigma_c)
            self.zeta_c_list.append(zeta_dot_c)
            tau_c = self.K_c @ signi(sigma_c, self.gamma) + self.zeta_c_list.sum() # Integral is the sum over time

            v_c = self.Lambda_c @ e_c + tau_c
            w_c = J_wcp @ v_c

            return w_c[0][0] # To get only the number
        else:
            A_c = [[L_y], [L_theta]] @ T_v 
            return (-J_wcp @ (self.Lambda_r @ e_c + A_c * self.v_d))[0] # To get only the number
