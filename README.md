## Master's Thesis - Erik Lykke Trier
This repository contains the code for a row following implementation presented in my master's thesis.

### The Controller
This code includes a robust Image-Based Visual Servoing (rIBVS) approach and a proportional IBVS approach, which can be switched between by setting the robust variable to True (Robust) or False (Not robust). The controller itself is implemented based on the work of Barbosa et al. (2021), "Robust Image-based Visual Servoing for Autonomous Row Crop Following with Wheeled Mobile Robots", which presents both robust and proportional controllers. The second paper is from Cherubini et al, 2011, "Visual servoing for path reaching with nonholonomic robots", which presents the adjoint matrix which maps robot motion to the feature vector coordinates $s = (x_v y_v \theta_v)$.

### How to use this code
1. Have a mobile robot that can take linear and angular velocity commands.
2. Attach a camera
3. Launch the robot with a camera in one of the worlds
4. Then run the node "vector_view_model.py" to control the robot with the deep learning model, changing, of course, the topic the velocity is to be published to and what topic the camera is being published to. 
