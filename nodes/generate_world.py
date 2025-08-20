#!/usr/bin/python3
import os
import rospkg
import numpy as np

def rng(mu, sig):
    s = np.random.normal(mu, sig/3, 1)
    return np.squeeze(s)

def generate_world(start_pos, size):
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('my_own_thorvald')
    output_path = package_path
    # Open output file
    output = open(os.path.join(output_path, 'worlds/v3_generated_world_'+size+'_flipped_sloped_bumpy.world'), 'w')

    # Write the same world header as in generate_sdf.py
    print('''<sdf version='1.6'>
  <world name='default'>
    <light name='sun' type='directional'>
      <cast_shadows>1</cast_shadows>
      <pose frame=''>0 0 10 0 -0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name='ground_plane'>
      <static>1</static>
      <link name='link_0'>
        <collision name='collision'>
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>100</mu>
                <mu2>50</mu2>
              </ode>
              <torsional>
                <ode/>
              </torsional>
            </friction>
            <contact>
              <ode/>
            </contact>
            <bounce/>
          </surface>
          <max_contacts>10</max_contacts>
        </collision>
        <visual name='visual'>
          <cast_shadows>0</cast_shadows>
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/Grey</name>
            </script>
          </material>
        </visual>
        <self_collide>0</self_collide>
        <enable_wind>0</enable_wind>
        <kinematic>0</kinematic>
      </link>
    </model>

    <model name='heightmap'>
      <static>1</static>
      <link name='link_0'>
        <collision name='collision'>
          <geometry>
            <heightmap>
              <uri>model://heightmap_/materials/textures/heightmap3.jpg</uri>
              <size>80 80 5.5</size>
              <pos>0 0 0</pos>
              <texture>
                <size>10</size>
                <diffuse>__default__</diffuse>
                <normal>__default__</normal>
              </texture>
              <blend>
                <min_height>0</min_height>
                <fade_dist>0</fade_dist>
              </blend>
            </heightmap>
          </geometry>
          <max_contacts>10</max_contacts>
          <surface>
            <contact>
              <ode/>
            </contact>
            <bounce/>
            <friction>
              <torsional>
                <ode/>
              </torsional>
              <ode/>
            </friction>
          </surface>
        </collision>
        <visual name='visual_abcedf'>
          <geometry>
            <heightmap>
              <use_terrain_paging>0</use_terrain_paging>
              <texture>
                <diffuse>file://media/materials/textures/grass_diffusespecular.png</diffuse>
                <normal>file://media/materials/textures/flat_normal.png</normal>
                <size>1</size>
              </texture>
              <uri>model://heightmap_/materials/textures/heightmap3.jpg</uri>
              <size>80 80 5.5</size>
              <pos>0 0 0</pos>
            </heightmap>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/Grass</name>
            </script>
          </material>
        </visual>
        <self_collide>0</self_collide>
        <enable_wind>0</enable_wind>
        <kinematic>0</kinematic>
      </link>
      <pose frame=''>-16.9724 8.52145 0 0 -0 0</pose>
    </model>''', file=output)

    # Generate patches of plants
    if size == 'small':
        scale=0.003
    elif size == 'medium':
        scale=0.005
    elif size == 'large':
        scale=0.02
    else:
        raise ValueError("Invalid size")
    # The following are tested for medium sized plants.
    # patch_width = 4
    # patch_height = 8
    # grid_width = 4
    # grid_height = 5


    # patch_spacing_x = 0.6
    # patch_spacing_y = 0.5
    # plant_spacing_x = 0.25
    # plant_spacing_y = 0.2

    patch_width = 8
    patch_height = 4
    grid_width = 5
    grid_height = 4
    patch_spacing_x = 0.5
    patch_spacing_y = 0.6
    plant_spacing_x = 0.2
    plant_spacing_y = 0.25

    idx = 0


    start_x, start_y, start_z = start_pos
    for patch_row in range(grid_height):
        for patch_col in range(grid_width):
            # Calculate patch center position
            patch_center_x = start_x + patch_col * (patch_spacing_x + patch_width * plant_spacing_x)
            patch_center_y = start_y + patch_row * (patch_spacing_y + patch_height * plant_spacing_y)
            
            for plant_row in range(patch_height):
                for plant_col in range(patch_width):
                    # Calculate plant position relative to patch center
                    if np.random.random() > 0.15:
                        x_rand = rng(0,0.1)
                        y_rand = rng(0, 0.075)
                    else:
                        x_rand, y_rand = 0, 0
                    x = patch_center_x + plant_col * plant_spacing_x + x_rand
                    y = patch_center_y + plant_row * plant_spacing_y + y_rand
                    z = start_z + 0.08 * x
                    rot = rng(0, np.pi)

                    print(f'''    <model name="plt_{idx}">
      <static>1</static>
      <link name='link_{idx:02d}'>
        <pose frame=''>{x:.3f} {y:.3f} {z:.3f} 0.0 0.0 {rot:.5f}</pose>
        <visual name='visual'>
          <cast_shadows>0</cast_shadows>
          <geometry>
            <mesh>
              <uri>model://plants_disorganized/meshes/acanthus.stl</uri>
              <scale>{scale:.3f} {scale:.3f} {scale:.3f} </scale>
            </mesh>
          </geometry>
          <material>
            <script>
              <uri>model://plants_disorganized/materials/scripts</uri>
              <uri>model://plants_disorganized/materials/textures</uri>
              <name>vrc/cotton</name>
            </script>
          </material>
        </visual>
        <self_collide>0</self_collide>
        <enable_wind>0</enable_wind>
        <kinematic>0</kinematic>
      </link>
    </model>''', file=output)
                    idx += 1

    # Close world
    print('  </world>', file=output)
    print('</sdf>', file=output)
    output.close()

    print("World generated successfully")

if __name__ == "__main__":
    # Start pos of first patch...
    # generate_world(start_pos=(-1, -20, 2.8), size='small')
    # generate_world(start_pos=(-1, -20, 5.0), size='medium') # Flatmap
    generate_world(start_pos=(-1, -20, 2.75), size='medium') # Slope 
    # generate_world(start_pos=(-1, -20, 2.8), size='large')