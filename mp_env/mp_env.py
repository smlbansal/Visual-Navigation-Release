from __future__ import print_function
import logging
import numpy as np
import sys
if sys.version_info[0] == 2:
    from . import map_utils as mu
else:
    from mp_env import map_utils as mu #py3

make_map = mu.make_map
resize_maps = mu.resize_maps
compute_traversibility = mu.compute_traversibility
pick_largest_cc = mu.pick_largest_cc

class Building():
  def __init__(self, dataset, name, robot, env, flip=False):
    self.restrict_to_largest_cc = True
    self.robot = robot
    self.env = env

    # Load the building meta data.
    env_paths = dataset.load_building(name)
    materials_scale = 1.0
    self.materials_scale = materials_scale
    
    shapess = dataset.load_building_meshes(env_paths, 
      materials_scale=materials_scale)
    if flip: 
      for shapes in shapess: 
        shapes.flip_shape()
    
    vs = []
    for shapes in shapess:
      vs.append(shapes.get_vertices()[0])
    vs = np.concatenate(vs, axis=0)
    map = make_map(env.padding, env.resolution, vertex=vs, sc=100.)
    map = compute_traversibility(
      map, robot.base, robot.height, robot.radius, env.valid_min,
      env.valid_max, env.num_point_threshold, shapess=shapess, sc=100.,
      n_samples_per_face=env.n_samples_per_face)

    self.env_paths = env_paths 
    self.shapess = shapess
    self.map = map
    self.traversible = map.traversible*1
    self.name = name 
    self.flipped = flip
    self.renderer_entitiy_ids = []
    if self.restrict_to_largest_cc:
      self.traversible = pick_largest_cc(self.traversible)

  def set_r_obj(self, r_obj):
    self.r_obj = r_obj

  def load_building_into_scene(self, dedup_tbo=False):
    assert(self.shapess is not None)
    
    # Loads the scene.
    self.renderer_entitiy_ids += self.r_obj.load_shapes(self.shapess, dedup_tbo)
    # Free up memory, we dont need the mesh or the materials anymore.
    self.shapess = None

  def to_actual_xyt(self, pqr):
    """Converts from node array to location array on the map."""
    out = pqr*1.
    # p = pqr[:,0:1]; q = pqr[:,1:2]; r = pqr[:,2:3];
    # out = np.concatenate((p + self.map.origin[0], q + self.map.origin[1], r), 1) 
    return out

  def set_building_visibility(self, visibility):
    self.r_obj.set_entity_visible(self.renderer_entitiy_ids, visibility)

  def render_nodes(self, nodes, perturb=None, aux_delta_theta=0.):
    # List of nodes to render.
    self.set_building_visibility(True)
    if perturb is None:
      perturb = np.zeros((len(nodes), 4))

    imgs = []
    r = 2
    elevation_z = r * np.tan(np.deg2rad(self.robot.camera_elevation_degree))

    for i in range(len(nodes)):
      xyt = self.to_actual_xyt(nodes[i][np.newaxis,:]*1.)[0,:]
      lookat_theta = 3.0 * np.pi / 2.0 - (xyt[2]+perturb[i,2]+aux_delta_theta) * (self.robot.delta_theta)
      nxy = np.array([xyt[0]+perturb[i,0], xyt[1]+perturb[i,1]]).reshape(1, -1)
      nxy = nxy * self.map.resolution
      nxy = nxy + self.map.origin
      camera_xyz = np.zeros((1, 3))
      camera_xyz[...] = [nxy[0, 0], nxy[0, 1], self.robot.sensor_height]
      camera_xyz = camera_xyz / 100.
      lookat_xyz = np.array([-r * np.sin(lookat_theta),
                             -r * np.cos(lookat_theta), elevation_z])
      lookat_xyz = lookat_xyz + camera_xyz[0, :]
      self.r_obj.position_camera(camera_xyz[0, :].tolist(), lookat_xyz.tolist(), 
        [0.0, 0.0, 1.0])
      img = self.r_obj.render(take_screenshot=True, output_type=0)
      img = [x for x in img if x is not None]
      img = np.concatenate(img, axis=2).astype(np.float32)
      if perturb[i,3]>0:
        img = img[:,::-1,:]
      imgs.append(img)

    self.set_building_visibility(False)
    return imgs
