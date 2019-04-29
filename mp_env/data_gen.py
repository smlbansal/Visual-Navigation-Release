from __future__ import print_function
from _logging import logging
import numpy as np, os, cv2, pickle
from render import swiftshader_renderer as sr
import sbpd, mp_env, map_utils as mu, utils

def dataset_mpc(dataset, imset, seed=1, n=100, out_dir=None, crop_size=80):
  # Load the building
  d = sbpd.get_dataset(dataset, imset)
  ns = d.get_imset()
  
  camera_param = utils.Foo(width=225, height=225, z_near=0.01, z_far=20.0,
    fov_horizontal=60., fov_vertical=60., modalities=['rgb'], img_channels=3,
    im_resize=1.)
  r_obj = sr.get_r_obj(camera_param)
 
  for name in ns:
    for flip in [True, False]:
 
      tt = d.load_data(name, flip)
      tt.set_r_obj(r_obj)
      tt.load_building_into_scene()

      traversible_cc, resolution = tt.traversible, tt.map.resolution
      traversible_map = tt.map.traversible*1.
      starts, ends = mu.sample_positions_on_map(seed, traversible_cc, resolution, n)
      
      # Crop out map from this location.
      loc = starts*1.; diff = ends - starts;
      theta = np.arctan2(diff[:,1], diff[:,0])[:,np.newaxis]
      x_axis = np.concatenate([np.cos(theta), np.sin(theta)], axis=1)
      y_axis = np.concatenate([np.cos(theta+np.pi/2.), np.sin(theta+np.pi/2.)], axis=1)
      crops = mu.generate_egocentric_maps([traversible_map], [1.0], [crop_size], 
        starts, x_axis, y_axis, dst_theta=np.pi/2.0)
      
      # Render out images for each location.
      nodes = np.concatenate([loc, theta/tt.robot.delta_theta], axis=1)
      imgs = tt.render_nodes(nodes)
      r_obj.clear_scene()
      
      # Write the maps and the images in a directory.
      if out_dir is not None:
        out_dir_ = os.path.join(out_dir, '{:d}'.format(seed), '{:s}_{:s}_{:d}'.format(dataset, name, flip))
        logging.error('out_dir: %s', out_dir_)
        utils.mkdir_if_missing(out_dir_)
        for i in range(n):
          file_name = os.path.join(out_dir_, 'img_{:06d}.jpg'.format(i))
          cv2.imwrite(file_name, imgs[i][:,:,::-1].astype(np.uint8))
          file_name = os.path.join(out_dir_, 'map_{:06d}.png'.format(i))
          cv2.imwrite(file_name, (crops[0][i,:,:]*255).astype(np.uint8))
        
        out_file_name = os.path.join(out_dir, 'maps', '{:s}_{:s}_{:d}.pkl'.format(dataset, name, flip)) 
        vv = vars(tt.map)
        vv['traversible_cc'] = tt.traversible
        # save_variables(out_file_name, list(vv.values()), list(vv.keys()), True)

        output_file = os.path.join(out_dir_, 'vis.html')
        create_webpage(output_file, n)

def create_webpage(output_file, n):
  from yattag import Doc, indent
  a = 0
  doc, tag, text = Doc().tagtext()
  with tag('html'):
    with tag('body'):
      with tag('table', style = 'width:100%', border="1"):
        for i in range(n):
          with tag('tr'):
            with tag('td'):
              with tag('img', width="100%", src=os.path.join('img_{:06d}.jpg'.format(i))):
                a = a+1
            with tag('td'):
              with tag('img', width="100%", src=os.path.join('map_{:06d}.png'.format(i))):
                  a = a+1
  r1 = doc.getvalue()
  r2 = indent(r1)
  with open(output_file, 'wt') as f:
    print(r2, file=f)

def save_variables(pickle_file_name, var, info, overwrite=False):
  if os.path.exists(pickle_file_name) and overwrite == False:
    raise Exception('{:s} exists and over write is false.'.format(pickle_file_name))
  # Construct the dictionary
  # assert(type(var) == list); assert(type(info) == list);
  for t in info: assert(type(t) == str), 'variable names are not strings'
  d = {}
  for i in range(len(var)):
    d[info[i]] = var[i]
  with open(pickle_file_name, 'w') as f:
    pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
  dataset_mpc('sbpd', 'test', seed=0, crop_size=160, out_dir='tmp/output', n=1000)
