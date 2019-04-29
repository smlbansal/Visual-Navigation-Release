# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Wrapper for selecting the navigation environment that we want to train and
test on.
"""
import os, glob, logging
import sys
# Py27 vs Py3 imports
if sys.version_info[0] == 2:
    from render import swiftshader_renderer as renderer
    import utils
    import mp_env
else:
    from mp_env.render import swiftshader_renderer as renderer 
    from mp_env import utils
    from mp_env import mp_env

def get_dataset(dataset_name, imset, data_dir):
  if dataset_name == 'sbpd':
    dataset = StanfordBuildingParserDataset(imset,
                                            data_dir=data_dir)
  else:
    logging.fatal('Not one of sbpd')
  return dataset

class Loader():
  def get_data_dir():
    pass

  def load_building(self, name, data_dir=None):
    if data_dir is None: data_dir = self.get_data_dir()
    out = {}
    out['name'] = name
    out['data_dir'] = data_dir
    return out

  def load_building_meshes(self, building, materials_scale=1.0):
    dir_name = os.path.join(building['data_dir'], 'mesh', building['name'])
    mesh_file_name = glob.glob1(dir_name, '*.obj')[0]
    mesh_file_name_full = os.path.join(dir_name, mesh_file_name)
    logging.error('Loading building from obj file: %s', mesh_file_name_full)
    shape = renderer.Shape(mesh_file_name_full, load_materials=True, 
      name_prefix=building['name']+'_',  materials_scale=materials_scale)
    return [shape]

  def load_data(self, name, robot, flip=False):
    env = utils.Foo(padding=10, resolution=5, num_point_threshold=2,
      valid_min=-10, valid_max=200, n_samples_per_face=200)
    building = mp_env.Building(self, name, robot, env, flip=flip)
    return building

class StanfordBuildingParserDataset(Loader):
  def __init__(self, imset, data_dir=None):
    self.imset = imset 
    self.data_dir = data_dir
  
  def get_data_dir(self):
    return self.data_dir

  def get_benchmark_sets(self):
    return self._get_benchmark_sets()

  def get_split(self):
    return self._get_split(self.imset)
  
  def get_imset(self):
    return self._get_split(self.imset)

  def _get_benchmark_sets(self):
    sets = ['train1', 'train2', 'val', 'test']
    return sets

  def _get_split(self, split_name):
    train = ['area1', 'area5a', 'area5b', 'area6']
    train1 = ['area1']
    train2 = ['area1', 'area5a']
    train2x2 = ['area1+area5a', 'area5b+area6']
    train1x4 = ['area1+area5a+area5b+area6']
    val = ['area3']
    test = ['area4']

    sets = {}
    sets['train'] = train
    sets['train1'] = train1
    sets['train2'] = train2
    sets['train2x2'] = train2x2
    sets['train1x4'] = train1x4
    sets['val'] = val
    sets['test'] = test
    sets['all'] = sorted(list(set(train + val + test)))
    return sets[split_name]
