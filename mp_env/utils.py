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

r"""Generaly Utilities.
"""

import numpy as np, os, time
import logging, hashlib
from contextlib import contextmanager

def get_time_str():
  return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

class Timer():
  def __init__(self, skip=0):
    self.calls = 0.
    self.start_time = 0.
    self.time_per_call = 0.
    self.time_ewma = 0.
    self.total_time = 0.
    self.last_log_time = 0.
    self.skip = skip

  def tic(self):
    self.start_time = time.time()
  
  def display(self, average=True, log_at=-1, log_str='', type='calls', mul=1, 
    current_time=None):
    if current_time is None: 
      current_time = time.time()
    if self.skip == 0:
      ewma = self.time_ewma * mul / np.maximum(0.01, (1.-0.99**self.calls))
      if type == 'calls' and log_at > 0 and np.mod(self.calls/mul, log_at) == 0:
        _ = []
        logging.info('%s: %f seconds / call, %d calls.', log_str, ewma, self.calls/mul)
      elif type == 'time' and log_at > 0 and current_time - self.last_log_time >= log_at:
        _ = []
        logging.info('%s: %f seconds / call, %d calls.', log_str, ewma, self.calls/mul)
        self.last_log_time = current_time
    # return self.time_per_call*mul
    return ewma

  def toc(self, average=True, log_at=-1, log_str='', type='calls', mul=1):
    if self.skip > 0:
      self.skip = self.skip-1
    else:
      if self.start_time == 0:
        logging.error('Timer not started by calling tic().')
      t = time.time()
      diff = time.time() - self.start_time
      self.total_time += diff; self.calls += 1.;
      self.time_per_call = self.total_time/self.calls
      alpha = 0.99
      self.time_ewma = self.time_ewma*alpha + (1-alpha)*diff
      self.display(average, log_at, log_str, type, mul, current_time=time)
    
    if average:
      return self.time_per_call*mul
    else:
      return diff
  
  @contextmanager
  def record(self):
    self.tic()
    yield
    self.toc()

class Foo(object):
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)
  def __str__(self):
    str_ = ''
    for v in vars(self).keys():
      a = getattr(self, v)
      if True: #isinstance(v, object):
        str__ = str(a)
        str__ = str__.replace('\n', '\n  ')
      else:
        str__ = str(a)
      str_ += '{:s}: {:s}'.format(v, str__)
      str_ += '\n'
    return str_

class TicTocPrint():
  def __init__(self, interval):
    self.interval = interval
    self.last_time = 0
  def log(self, *args):
    t = time.time()
    if t - self.last_time > self.interval:
      logging.error(*args)
      self.last_time = t

def mkdir_if_missing(output_dir):
  if not os.path.exists(output_dir):
    try:
      os.makedirs(output_dir)
    except:
      logging.error("Something went wrong in mkdir_if_missing. "
        "Probably some other process created the directory already.")
