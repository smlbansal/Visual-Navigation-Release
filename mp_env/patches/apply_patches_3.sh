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


# Extract the base directory of this anaconda environment
conda_info_json="$(conda info --json | grep active_prefix)"
IFS=',' read -r active_info other <<< "$conda_info_json"
IFS=':' read -r other conda_dir <<< "$active_info"

# remove unnecesary quotation marks
conda_dir="${conda_dir//\"}"

# Add the file path to the OpenGL file to be patched
file_to_patch="$conda_dir/lib/python3.6/site-packages/OpenGL/GLES2/VERSION/GLES2_2_0.py"
patch $file_to_patch patches/GLES2_2_0.py.patch

# If the above does not work the intended command is shown below.
# The provided patch should be applied to GLES_2_2.0.py
# for the python installation which the user intends to use
# for this project

# patch /home/user_name/anaconda3/envs/venv-mpc/lib/python3.6/site-packages/OpenGL/GLES2/VERSION/GLES2_2_0.py patches/GLES2_2_0.py.patch
