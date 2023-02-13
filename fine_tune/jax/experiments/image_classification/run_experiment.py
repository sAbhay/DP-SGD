# coding=utf-8
# Copyright 2022 DeepMind Technologies Limited.
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

"""Main script to run an experiment.

Usage example (run from this directory):
  python run_experiment.py --config=configs/cifar10_wrn.py
"""

import functools

from absl import app
from absl import flags
from jax_privacy.src.training.image_classification import experiment
from jaxline import platform
import os


if __name__ == '__main__':
  # os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # prone to memory fragmentation
  # os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform' # TODO: remove flag, very slow
  # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

  flags.mark_flag_as_required('config')
  app.run(functools.partial(platform.main, experiment.Experiment))
