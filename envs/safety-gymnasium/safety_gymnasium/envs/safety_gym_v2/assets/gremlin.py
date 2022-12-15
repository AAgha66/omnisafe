# Copyright 2022 OmniSafe Team. All Rights Reserved.
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
"""gremlin"""

import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.color import COLOR
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP


def get_gremlin(
    index,
    layout,
    rot,
    density=0.001,
    size=0.1,
):
    name = f'gremlin{index}obj'
    object = {
        'name': name,
        'size': np.ones(3) * size,
        'type': 'box',
        'density': density,
        'pos': np.r_[layout[name.replace('obj', '')], size],
        'rot': rot,
        'group': GROUP['gremlin'],
        'rgba': COLOR['gremlin'],
    }
    return object