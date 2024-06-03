import clearml
import pathlib

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train


def main(env: str, algo: str, seed: int):
    eg = ExperimentGrid(exp_name="Benchmark_Safety_Velocity")
    mujoco_envs = [env]

    eg.add("env_id", mujoco_envs)
    # Set the device.
    gpu_id = [0]

    eg.add("algo", algo)
    eg.add("logger_cfgs:use_wandb", [False])
    eg.add("logger_cfgs:use_tensorboard", [True])
    eg.add("train_cfgs:vector_env_nums", [1])
    eg.add("train_cfgs:torch_threads", [1])
    eg.add("algo_cfgs:steps_per_epoch", [2000])
    eg.add("train_cfgs:total_steps", [300000])
    eg.add("seed", [seed])

    # total experiment num must can be divided by num_pool
    # meanwhile, users should decide this value according to their machine
    eg.run(train, num_pool=1, gpu_id=gpu_id)


if __name__ == "__main__":
    task = clearml.Task.init()
    task_logger = task.get_logger()
    task_params = task.get_parameters_as_dict(cast=True)
    d = task_params["internal"]
    print(d)
    main(
        seed=d["seed"],
        env=d["env"],
        algo=d["algo"],
    )
