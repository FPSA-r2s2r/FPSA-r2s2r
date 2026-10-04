# Simulation tasks

Each task keeps its simulation, Gym environment, and one shared YAML
configuration in this directory. Both data collection and policy evaluation use
the same `make_scene` settings.

| Task | Simulation | Environment | Runner | Shared config |
| --- | --- | --- | --- | --- |
| Gear | `Sim_Gear.py` | `Env_Gear.py` | `FPSA_Gear_runner.py` | `Config_Gear.yaml` |
| Wrench | `Sim_Wrench.py` | `Env_Wrench.py` | `FPSA_wrench_runner.py` | `Config_Wrench.yaml` |
| Assembly | `Sim_Assembly.py` | `Env_Assembly.py` | `FPSA_Assembly_runner.py` | `Config_Assembly.yaml` |
| Gear Horizon | `Sim_GearHorizon.py` | `Env_GearHorizon.py` | `FPSA_GearHorizon_runner.py` | `Config_GearHorizon.yaml` |
| Pick Up | `Sim_PickUp.py` | `Env_PickUp.py` | `FPSA_PickUp_runner.py` | `Config_PickUp.yaml` |

The `simulation` field selects the Sim class. `sim_init`, `demo`, `mp_collect`,
`make_scene`, and `collect_observation` provide everything needed by both common
entry points. Run them from the repository root, for example:

```bash
python demo.py Config_Gear.yaml
python MP_collect.py Config_Gear.yaml --num-episodes 3000
```

All task configs set `sim_init.use_egl: true`, so `DIRECT` environments render
through PyBullet's EGL OpenGL plugin. The shared visual randomizers live under
`VisualDR/`. Common runtime helpers (`episode_writer.py`,
`pybullet_utility.py`, `task_config.py`, `task_runtime.py`, and `ultility.py`)
live under `src/`. The shared environment and simulation bases are
`Base_Env.py` and `Base_Simulation.py`; the user-facing `demo.py` and
`MP_collect.py` entry points live at the repository root.
