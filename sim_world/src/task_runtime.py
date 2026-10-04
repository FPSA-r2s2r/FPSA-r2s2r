"""Shared task discovery and simulation construction."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple, Type


SIMULATION_REGISTRY: Dict[str, Tuple[str, str]] = {
    "Sim_Gear": ("sim_world.Sim_Gear", "GearSim"),
    "Sim_Wrench": ("sim_world.Sim_Wrench", "WrenchSim"),
    "Sim_Assembly": ("sim_world.Sim_Assembly", "AssemblySim"),
    "Sim_GearHorizon": ("sim_world.Sim_GearHorizon", "GearHorizonSim"),
    "Sim_PickUp": ("sim_world.Sim_PickUp", "PickUpSim"),
}


def get_simulation_class(name: str) -> Type[Any]:
    """Resolve a configured simulation name from the supported task registry."""
    try:
        module_name, class_name = SIMULATION_REGISTRY[name]
    except KeyError as exc:
        supported = ", ".join(sorted(SIMULATION_REGISTRY))
        raise ValueError(f"Unknown simulation {name!r}; choose one of: {supported}") from exc
    return getattr(import_module(module_name), class_name)


def build_simulation(
    task_config: Dict[str, Any],
    bullet_client: Any,
    client_id: int,
    *,
    seed: int,
    control_dt: float,
) -> Any:
    """Instantiate the simulation selected by a task config."""
    simulation_name = task_config.get("simulation")
    if not isinstance(simulation_name, str):
        raise ValueError("Task config must define a string 'simulation' name")
    sim_class = get_simulation_class(simulation_name)
    init_kwargs = dict(task_config.get("sim_init", {}))
    init_kwargs.update(control_dt=control_dt, seed=seed)
    return sim_class(bullet_client, client_id, **init_kwargs)
