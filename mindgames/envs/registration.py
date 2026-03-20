import importlib
import random
import re
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field

from mindgames.core import Wrapper
from mindgames.prompting.templates import PromptProfile


# Global environment registry
ENV_REGISTRY: Dict[str, Callable] = {}


def _resolve_import_string(ref: Any) -> Any:
    if not isinstance(ref, str):
        return ref

    module_path, attr_name = ref.split(":")
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not import {module_path}.{attr_name}. Error: {e}") from e

@dataclass
class EnvSpec:
    """A specification for creating environments."""
    id: str
    entry_point: Callable
    default_wrappers: Optional[List[type[Wrapper]]]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    prompt_profile: Optional[PromptProfile] = None
    action_parser: Optional[Callable[..., List[str]] | str] = None
    reward_mode: str = "terminal"
    obs_mode: str = "llm"

    def make(self, **kwargs) -> Any:
        """Create an environment instance."""
        all_kwargs = {**self.kwargs, **kwargs}
        return self.entry_point(**all_kwargs)

    def resolve_action_parser(self) -> Optional[Callable[..., List[str]]]:
        if self.action_parser is None:
            return None
        return _resolve_import_string(self.action_parser)


def register(
    id: str,
    entry_point: Callable,
    default_wrappers: Optional[List[type[Wrapper]]] = None,
    *,
    prompt_profile: Optional[PromptProfile] = None,
    action_parser: Optional[Callable[..., List[str]] | str] = None,
    reward_mode: str = "terminal",
    obs_mode: str = "llm",
    **kwargs: Any,
):
    """Register an environment with a given ID."""
    if id in ENV_REGISTRY:
        raise ValueError(f"Environment {id} already registered.")
    ENV_REGISTRY[id] = EnvSpec(
        id=id,
        entry_point=entry_point,
        default_wrappers=default_wrappers,
        kwargs=kwargs,
        prompt_profile=prompt_profile,
        action_parser=action_parser,
        reward_mode=reward_mode,
        obs_mode=obs_mode,
    )


def register_with_versions(
    id: str,
    entry_point: Callable,
    wrappers: Optional[Dict[str, List[type[Wrapper]]]] = None,
    *,
    prompt_profile: Optional[PromptProfile] = None,
    action_parser: Optional[Callable[..., List[str]] | str] = None,
    reward_mode: str = "terminal",
    obs_mode: str = "llm",
    **kwargs: Any,
):
    """Register an environment with a given ID."""
    if id in ENV_REGISTRY:
        raise ValueError(f"Environment {id} already registered.")

    # first register default version
    ENV_REGISTRY[id] = EnvSpec(
        id=id,
        entry_point=entry_point,
        default_wrappers=wrappers.get("default"),
        kwargs=kwargs,
        prompt_profile=prompt_profile,
        action_parser=action_parser,
        reward_mode=reward_mode,
        obs_mode=obs_mode,
    )
    for wrapper_version_key in list(wrappers.keys()) + ["-raw"]:
        if wrapper_version_key == "default":
            continue
        ENV_REGISTRY[f"{id}{wrapper_version_key}"] = EnvSpec(
            id=f"{id}{wrapper_version_key}",
            entry_point=entry_point,
            default_wrappers=wrappers.get(wrapper_version_key),
            kwargs=kwargs,
            prompt_profile=prompt_profile,
            action_parser=action_parser,
            reward_mode=reward_mode,
            obs_mode=obs_mode,
        )

def pprint_registry_detailed():
    """Pretty print the registry with additional details like kwargs."""
    if not ENV_REGISTRY:
        print("No environments registered.")
    else:
        print("Detailed Registered Environments:")
        for env_id, env_spec in ENV_REGISTRY.items():
            print(f"  - {env_id}:")
            print(f"      Entry Point: {env_spec.entry_point}")
            print(f"      Kwargs:      {env_spec.kwargs}")
            print(f"      Wrappers:    {env_spec.default_wrappers}")
            print(f"      Prompt:      {env_spec.prompt_profile}")
            print(f"      Parser:      {env_spec.action_parser}")
            print(f"      RewardMode:  {env_spec.reward_mode}")
            print(f"      ObsMode:     {env_spec.obs_mode}")

def check_env_exists(env_id: str):
    """Check if an environment exists in the registry."""
    if env_id not in ENV_REGISTRY:
        raise ValueError(f"Environment {env_id} is not registered.")
    else:
        print(f"Environment {env_id} is registered.")

def make(env_id: Union[str, List[str]], **kwargs) -> Any:
    """Create an environment instance using the registered ID."""
    # If env_id is a list, randomly select one environment ID
    if isinstance(env_id, list):
        if not env_id:
            raise ValueError("Empty list of environment IDs provided.")
        env_id = random.choice(env_id)
    
    # Continue with the existing implementation
    if env_id not in ENV_REGISTRY:
        raise ValueError(f"Environment {env_id} not found in registry.")
    
    env_spec = ENV_REGISTRY[env_id]

    # Resolve the entry point if it's a string
    if isinstance(env_spec.entry_point, str):
        env_class = _resolve_import_string(env_spec.entry_point)
    else:
        env_class = env_spec.entry_point

    env = env_class(**{**env_spec.kwargs, **kwargs})

    # Dynamically attach the env_id
    env.env_id = env_id
    env.env_spec = env_spec
    env.entry_point = env_spec.entry_point
    env.prompt_profile = env_spec.prompt_profile
    env.action_parser = env_spec.resolve_action_parser()
    env.reward_mode = env_spec.reward_mode
    env.obs_mode = env_spec.obs_mode

    # wrap the environment
    if env_spec.default_wrappers is not None and len(env_spec.default_wrappers) > 0:
        for wrapper in env_spec.default_wrappers:
            env = wrapper(env)

    return env


def get_env_spec(env_or_env_id: Union[str, Any]) -> EnvSpec:
    if isinstance(env_or_env_id, str):
        if env_or_env_id not in ENV_REGISTRY:
            raise ValueError(f"Environment {env_or_env_id} not found in registry.")
        return ENV_REGISTRY[env_or_env_id]

    current = env_or_env_id
    while current is not None:
        env_spec = getattr(current, "env_spec", None)
        if isinstance(env_spec, EnvSpec):
            return env_spec
        current = getattr(current, "env", None)

    env_id = getattr(env_or_env_id, "env_id", None)
    if isinstance(env_id, str) and env_id in ENV_REGISTRY:
        return ENV_REGISTRY[env_id]
    raise ValueError("Could not resolve env spec from the provided environment.")


def get_prompt_profile(env_or_env_id: Union[str, Any]) -> Optional[PromptProfile]:
    return get_env_spec(env_or_env_id).prompt_profile


def get_action_parser(env_or_env_id: Union[str, Any]) -> Optional[Callable[..., List[str]]]:
    return get_env_spec(env_or_env_id).resolve_action_parser()
