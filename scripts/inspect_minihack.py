from __future__ import annotations

from typing import Any

import gymnasium as gym
import minihack  # noqa: F401
import numpy as np


ENVIRONMENT_NAME = "MiniHack-Room-5x5-v0"
SEED = 42

OBSERVATION_KEYS = (
    "glyphs",
    "chars",
    "colors",
    "blstats",
    "message",
)


def decode_message(value: Any) -> str:
    """
    Преобразует массив чисел из NLE в обычную строку.
    """
    if not isinstance(value, np.ndarray):
        return ""

    raw_bytes = bytes(
        int(item)
        for item in value
        if int(item) != 0
    )

    return raw_bytes.decode(
        "utf-8",
        errors="replace",
    ).strip()


def describe_array(
    name: str,
    value: np.ndarray,
) -> None:
    """
    Печатает краткую информацию о массиве наблюдений.
    """
    print(f"\n{name}")
    print("-" * 70)
    print(f"shape: {value.shape}")
    print(f"dtype: {value.dtype}")
    print(f"size:  {value.size}")

    if value.size == 0:
        return

    flattened = value.reshape(-1)

    print(f"preview: {flattened[:20]}")

    if np.issubdtype(value.dtype, np.number):
        print(f"minimum: {value.min()}")
        print(f"maximum: {value.max()}")


def describe_observation(
    observation: dict[str, Any],
) -> None:
    """
    Выводит все поля наблюдения MiniHack.
    """
    print("\nOBSERVATION")
    print("=" * 70)

    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            describe_array(key, value)
        else:
            print(f"\n{key}")
            print("-" * 70)
            print(value)

    message = decode_message(
        observation.get("message")
    )

    print("\nDecoded message")
    print("-" * 70)
    print(message or "<empty>")


def main() -> None:
    print("=" * 70)
    print("MINIHACK INITIAL CHECK")
    print("=" * 70)
    print(f"Environment: {ENVIRONMENT_NAME}")
    print(f"Seed:        {SEED}")

    environment = gym.make(
        ENVIRONMENT_NAME,
        observation_keys=OBSERVATION_KEYS,
    )

    try:
        observation, info = environment.reset(
            seed=SEED
        )

        if not isinstance(observation, dict):
            raise TypeError(
                "Expected MiniHack observation to be a dictionary, "
                f"received {type(observation).__name__}."
            )

        print("\nEnvironment successfully created.")
        print(f"Action space:      {environment.action_space}")
        print(f"Observation space: {environment.observation_space}")
        print(f"Info keys:         {list(info.keys())}")

        describe_observation(observation)

        action = environment.action_space.sample()

        print("\nRANDOM STEP")
        print("=" * 70)
        print(f"Selected action: {action}")

        (
            next_observation,
            reward,
            terminated,
            truncated,
            next_info,
        ) = environment.step(action)

        print(f"Reward:     {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated:  {truncated}")
        print(f"Info keys:  {list(next_info.keys())}")

        if isinstance(next_observation, dict):
            next_message = decode_message(
                next_observation.get("message")
            )

            print(
                "Message:",
                next_message or "<empty>",
            )

        print("\nMiniHack check completed successfully.")

    finally:
        environment.close()


if __name__ == "__main__":
    main()