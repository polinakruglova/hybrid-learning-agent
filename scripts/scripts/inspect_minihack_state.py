from __future__ import annotations

import gymnasium as gym
import minihack  # noqa: F401

from src.environments.minihack import MiniHackStateParser


ENVIRONMENT_NAME = "MiniHack-Room-5x5-v0"


def main() -> None:
    environment = gym.make(
        ENVIRONMENT_NAME,
        observation_keys=(
            "glyphs",
            "chars",
            "colors",
            "blstats",
            "message",
        ),
    )

    parser = MiniHackStateParser()

    try:
        observation, info = environment.reset(
            seed=42
        )

        state = parser.parse(
            observation,
            step=0,
            info=info,
        )

        print("=" * 70)
        print("MINIHACK WORLD STATE")
        print("=" * 70)

        for key, value in state.items():
            print(f"{key:22}: {value}")

        print("=" * 70)

    finally:
        environment.close()


if __name__ == "__main__":
    main()