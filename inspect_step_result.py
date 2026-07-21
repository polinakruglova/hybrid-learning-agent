from __future__ import annotations

from src.environments import KeyDoorEnvironment


def print_object_data(title: str, obj: object) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print("Type:")
    print(type(obj))

    print()
    print("Representation:")
    print(repr(obj))

    print()
    print("vars():")

    try:
        data = vars(obj)
    except TypeError:
        data = None

    if data is None:
        print("vars() unavailable")
    elif not data:
        print("empty")
    else:
        for name, value in data.items():
            print(f"{name} = {value!r}")

    print()
    print("Public attributes:")

    for name in dir(obj):
        if name.startswith("_"):
            continue

        try:
            value = getattr(obj, name)
        except Exception as error:
            print(f"{name}: error: {error}")
            continue

        if callable(value):
            continue

        print(f"{name} = {value!r}")


def main() -> None:
    environment = KeyDoorEnvironment(
        width=7,
        height=7,
        max_steps=100,
    )

    state = environment.reset()

    print_object_data(
        title="INITIAL STATE",
        obj=state,
    )

    step_result = environment.step(
        KeyDoorEnvironment.ACTION_WAIT
    )

    print_object_data(
        title="STEP RESULT",
        obj=step_result,
    )

    print()
    print("=" * 80)
    print("ENVIRONMENT AFTER STEP")
    print("=" * 80)

    print(f"step_count = {environment.step_count}")
    print(f"done = {environment.done}")
    print(f"success = {environment.success}")

    print()
    print(environment.render())


if __name__ == "__main__":
    main()