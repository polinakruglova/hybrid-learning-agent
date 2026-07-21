from __future__ import annotations

import inspect

from src.agents import HybridAgent, RuleAgent
from src.environments import KeyDoorEnvironment
from src.environments.key_door import StepResult
from src.learning import LearningController
from src.memory import ExperienceBuffer
from src.predicates import PredicateGenerator
from src.rl import QTablePolicy
from src.rules import RuleMiner, RuleStore
from src.state import State
from src.training import (
    EpisodeResult,
    TrainingHistory,
    TrainingRunner,
)


OBJECTS = [
    State,
    StepResult,
    KeyDoorEnvironment,
    PredicateGenerator,
    RuleStore,
    RuleMiner,
    ExperienceBuffer,
    RuleAgent,
    QTablePolicy,
    HybridAgent,
    LearningController,
    EpisodeResult,
    TrainingHistory,
    TrainingRunner,
]


def print_class_api(
    class_object: type,
) -> None:
    """
    Выводит публичный API класса:

    - сигнатуру конструктора;
    - публичные методы;
    - публичные атрибуты и свойства.
    """

    print()
    print("=" * 80)
    print(
        class_object.__module__,
        class_object.__name__,
    )
    print("=" * 80)

    print("Constructor:")

    try:
        print(inspect.signature(class_object))
    except (TypeError, ValueError):
        print("Constructor signature unavailable")

    print()
    print("Public methods:")

    method_found = False

    for name, member in inspect.getmembers(
        class_object
    ):
        if name.startswith("_"):
            continue

        if not callable(member):
            continue

        method_found = True

        try:
            signature = inspect.signature(member)
        except (TypeError, ValueError):
            signature = "(signature unavailable)"

        print(f"  {name}{signature}")

    if not method_found:
        print("  no public methods")

    print()
    print("Public class attributes:")

    attribute_found = False

    for name, value in vars(
        class_object
    ).items():
        if name.startswith("_"):
            continue

        if callable(value):
            continue

        attribute_found = True

        print(f"  {name} = {value!r}")

    if not attribute_found:
        print("  no public class attributes")


def print_instance_data(
    title: str,
    instance: object,
) -> None:
    """
    Печатает внутренние данные объекта.

    Поддерживает обычные классы, dataclass
    и классы со slots.
    """

    print()
    print(title)
    print("-" * 80)
    print(f"Type: {type(instance)}")
    print(f"Representation: {instance!r}")

    try:
        instance_vars = vars(instance)
    except TypeError:
        instance_vars = None

    if instance_vars is not None:
        print("vars():")

        if instance_vars:
            for name, value in instance_vars.items():
                print(f"  {name} = {value!r}")
        else:
            print("  empty")

    slots = getattr(
        type(instance),
        "__slots__",
        None,
    )

    if slots is not None:
        print("__slots__ values:")

        if isinstance(slots, str):
            slot_names = [slots]
        else:
            slot_names = list(slots)

        for name in slot_names:
            if hasattr(instance, name):
                value = getattr(instance, name)
                print(f"  {name} = {value!r}")


def print_named_attributes(
    instance: object,
    attribute_names: list[str],
) -> None:
    """
    Проверяет наличие ожидаемых полей объекта.
    """

    print()
    print("Expected attributes:")

    for name in attribute_names:
        if hasattr(instance, name):
            value = getattr(instance, name)

            print(
                f"  {name:<20} "
                f"present: {value!r}"
            )
        else:
            print(
                f"  {name:<20} "
                "missing"
            )


def inspect_environment_step() -> None:
    """
    Выполняет реальный шаг среды и показывает,
    как устроены State и StepResult.
    """

    print()
    print("=" * 80)
    print("REAL ENVIRONMENT STEP")
    print("=" * 80)

    environment = KeyDoorEnvironment(
        width=7,
        height=7,
        max_steps=100,
    )

    initial_state = environment.reset()

    print_instance_data(
        title="Initial State",
        instance=initial_state,
    )

    print_named_attributes(
        instance=initial_state,
        attribute_names=[
            "agent_x",
            "agent_y",
            "agent_direction",
            "agent_dir",
            "key_x",
            "key_y",
            "door_x",
            "door_y",
            "goal_x",
            "goal_y",
            "has_key",
            "door_open",
            "door_locked",
        ],
    )

    result = environment.step(
        KeyDoorEnvironment.ACTION_WAIT
    )

    print_instance_data(
        title="StepResult after ACTION_WAIT",
        instance=result,
    )

    print_named_attributes(
        instance=result,
        attribute_names=[
            "state",
            "next_state",
            "reward",
            "done",
            "success",
            "event",
            "info",
        ],
    )

    print()
    print("Environment properties after step:")
    print(
        f"  step_count = "
        f"{environment.step_count!r}"
    )
    print(
        f"  done = "
        f"{environment.done!r}"
    )
    print(
        f"  success = "
        f"{environment.success!r}"
    )

    print()
    print("Rendered environment:")
    print(environment.render())


def inspect_state_serialization() -> None:
    """
    Проверяет, какие способы преобразования
    состояния доступны.
    """

    print()
    print("=" * 80)
    print("STATE SERIALIZATION")
    print("=" * 80)

    environment = KeyDoorEnvironment()
    state = environment.reset()

    methods_to_check = [
        "to_dict",
        "as_dict",
        "items",
        "get",
        "copy",
    ]

    for method_name in methods_to_check:
        method = getattr(
            state,
            method_name,
            None,
        )

        if method is None:
            print(
                f"{method_name:<15} missing"
            )
            continue

        print(
            f"{method_name:<15} "
            f"present: {method!r}"
        )

        if method_name in {
            "to_dict",
            "as_dict",
        }:
            try:
                result = method()
                print(f"  result: {result!r}")
            except Exception as error:
                print(
                    "  call failed: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )


def main() -> None:
    print("=" * 80)
    print("BENCHMARK API INSPECTION")
    print("=" * 80)

    for class_object in OBJECTS:
        print_class_api(class_object)

    inspect_environment_step()
    inspect_state_serialization()

    print()
    print("=" * 80)
    print("INSPECTION FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()