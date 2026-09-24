import gymnasium as gym
import minihack

from src.environments.minihack import MiniHackStateParser

from src.transformations.change_detector import ChangeDetector
from src.transformations.transformation_tracker import TransformationTracker
from src.transformations.transition_memory import TransitionMemory
from src.transformations.effect_grouper import EffectGrouper
from src.transformations.condition_discovery import ConditionDiscovery
from src.transformations.transformation_rule import TransformationRule


# ============================================================
# НАСТРОЙКИ
# ============================================================

STEPS = 1000

MIN_COUNT = 3
MIN_CONFIDENCE = 0.50

MIN_SUPPORT = 0.20
MIN_SCORE = 0.30


# ============================================================
# ПРЕОБРАЗОВАНИЕ СОСТОЯНИЯ В DICT
# ============================================================

def state_to_dict(state):

    if isinstance(state, dict):
        return dict(state)

    if hasattr(state, "_asdict"):
        return dict(state._asdict())

    if hasattr(state, "items"):
        return dict(state.items())

    if hasattr(state, "__dict__"):
        return dict(vars(state))

    return dict(state)


# ============================================================
# EFFECT SIGNATURE -> EFFECTS
# ============================================================

def signature_to_effects(signature):

    effects = {}

    if not signature:
        return effects

    if signature == ("NO_CHANGE",):
        return effects

    # одиночная сигнатура:
    #
    # ("x", "delta", 1)

    if (
        isinstance(signature, tuple)
        and len(signature) == 3
        and isinstance(signature[0], str)
    ):

        feature, effect_type, value = signature

        effects[feature] = (
            effect_type,
            value,
        )

        return effects

    # набор эффектов:
    #
    # (
    #     ("x", "delta", 1),
    #     ("time", "delta", 1),
    # )

    if isinstance(signature, tuple):

        for item in signature:

            if not isinstance(item, tuple):
                continue

            if len(item) != 3:
                continue

            feature, effect_type, value = item

            effects[feature] = (
                effect_type,
                value,
            )

    return effects


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. MINIHACK
    # ========================================================

    env = gym.make(
        "MiniHack-Room-5x5-v0",
        observation_keys=(
            "blstats",
            "message",
            "chars",
        ),
    )

    # ========================================================
    # 2. МОДУЛИ
    # ========================================================

    parser = MiniHackStateParser()

    detector = ChangeDetector()

    tracker = TransformationTracker()

    memory = TransitionMemory()

    grouper = EffectGrouper()

    condition_discovery = ConditionDiscovery()

    # ========================================================
    # 3. НАЧАЛЬНОЕ СОСТОЯНИЕ
    # ========================================================

    observation, info = env.reset(
        seed=42
    )

    previous_state = parser.parse(
        observation
    )

    previous_data = state_to_dict(
        previous_state
    )

    print()
    print("=" * 70)
    print("START WORLD EXPLORATION")
    print("=" * 70)

    print(
        f"Исследуем MiniHack: "
        f"{STEPS} шагов"
    )

    # ========================================================
    # 4. ИССЛЕДУЕМ МИР
    # ========================================================

    for step_number in range(STEPS):

        # ----------------------------------------------------
        # ACTION
        # ----------------------------------------------------

        action = env.action_space.sample()

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        # ----------------------------------------------------
        # STATE AFTER
        # ----------------------------------------------------

        current_state = parser.parse(
            observation
        )

        current_data = state_to_dict(
            current_state
        )

        # ----------------------------------------------------
        # Не используем terminal/reset как обычный переход
        # мира.
        # ----------------------------------------------------

        if terminated or truncated:

            print(
                f"EPISODE FINISHED "
                f"step={step_number}, "
                f"reward={reward}"
            )

            observation, info = env.reset()

            previous_state = parser.parse(
                observation
            )

            previous_data = state_to_dict(
                previous_state
            )

            continue

        # ----------------------------------------------------
        # CHANGE DETECTOR
        # ----------------------------------------------------

        changes = detector.detect(
            previous_data,
            current_data,
        )

        # ----------------------------------------------------
        # TRANSFORMATION TRACKER
        #
        # ACTION -> повторяющийся EFFECT
        # ----------------------------------------------------

        tracker.observe(
            action,
            changes,
        )

        # ----------------------------------------------------
        # TRANSITION MEMORY
        #
        # Сохраняем полный переход:
        #
        # StateBefore
        #      +
        #    Action
        #      ↓
        # StateAfter
        # ----------------------------------------------------

        memory.add(
            before=previous_data,
            action=action,
            after=current_data,
            changes=changes,
        )

        # ----------------------------------------------------
        # ПРОГРЕСС
        # ----------------------------------------------------

        if step_number % 100 == 0:

            print(
                f"step={step_number}, "
                f"action={action}, "
                f"changes={len(changes)}"
            )

        previous_data = current_data

    # ========================================================
    # 5. ПОВТОРЯЮЩИЕСЯ ТРАНСФОРМАЦИИ
    # ========================================================

    print()
    print("=" * 70)
    print("REPEATED TRANSFORMATIONS")
    print("=" * 70)

    repeated = tracker.get_repeated(
        min_count=MIN_COUNT
    )

    if not repeated:

        print(
            "Повторяющиеся трансформации "
            "не найдены."
        )

    else:

        for signature, count in repeated:

            print(
                f"{signature} "
                f"-> observed {count} times"
            )

    # ========================================================
    # 6. БАЗОВЫЕ RULES
    # ========================================================

    print()
    print("=" * 70)
    print("BASE TRANSFORMATION RULES")
    print("=" * 70)

    base_rules = tracker.build_rules(
        min_count=MIN_COUNT,
        min_confidence=MIN_CONFIDENCE,
    )

    if not base_rules:

        print(
            "Базовые правила не найдены."
        )

    else:

        for rule in base_rules:

            print(rule)

    print()
    print(
        f"Базовых правил: "
        f"{len(base_rules)}"
    )

    # ========================================================
    # 7. EFFECT GROUPS
    #
    # Теперь используем накопленные Transition.
    # ========================================================

    print()
    print("=" * 70)
    print("EFFECT GROUPS")
    print("=" * 70)

    all_groups = grouper.group_all(
        memory.transitions
    )

    total_groups = 0

    for action, groups in all_groups.items():

        print()
        print("-" * 70)
        print(f"ACTION {action}")
        print("-" * 70)

        for group_number, group in enumerate(
            groups,
            start=1,
        ):

            total_groups += 1

            print(
                f"GROUP {group_number}"
            )

            print(
                f"observations = "
                f"{group.count}"
            )

            print(
                f"effect = "
                f"{group.effect_signature}"
            )

    print()
    print(
        f"Всего групп эффектов: "
        f"{total_groups}"
    )

    # ========================================================
    # 8. CONDITION DISCOVERY
    #
    # Одинаковый action
    #        ↓
    # разные EffectGroup
    #        ↓
    # сравниваем StateBefore
    #        ↓
    # ConditionCandidate
    # ========================================================

    print()
    print("=" * 70)
    print("CONDITION DISCOVERY")
    print("=" * 70)

    discovered_comparisons = []

    for action, groups in all_groups.items():

        if len(groups) < 2:
            continue

        print()
        print("#" * 70)
        print(f"ACTION {action}")
        print("#" * 70)

        # Самая большая группа —
        # основной результат действия.
        main_group = groups[0]

        print()
        print("MAIN EFFECT:")

        print(
            main_group.effect_signature
        )

        print(
            f"observations = "
            f"{main_group.count}"
        )

        # ----------------------------------------------------
        # Основной результат сравниваем
        # со всеми альтернативными.
        # ----------------------------------------------------

        for group_number, other_group in enumerate(
            groups[1:],
            start=2,
        ):

            if other_group.count < MIN_COUNT:
                continue

            print()
            print("-" * 70)

            print(
                f"COMPARE WITH GROUP "
                f"{group_number}"
            )

            print(
                "ALTERNATIVE EFFECT:"
            )

            print(
                other_group.effect_signature
            )

            print(
                f"observations = "
                f"{other_group.count}"
            )

            candidates = (
                condition_discovery.compare_groups(
                    main_group,
                    other_group,
                    min_support=MIN_SUPPORT,
                    min_score=MIN_SCORE,
                )
            )

            if not candidates:

                print(
                    "Отличающих условий "
                    "не найдено."
                )

                continue

            print()
            print("POSSIBLE CONDITIONS:")

            for candidate in candidates[:10]:

                print(
                    "  ",
                    candidate,
                )

            discovered_comparisons.append(
                (
                    action,
                    main_group,
                    other_group,
                    candidates,
                )
            )

    # ========================================================
    # 9. CONDITIONAL TRANSFORMATION RULES
    # ========================================================

    print()
    print("=" * 70)
    print("CONDITIONAL TRANSFORMATION RULES")
    print("=" * 70)

    conditional_rules = []

    for (
        action,
        group_a,
        group_b,
        candidates,
    ) in discovered_comparisons:

        if not candidates:
            continue

        # Пока берём лучший найденный condition.
        best = candidates[0]

        # ----------------------------------------------------
        # Определяем, для какой группы
        # это условие характернее.
        # ----------------------------------------------------

        if (
            best.probability_a
            >= best.probability_b
        ):

            target_group = group_a

            confidence = (
                best.probability_a
            )

        else:

            target_group = group_b

            confidence = (
                best.probability_b
            )

        effects = signature_to_effects(
            target_group.effect_signature
        )

        rule = TransformationRule(
            action=action,

            conditions={
                best.feature:
                    best.value
            },

            effects=effects,

            observations=(
                target_group.count
            ),

            confidence=confidence,
        )

        conditional_rules.append(
            rule
        )

    if not conditional_rules:

        print(
            "Условные правила пока "
            "не найдены."
        )

    else:

        for rule in conditional_rules:

            print(rule)

    # ========================================================
    # 10. SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Базовых правил: "
        f"{len(base_rules)}"
    )

    print(
        f"Условных правил: "
        f"{len(conditional_rules)}"
    )

    print(
        f"Всего правил: "
        f"{len(base_rules) + len(conditional_rules)}"
    )

    print("=" * 70)

    env.close()
    return base_rules + conditional_rules

if __name__ == "__main__":
    main()