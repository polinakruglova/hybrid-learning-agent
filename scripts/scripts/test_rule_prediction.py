import gymnasium as gym
import minihack

from src.environments.minihack import MiniHackStateParser

from src.transformations.change_detector import ChangeDetector
from src.transformations.transformation_tracker import TransformationTracker
from src.transformations.transition_memory import TransitionMemory
from src.transformations.effect_grouper import EffectGrouper
from src.transformations.condition_discovery import ConditionDiscovery
from src.transformations.transformation_rule import TransformationRule
from src.transformations.rule_predictor import RulePredictor


# ============================================================
# НАСТРОЙКИ
# ============================================================

LEARNING_STEPS = 1000
TEST_STEPS = 100

MIN_COUNT = 3
MIN_CONFIDENCE = 0.50

MIN_SUPPORT = 0.20
MIN_SCORE = 0.30

MAX_SHOWN_ERRORS = 10


# ============================================================
# STATE -> DICT
# ============================================================

def state_to_dict(state):
    """
    Приводит состояние к обычному словарю.
    """

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
    """
    Превращает сигнатуру EffectGroup
    в effects для TransformationRule.
    """

    effects = {}

    if not signature:
        return effects

    if signature == ("NO_CHANGE",):
        return effects

    # --------------------------------------------------------
    # Один эффект:
    #
    # ("x", "delta", 1)
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Несколько эффектов:
    #
    # (
    #     ("x", "delta", 1),
    #     ("time", "delta", 1),
    # )
    # --------------------------------------------------------

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
# СОЗДАНИЕ УСЛОВНЫХ ПРАВИЛ
# ============================================================

def build_conditional_rules(
    all_groups,
    condition_discovery,
):
    """
    Для каждого action:

        берём разные EffectGroup
                ↓
        сравниваем StateBefore
                ↓
        ищем ConditionCandidate
                ↓
        создаём TransformationRule
    """

    rules = []

    for action, groups in all_groups.items():

        if len(groups) < 2:
            continue

        # Самая частая группа считается
        # основным результатом действия.
        main_group = groups[0]

        for other_group in groups[1:]:

            if other_group.count < MIN_COUNT:
                continue

            candidates = (
                condition_discovery.compare_groups(
                    main_group,
                    other_group,
                    min_support=MIN_SUPPORT,
                    min_score=MIN_SCORE,
                )
            )

            if not candidates:
                continue

            # Пока используем самый сильный
            # найденный condition.
            best = candidates[0]

            # ------------------------------------------------
            # Выбираем ту группу, для которой
            # condition характернее.
            # ------------------------------------------------

            if (
                best.probability_a
                >= best.probability_b
            ):
                target_group = main_group
                confidence = best.probability_a

            else:
                target_group = other_group
                confidence = best.probability_b

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

                observations=target_group.count,

                confidence=confidence,
            )

            rules.append(rule)

    return rules


# ============================================================
# СРАВНЕНИЕ ПРОГНОЗА И РЕАЛЬНОСТИ
# ============================================================

def compare_states(
    predicted,
    actual,
):
    """
    Сравнивает predicted_state
    с реальным actual_state.

    Возвращает:

        checked
        correct
        differences
    """

    checked = 0
    correct = 0

    differences = []

    for key, predicted_value in predicted.items():

        if key not in actual:
            continue

        actual_value = actual[key]

        checked += 1

        if predicted_value == actual_value:
            correct += 1

        else:
            differences.append(
                (
                    key,
                    predicted_value,
                    actual_value,
                )
            )

    return (
        checked,
        correct,
        differences,
    )


# ============================================================
# КРАСИВЫЙ ВЫВОД STATE
# ============================================================

def print_state(
    title,
    state,
):
    print()
    print(title)

    for key, value in state.items():

        print(
            f"  {key}: "
            f"{value!r}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. СОЗДАЁМ MINIHACK
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
    # 2. СОЗДАЁМ МОДУЛИ
    # ========================================================

    parser = MiniHackStateParser()

    detector = ChangeDetector()

    tracker = TransformationTracker()

    memory = TransitionMemory()

    grouper = EffectGrouper()

    condition_discovery = (
        ConditionDiscovery()
    )

    predictor = RulePredictor()

    # ========================================================
    # 3. ИССЛЕДОВАНИЕ МИРА
    #
    # Здесь агент собирает опыт.
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
    print("LEARNING WORLD RULES")
    print("=" * 70)

    print(
        f"Learning steps: "
        f"{LEARNING_STEPS}"
    )

    for step_number in range(
        LEARNING_STEPS
    ):

        # ----------------------------------------------------
        # Выбираем действие.
        # ----------------------------------------------------

        action = env.action_space.sample()

        # ----------------------------------------------------
        # Выполняем действие в реальной среде.
        # ----------------------------------------------------

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        current_state = parser.parse(
            observation
        )

        current_data = state_to_dict(
            current_state
        )

        # ----------------------------------------------------
        # Если эпизод закончился —
        # reset не считаем обычным правилом мира.
        # ----------------------------------------------------

        if terminated or truncated:

            observation, info = env.reset()

            previous_state = parser.parse(
                observation
            )

            previous_data = state_to_dict(
                previous_state
            )

            continue

        # ----------------------------------------------------
        # Находим изменения.
        # ----------------------------------------------------

        changes = detector.detect(
            previous_data,
            current_data,
        )

        # ----------------------------------------------------
        # Общая статистика:
        #
        # ACTION -> EFFECT
        # ----------------------------------------------------

        tracker.observe(
            action,
            changes,
        )

        # ----------------------------------------------------
        # Полный опыт:
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
        # Небольшой прогресс.
        # ----------------------------------------------------

        if step_number % 100 == 0:

            print(
                f"step={step_number}"
            )

        previous_data = current_data

    # ========================================================
    # 4. СТРОИМ БАЗОВЫЕ ПРАВИЛА
    # ========================================================

    base_rules = tracker.build_rules(
        min_count=MIN_COUNT,
        min_confidence=MIN_CONFIDENCE,
    )

    # ========================================================
    # 5. ГРУППИРУЕМ ЭФФЕКТЫ
    # ========================================================

    all_groups = grouper.group_all(
        memory.transitions
    )

    # ========================================================
    # 6. ИЩЕМ УСЛОВНЫЕ ПРАВИЛА
    # ========================================================

    conditional_rules = (
        build_conditional_rules(
            all_groups,
            condition_discovery,
        )
    )

    # ========================================================
    # 7. ВСЕ ПРАВИЛА
    # ========================================================

    all_rules = (
        base_rules
        + conditional_rules
    )

    print()
    print("=" * 70)
    print("DISCOVERED RULES")
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
        f"{len(all_rules)}"
    )

    # ========================================================
    # 8. НОВЫЙ ЭПИЗОД
    #
    # Здесь правила больше НЕ обучаются.
    #
    # Только:
    #
    # state + action
    #       ↓
    # RulePredictor
    #       ↓
    # predicted_state
    #       ↓
    # env.step(action)
    #       ↓
    # actual_state
    #       ↓
    # сравнение
    # ========================================================

    observation, info = env.reset(
        seed=123
    )

    current_state = parser.parse(
        observation
    )

    current_data = state_to_dict(
        current_state
    )

    # ========================================================
    # СТАТИСТИКА
    # ========================================================

    total_predictions = 0

    exact_predictions = 0

    total_features = 0

    correct_features = 0

    shown_errors = 0

    print()
    print("=" * 70)
    print("RULE PREDICTION TEST")
    print("=" * 70)

    print(
        f"Test steps: "
        f"{TEST_STEPS}"
    )

    # ========================================================
    # 9. ТЕСТИРОВАНИЕ
    # ========================================================

    for step_number in range(
        TEST_STEPS
    ):

        action = env.action_space.sample()

        # ----------------------------------------------------
        # СНАЧАЛА прогнозируем.
        #
        # Важно:
        # env.step ещё НЕ вызван.
        # ----------------------------------------------------

        matching_rules = (
            predictor.find_matching_rules(
                state=current_data,
                action=action,
                rules=all_rules,
            )
        )

        predicted_state = (
            predictor.predict(
                state=current_data,
                action=action,
                rules=all_rules,
            )
        )

        # ----------------------------------------------------
        # Теперь выполняем действие
        # в настоящем MiniHack.
        # ----------------------------------------------------

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        next_state = parser.parse(
            observation
        )

        actual_state = state_to_dict(
            next_state
        )

        # ----------------------------------------------------
        # Если для action были подходящие rules,
        # оцениваем прогноз.
        # ----------------------------------------------------

        if matching_rules:

            (
                checked,
                correct,
                differences,
            ) = compare_states(
                predicted_state,
                actual_state,
            )

            total_predictions += 1

            total_features += checked

            correct_features += correct

            exact = (
                len(differences) == 0
            )

            if exact:

                exact_predictions += 1

            # =================================================
            # ДИАГНОСТИКА ОШИБОК
            #
            # Показываем первые 10
            # НЕТОЧНЫХ прогнозов.
            # =================================================

            if differences:

                if (
                    shown_errors
                    < MAX_SHOWN_ERRORS
                ):

                    shown_errors += 1

                    print()
                    print("=" * 70)

                    print(
                        f"PREDICTION ERROR "
                        f"#{shown_errors}"
                    )

                    print("=" * 70)

                    # -----------------------------------------
                    # STATE BEFORE
                    # -----------------------------------------

                    print_state(
                        "STATE BEFORE:",
                        current_data,
                    )

                    # -----------------------------------------
                    # ACTION
                    # -----------------------------------------

                    print()
                    print(
                        f"ACTION: {action}"
                    )

                    # -----------------------------------------
                    # MATCHING RULES
                    # -----------------------------------------

                    print()
                    print(
                        "MATCHING RULES: "
                        f"{len(matching_rules)}"
                    )

                    for (
                        rule_number,
                        rule,
                    ) in enumerate(
                        matching_rules,
                        start=1,
                    ):

                        print()
                        print(
                            f"RULE "
                            f"{rule_number}:"
                        )

                        print(
                            "  conditions = "
                            f"{rule.conditions}"
                        )

                        print(
                            "  effects = "
                            f"{rule.effects}"
                        )

                        print(
                            "  confidence = "
                            f"{rule.confidence:.2f}"
                        )

                        print(
                            "  observations = "
                            f"{rule.observations}"
                        )

                    # -----------------------------------------
                    # PREDICTED
                    # -----------------------------------------

                    print_state(
                        "PREDICTED STATE:",
                        predicted_state,
                    )

                    # -----------------------------------------
                    # ACTUAL
                    # -----------------------------------------

                    print_state(
                        "ACTUAL STATE:",
                        actual_state,
                    )

                    # -----------------------------------------
                    # DIFFERENCES
                    # -----------------------------------------

                    print()
                    print("DIFFERENCES:")

                    for (
                        feature,
                        predicted_value,
                        actual_value,
                    ) in differences:

                        print(
                            f"  {feature}: "
                            f"predicted="
                            f"{predicted_value!r}, "
                            f"actual="
                            f"{actual_value!r}"
                        )

                    # -----------------------------------------
                    # FEATURE ACCURACY ЭТОГО ПРОГНОЗА
                    # -----------------------------------------

                    if checked:

                        step_accuracy = (
                            correct
                            / checked
                        )

                        print()
                        print(
                            "STEP FEATURE ACCURACY: "
                            f"{step_accuracy:.2%}"
                        )

        # ----------------------------------------------------
        # Следующее состояние становится текущим.
        # ----------------------------------------------------

        current_data = actual_state

        # ----------------------------------------------------
        # Если эпизод закончился —
        # начинаем новый.
        # ----------------------------------------------------

        if terminated or truncated:

            observation, info = env.reset()

            current_state = parser.parse(
                observation
            )

            current_data = state_to_dict(
                current_state
            )

    # ========================================================
    # 10. ИТОГОВАЯ СТАТИСТИКА
    # ========================================================

    print()
    print("=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)

    print(
        f"Проверено прогнозов: "
        f"{total_predictions}"
    )

    print(
        f"Полностью точных прогнозов: "
        f"{exact_predictions}"
    )

    # --------------------------------------------------------
    # Exact accuracy
    # --------------------------------------------------------

    if total_predictions:

        exact_accuracy = (
            exact_predictions
            / total_predictions
        )

        print(
            f"Exact accuracy: "
            f"{exact_accuracy:.2%}"
        )

    else:

        print(
            "Exact accuracy: "
            "нет данных"
        )

    # --------------------------------------------------------
    # Feature accuracy
    # --------------------------------------------------------

    if total_features:

        feature_accuracy = (
            correct_features
            / total_features
        )

        print(
            f"Feature accuracy: "
            f"{feature_accuracy:.2%}"
        )

    else:

        print(
            "Feature accuracy: "
            "нет данных"
        )

    print()
    print(
        f"Показано ошибок: "
        f"{shown_errors}"
    )

    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()