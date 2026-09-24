from __future__ import annotations

import runpy
from pathlib import Path
from statistics import mean
from scripts.scripts.discover_transformations import main as discover_rules

import gymnasium as gym

from src.agents.causal_agent import CausalAgent
from src.environments.minihack.state_parser import MiniHackStateParser
from src.transformations.causal_chain_finder import CausalChainFinder
from src.transformations.transformation_rule import TransformationRule


# ============================================================
# НАСТРОЙКИ
# ============================================================

DISCOVERY_SCRIPT = Path(
    "scripts/scripts/discover_transformations.py"
)

# Если discover_transformations.py не позволит автоматически
# определить среду, используем эту.
#
# Если у тебя там другая MiniHack-среда,
# потом просто поменяем эту строку.
FALLBACK_ENV_ID = "MiniHack-Room-5x5-v0"


TRAINING_EPISODES = 20

MAX_STEPS_PER_EPISODE = 200

# Доля случайных действий агента.
#
# 0.15 =
# 15% действий случайные
# 85% агент пытается использовать правила.
EPSILON = 0.15

SEED = 42


# ============================================================
# ЗАПУСК ОБУЧЕНИЯ ПРАВИЛ
# ============================================================

def run_rule_discovery():
    """
    Запускает обучение правил в MiniHack
    и получает найденные TransformationRule.
    """

    print()
    print("=" * 70)
    print("PHASE 1")
    print("ОБУЧЕНИЕ ПРАВИЛ")
    print("=" * 70)

    rules = discover_rules()

    if rules is None:
        return []

    return list(rules)
# ============================================================
# ИЗВЛЕЧЕНИЕ ПРАВИЛ
# ============================================================

def extract_rules(
    namespace,
):
    """
    Ищет все TransformationRule,
    которые были созданы внутри discover_transformations.py.

    Мы специально не привязываемся к имени переменной:

        rules
        all_rules
        conditional_rules
        transformation_rules

    и т.д.

    Поэтому текущий discover_transformations.py
    можно оставить почти без изменений.
    """

    found_rules = []

    seen_ids = set()

    for name, value in namespace.items():

        # --------------------------------------------------------
        # Один отдельный TransformationRule
        # --------------------------------------------------------

        if isinstance(
            value,
            TransformationRule,
        ):

            if id(value) not in seen_ids:

                found_rules.append(
                    value
                )

                seen_ids.add(
                    id(value)
                )

            continue

        # --------------------------------------------------------
        # Список / tuple / set правил
        # --------------------------------------------------------

        if isinstance(
            value,
            (list, tuple, set),
        ):

            for item in value:

                if not isinstance(
                    item,
                    TransformationRule,
                ):
                    continue

                if id(item) in seen_ids:
                    continue

                found_rules.append(
                    item
                )

                seen_ids.add(
                    id(item)
                )

        # --------------------------------------------------------
        # Словарь
        # --------------------------------------------------------

        if isinstance(
            value,
            dict,
        ):

            for item in value.values():

                if not isinstance(
                    item,
                    TransformationRule,
                ):
                    continue

                if id(item) in seen_ids:
                    continue

                found_rules.append(
                    item
                )

                seen_ids.add(
                    id(item)
                )

    return found_rules


# ============================================================
# ПОПЫТКА ОПРЕДЕЛИТЬ MINI HACK ENVIRONMENT
# ============================================================

def detect_environment_id(
    namespace,
):
    """
    Пытается определить MiniHack environment,
    который использовал discover_transformations.py.
    """

    # --------------------------------------------------------
    # Сначала ищем готовый env
    # --------------------------------------------------------

    for value in namespace.values():

        spec = getattr(
            value,
            "spec",
            None,
        )

        if spec is None:
            continue

        env_id = getattr(
            spec,
            "id",
            None,
        )

        if isinstance(
            env_id,
            str,
        ):

            if "MiniHack" in env_id:

                return env_id

    # --------------------------------------------------------
    # Затем ищем строковые переменные
    # --------------------------------------------------------

    possible_names = (
        "ENV_ID",
        "ENV_NAME",
        "ENVIRONMENT_ID",
        "environment_id",
        "env_id",
        "env_name",
    )

    for name in possible_names:

        value = namespace.get(
            name
        )

        if (
            isinstance(value, str)
            and "MiniHack" in value
        ):

            return value

    # --------------------------------------------------------
    # Запасной вариант
    # --------------------------------------------------------

    return FALLBACK_ENV_ID


# ============================================================
# ПАРСИНГ СОСТОЯНИЯ
# ============================================================

def parse_state(
    parser,
    observation,
    step,
):
    """
    Небольшой адаптер для MiniHackStateParser.

    Поддерживает несколько вариантов интерфейса,
    чтобы скрипт не зависел от одного конкретного
    названия аргументов.
    """

    parse_method = getattr(
        parser,
        "parse",
        None,
    )

    if parse_method is None:

        raise AttributeError(
            "MiniHackStateParser не содержит метода parse()."
        )

    # --------------------------------------------------------
    # Вариант:
    #
    # parser.parse(observation, step=step)
    # --------------------------------------------------------

    try:

        return parse_method(
            observation,
            step=step,
        )

    except TypeError:
        pass

    # --------------------------------------------------------
    # Вариант:
    #
    # parser.parse(observation)
    # --------------------------------------------------------

    return parse_method(
        observation
    )


# ============================================================
# СОЗДАНИЕ MINI HACK ENVIRONMENT
# ============================================================

def create_environment(
    env_id,
):
    """
    Создаёт новую MiniHack-среду
    для запуска уже обученного агента.
    """

    print()
    print(
        f"Создаём environment: {env_id}"
    )

    try:

        env = gym.make(
            env_id,
            observation_keys=(
                "blstats",
                "message",
                "glyphs",
            ),
        )

    except Exception:

        print(
            "Не удалось создать env "
            "с observation_keys."
        )

        print(
            "Пробуем стандартную конфигурацию..."
        )

        env = gym.make(
            env_id
        )

    return env


# ============================================================
# КРАСИВАЯ ПЕЧАТЬ ПРИЧИННЫХ ЦЕПОЧЕК
# ============================================================

def print_causal_chains(
    chains,
    limit=10,
):
    print()
    print("=" * 70)
    print("НАЙДЕННЫЕ ПРИЧИННЫЕ ЦЕПОЧКИ")
    print("=" * 70)

    if not chains:

        print(
            "Причинные цепочки не найдены."
        )

        return

    # --------------------------------------------------------
    # Сначала самые длинные
    # --------------------------------------------------------

    chains = sorted(
        chains,
        key=len,
        reverse=True,
    )

    for chain_index, chain in enumerate(
        chains[:limit],
        start=1,
    ):

        print()
        print(
            f"CHAIN {chain_index}"
        )

        for link in chain:

            cause_action = getattr(
                link.cause_rule,
                "action",
                "?",
            )

            effect_action = getattr(
                link.effect_rule,
                "action",
                "?",
            )

            print(
                f"action {cause_action}"
            )

            print(
                f"   ↓ {link.connecting_feature}"
            )

            print(
                f"action {effect_action}"
            )

            print(
                f"   confidence: "
                f"{link.confidence:.3f}"
            )


# ============================================================
# ЗАПУСК CAUSAL AGENT
# ============================================================

def run_agent(
    env,
    parser,
    agent,
):
    """
    Запускает CausalAgent в новой серии эпизодов.
    """

    print()
    print("=" * 70)
    print("PHASE 2")
    print("ЗАПУСК CAUSAL AGENT")
    print("=" * 70)

    rewards = []

    episode_lengths = []

    successful_episodes = 0

    # --------------------------------------------------------
    # Эпизоды
    # --------------------------------------------------------

    for episode in range(
        1,
        TRAINING_EPISODES + 1,
    ):

        observation, info = env.reset(
            seed=SEED + episode
        )

        state = parse_state(
            parser=parser,
            observation=observation,
            step=0,
        )

        total_reward = 0.0

        terminated = False

        truncated = False

        steps = 0

        # ----------------------------------------------------
        # Один episode
        # ----------------------------------------------------

        while (
            not terminated
            and not truncated
            and steps
            < MAX_STEPS_PER_EPISODE
        ):

            action = agent.choose_action(
                state
            )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            steps += 1

            total_reward += float(
                reward
            )

            state = parse_state(
                parser=parser,
                observation=next_observation,
                step=steps,
            )

        # ----------------------------------------------------
        # Статистика
        # ----------------------------------------------------

        rewards.append(
            total_reward
        )

        episode_lengths.append(
            steps
        )

        if terminated:
            successful_episodes += 1

        print(
            f"Episode {episode:3d} | "
            f"reward={total_reward:8.3f} | "
            f"steps={steps:4d} | "
            f"terminated={terminated}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("CAUSAL AGENT SUMMARY")
    print("=" * 70)

    print(
        f"Эпизодов: "
        f"{len(rewards)}"
    )

    if rewards:

        print(
            f"Средняя награда: "
            f"{mean(rewards):.3f}"
        )

        print(
            f"Лучшая награда: "
            f"{max(rewards):.3f}"
        )

        print(
            f"Худшая награда: "
            f"{min(rewards):.3f}"
        )

    if episode_lengths:

        print(
            f"Среднее число шагов: "
            f"{mean(episode_lengths):.2f}"
        )

    print(
        f"Завершённых эпизодов: "
        f"{successful_episodes}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. ОБУЧЕНИЕ / ОБНАРУЖЕНИЕ ПРАВИЛ
    # ========================================================

    rules = run_rule_discovery()

    print()
    print("=" * 70)
    print("RULE EXTRACTION")
    print("=" * 70)

    print(
        f"Найдено TransformationRule: "
        f"{len(rules)}"
    )

    if not rules:

        raise RuntimeError(
            "\n"
            "Не удалось получить TransformationRule "
            "из discover_transformations.py.\n"
            "\n"
            "Сам discover_transformations.py отработал, "
            "но правила находятся внутри функции и "
            "не остались в глобальных переменных.\n"
            "\n"
            "Если увидишь эту ошибку — это не проблема "
            "CausalAgent. Тогда мы просто вынесем "
            "обнаружение правил в отдельную функцию "
            "discover_rules()."
        )

    # ========================================================
    # 3. ПРИЧИННЫЕ СВЯЗИ
    # ========================================================

    finder = CausalChainFinder(
        minimum_confidence=0.0
    )

    causal_links = finder.find_links(
        rules
    )

    causal_chains = finder.find_chains(
        causal_links,
        max_depth=5,
    )

    print()
    print("=" * 70)
    print("CAUSAL DISCOVERY")
    print("=" * 70)

    print(
        f"Правил: "
        f"{len(rules)}"
    )

    print(
        f"Причинных связей: "
        f"{len(causal_links)}"
    )

    print(
        f"Причинных цепочек: "
        f"{len(causal_chains)}"
    )

    print_causal_chains(
        causal_chains,
        limit=10,
    )

    # ========================================================
    # 4. ОПРЕДЕЛЯЕМ ENVIRONMENT
    # ========================================================

    env_id = FALLBACK_ENV_ID

    print()
    print(
        f"Environment для агента: "
        f"{env_id}"
    )

    # ========================================================
    # 5. СОЗДАЁМ НОВУЮ СРЕДУ
    # ========================================================

    env = create_environment(
        env_id
    )

    # ========================================================
    # 6. ACTION SPACE
    # ========================================================

    if not hasattr(
        env.action_space,
        "n",
    ):

        raise TypeError(
            "Пока CausalAgent ожидает "
            "дискретный action_space."
        )

    actions = list(
        range(
            env.action_space.n
        )
    )

    print(
        f"Количество действий: "
        f"{len(actions)}"
    )

    # ========================================================
    # 7. PARSER
    # ========================================================

    parser = MiniHackStateParser()

    # ========================================================
    # 8. CAUSAL AGENT
    # ========================================================

    agent = CausalAgent(
        actions=actions,
        rules=rules,
        causal_links=causal_links,
        epsilon=EPSILON,
        seed=SEED,
    )

    print()
    print(
        "CausalAgent создан."
    )

    print(
        f"rules = "
        f"{len(rules)}"
    )

    print(
        f"causal_links = "
        f"{len(causal_links)}"
    )

    print(
        f"epsilon = "
        f"{EPSILON}"
    )

    # ========================================================
    # 9. ЗАПУСК
    # ========================================================

    try:

        run_agent(
            env=env,
            parser=parser,
            agent=agent,
        )

    finally:

        env.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()