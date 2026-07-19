from __future__ import annotations

from typing import Any

from src.agents import HybridAgent, RuleAgent
from src.environments import KeyDoorEnvironment
from src.learning import LearningController
from src.memory import ExperienceBuffer
from src.predicates import PredicateGenerator
from src.rl import QTablePolicy
from src.rules import RuleMiner, RuleStore
from src.training import TrainingHistory, TrainingRunner
from src.persistence import TrainingExporter
from src.visualization import TrainingVisualizer


# ============================================================
# НАСТРОЙКИ ОБУЧЕНИЯ
# ============================================================

EPISODES = 1000
LOG_INTERVAL = 100

LEARNING_RATE = 0.15
DISCOUNT_FACTOR = 0.95

INITIAL_EPSILON = 1.0
EPSILON_DECAY = 0.995
MINIMUM_EPSILON = 0.05

BUFFER_CAPACITY = 50_000
RANDOM_SEED = 42

ENVIRONMENT_WIDTH = 7
ENVIRONMENT_HEIGHT = 7
MAX_STEPS_PER_EPISODE = 100


def create_training_system() -> tuple[
    KeyDoorEnvironment,
    PredicateGenerator,
    RuleStore,
    RuleMiner,
    ExperienceBuffer,
    RuleAgent,
    QTablePolicy,
    HybridAgent,
    LearningController,
    TrainingRunner,
]:
    """
    Создаёт и соединяет все компоненты гибридной системы.
    """

    # --------------------------------------------------------
    # 1. Среда
    # --------------------------------------------------------

    environment = KeyDoorEnvironment(
        width=ENVIRONMENT_WIDTH,
        height=ENVIRONMENT_HEIGHT,
        max_steps=MAX_STEPS_PER_EPISODE,
    )

    # --------------------------------------------------------
    # 2. Генератор признаков
    # --------------------------------------------------------

    predicate_generator = PredicateGenerator()

    # --------------------------------------------------------
    # 3. Хранилище правил
    # --------------------------------------------------------

    rule_store = RuleStore()

    # --------------------------------------------------------
    # 4. Генератор новых правил
    # --------------------------------------------------------

    rule_miner = RuleMiner(
        predicate_generator=predicate_generator,

        # Правила создаются только из опыта
        # с положительной наградой.
        minimum_reward=0.0,
    )

    # --------------------------------------------------------
    # 5. Память опыта
    # --------------------------------------------------------

    experience_buffer = ExperienceBuffer(
        capacity=BUFFER_CAPACITY,
        seed=RANDOM_SEED,
    )

    # --------------------------------------------------------
    # 6. Список возможных действий
    # --------------------------------------------------------

    actions = [
        KeyDoorEnvironment.ACTION_UP,
        KeyDoorEnvironment.ACTION_RIGHT,
        KeyDoorEnvironment.ACTION_DOWN,
        KeyDoorEnvironment.ACTION_LEFT,
        KeyDoorEnvironment.ACTION_PICKUP,
        KeyDoorEnvironment.ACTION_OPEN_DOOR,
        KeyDoorEnvironment.ACTION_WAIT,
    ]

    # --------------------------------------------------------
    # 7. Q-learning
    # --------------------------------------------------------

    q_policy = QTablePolicy(
        actions=actions,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=INITIAL_EPSILON,
        seed=RANDOM_SEED,
    )

    # --------------------------------------------------------
    # 8. Агент правил
    # --------------------------------------------------------

    rule_agent = RuleAgent(
        rule_store=rule_store,
        predicate_generator=predicate_generator,

        # Если подходящего правила нет, RuleAgent
        # возвращает None, и управление переходит
        # к QTablePolicy.
        fallback_action=None,
        fallback_policy=None,
    )

    # --------------------------------------------------------
    # 9. Гибридный агент
    # --------------------------------------------------------

    hybrid_agent = HybridAgent(
        rule_agent=rule_agent,
        rl_policy=q_policy,
    )

    # --------------------------------------------------------
    # 10. Контроллер обучения
    # --------------------------------------------------------

    controller = LearningController(
        agent=hybrid_agent,
        q_policy=q_policy,
        experience_buffer=experience_buffer,
        rule_miner=rule_miner,
        rule_store=rule_store,
    )

    # --------------------------------------------------------
    # 11. Цикл обучения
    # --------------------------------------------------------

    runner = TrainingRunner(
        environment=environment,
        controller=controller,
        epsilon_decay=EPSILON_DECAY,
        minimum_epsilon=MINIMUM_EPSILON,
    )

    return (
        environment,
        predicate_generator,
        rule_store,
        rule_miner,
        experience_buffer,
        rule_agent,
        q_policy,
        hybrid_agent,
        controller,
        runner,
    )


def print_header() -> None:
    print()
    print("=" * 70)
    print("HYBRID LEARNING AGENT")
    print("=" * 70)

    print(f"Episodes:             {EPISODES}")
    print(f"Environment size:     {ENVIRONMENT_WIDTH} x {ENVIRONMENT_HEIGHT}")
    print(f"Max episode steps:    {MAX_STEPS_PER_EPISODE}")
    print(f"Learning rate:        {LEARNING_RATE}")
    print(f"Discount factor:      {DISCOUNT_FACTOR}")
    print(f"Initial epsilon:      {INITIAL_EPSILON}")
    print(f"Epsilon decay:        {EPSILON_DECAY}")
    print(f"Minimum epsilon:      {MINIMUM_EPSILON}")

    print("=" * 70)
    print()


def print_training_summary(
    history: TrainingHistory,
    controller: LearningController,
    experience_buffer: ExperienceBuffer,
) -> None:
    """
    Выводит итоговую статистику обучения.
    """

    history_summary = history.summary()
    controller_statistics = controller.statistics()
    buffer_statistics = experience_buffer.statistics()

    print()
    print("=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    print(
        f"Episodes:              "
        f"{history_summary['episode_count']}"
    )

    print(
        f"Successful episodes:   "
        f"{history_summary['success_count']}"
    )

    print(
        f"Success rate:          "
        f"{history_summary['success_rate']:.2%}"
    )

    print(
        f"Average reward:        "
        f"{history_summary['average_reward']:.3f}"
    )

    print(
        f"Best reward:           "
        f"{format_optional_number(history_summary['best_reward'])}"
    )

    print(
        f"Worst reward:          "
        f"{format_optional_number(history_summary['worst_reward'])}"
    )

    print(
        f"Average steps:         "
        f"{history_summary['average_steps']:.2f}"
    )

    print(
        f"Last 100 success rate: "
        f"{history.recent_success_rate(window=100):.2%}"
    )

    print(
        f"Last 100 avg reward:   "
        f"{history.recent_average_reward(window=100):.3f}"
    )

    print("-" * 70)

    print_dictionary(
        title="Controller statistics",
        data=controller_statistics,
    )

    print_dictionary(
        title="Experience buffer statistics",
        data=buffer_statistics,
    )

    print("=" * 70)


def format_optional_number(
    value: float | None,
) -> str:
    if value is None:
        return "None"

    return f"{value:.3f}"


def print_dictionary(
    title: str,
    data: dict[str, Any],
) -> None:
    print(f"\n{title}:")

    if not data:
        print("  no data")
        return

    for key, value in data.items():
        formatted_key = key.replace("_", " ").capitalize()

        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)

        print(
            f"  {formatted_key:<25} "
            f"{formatted_value}"
        )


def print_last_episode(
    history: TrainingHistory,
) -> None:
    if not history.episodes:
        return

    result = history.episodes[-1]

    print()
    print("Last episode:")
    print(f"  Episode:       {result.episode}")
    print(f"  Reward:        {result.total_reward:.3f}")
    print(f"  Steps:         {result.steps}")
    print(f"  Success:       {result.success}")
    print(f"  Final event:   {result.final_event}")
    print(f"  Epsilon:       {result.epsilon:.4f}")
    print(f"  Rules:         {result.rule_count}")
    print(f"  Q states:      {result.q_state_count}")


def main() -> None:
    print_header()

    (
        environment,
        predicate_generator,
        rule_store,
        rule_miner,
        experience_buffer,
        rule_agent,
        q_policy,
        hybrid_agent,
        controller,
        runner,
    ) = create_training_system()

    print("System components created successfully.")
    print()
    print("Initial environment:")
    print(environment.render())
    print()
    print("Training started...")
    print()

    history = runner.train(
        episodes=EPISODES,
        render=False,
        log_interval=LOG_INTERVAL,
    )

    print_training_summary(
        history=history,
        controller=controller,
        experience_buffer=experience_buffer,
    )

    print_last_episode(history)

    print()
    print("Final environment state:")
    print(environment.render())

    # ========================================================
    # СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ========================================================

    print()
    print("Saving training results...")

    exporter = TrainingExporter(
        output_directory="outputs/hybrid"
    )

    exported_paths = exporter.export_all(
        history=history,
        rule_store=rule_store,
        q_policy=q_policy,
        controller_statistics=controller.statistics(),
        buffer_statistics=experience_buffer.statistics(),
        configuration={
            "agent": "hybrid",
            "episodes": EPISODES,
            "learning_rate": LEARNING_RATE,
            "discount_factor": DISCOUNT_FACTOR,
            "initial_epsilon": INITIAL_EPSILON,
            "epsilon_decay": EPSILON_DECAY,
            "minimum_epsilon": MINIMUM_EPSILON,
            "seed": RANDOM_SEED,
            "environment_width": ENVIRONMENT_WIDTH,
            "environment_height": ENVIRONMENT_HEIGHT,
            "max_steps": MAX_STEPS_PER_EPISODE,
        },
    )

    # ========================================================
    # СОЗДАНИЕ ГРАФИКОВ
    # ========================================================

    print("Creating training plots...")

    visualizer = TrainingVisualizer(
        output_directory="outputs/hybrid/plots",
        rolling_window=50,
    )

    plot_paths = visualizer.plot_all(
        csv_path=exported_paths["history_csv"],
        experiment_name="Hybrid Agent",
    )

    print()
    print("Created data files:")

    for name, path in exported_paths.items():
        print(f"  {name:<20} {path}")

    print()
    print("Created plots:")

    for name, path in plot_paths.items():
        print(f"  {name:<20} {path}")


if __name__ == "__main__":
    main()