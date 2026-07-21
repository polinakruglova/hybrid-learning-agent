from __future__ import annotations

from typing import Any

from src.agents import HybridAgent, RuleAgent
from src.environments import KeyDoorEnvironment
from src.learning import LearningController
from src.memory import ExperienceBuffer
from src.persistence import TrainingExporter
from src.predicates import PredicateGenerator
from src.rl import QTablePolicy
from src.rules import (
    NeuralPatternMiner,
    NeuralRuleTrainer,
    RuleStore,
)
from src.training import TrainingHistory, TrainingRunner
from src.visualization import TrainingVisualizer


# ============================================================
# НАСТРОЙКИ ОБУЧЕНИЯ
# ============================================================

# Сначала агент собирает опыт без нейронных правил.
EXPERIENCE_EPISODES = 800

# Затем нейросеть создаёт правила, и агент продолжает обучение,
# уже используя найденные правила.
RULE_EPISODES = 200

TOTAL_EPISODES = (
    EXPERIENCE_EPISODES
    + RULE_EPISODES
)

LOG_INTERVAL = 100


# ============================================================
# Q-LEARNING
# ============================================================

LEARNING_RATE = 0.15
DISCOUNT_FACTOR = 0.95

INITIAL_EPSILON = 1.0
EPSILON_DECAY = 0.995
MINIMUM_EPSILON = 0.05


# ============================================================
# БУФЕР ОПЫТА
# ============================================================

BUFFER_CAPACITY = 50_000
RANDOM_SEED = 42


# ============================================================
# СРЕДА
# ============================================================

ENVIRONMENT_WIDTH = 7
ENVIRONMENT_HEIGHT = 7
MAX_STEPS_PER_EPISODE = 100


# ============================================================
# НЕЙРОСЕТЕВАЯ ГЕНЕРАЦИЯ ПРАВИЛ
# ============================================================

NEURAL_EPOCHS = 100
MINIMUM_NEURAL_EXAMPLES = 10

MINIMUM_RULE_CONFIDENCE = 0.70
MINIMUM_RULE_SUPPORT = 2
MINIMUM_RULE_SUCCESS_RATE = 0.50

PREDICATE_FREQUENCY = 0.60
MAXIMUM_RULE_CONDITIONS = 5


def create_training_system() -> tuple[
    KeyDoorEnvironment,
    PredicateGenerator,
    RuleStore,
    ExperienceBuffer,
    NeuralPatternMiner,
    NeuralRuleTrainer,
    RuleAgent,
    QTablePolicy,
    HybridAgent,
    LearningController,
    TrainingRunner,
]:
    """
    Создаёт и соединяет компоненты гибридной системы.

    Архитектура:

        Environment
            ↓
        LearningController
            ├── ExperienceBuffer
            └── QTablePolicy

        ExperienceBuffer
            ↓
        NeuralRuleTrainer
            ↓
        NeuralPatternMiner
            ↓
        RuleStore
            ↓
        RuleAgent
            ↓
        HybridAgent
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
    # 2. Генератор предикатов
    # --------------------------------------------------------

    predicate_generator = PredicateGenerator()

    # --------------------------------------------------------
    # 3. Хранилище символических правил
    # --------------------------------------------------------

    rule_store = RuleStore()

    # --------------------------------------------------------
    # 4. Буфер опыта
    # --------------------------------------------------------

    experience_buffer = ExperienceBuffer(
        capacity=BUFFER_CAPACITY,
        seed=RANDOM_SEED,
    )

    # --------------------------------------------------------
    # 5. Действия среды
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
    # 6. Q-learning
    # --------------------------------------------------------

    q_policy = QTablePolicy(
        actions=actions,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=INITIAL_EPSILON,
        seed=RANDOM_SEED,
    )

    # --------------------------------------------------------
    # 7. Нейросеть поиска закономерностей
    # --------------------------------------------------------

    neural_pattern_miner = NeuralPatternMiner(
        actions=actions,
        seed=RANDOM_SEED,
    )

    # --------------------------------------------------------
    # 8. Нейросетевой тренер правил
    # --------------------------------------------------------

    neural_rule_trainer = NeuralRuleTrainer(
        experience_buffer=experience_buffer,
        predicate_generator=predicate_generator,
        pattern_miner=neural_pattern_miner,
        rule_store=rule_store,
        minimum_confidence=MINIMUM_RULE_CONFIDENCE,
        minimum_support=MINIMUM_RULE_SUPPORT,
        minimum_success_rate=(
            MINIMUM_RULE_SUCCESS_RATE
        ),
        predicate_frequency=PREDICATE_FREQUENCY,
        maximum_conditions=(
            MAXIMUM_RULE_CONDITIONS
        ),

        # Для обучения правил используются только
        # положительные или успешные переходы.
        positive_only=True,
    )

    # --------------------------------------------------------
    # 9. Агент символических правил
    # --------------------------------------------------------

    rule_agent = RuleAgent(
        rule_store=rule_store,
        predicate_generator=predicate_generator,

        # При отсутствии подходящего правила управление
        # передаётся Q-learning-политике.
        fallback_action=None,
        fallback_policy=None,
    )

    # --------------------------------------------------------
    # 10. Гибридный агент
    # --------------------------------------------------------

    hybrid_agent = HybridAgent(
        rule_agent=rule_agent,
        rl_policy=q_policy,
    )

    # --------------------------------------------------------
    # 11. Контроллер обучения
    # --------------------------------------------------------

    controller = LearningController(
        agent=hybrid_agent,
        q_policy=q_policy,
        experience_buffer=experience_buffer,
        rule_store=rule_store,
    )

    # --------------------------------------------------------
    # 12. Цикл обучения
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
        experience_buffer,
        neural_pattern_miner,
        neural_rule_trainer,
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

    print(
        f"Experience episodes:   "
        f"{EXPERIENCE_EPISODES}"
    )
    print(
        f"Rule episodes:         "
        f"{RULE_EPISODES}"
    )
    print(
        f"Total episodes:        "
        f"{TOTAL_EPISODES}"
    )

    print(
        f"Environment size:      "
        f"{ENVIRONMENT_WIDTH} x "
        f"{ENVIRONMENT_HEIGHT}"
    )
    print(
        f"Max episode steps:     "
        f"{MAX_STEPS_PER_EPISODE}"
    )

    print(
        f"Learning rate:         "
        f"{LEARNING_RATE}"
    )
    print(
        f"Discount factor:       "
        f"{DISCOUNT_FACTOR}"
    )
    print(
        f"Initial epsilon:       "
        f"{INITIAL_EPSILON}"
    )
    print(
        f"Epsilon decay:         "
        f"{EPSILON_DECAY}"
    )
    print(
        f"Minimum epsilon:       "
        f"{MINIMUM_EPSILON}"
    )

    print(
        f"Neural epochs:         "
        f"{NEURAL_EPOCHS}"
    )
    print(
        f"Min neural examples:   "
        f"{MINIMUM_NEURAL_EXAMPLES}"
    )
    print(
        f"Rule confidence:       "
        f"{MINIMUM_RULE_CONFIDENCE}"
    )
    print(
        f"Rule support:          "
        f"{MINIMUM_RULE_SUPPORT}"
    )

    print("=" * 70)
    print()


def train_neural_rules(
    neural_rule_trainer: NeuralRuleTrainer,
    experience_buffer: ExperienceBuffer,
    rule_store: RuleStore,
) -> None:
    """
    Обучает NeuralPatternMiner на накопленном опыте
    и сохраняет найденные правила в RuleStore.
    """

    print()
    print("=" * 70)
    print("NEURAL RULE TRAINING")
    print("=" * 70)

    print(
        f"Experiences in buffer: "
        f"{len(experience_buffer)}"
    )

    successful_count = len(
        experience_buffer.successful()
    )
    positive_count = len(
        experience_buffer.positive()
    )

    print(
        f"Successful transitions: "
        f"{successful_count}"
    )
    print(
        f"Positive transitions:   "
        f"{positive_count}"
    )

    result = neural_rule_trainer.train_from_buffer(
        epochs=NEURAL_EPOCHS,
        minimum_examples=MINIMUM_NEURAL_EXAMPLES,
    )

    if result is None:
        print()
        print(
            "Недостаточно положительного или успешного "
            "опыта для обучения нейросети."
        )
        print(
            f"Нужно минимум: "
            f"{MINIMUM_NEURAL_EXAMPLES}"
        )
        print("=" * 70)
        return

    print()
    print("Neural training finished.")

    print(
        f"Examples used:         "
        f"{result.examples_count}"
    )
    print(
        f"Candidates found:      "
        f"{result.candidates_count}"
    )
    print(
        f"Rules created:         "
        f"{result.created_count}"
    )
    print(
        f"Rules updated:         "
        f"{result.updated_count}"
    )
    print(
        f"Candidates rejected:   "
        f"{result.rejected_count}"
    )
    print(
        f"Rules in store:        "
        f"{len(rule_store)}"
    )

    final_loss = result.final_loss

    if final_loss is None:
        print("Final neural loss:     None")
    else:
        print(
            f"Final neural loss:     "
            f"{final_loss:.6f}"
        )

    print("=" * 70)


def print_rules(
    rule_store: RuleStore,
    limit: int = 20,
) -> None:
    """
    Выводит найденные нейросетевые правила.
    """

    print()
    print("=" * 70)
    print("NEURAL SYMBOLIC RULES")
    print("=" * 70)

    rules = rule_store.rules

    if not rules:
        print("Правила не найдены.")
        print("=" * 70)
        return

    for index, rule in enumerate(
        rules[:limit],
        start=1,
    ):
        print()
        print(f"Rule #{index}")

        print("  IF:")

        for name, value in rule.conditions.items():
            print(
                f"    {name} = {value}"
            )

        print(
            f"  ACTION:      "
            f"{rule.action}"
        )
        print(
            f"  SUPPORT:     "
            f"{rule.support}"
        )
        print(
            f"  CONFIDENCE:  "
            f"{rule.confidence:.3f}"
        )
        print(
            f"  SPECIFICITY: "
            f"{rule.specificity}"
        )

        neural_confidence = rule.metadata.get(
            "neural_confidence"
        )

        if neural_confidence is not None:
            print(
                f"  NEURAL CONF: "
                f"{float(neural_confidence):.3f}"
            )

    if len(rules) > limit:
        print()
        print(
            f"Показано {limit} из "
            f"{len(rules)} правил."
        )

    print("=" * 70)


def print_training_summary(
    history: TrainingHistory,
    controller: LearningController,
    experience_buffer: ExperienceBuffer,
    neural_rule_trainer: NeuralRuleTrainer,
) -> None:
    """
    Выводит итоговую статистику обучения.
    """

    history_summary = history.summary()
    controller_statistics = controller.statistics()
    buffer_statistics = experience_buffer.statistics()
    neural_statistics = neural_rule_trainer.statistics()

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

    print_dictionary(
        title="Neural rule statistics",
        data=neural_statistics,
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
    print()
    print(f"{title}:")

    if not data:
        print("  no data")
        return

    for key, value in data.items():
        formatted_key = (
            key
            .replace("_", " ")
            .capitalize()
        )

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
    print(
        f"  Reward:        "
        f"{result.total_reward:.3f}"
    )
    print(f"  Steps:         {result.steps}")
    print(f"  Success:       {result.success}")
    print(
        f"  Final event:   "
        f"{result.final_event}"
    )
    print(
        f"  Epsilon:       "
        f"{result.epsilon:.4f}"
    )
    print(
        f"  Rules:         "
        f"{result.rule_count}"
    )
    print(
        f"  Q states:      "
        f"{result.q_state_count}"
    )


def main() -> None:
    print_header()

    (
        environment,
        predicate_generator,
        rule_store,
        experience_buffer,
        neural_pattern_miner,
        neural_rule_trainer,
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

    # ========================================================
    # ФАЗА 1: НАКОПЛЕНИЕ ОПЫТА
    # ========================================================

    print()
    print("=" * 70)
    print("PHASE 1: EXPERIENCE COLLECTION")
    print("=" * 70)
    print()

    history = runner.train(
        episodes=EXPERIENCE_EPISODES,
        render=False,
        log_interval=LOG_INTERVAL,
    )

    # ========================================================
    # ФАЗА 2: НЕЙРОСЕТЕВАЯ ГЕНЕРАЦИЯ ПРАВИЛ
    # ========================================================

    train_neural_rules(
        neural_rule_trainer=neural_rule_trainer,
        experience_buffer=experience_buffer,
        rule_store=rule_store,
    )

    print_rules(
        rule_store=rule_store,
    )

    # ========================================================
    # ФАЗА 3: ОБУЧЕНИЕ ГИБРИДНОГО АГЕНТА С ПРАВИЛАМИ
    # ========================================================

    if RULE_EPISODES > 0:
        print()
        print("=" * 70)
        print("PHASE 2: HYBRID TRAINING WITH RULES")
        print("=" * 70)
        print()

        history = runner.train(
            episodes=RULE_EPISODES,
            render=False,
            log_interval=LOG_INTERVAL,
        )

    # ========================================================
    # ИТОГОВАЯ СТАТИСТИКА
    # ========================================================

    print_training_summary(
        history=history,
        controller=controller,
        experience_buffer=experience_buffer,
        neural_rule_trainer=neural_rule_trainer,
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

    controller_statistics = (
        controller.statistics()
    )

    controller_statistics.update(
        {
            f"neural_{key}": value
            for key, value
            in neural_rule_trainer.statistics().items()
        }
    )

    exported_paths = exporter.export_all(
        history=history,
        rule_store=rule_store,
        q_policy=q_policy,
        controller_statistics=(
            controller_statistics
        ),
        buffer_statistics=(
            experience_buffer.statistics()
        ),
        configuration={
            "agent": "neural_hybrid",
            "experience_episodes": (
                EXPERIENCE_EPISODES
            ),
            "rule_episodes": RULE_EPISODES,
            "total_episodes": TOTAL_EPISODES,
            "learning_rate": LEARNING_RATE,
            "discount_factor": (
                DISCOUNT_FACTOR
            ),
            "initial_epsilon": INITIAL_EPSILON,
            "epsilon_decay": EPSILON_DECAY,
            "minimum_epsilon": MINIMUM_EPSILON,
            "seed": RANDOM_SEED,
            "environment_width": (
                ENVIRONMENT_WIDTH
            ),
            "environment_height": (
                ENVIRONMENT_HEIGHT
            ),
            "max_steps": (
                MAX_STEPS_PER_EPISODE
            ),
            "neural_epochs": NEURAL_EPOCHS,
            "minimum_neural_examples": (
                MINIMUM_NEURAL_EXAMPLES
            ),
            "minimum_rule_confidence": (
                MINIMUM_RULE_CONFIDENCE
            ),
            "minimum_rule_support": (
                MINIMUM_RULE_SUPPORT
            ),
            "minimum_rule_success_rate": (
                MINIMUM_RULE_SUCCESS_RATE
            ),
            "predicate_frequency": (
                PREDICATE_FREQUENCY
            ),
            "maximum_rule_conditions": (
                MAXIMUM_RULE_CONDITIONS
            ),
        },
    )

    # ========================================================
    # СОЗДАНИЕ ГРАФИКОВ
    # ========================================================

    print("Creating training plots...")

    visualizer = TrainingVisualizer(
        output_directory=(
            "outputs/hybrid/plots"
        ),
        rolling_window=50,
    )

    plot_paths = visualizer.plot_all(
        csv_path=exported_paths["history_csv"],
        experiment_name=(
            "Neural Hybrid Agent"
        ),
    )

    print()
    print("Created data files:")

    for name, path in exported_paths.items():
        print(
            f"  {name:<20} {path}"
        )

    print()
    print("Created plots:")

    for name, path in plot_paths.items():
        print(
            f"  {name:<20} {path}"
        )


if __name__ == "__main__":
    main()