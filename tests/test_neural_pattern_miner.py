import pytest
import torch

from src.rules import (
    NeuralPatternMiner,
    NeuralRuleCandidate,
    PatternExample,
    PatternNetwork,
    PredicateVectorizer,
)


def make_examples():
    return [
        PatternExample(
            predicates={
                "has_key": True,
                "near_door": True,
                "door_locked": True,
            },
            action=5,
            reward=5.0,
            success=True,
        ),
        PatternExample(
            predicates={
                "has_key": True,
                "near_door": True,
                "door_locked": True,
                "goal_visible": True,
            },
            action=5,
            reward=4.0,
            success=True,
        ),
        PatternExample(
            predicates={
                "has_key": True,
                "near_door": True,
                "door_locked": True,
                "key_visible": False,
            },
            action=5,
            reward=3.0,
            success=True,
        ),
        PatternExample(
            predicates={
                "has_key": False,
                "near_key": True,
                "key_visible": True,
            },
            action=4,
            reward=2.0,
            success=True,
        ),
        PatternExample(
            predicates={
                "has_key": False,
                "near_key": True,
                "key_visible": True,
                "door_visible": True,
            },
            action=4,
            reward=2.0,
            success=True,
        ),
        PatternExample(
            predicates={
                "has_key": False,
                "near_key": True,
                "key_visible": True,
                "goal_visible": True,
            },
            action=4,
            reward=2.0,
            success=True,
        ),
    ]


def test_pattern_example_creation():
    example = PatternExample(
        predicates={"has_key": True},
        action=1,
        reward=2.0,
        success=True,
    )

    assert example.predicates == {"has_key": True}
    assert example.action == 1
    assert example.reward == 2.0
    assert example.success is True


def test_vectorizer_fits_predicates():
    vectorizer = PredicateVectorizer()

    vectorizer.fit(
        [
            {"has_key": True},
            {"door_locked": False},
        ]
    )

    assert vectorizer.is_fitted is True
    assert vectorizer.feature_count == 2


def test_vectorizer_transforms_predicates():
    vectorizer = PredicateVectorizer()

    vectorizer.fit(
        [
            {
                "has_key": True,
                "door_locked": False,
            }
        ]
    )

    vector = vectorizer.transform(
        {
            "has_key": True,
            "door_locked": False,
        }
    )

    assert isinstance(vector, torch.Tensor)
    assert vector.dtype == torch.float32
    assert vector.shape == (2,)
    assert vector.sum().item() == 2.0


def test_vectorizer_ignores_unknown_token():
    vectorizer = PredicateVectorizer()
    vectorizer.fit([{"has_key": True}])

    vector = vectorizer.transform(
        {
            "has_key": True,
            "unknown_predicate": True,
        }
    )

    assert vector.shape == (1,)
    assert vector.sum().item() == 1.0


def test_vectorizer_requires_fit():
    vectorizer = PredicateVectorizer()

    with pytest.raises(RuntimeError):
        vectorizer.transform({"has_key": True})


def test_vectorizer_rejects_empty_training_data():
    vectorizer = PredicateVectorizer()

    with pytest.raises(ValueError):
        vectorizer.fit([])


def test_token_round_trip():
    token = PredicateVectorizer.make_token(
        "door_locked",
        True,
    )

    name, value = PredicateVectorizer.parse_token(token)

    assert name == "door_locked"
    assert value is True


def test_pattern_network_output_shape():
    network = PatternNetwork(
        input_size=5,
        action_count=7,
        hidden_size=16,
    )

    inputs = torch.zeros((3, 5))

    outputs = network(inputs)

    assert outputs.shape == (3, 7)


def test_miner_creation():
    miner = NeuralPatternMiner(
        actions=[0, 1, 2, 3, 4, 5, 6]
    )

    assert miner.actions == (0, 1, 2, 3, 4, 5, 6)
    assert miner.is_trained is False


def test_miner_rejects_empty_actions():
    with pytest.raises(ValueError):
        NeuralPatternMiner(actions=[])


def test_miner_rejects_duplicate_actions():
    with pytest.raises(ValueError):
        NeuralPatternMiner(actions=[0, 1, 1])


def test_miner_trains_on_examples():
    miner = NeuralPatternMiner(
        actions=[0, 1, 2, 3, 4, 5, 6],
        hidden_size=16,
        learning_rate=0.01,
        batch_size=3,
        seed=42,
    )

    history = miner.fit(
        make_examples(),
        epochs=40,
    )

    assert miner.is_trained is True
    assert len(history) == 40
    assert all(loss >= 0.0 for loss in history)


def test_miner_predicts_known_pattern():
    miner = NeuralPatternMiner(
        actions=[0, 1, 2, 3, 4, 5, 6],
        hidden_size=32,
        learning_rate=0.01,
        batch_size=3,
        seed=42,
    )

    miner.fit(
        make_examples(),
        epochs=120,
    )

    action, confidence = miner.predict(
        {
            "has_key": True,
            "near_door": True,
            "door_locked": True,
        }
    )

    assert action == 5
    assert 0.0 <= confidence <= 1.0


def test_miner_requires_training_before_prediction():
    miner = NeuralPatternMiner(
        actions=[0, 1, 2]
    )

    with pytest.raises(RuntimeError):
        miner.predict({"has_key": True})


def test_miner_extracts_rule_candidates():
    miner = NeuralPatternMiner(
        actions=[0, 1, 2, 3, 4, 5, 6],
        hidden_size=32,
        learning_rate=0.01,
        batch_size=3,
        seed=42,
    )

    examples = make_examples()

    miner.fit(
        examples,
        epochs=120,
    )

    candidates = miner.extract_candidates(
        examples,
        minimum_confidence=0.50,
        minimum_support=2,
        predicate_frequency=0.60,
        maximum_conditions=4,
    )

    assert candidates
    assert all(
        isinstance(candidate, NeuralRuleCandidate)
        for candidate in candidates
    )


def test_candidate_contains_common_door_conditions():
    miner = NeuralPatternMiner(
        actions=[0, 1, 2, 3, 4, 5, 6],
        hidden_size=32,
        learning_rate=0.01,
        batch_size=3,
        seed=42,
    )

    examples = make_examples()

    miner.fit(
        examples,
        epochs=120,
    )

    candidates = miner.extract_candidates(
        examples,
        minimum_confidence=0.50,
        minimum_support=2,
        predicate_frequency=0.80,
        maximum_conditions=5,
    )

    door_candidates = [
        candidate
        for candidate in candidates
        if candidate.action == 5
    ]

    assert door_candidates

    conditions = door_candidates[0].conditions

    assert conditions["has_key"] is True
    assert conditions["near_door"] is True
    assert conditions["door_locked"] is True


def test_candidate_to_dict():
    candidate = NeuralRuleCandidate(
        conditions={"has_key": True},
        action=5,
        confidence=0.9,
        support=10,
        success_rate=0.8,
    )

    data = candidate.to_dict()

    assert data["conditions"] == {"has_key": True}
    assert data["action"] == 5
    assert data["confidence"] == 0.9
    assert data["support"] == 10
    assert data["success_rate"] == 0.8
    assert data["specificity"] == 1


def test_feature_importance_returns_active_features():
    miner = NeuralPatternMiner(
        actions=[0, 1, 2, 3, 4, 5, 6],
        hidden_size=32,
        learning_rate=0.01,
        batch_size=3,
        seed=42,
    )

    examples = make_examples()

    miner.fit(
        examples,
        epochs=120,
    )

    importance = miner.feature_importance(
        {
            "has_key": True,
            "near_door": True,
            "door_locked": True,
        },
        action=5,
    )

    assert isinstance(importance, dict)

    assert set(importance).issubset(
        {
            "has_key=True",
            "near_door=True",
            "door_locked=True",
        }
    )

    assert all(
        value >= 0.0
        for value in importance.values()
    )


def test_fit_rejects_empty_examples():
    miner = NeuralPatternMiner(
        actions=[0, 1]
    )

    with pytest.raises(ValueError):
        miner.fit([])


def test_fit_rejects_unknown_action():
    miner = NeuralPatternMiner(
        actions=[0, 1]
    )

    examples = [
        PatternExample(
            predicates={"has_key": True},
            action=7,
            reward=1.0,
            success=True,
        )
    ]

    with pytest.raises(ValueError):
        miner.fit(examples)


def test_fit_requires_positive_or_successful_example():
    miner = NeuralPatternMiner(
        actions=[0, 1]
    )

    examples = [
        PatternExample(
            predicates={"has_key": False},
            action=0,
            reward=-1.0,
            success=False,
        )
    ]

    with pytest.raises(ValueError):
        miner.fit(examples)


def test_extract_candidates_validates_parameters():
    miner = NeuralPatternMiner(
        actions=[0, 1],
        hidden_size=8,
        learning_rate=0.01,
    )

    examples = [
        PatternExample(
            predicates={"has_key": True},
            action=1,
            reward=1.0,
            success=True,
        ),
        PatternExample(
            predicates={"has_key": True},
            action=1,
            reward=1.0,
            success=True,
        ),
    ]

    miner.fit(examples, epochs=20)

    with pytest.raises(ValueError):
        miner.extract_candidates(
            examples,
            minimum_confidence=1.5,
        )

    with pytest.raises(ValueError):
        miner.extract_candidates(
            examples,
            minimum_support=0,
        )

    with pytest.raises(ValueError):
        miner.extract_candidates(
            examples,
            predicate_frequency=0.0,
        )

    with pytest.raises(ValueError):
        miner.extract_candidates(
            examples,
            maximum_conditions=0,
        )