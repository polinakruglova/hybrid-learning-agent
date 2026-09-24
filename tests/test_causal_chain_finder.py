from src.transformations.causal_chain_finder import CausalChainFinder
from src.transformations.transformation_rule import TransformationRule


def test_causal_chain_a_b_c():

    # ---------------------------------------------------------
    # RULE A
    # ---------------------------------------------------------
    # После действия появляется ключ
    # ---------------------------------------------------------

    rule_a = TransformationRule(
        action=1,
        conditions={},
        effects={
            "has_key": ("set", True),
        },
        observations=20,
        confidence=0.95,
    )

    # ---------------------------------------------------------
    # RULE B
    # ---------------------------------------------------------
    # Если ключ уже есть,
    # действие открывает дверь
    # ---------------------------------------------------------

    rule_b = TransformationRule(
        action=2,
        conditions={
            "has_key": True,
        },
        effects={
            "door_open": ("set", True),
        },
        observations=15,
        confidence=0.90,
    )

    # ---------------------------------------------------------
    # RULE C
    # ---------------------------------------------------------
    # Если дверь открыта,
    # действие позволяет пройти дальше
    # ---------------------------------------------------------

    rule_c = TransformationRule(
        action=3,
        conditions={
            "door_open": True,
        },
        effects={
            "passed_door": ("set", True),
        },
        observations=12,
        confidence=0.88,
    )

    rules = [
        rule_a,
        rule_b,
        rule_c,
    ]

    # ---------------------------------------------------------
    # ПОИСК СВЯЗЕЙ
    # ---------------------------------------------------------

    finder = CausalChainFinder(
        minimum_confidence=0.5,
    )

    links = finder.find_links(
        rules
    )

    print()
    print("FOUND LINKS")
    print("=" * 60)

    for link in links:

        print(
            f"action {link.cause_rule.action}"
            f" --[{link.connecting_feature}]--> "
            f"action {link.effect_rule.action}"
            f" | confidence={link.confidence:.2f}"
        )

    # ---------------------------------------------------------
    # ПОИСК ЦЕПОЧЕК
    # ---------------------------------------------------------

    chains = finder.find_chains(
        links=links,
        max_depth=5,
    )

    print()
    print("FOUND CHAINS")
    print("=" * 60)

    for index, chain in enumerate(
        chains,
        start=1,
    ):

        print(
            f"\nCHAIN {index}"
        )

        first_rule = chain[0].cause_rule

        print(
            f"action {first_rule.action}"
        )

        for link in chain:

            print(
                f"  ↓ {link.connecting_feature}"
            )

            print(
                f"action {link.effect_rule.action}"
            )

    # ---------------------------------------------------------
    # ПРОВЕРКИ
    # ---------------------------------------------------------

    assert len(links) == 2

    assert links[0].cause_rule is rule_a
    assert links[0].effect_rule is rule_b

    assert links[1].cause_rule is rule_b
    assert links[1].effect_rule is rule_c

    # должна существовать цепочка:
    #
    # A -> B -> C

    long_chains = [
        chain
        for chain in chains
        if len(chain) == 2
    ]

    assert len(long_chains) >= 1

    chain = long_chains[0]

    assert chain[0].cause_rule is rule_a
    assert chain[0].effect_rule is rule_b

    assert chain[1].cause_rule is rule_b
    assert chain[1].effect_rule is rule_c