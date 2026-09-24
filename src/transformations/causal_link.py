from dataclasses import dataclass

from src.transformations.transformation_rule import TransformationRule


@dataclass
class CausalLink:
    """
    Причинная связь между двумя правилами.

    cause_rule:
        правило, результат которого может создать условие
        для следующего правила.

    effect_rule:
        правило, которое может стать применимым
        после cause_rule.

    connecting_feature:
        свойство, через которое связаны правила.

    confidence:
        уверенность в причинной связи от 0.0 до 1.0.
    """

    cause_rule: TransformationRule
    effect_rule: TransformationRule

    connecting_feature: str

    confidence: float = 0.0

    observations: int = 0

    def __repr__(self):
        return (
            "CausalLink("
            f"feature={self.connecting_feature!r}, "
            f"confidence={self.confidence:.3f}, "
            f"observations={self.observations}"
            ")"
        )