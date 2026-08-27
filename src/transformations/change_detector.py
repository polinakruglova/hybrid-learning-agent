from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FeatureChange:
    name: str
    before: Any
    after: Any
    delta: Any = None


class ChangeDetector:
    """
    Сравнивает два состояния мира
    и возвращает только реально изменившиеся признаки.
    """

    def detect(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> list[FeatureChange]:

        changes: list[FeatureChange] = []

        all_keys = set(before) | set(after)

        for key in sorted(all_keys):
            before_value = before.get(key)
            after_value = after.get(key)

            if before_value == after_value:
                continue

            delta = self._calculate_delta(
                before_value,
                after_value,
            )

            changes.append(
                FeatureChange(
                    name=key,
                    before=before_value,
                    after=after_value,
                    delta=delta,
                )
            )

        return changes

    @staticmethod
    def _calculate_delta(
        before: Any,
        after: Any,
    ) -> Any:

        if (
            isinstance(before, (int, float))
            and isinstance(after, (int, float))
            and not isinstance(before, bool)
            and not isinstance(after, bool)
        ):
            return after - before

        return None 