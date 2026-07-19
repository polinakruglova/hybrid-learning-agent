from math import sqrt
from typing import Any

from src.state import Position, State


class PredicateGenerator:
    """
    Создаёт логические и числовые признаки из состояния среды.
    """

    def generate(self, state: State) -> dict[str, Any]:
        predicates: dict[str, Any] = {
            "has_key": state.has_key,
            "door_open": state.door_open,
            "door_locked": state.door_locked,
            "key_visible": state.key_visible,
            "door_visible": state.door_visible,
            "goal_visible": state.goal_visible,
        }

        predicates.update(
            self._generate_object_predicates(
                agent_position=state.agent_position,
                object_position=state.key_position,
                object_name="key",
            )
        )

        predicates.update(
            self._generate_object_predicates(
                agent_position=state.agent_position,
                object_position=state.door_position,
                object_name="door",
            )
        )

        predicates.update(
            self._generate_object_predicates(
                agent_position=state.agent_position,
                object_position=state.goal_position,
                object_name="goal",
            )
        )

        return predicates

    def _generate_object_predicates(
        self,
        agent_position: Position,
        object_position: Position | None,
        object_name: str,
    ) -> dict[str, Any]:
        if object_position is None:
            return self._unknown_object_predicates(object_name)

        agent_x, agent_y = agent_position
        object_x, object_y = object_position

        dx = object_x - agent_x
        dy = object_y - agent_y

        manhattan_distance = abs(dx) + abs(dy)
        euclidean_distance = sqrt(dx**2 + dy**2)

        return {
            f"distance_to_{object_name}": manhattan_distance,
            f"euclidean_distance_to_{object_name}": euclidean_distance,

            f"near_{object_name}": manhattan_distance <= 1,
            f"very_close_to_{object_name}": manhattan_distance == 1,
            f"at_{object_name}": manhattan_distance == 0,
            f"far_from_{object_name}": manhattan_distance >= 4,

            f"{object_name}_left": dx < 0,
            f"{object_name}_right": dx > 0,
            f"{object_name}_above": dy < 0,
            f"{object_name}_below": dy > 0,

            f"same_row_with_{object_name}": dy == 0,
            f"same_column_with_{object_name}": dx == 0,
        }

    @staticmethod
    def _unknown_object_predicates(
        object_name: str,
    ) -> dict[str, Any]:
        return {
            f"distance_to_{object_name}": None,
            f"euclidean_distance_to_{object_name}": None,

            f"near_{object_name}": False,
            f"very_close_to_{object_name}": False,
            f"at_{object_name}": False,
            f"far_from_{object_name}": False,

            f"{object_name}_left": False,
            f"{object_name}_right": False,
            f"{object_name}_above": False,
            f"{object_name}_below": False,

            f"same_row_with_{object_name}": False,
            f"same_column_with_{object_name}": False,
        }