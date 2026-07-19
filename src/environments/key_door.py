from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.state import State


Position = tuple[int, int]


@dataclass(frozen=True)
class StepResult:
    """
    Результат одного шага среды.

    state:
        Новое состояние после действия.

    reward:
        Награда за действие.

    done:
        Завершён ли эпизод.

    success:
        Завершился ли эпизод успешно.

    info:
        Дополнительная информация о событии.
    """

    state: State
    reward: float
    done: bool
    success: bool
    info: dict[str, Any]


class KeyDoorEnvironment:
    """
    Простая клеточная среда с ключом, дверью и целью.

    Обозначения при render():

        # — стена
        . — пустая клетка
        A — агент
        K — ключ
        D — закрытая дверь
        d — открытая дверь
        G — цель

    Действия:

        0 — движение вверх
        1 — движение вправо
        2 — движение вниз
        3 — движение влево
        4 — подобрать ключ
        5 — открыть дверь
        6 — ждать
    """

    ACTION_UP = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_PICKUP = 4
    ACTION_OPEN_DOOR = 5
    ACTION_WAIT = 6

    ACTIONS: tuple[int, ...] = (
        ACTION_UP,
        ACTION_RIGHT,
        ACTION_DOWN,
        ACTION_LEFT,
        ACTION_PICKUP,
        ACTION_OPEN_DOOR,
        ACTION_WAIT,
    )

    ACTION_NAMES: dict[int, str] = {
        ACTION_UP: "up",
        ACTION_RIGHT: "right",
        ACTION_DOWN: "down",
        ACTION_LEFT: "left",
        ACTION_PICKUP: "pickup",
        ACTION_OPEN_DOOR: "open_door",
        ACTION_WAIT: "wait",
    }

    DIRECTION_VECTORS: dict[int, Position] = {
        ACTION_UP: (0, -1),
        ACTION_RIGHT: (1, 0),
        ACTION_DOWN: (0, 1),
        ACTION_LEFT: (-1, 0),
    }

    def __init__(
        self,
        width: int = 7,
        height: int = 7,
        max_steps: int = 100,
    ) -> None:
        self._validate_configuration(
            width=width,
            height=height,
            max_steps=max_steps,
        )

        self._width = width
        self._height = height
        self._max_steps = max_steps

        self._start_position: Position = (1, 1)

        self._key_start_position: Position = (
            1,
            height - 2,
        )

        self._door_position: Position = (
            width // 2,
            height // 2,
        )

        self._goal_position: Position = (
            width - 2,
            height - 2,
        )

        self._walls = self._create_walls()

        self._agent_position: Position
        self._agent_direction: int
        self._key_position: Position | None
        self._has_key: bool
        self._door_open: bool
        self._step_count: int
        self._done: bool
        self._success: bool

        self.reset()

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def agent_position(self) -> Position:
        return self._agent_position

    @property
    def agent_direction(self) -> int:
        return self._agent_direction

    @property
    def key_position(self) -> Position | None:
        return self._key_position

    @property
    def door_position(self) -> Position:
        return self._door_position

    @property
    def goal_position(self) -> Position:
        return self._goal_position

    @property
    def has_key(self) -> bool:
        return self._has_key

    @property
    def door_open(self) -> bool:
        return self._door_open

    @property
    def door_locked(self) -> bool:
        return not self._door_open

    @property
    def done(self) -> bool:
        return self._done

    @property
    def success(self) -> bool:
        return self._success

    @property
    def walls(self) -> frozenset[Position]:
        return frozenset(self._walls)

    def reset(self) -> State:
        """
        Возвращает среду в исходное состояние.
        """
        self._agent_position = self._start_position
        self._agent_direction = self.ACTION_RIGHT

        self._key_position = self._key_start_position
        self._has_key = False

        self._door_open = False

        self._step_count = 0
        self._done = False
        self._success = False

        return self.get_state()

    def get_state(self) -> State:
        """
        Создаёт State из текущего состояния среды.
        """
        return State(
            agent_position=self._agent_position,
            agent_direction=self._agent_direction,
            key_position=self._key_position,
            door_position=self._door_position,
            goal_position=self._goal_position,
            has_key=self._has_key,
            door_open=self._door_open,
            door_locked=not self._door_open,
            key_visible=self._key_position is not None,
            door_visible=True,
            goal_visible=True,
        )

    def step(
        self,
        action: int,
    ) -> StepResult:
        """
        Выполняет одно действие агента.
        """
        self._validate_action(action)

        if self._done:
            raise RuntimeError(
                "Эпизод уже завершён. "
                "Перед новым эпизодом вызови reset()."
            )

        self._step_count += 1

        reward = -0.01
        event = "step"

        if action in self.DIRECTION_VECTORS:
            reward, event = self._move(action)

        elif action == self.ACTION_PICKUP:
            reward, event = self._pickup_key()

        elif action == self.ACTION_OPEN_DOOR:
            reward, event = self._open_door()

        elif action == self.ACTION_WAIT:
            reward = -0.02
            event = "wait"

        if (
            self._agent_position == self._goal_position
            and not self._done
        ):
            reward = 10.0
            self._done = True
            self._success = True
            event = "goal_reached"

        if (
            self._step_count >= self._max_steps
            and not self._done
        ):
            reward -= 1.0
            self._done = True
            self._success = False
            event = "max_steps_reached"

        return StepResult(
            state=self.get_state(),
            reward=float(reward),
            done=self._done,
            success=self._success,
            info={
                "event": event,
                "step_count": self._step_count,
                "action": action,
                "action_name": self.ACTION_NAMES[action],
                "has_key": self._has_key,
                "door_open": self._door_open,
                "door_locked": not self._door_open,
            },
        )

    def render(self) -> str:
        """
        Возвращает текстовую карту среды.
        """
        rows: list[str] = []

        for y in range(self._height):
            row: list[str] = []

            for x in range(self._width):
                position = (x, y)

                if position == self._agent_position:
                    symbol = "A"

                elif position == self._key_position:
                    symbol = "K"

                elif position == self._door_position:
                    symbol = (
                        "d"
                        if self._door_open
                        else "D"
                    )

                elif position == self._goal_position:
                    symbol = "G"

                elif position in self._walls:
                    symbol = "#"

                else:
                    symbol = "."

                row.append(symbol)

            rows.append("".join(row))

        return "\n".join(rows)

    def _move(
        self,
        action: int,
    ) -> tuple[float, str]:
        """
        Перемещает агента, если клетка доступна.
        """
        self._agent_direction = action

        dx, dy = self.DIRECTION_VECTORS[action]

        target_position = (
            self._agent_position[0] + dx,
            self._agent_position[1] + dy,
        )

        if target_position in self._walls:
            return -0.10, "wall_collision"

        if (
            target_position == self._door_position
            and not self._door_open
        ):
            return -0.20, "closed_door_collision"

        self._agent_position = target_position

        return -0.01, "moved"

    def _pickup_key(self) -> tuple[float, str]:
        """
        Подбирает ключ, когда он находится рядом.
        """
        if self._has_key:
            return -0.05, "already_has_key"

        if self._key_position is None:
            return -0.05, "key_unavailable"

        if not self._is_adjacent_or_same(
            self._agent_position,
            self._key_position,
        ):
            return -0.05, "key_not_near"

        self._has_key = True
        self._key_position = None

        return 1.0, "key_picked_up"

    def _open_door(self) -> tuple[float, str]:
        """
        Открывает дверь, когда агент находится рядом
        и уже имеет ключ.
        """
        if self._door_open:
            return -0.05, "door_already_open"

        if not self._has_key:
            return -0.20, "door_locked"

        if not self._is_adjacent_or_same(
            self._agent_position,
            self._door_position,
        ):
            return -0.05, "door_not_near"

        self._door_open = True

        return 2.0, "door_opened"

    def _create_walls(self) -> set[Position]:
        """
        Создаёт внешние стены и центральную перегородку.

        В перегородке оставляется клетка двери.
        """
        walls: set[Position] = set()

        for x in range(self._width):
            walls.add((x, 0))
            walls.add((x, self._height - 1))

        for y in range(self._height):
            walls.add((0, y))
            walls.add((self._width - 1, y))

        divider_x = self._door_position[0]

        for y in range(1, self._height - 1):
            position = (divider_x, y)

            if position != self._door_position:
                walls.add(position)

        return walls

    @staticmethod
    def _is_adjacent_or_same(
        first: Position,
        second: Position,
    ) -> bool:
        """
        Проверяет, находятся ли две позиции рядом
        по Манхэттенскому расстоянию.
        """
        distance = (
            abs(first[0] - second[0])
            + abs(first[1] - second[1])
        )

        return distance <= 1

    @staticmethod
    def _validate_configuration(
        width: int,
        height: int,
        max_steps: int,
    ) -> None:
        if not isinstance(width, int):
            raise TypeError(
                "width должен иметь тип int."
            )

        if not isinstance(height, int):
            raise TypeError(
                "height должен иметь тип int."
            )

        if not isinstance(max_steps, int):
            raise TypeError(
                "max_steps должен иметь тип int."
            )

        if width < 5:
            raise ValueError(
                "width должен быть не меньше 5."
            )

        if height < 5:
            raise ValueError(
                "height должен быть не меньше 5."
            )

        if max_steps <= 0:
            raise ValueError(
                "max_steps должен быть положительным."
            )

    def _validate_action(
        self,
        action: int,
    ) -> None:
        if not isinstance(action, int):
            raise TypeError(
                "action должен иметь тип int."
            )

        if action not in self.ACTIONS:
            raise ValueError(
                f"Неизвестное действие: {action}."
            )