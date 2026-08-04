"""Optional action lifecycle hooks for game integrations such as a UI."""

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from Skyjo.src.action import Action
    from Skyjo.src.players.player import Player
    from Skyjo.src.skyjo_game import SkyjoGame


class GameActionHooks(Protocol):
    """Receives action lifecycle events without coupling them to players."""

    def before_action(
        self, game: "SkyjoGame", player: "Player", action: "Action"
    ) -> None: ...

    def after_action(
        self, game: "SkyjoGame", player: "Player", action: "Action"
    ) -> None: ...
