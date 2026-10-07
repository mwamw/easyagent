"""Public optional UI API; terminal libraries are loaded only when requested."""
from ui import BaseAgentUI
from ui.configuration import UIRegistry, default_ui_registry, load_ui
from runtime.interactions import InteractionReply

__all__ = ["BaseAgentUI", "TerminalUI", "UIRegistry", "default_ui_registry", "load_ui", "InteractionReply"]


def __getattr__(name: str):
    if name == "TerminalUI":
        from ui import TerminalUI
        return TerminalUI
    raise AttributeError(name)


if __name__ == "__main__":
    from ui.__main__ import main
    raise SystemExit(main())
