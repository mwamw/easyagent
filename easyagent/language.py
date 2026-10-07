"""Public language API, independent of the existing Python/YAML entrypoints."""
from language import CompiledProgram, LanguageError, Span, compile_file, parse

__all__ = ["compile_file", "parse", "CompiledProgram", "LanguageError", "Span"]

if __name__ == "__main__":
    from language.__main__ import main
    raise SystemExit(main())
