# EasyAgent Language syntax support

This local VS Code extension contributes `.ea` syntax highlighting, bracket pairs,
and line comments. It contains no executable extension code and does not start an Agent.
It is not a language server: use `easyagent-lang check file.ea` for semantic diagnostics.

From the repository root, open a development window with:

```bash
code --extensionDevelopmentPath="$PWD/tooling/vscode" example/language/minimal.ea
```

This command is for the user to run in their graphical environment. No marketplace
installation or background model access is required. See `docs/language.md` for syntax.
