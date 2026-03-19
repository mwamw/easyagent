"""Simple manual test runner with MemoryManage-like output style."""

from __future__ import annotations

import sys
import time
import traceback
import types
from typing import Iterable, List, Type


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_section(title: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}{Colors.END}")


def print_pass(msg: str) -> None:
    print(f"  {Colors.GREEN}PASS{Colors.END} - {msg}")


def print_fail(msg: str, detail: str = "") -> None:
    suffix = f" ({detail})" if detail else ""
    print(f"  {Colors.RED}FAIL{Colors.END} - {msg}{suffix}")


def print_case_start(index: int, total: int, name: str, desc: str = "") -> None:
    suffix = f" | {desc}" if desc else ""
    print(f"  {Colors.YELLOW}[{index}/{total}]{Colors.END} RUN  - {name}{suffix}")


def _short(value: object, max_len: int = 100) -> str:
    text = repr(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_assert_observation(method_name: str, args: tuple) -> str:
    if method_name in {"assertEqual", "assertNotEqual"} and len(args) >= 2:
        return f"actual={_short(args[0])}, expected={_short(args[1])}"
    if method_name in {"assertIn", "assertNotIn"} and len(args) >= 2:
        return f"member={_short(args[0])}, container={_short(args[1])}"
    if method_name in {"assertTrue", "assertFalse"} and len(args) >= 1:
        return f"value={_short(args[0])}"
    if method_name in {"assertGreater", "assertGreaterEqual", "assertLess", "assertLessEqual"} and len(args) >= 2:
        return f"left={_short(args[0])}, right={_short(args[1])}"
    if method_name in {"assertIsInstance"} and len(args) >= 2:
        return f"obj={_short(args[0])}, type={_short(args[1])}"
    return f"args={_short(args)}"


def _attach_assert_logging(case: object) -> None:
    methods = [
        "assertEqual",
        "assertNotEqual",
        "assertTrue",
        "assertFalse",
        "assertIn",
        "assertNotIn",
        "assertGreater",
        "assertGreaterEqual",
        "assertLess",
        "assertLessEqual",
        "assertIsInstance",
    ]

    for method_name in methods:
        original = getattr(case, method_name, None)
        if not callable(original):
            continue

        def _make_wrapper(name: str, fn):
            def _wrapper(self, *args, **kwargs):
                observation = _format_assert_observation(name, args)
                print(f"    CHECK {name} - {observation}")
                return fn(*args, **kwargs)

            return _wrapper

        wrapped = _make_wrapper(method_name, original)
        setattr(case, method_name, types.MethodType(wrapped, case))


def _iter_test_methods(case_instance: object) -> List[str]:
    names = []
    for name in dir(case_instance):
        if name.startswith("test_") and callable(getattr(case_instance, name)):
            names.append(name)
    return sorted(names)


def run_manual_tests(test_classes: Iterable[Type[object]], title: str) -> bool:
    """Run unittest-style classes manually and print a compact summary."""
    passed = 0
    failed = 0
    errors: List[str] = []

    print_section(title)

    for test_class in test_classes:
        class_name = getattr(test_class, "__name__", str(test_class))
        print_section(f"Running {class_name}")
        class_passed = 0
        class_failed = 0

        try:
            case = test_class()
            _attach_assert_logging(case)
        except Exception as exc:  # pragma: no cover - defensive
            failed += 1
            msg = f"{class_name} __init__"
            print_fail(msg, str(exc))
            errors.append(f"{msg}: {exc}")
            traceback.print_exc()
            continue

        methods = _iter_test_methods(case)
        total_methods = len(methods)

        for index, method_name in enumerate(methods, 1):
            method = getattr(case, method_name)
            desc = (getattr(method, "__doc__", "") or "").strip().replace("\n", " ")
            print_case_start(index, total_methods, f"{class_name}.{method_name}", desc)

            start = time.perf_counter()
            try:
                setup = getattr(case, "setUp", None)
                if callable(setup):
                    setup()

                method()
                elapsed = time.perf_counter() - start
                print_pass(f"{class_name}.{method_name} ({elapsed:.3f}s)")
                passed += 1
                class_passed += 1
            except Exception as exc:
                elapsed = time.perf_counter() - start
                print_fail(f"{class_name}.{method_name} ({elapsed:.3f}s)", str(exc))
                traceback.print_exc()
                failed += 1
                class_failed += 1
                errors.append(f"{class_name}.{method_name}: {exc}")
            finally:
                teardown = getattr(case, "tearDown", None)
                if callable(teardown):
                    try:
                        teardown()
                    except Exception as exc:
                        print_fail(f"{class_name}.{method_name}.tearDown", str(exc))
                        traceback.print_exc()
                        failed += 1
                        errors.append(f"{class_name}.{method_name}.tearDown: {exc}")

        print(
            f"  {Colors.CYAN}Class Summary{Colors.END} - {class_name}: "
            f"passed={class_passed}, failed={class_failed}"
        )

    print_section("Summary")
    total = passed + failed
    print(f"  Total: {total}")
    print(f"  {Colors.GREEN}Passed: {passed}{Colors.END}")
    print(f"  {Colors.RED}Failed: {failed}{Colors.END}")

    if errors:
        print(f"\n  {Colors.YELLOW}Failed Cases:{Colors.END}")
        for i, err in enumerate(errors, 1):
            print(f"    {i}. {err}")

    return failed == 0


def exit_with_status(ok: bool) -> None:
    sys.exit(0 if ok else 1)
