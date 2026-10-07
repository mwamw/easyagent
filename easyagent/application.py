"""Public application generation and startup API."""
from application import (
    ApplicationPlan, build_application, create_application_agent,
    plan_application, run_application, write_application,
)

__all__ = ["ApplicationPlan", "build_application", "plan_application", "write_application",
           "create_application_agent", "run_application"]

if __name__ == "__main__":
    from application.__main__ import main
    raise SystemExit(main())
