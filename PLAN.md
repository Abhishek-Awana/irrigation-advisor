Work on ONE step of PLAN.md at a time. Stop when its "done when" check passes.
Write a pytest test for each new function. Run tests before saying a step is done.
Do not add dependencies without asking.
Never hardcode secrets. Use .env.
Irrigation decisions live in src/engine.py only. The LLM never decides, it only explains.
Keep functions small, add type hints and short docstrings.
Show units (mm, degrees, days) in variable names.