---
description: 'Loguru conventions: singleton logger usage, sink configuration at the entry point, library interception, structured context binding, and stdlib-logging idioms that do not transfer. Apply when writing or reviewing logging setup or log statements in repos that adopt loguru.'
paths:
- '**/*.py'
provenance: shared/rules/stack/loguru.md @ 85dde8d
---

You are an expert in application logging with loguru. In repos that declare loguru, this rule overrides the lang/python directive to use stdlib `logging` with `__name__` loggers; the rest of that rule (logging over print, boundaries, levels) stands.

## Principles

1. One logger, configured once: sinks are added at the application entry point and nowhere else.
2. Log records carry structure (bound context), never data interpolated into the message string.

## Directives

- `from loguru import logger` everywhere; modules log, only `main` configures.
- At the entry point, `logger.remove()` the default handler, then `logger.add(...)` each sink with explicit `level`, `rotation`/`retention` for files, and `serialize=True` where logs are machine-consumed.
- Use brace placeholders with arguments (`logger.info("fit complete in {:.1f}s", dt)`) so formatting is deferred; expensive values go through `logger.opt(lazy=True)` with callables.
- Attach context with `logger.bind(run_id=...)` (or `logger.contextualize`) instead of embedding identifiers in every message.
- In exception handlers, `logger.exception(...)` captures the traceback; `@logger.catch` only at outermost boundaries where a crash should be logged rather than propagated.
- Pass `enqueue=True` on sinks written from multiple processes.
- Route third-party stdlib logging through loguru with the documented `InterceptHandler` set via `logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)`.

## Anti-hallucination

| Banned | Correct |
|---|---|
| `logging.getLogger(__name__)` in a loguru repo | `from loguru import logger` |
| `logger.info("x=%s", x)` (printf style) | `logger.info("x={}", x)` |
| f-string interpolation of expensive values in log calls | brace placeholders or `logger.opt(lazy=True)` |
| adding sinks inside library modules | sinks only at the entry point |
| `logger.warn(...)` | `logger.warning(...)` |
