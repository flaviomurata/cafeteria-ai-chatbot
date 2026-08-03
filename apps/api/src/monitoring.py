import json
import logging
import time
from datetime import UTC, datetime


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        extra_data = getattr(record, "extra_data", None)
        if extra_data:
            log_obj.update(extra_data)
        return json.dumps(log_obj)


def get_logger(name: str = "production-api") -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


class MetricsCollector:
    def __init__(self):
        self._requests_total = 0
        self._errors_total = 0
        self._latency_sum = 0.0
        self._latency_count = 0
        self._tokens_input = 0
        self._tokens_output = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: bool = False,
        cache_hit: bool = False,
    ):
        self._requests_total += 1
        self._latency_sum += latency_ms
        self._latency_count += 1
        self._tokens_input += input_tokens
        self._tokens_output += output_tokens

        if error:
            self._errors_total += 1
        if cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

    @property
    def summary(self) -> dict:
        avg_latency = (
            self._latency_sum / self._latency_count if self._latency_count > 0 else 0.0
        )
        error_rate = (
            self._errors_total / self._requests_total
            if self._requests_total > 0
            else 0.0
        )
        cache_total = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / cache_total if cache_total > 0 else 0.0

        return {
            "total_requests": self._requests_total,
            "total_errors": self._errors_total,
            "error_rate": f"{error_rate:.2%}",
            "avg_latency_ms": round(avg_latency, 2),
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
            "total_input_tokens": self._tokens_input,
            "total_output_tokens": self._tokens_output,
        }


class RequestTimer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start) * 1000


# uv run python -c "
# from src.monitoring import get_logger, MetricsCollector, RequestTimer
# import time
#
# logger = get_logger()
# metrics = MetricsCollector()
#
# print('=== STRUCTURED LOGGING ===')
# print()
# logger.info('Application starting')
# logger.info(
#     'Processing request',
#     extra={'extra_data': {'user_id': 'user-123', 'thread_id': 'thread-456'}},
# )
# logger.warning(
#     'Rate limit approaching',
#     extra={'extra_data': {'current_rate': 18, 'limit': 20}},
# )
#
# print()
# print('=== METRICS COLLECTION ===')
# print()
#
# # Simulate some requests
# with RequestTimer() as timer:
#     time.sleep(0.1)  # Simulate work
# metrics.record_request(
#     latency_ms=timer.elapsed_ms, input_tokens=50, output_tokens=100, cache_hit=False
# )
# print(f'Request 1: {timer.elapsed_ms:.1f}ms')
#
# with RequestTimer() as timer:
#     time.sleep(0.05)
# metrics.record_request(
#     latency_ms=timer.elapsed_ms, input_tokens=30, output_tokens=80, cache_hit=True
# )
# print(f'Request 2: {timer.elapsed_ms:.1f}ms (cache hit)')
#
# metrics.record_request(latency_ms=5.0, error=True)
# print(f'Request 3: error')
#
# print()
# print('=== METRICS SUMMARY ===')
# import json
# print(json.dumps(metrics.summary, indent=2))
# "
