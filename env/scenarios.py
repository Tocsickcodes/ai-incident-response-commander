# env/scenarios.py
# Defines all incident scenarios the environment can simulate

SCENARIOS = {
    "memory_leak": {
        "name": "Memory Leak",
        "alerts": (
            "🚨 ALERT: Service 'api-gateway' memory usage at 94% (threshold: 80%)\n"
            "🚨 ALERT: Response times degraded — p99 latency > 3000ms\n"
            "⚠️  ALERT: OOMKiller triggered on pod api-gateway-7f9b4"
        ),
        "logs": (
            "ERROR [api-gateway] Heap size exceeded: allocated 7.8GB of 8GB\n"
            "WARN  [api-gateway] GC pause > 500ms — 23 occurrences in last 5 min\n"
            "ERROR [api-gateway] java.lang.OutOfMemoryError: Java heap space\n"
            "INFO  [api-gateway] Request queue depth: 1842 (normal: <50)\n"
            "WARN  [api-gateway] Connection pool exhausted — threads waiting"
        ),
        "metrics": {
            "memory_usage_pct": 94,
            "cpu_usage_pct": 41,
            "error_rate_pct": 18,
            "latency_p99_ms": 3200,
            "recent_deployments": "None in last 72h",
            "gc_pause_count": 23,
        },
        "root_cause": "memory_leak",
        "correct_action": "restart_service",
        "fix_hint_in_logs": "OutOfMemoryError",
        "fix_hint_in_metrics": "memory_usage_pct",
    },

    "bad_deployment": {
        "name": "Bad Deployment",
        "alerts": (
            "🚨 ALERT: Error rate spiked to 32% (threshold: 5%)\n"
            "🚨 ALERT: Service 'checkout-service' health check failing\n"
            "⚠️  ALERT: Deployment 'checkout-service:v2.4.1' completed 8 minutes ago"
        ),
        "logs": (
            "ERROR [checkout-service] NullPointerException at PaymentProcessor.java:142\n"
            "ERROR [checkout-service] Failed to process payment — unexpected null cart_id\n"
            "WARN  [checkout-service] Schema mismatch: field 'discount_code' not found\n"
            "ERROR [checkout-service] 500 Internal Server Error — /api/checkout (x847)\n"
            "INFO  [checkout-service] Version: v2.4.1 deployed at 14:22 UTC"
        ),
        "metrics": {
            "memory_usage_pct": 52,
            "cpu_usage_pct": 38,
            "error_rate_pct": 32,
            "latency_p99_ms": 980,
            "recent_deployments": "checkout-service:v2.4.1 deployed 8 min ago",
            "gc_pause_count": 0,
        },
        "root_cause": "bad_deployment",
        "correct_action": "rollback_deployment",
        "fix_hint_in_logs": "NullPointerException",
        "fix_hint_in_metrics": "recent_deployments",
    },

    "network_issue": {
        "name": "Network Issue",
        "alerts": (
            "🚨 ALERT: Inter-service timeout rate > 40% between 'orders' and 'inventory'\n"
            "🚨 ALERT: Packet loss detected on subnet 10.0.4.0/24\n"
            "⚠️  ALERT: Circuit breaker OPEN on inventory-service client"
        ),
        "logs": (
            "ERROR [orders-service] Timeout connecting to inventory-service (5000ms)\n"
            "ERROR [orders-service] java.net.SocketTimeoutException: Read timed out\n"
            "WARN  [orders-service] Retry attempt 3/3 — all failed\n"
            "ERROR [orders-service] Circuit breaker tripped — falling back to cache\n"
            "INFO  [orders-service] Memory: 41%, CPU: 29% — service itself is healthy"
        ),
        "metrics": {
            "memory_usage_pct": 41,
            "cpu_usage_pct": 29,
            "error_rate_pct": 43,
            "latency_p99_ms": 5100,
            "recent_deployments": "None in last 48h",
            "packet_loss_pct": 38,
        },
        "root_cause": "network_issue",
        "correct_action": "escalate",
        "fix_hint_in_logs": "SocketTimeoutException",
        "fix_hint_in_metrics": "packet_loss_pct",
    },
}