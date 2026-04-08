class Agent:

    def act(self, observation) -> str:
        logs    = observation.logs    if hasattr(observation, "logs")    else observation.get("logs")
        metrics = observation.metrics if hasattr(observation, "metrics") else observation.get("metrics")

        if logs is None:
            return "inspect_logs"
        if metrics is None:
            return "check_metrics"
        return self._diagnose(logs, metrics)

    def _diagnose(self, logs: str, metrics) -> str:
        mem = metrics.get("memory_usage_pct", 0) if isinstance(metrics, dict) else getattr(metrics, "memory_usage_pct", 0)
        deploys = metrics.get("recent_deployments", "None") if isinstance(metrics, dict) else getattr(metrics, "recent_deployments", "None")
        has_packet_loss = ("packet_loss_pct" in metrics) if isinstance(metrics, dict) else hasattr(metrics, "packet_loss_pct")

        if ("OutOfMemoryError" in logs or "heap" in logs.lower()) and mem > 85:
            return "restart_service"

        if ("NullPointerException" in logs or "Schema mismatch" in logs) and "None" not in str(deploys):
            return "rollback_deployment"

        if ("SocketTimeoutException" in logs or "Timeout" in logs) and has_packet_loss:
            return "escalate"

        return "escalate"