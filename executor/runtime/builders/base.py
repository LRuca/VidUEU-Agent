from __future__ import annotations

from typing import Any, Dict


class BaseResourceBuilder:
    def build(
        self,
        *,
        manager: Any,
        resource_name: str,
        resource_spec: Dict[str, Any],
        config: Dict[str, Any],
        resolved_device: str | None,
    ) -> Any:
        raise NotImplementedError
