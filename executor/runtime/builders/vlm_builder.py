from __future__ import annotations

from executor.runtime.builders.base import BaseResourceBuilder


class VLMBuilder(BaseResourceBuilder):
    def build(self, *, manager, resource_name, resource_spec, config, resolved_device):
        return manager._build_vlm()
