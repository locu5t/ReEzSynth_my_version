"""Public pipeline entry point; legacy projects keep their original behavior."""
from .legacy_pipeline import SynthesisPipeline as LegacySynthesisPipeline


class SynthesisPipeline(LegacySynthesisPipeline):
    def run(self):
        if not self.config.temporal.enabled:
            return super().run()
        from .temporal.pipeline import TemporalPipeline
        return TemporalPipeline(self.config, self.data, self.synthesis_engine).run()
