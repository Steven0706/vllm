# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from prometheus_client import REGISTRY

from vllm.config import ModelConfig, ObservabilityConfig, VllmConfig
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.metrics.stats import SchedulerIterationDetails, SchedulerStats

pytestmark = pytest.mark.cpu_test

TEST_MODEL = "distilbert/distilgpt2"


def make_config(engine_step_metrics: bool) -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(model=TEST_MODEL),
        observability_config=ObservabilityConfig(
            engine_step_metrics=engine_step_metrics
        ),
    )


def engine_step_samples() -> dict[tuple[str, str], tuple[float, float]]:
    """(phase, step_kind) -> (count, sum) of vllm:engine_step_seconds."""
    samples: dict[tuple[str, str], tuple[float, float]] = {}
    for metric in REGISTRY.collect():
        if metric.name != "vllm:engine_step_seconds":
            continue
        for sample in metric.samples:
            key = (sample.labels["phase"], sample.labels["step_kind"])
            count, total = samples.get(key, (0.0, 0.0))
            if sample.name.endswith("_count"):
                count = sample.value
            elif sample.name.endswith("_sum"):
                total = sample.value
            samples[key] = (count, total)
    return samples


def test_engine_step_histogram_not_registered_by_default():
    try:
        PrometheusStatLogger(make_config(engine_step_metrics=False))
        assert not engine_step_samples()
    finally:
        unregister_vllm_metrics()


def test_engine_step_histogram_observes_phases_by_step_kind():
    decode_step = SchedulerIterationDetails(
        iteration_index=7,
        num_ctx_requests=0,
        num_ctx_tokens=0,
        num_generation_requests=4,
        num_generation_tokens=4,
        elapsed_ms=20.0,
        schedule_ms=1.5,
        update_ms=0.5,
    )
    dummy_step = SchedulerIterationDetails(
        iteration_index=8,
        num_ctx_requests=0,
        num_ctx_tokens=0,
        num_generation_requests=0,
        num_generation_tokens=0,
        elapsed_ms=3.0,
        is_dummy=True,
    )
    try:
        logger = PrometheusStatLogger(make_config(engine_step_metrics=True))
        logger.record(SchedulerStats(iteration_details=decode_step), None)
        logger.record(SchedulerStats(iteration_details=dummy_step), None)
        logger.record(SchedulerStats(), None)

        samples = engine_step_samples()
        assert samples[("schedule", "decode")] == (1.0, pytest.approx(0.0015))
        assert samples[("execute", "decode")] == (1.0, pytest.approx(0.020))
        assert samples[("update", "decode")] == (1.0, pytest.approx(0.0005))
        for (phase, step_kind), (count, _) in samples.items():
            if step_kind != "decode":
                assert count == 0.0, (phase, step_kind)
    finally:
        unregister_vllm_metrics()
