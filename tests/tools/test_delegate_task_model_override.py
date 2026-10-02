#!/usr/bin/env python3
"""Per-task model/provider/reasoning_effort overrides on delegate_task.

A task may pin its child to another configured provider:model pair. The
override re-resolves credentials from a copy of the batch routing config;
override-less tasks reuse the batch resolution untouched (byte-identical
behavior and error text), and an unknown provider fails the whole call
before any child spawns.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tools.delegate_tool as delegate_tool_mod
from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_children,
    _strip_model_hidden_task_fields,
)
from tools.delegate_tool_tasks import _normalize_task_list

_GOAL = "Review the delegation module for regressions"
_OVERRIDE_FIELDS = ("model", "provider", "reasoning_effort")


def _parent_stub():
    return SimpleNamespace(model="parent-model", request_overrides={})


def _creds_recorder(calls):
    """Distinct credential bundle per cfg; 'bogus' provider fails like the real preflight."""
    def record(cfg, _parent):
        calls.append(cfg)
        if cfg.get("provider") == "bogus":
            raise ValueError(
                "Delegation provider 'bogus' is not a configured provider."
            )
        return {
            "model": cfg.get("model") or "batch-model",
            "provider": cfg.get("provider") or "batch-provider",
            "base_url": "https://example.invalid/v1",
            "api_key": "***",
            "api_mode": "chat_completions",
            "request_overrides": {},
        }
    return record


def _child_builder_recorder(built):
    def build(**kwargs):
        built.append(kwargs)
        return SimpleNamespace(model=kwargs.get("model"))
    return build


class TestTaskOverrideSchema(unittest.TestCase):

    def test_override_fields_on_task_items(self):
        items = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
        task_props = items["properties"]
        for field in _OVERRIDE_FIELDS:
            self.assertIn(field, task_props)
            self.assertEqual(task_props[field]["type"], "string")
            self.assertNotIn(field, items["required"])

    def test_override_fields_not_top_level(self):
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        for field in _OVERRIDE_FIELDS:
            self.assertNotIn(field, props)

    def test_hidden_field_stripping_leaves_override_fields(self):
        tasks = [{
            "goal": _GOAL, "model": "qwen3.8-27b", "provider": "snowball",
            "reasoning_effort": "high", "acp_command": "copilot", "acp_args": ["--acp"],
        }]
        stripped = _strip_model_hidden_task_fields(tasks)
        self.assertNotIn("acp_command", stripped[0])
        self.assertNotIn("acp_args", stripped[0])
        for field in _OVERRIDE_FIELDS:
            self.assertIn(field, stripped[0])


class TestNormalizeTaskListOverrides(unittest.TestCase):

    def test_valid_overrides_stripped_and_carried(self):
        tasks = [{
            "goal": _GOAL, "model": " qwen3.8-27b ", "provider": " snowball ",
            "reasoning_effort": " high ",
        }]
        normalized, error = _normalize_task_list(None, None, tasks, None, "leaf", 3)
        self.assertIsNone(error)
        assert normalized is not None
        self.assertEqual(normalized[0]["model"], "qwen3.8-27b")
        self.assertEqual(normalized[0]["provider"], "snowball")
        self.assertEqual(normalized[0]["reasoning_effort"], "high")

    def test_empty_string_drops_key(self):
        tasks = [{"goal": _GOAL, "model": "", "provider": "", "reasoning_effort": ""}]
        normalized, error = _normalize_task_list(None, None, tasks, None, "leaf", 3)
        self.assertIsNone(error)
        assert normalized is not None
        for field in _OVERRIDE_FIELDS:
            self.assertNotIn(field, normalized[0])

    def test_absent_fields_left_absent(self):
        normalized, error = _normalize_task_list(None, None, [{"goal": _GOAL}], None, "leaf", 3)
        self.assertIsNone(error)
        assert normalized is not None
        for field in _OVERRIDE_FIELDS:
            self.assertNotIn(field, normalized[0])

    def test_whitespace_only_rejected(self):
        for field in _OVERRIDE_FIELDS:
            _, error = _normalize_task_list(None, None, [{"goal": _GOAL, field: "   "}], None, "leaf", 3)
            self.assertEqual(error, f"Task 0 '{field}' must be a non-empty string")

    def test_non_string_rejected(self):
        for field in _OVERRIDE_FIELDS:
            _, error = _normalize_task_list(None, None, [{"goal": _GOAL, field: {"x": 1}}], None, "leaf", 3)
            self.assertEqual(error, f"Task 0 '{field}' must be a non-empty string")

    def test_error_names_task_index(self):
        tasks = [{"goal": _GOAL}, {"goal": _GOAL, "model": ["not-a-string"]}]
        _, error = _normalize_task_list(None, None, tasks, None, "leaf", 3)
        self.assertEqual(error, "Task 1 'model' must be a non-empty string")


def _batch_creds(routing_cfg):
    """Full bundle as _resolve_delegation_credentials would return for routing_cfg."""
    return {
        "model": routing_cfg.get("model") or "batch-model",
        "provider": routing_cfg.get("provider") or "batch-provider",
        "base_url": routing_cfg.get("base_url") or "https://example.invalid/v1",
        "api_key": "batch-key", "api_mode": "chat_completions",
        "request_overrides": {},
    }


class TestBuildChildrenPerTaskResolution(unittest.TestCase):

    def _build(self, task_list, routing_cfg, creds, parent):
        return _build_children(
            task_list, [None] * len(task_list), creds, top_role="leaf", max_iterations=10,
            parent_agent=parent, routing_cfg=routing_cfg, live_deleg_id=None, live_writers=[],
            task_images=None,
        )

    def _run(self, task_list, routing_cfg, parent):
        """Batch resolution (as delegate_task() performs) + _build_children under patched seams."""
        calls, built = [], []
        with (
            patch("tools.delegate_tool._resolve_delegation_credentials", _creds_recorder(calls)),
            patch("tools.delegate_tool._build_child_preserving_parent_tools", _child_builder_recorder(built)),
        ):
            creds = delegate_tool_mod._resolve_delegation_credentials(routing_cfg, parent)
            children, error = self._build(task_list, routing_cfg, creds, parent)
        return calls, built, children, error

    def test_override_less_tasks_reuse_batch_creds(self):
        routing_cfg = {"model": None, "provider": None, "base_url": None}
        calls, built, children, error = self._run([{"goal": _GOAL}, {"goal": _GOAL}], routing_cfg, _parent_stub())
        self.assertIsNone(error)
        self.assertEqual(len(children), 2)
        # No second resolution: override-less tasks reuse the batch creds exactly.
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], routing_cfg)
        for kwargs in built:
            self.assertEqual(kwargs["model"], "batch-model")
            self.assertEqual(kwargs["override_provider"], "batch-provider")
            self.assertIs(kwargs["routing_cfg"], routing_cfg)

    def test_task_override_triggers_second_resolution(self):
        routing_cfg = {"model": None, "provider": None, "base_url": None}
        tasks = [{"goal": _GOAL}, {"goal": _GOAL, "model": "other-model", "provider": "other-provider"}]
        calls, built, children, error = self._run(tasks, routing_cfg, _parent_stub())
        self.assertIsNone(error)
        self.assertEqual(len(children), 2)
        self.assertEqual(len(calls), 2)
        task_cfg = calls[1]
        self.assertEqual(task_cfg["model"], "other-model")
        self.assertEqual(task_cfg["provider"], "other-provider")
        # A provider override drops the batch endpoint pin (base_url short-circuits
        # provider resolution); with no pin left the untouched keys are simply absent.
        self.assertNotIn("base_url", task_cfg)
        # First child kept the batch route; second child got the task route.
        self.assertEqual(built[0]["model"], "batch-model")
        self.assertEqual(built[1]["model"], "other-model")
        self.assertEqual(built[1]["override_provider"], "other-provider")
        # Fallback policy ownership stays with the batch routing config.
        self.assertIs(built[1]["routing_cfg"], routing_cfg)

    def test_unknown_provider_fails_whole_call(self):
        routing_cfg = {"model": None, "provider": None, "base_url": None}
        tasks = [{"goal": _GOAL}, {"goal": _GOAL, "provider": "bogus"}]
        _, built, children, error = self._run(tasks, routing_cfg, _parent_stub())
        self.assertEqual(children, [])
        self.assertEqual(error, "Delegation provider 'bogus' is not a configured provider.")
        # Even though an earlier child built fine, the call returns zero children.
        self.assertEqual(len(built), 1)

    def test_child_built_with_task_model(self):
        routing_cfg = {"model": None, "provider": None, "base_url": None}
        tasks = [{"goal": _GOAL, "model": "other-model"}]
        _, built, children, error = self._run(tasks, routing_cfg, _parent_stub())
        self.assertIsNone(error)
        self.assertEqual(children[0][2].model, "other-model")
        self.assertEqual(built[0]["model"], "other-model")

    def test_provider_override_drops_batch_endpoint_pin(self):
        """A per-task provider override must drop the batch pin's base_url/api_key/api_mode.

        Those belong to the delegation pin's endpoint; keeping them sends the task's
        provider name to the wrong host because base_url short-circuits provider
        resolution in _resolve_delegation_credentials. Live-caught: a snowball task
        resolved to kimi's base_url and the child self-reported the right model while
        actually serving from the wrong endpoint.
        """
        routing_cfg = {
            "model": "batch-model", "provider": "batch-provider",
            "base_url": "https://pinned.example/v1", "api_key": "pinned-key",
            "api_mode": "chat_completions",
        }
        tasks = [{"goal": _GOAL, "model": "other-model", "provider": "snowball"}]
        calls = []
        built = []
        with patch.object(
            delegate_tool_mod, "_resolve_delegation_credentials", _creds_recorder(calls),
        ), patch.object(
            delegate_tool_mod, "_build_child_preserving_parent_tools", _child_builder_recorder(built),
        ):
            children, error = _build_children(
                tasks, [None], _batch_creds(routing_cfg),
                top_role="leaf", max_iterations=250, parent_agent=_parent_stub(),
                routing_cfg=routing_cfg, live_deleg_id=None, live_writers=[None],
            )
        self.assertIsNone(error)
        # The task resolution saw NO endpoint pin: provider resolution must be reachable.
        self.assertIsNone(calls[0].get("base_url"))
        self.assertEqual(calls[0]["provider"], "snowball")
        self.assertEqual(built[0]["override_base_url"], "https://example.invalid/v1")
        # The BATCH routing_cfg passed to the child keeps the pin untouched.
        self.assertIs(built[0]["routing_cfg"], routing_cfg)

    def test_model_only_override_keeps_batch_endpoint_pin(self):
        """A model-only override keeps the pinned endpoint: the lane stays, the model changes."""
        routing_cfg = {
            "model": "batch-model", "provider": "batch-provider",
            "base_url": "https://pinned.example/v1", "api_key": "pinned-key",
            "api_mode": "chat_completions",
        }
        tasks = [{"goal": _GOAL, "model": "other-model"}]
        calls = []
        built = []
        with patch.object(
            delegate_tool_mod, "_resolve_delegation_credentials", _creds_recorder(calls),
        ), patch.object(
            delegate_tool_mod, "_build_child_preserving_parent_tools", _child_builder_recorder(built),
        ):
            children, error = _build_children(
                tasks, [None], _batch_creds(routing_cfg),
                top_role="leaf", max_iterations=250, parent_agent=_parent_stub(),
                routing_cfg=routing_cfg, live_deleg_id=None, live_writers=[None],
            )
        self.assertIsNone(error)
        self.assertEqual(calls[0].get("base_url"), "https://pinned.example/v1")
        self.assertEqual(built[0]["override_base_url"], "https://example.invalid/v1")


if __name__ == "__main__":
    unittest.main()
