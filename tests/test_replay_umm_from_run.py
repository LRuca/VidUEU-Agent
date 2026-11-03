from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from replay_umm_from_run import DeepSeekLLM, replay_bundles, _resolve_deepseek_settings


class ReplayUMMFromRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("logs") / "test_tmp" / f"umm_replay_{uuid.uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _bundle(self) -> dict:
        return {
            "segment": {"index": 0, "video_id": "demo", "video_path": "videos/demo.mp4", "segment_path": "segments/demo_seg000.mp4"},
            "t1_out": {},
            "t2_out": {},
            "t3_out": {"segment_summary": "A cooking segment shows vegetables being chopped and moved into a pan."},
            "frames_aligned": [
                {"pos": 0, "origin_index": 10, "img": "f0.jpg", "caption": "A person prepares vegetables on a cutting board.", "shot_id": "shot_01", "chapter_id": "chapter_01"},
                {"pos": 1, "origin_index": 11, "img": "f1.jpg", "caption": "More chopped vegetables appear in a bowl beside the board.", "shot_id": "shot_01", "chapter_id": "chapter_01"},
                {"pos": 2, "origin_index": 12, "img": "f2.jpg", "caption": "The bowl is fuller and the knife is set down near the board.", "shot_id": "shot_01", "chapter_id": "chapter_01"},
                {"pos": 3, "origin_index": 13, "img": "f3.jpg", "caption": "The vegetables move into a pan on the stove.", "shot_id": "shot_02", "chapter_id": "chapter_01"},
            ],
            "evidence": {
                "video_id": "demo",
                "segment_index": 0,
                "segment_summary": "A cooking segment shows vegetables being chopped and moved into a pan.",
                "frames": [
                    {"pos": 0, "origin_index": 10, "img": "f0.jpg", "caption": "A person prepares vegetables on a cutting board.", "shot_id": "shot_01", "chapter_id": "chapter_01"},
                    {"pos": 1, "origin_index": 11, "img": "f1.jpg", "caption": "More chopped vegetables appear in a bowl beside the board.", "shot_id": "shot_01", "chapter_id": "chapter_01"},
                    {"pos": 2, "origin_index": 12, "img": "f2.jpg", "caption": "The bowl is fuller and the knife is set down near the board.", "shot_id": "shot_01", "chapter_id": "chapter_01"},
                    {"pos": 3, "origin_index": 13, "img": "f3.jpg", "caption": "The vegetables move into a pan on the stove.", "shot_id": "shot_02", "chapter_id": "chapter_01"},
                ],
                "evidence_units": [
                    {"evidence_id": "S_SEGEMENT_SUM", "text": "A cooking segment shows vegetables being chopped and moved into a pan."},
                    {"evidence_id": "S_SHOT_00_SUM", "text": "Shot 01 focuses on prep on the cutting board.", "shot_id": "shot_01"},
                    {"evidence_id": "S_CHAPTER_00_SUM", "text": "Chapter 01 covers meal preparation.", "chapter_id": "chapter_01"},
                    {"evidence_id": "F00_CAP", "text": "Frame F00: A person prepares vegetables on a cutting board."},
                    {"evidence_id": "F01_CAP", "text": "Frame F01: More chopped vegetables appear in a bowl beside the board."},
                    {"evidence_id": "F02_CAP", "text": "Frame F02: The bowl is fuller and the knife is set down near the board."},
                    {"evidence_id": "F03_CAP", "text": "Frame F03: The vegetables move into a pan on the stove."},
                    {"evidence_id": "P00_TRANS", "text": "Pair F00->F01: chopped vegetables accumulate in the bowl.", "pair_from_pos": 0, "pair_to_pos": 1},
                    {"evidence_id": "P00_REVIEW", "text": "Reviewed pair P00: a small local prep progression.", "pair_from_pos": 0, "pair_to_pos": 1},
                    {"evidence_id": "P01_TRANS", "text": "Pair F01->F02: the bowl becomes fuller and the knife is put down.", "pair_from_pos": 1, "pair_to_pos": 2},
                    {"evidence_id": "P02_TRANS", "text": "Pair F02->F03: prepared vegetables move from the board area into the pan.", "pair_from_pos": 2, "pair_to_pos": 3},
                ],
                "aligned_pairs": [
                    {"pair_from_pos": 0, "pair_to_pos": 1, "pair_key": "F00->F01", "transition_text": "chopped vegetables accumulate in the bowl.", "review": {"instruction": "Add more chopped vegetables into the bowl.", "edits": ["more chopped vegetables"]}},
                    {"pair_from_pos": 1, "pair_to_pos": 2, "pair_key": "F01->F02", "transition_text": "the bowl becomes fuller and the knife is put down."},
                    {"pair_from_pos": 2, "pair_to_pos": 3, "pair_key": "F02->F03", "transition_text": "prepared vegetables move from the board area into the pan."},
                ],
                "candidate_pairs": [{"from": 0, "to": 1}, {"from": 1, "to": 2}, {"from": 2, "to": 3}],
                "reviewed_ok_pairs": [{"from": 0, "to": 1}],
            },
            "evidence_units": [
                {"evidence_id": "S_SEGEMENT_SUM", "text": "A cooking segment shows vegetables being chopped and moved into a pan."},
                {"evidence_id": "S_SHOT_00_SUM", "text": "Shot 01 focuses on prep on the cutting board.", "shot_id": "shot_01"},
                {"evidence_id": "S_CHAPTER_00_SUM", "text": "Chapter 01 covers meal preparation.", "chapter_id": "chapter_01"},
                {"evidence_id": "F00_CAP", "text": "Frame F00: A person prepares vegetables on a cutting board."},
                {"evidence_id": "F01_CAP", "text": "Frame F01: More chopped vegetables appear in a bowl beside the board."},
                {"evidence_id": "F02_CAP", "text": "Frame F02: The bowl is fuller and the knife is set down near the board."},
                {"evidence_id": "F03_CAP", "text": "Frame F03: The vegetables move into a pan on the stove."},
                {"evidence_id": "P00_TRANS", "text": "Pair F00->F01: chopped vegetables accumulate in the bowl.", "pair_from_pos": 0, "pair_to_pos": 1},
                {"evidence_id": "P00_REVIEW", "text": "Reviewed pair P00: a small local prep progression.", "pair_from_pos": 0, "pair_to_pos": 1},
                {"evidence_id": "P01_TRANS", "text": "Pair F01->F02: the bowl becomes fuller and the knife is put down.", "pair_from_pos": 1, "pair_to_pos": 2},
                {"evidence_id": "P02_TRANS", "text": "Pair F02->F03: prepared vegetables move from the board area into the pan.", "pair_from_pos": 2, "pair_to_pos": 3},
            ],
            "pair_records": [
                {"pair_from_pos": 0, "pair_to_pos": 1, "pair_key": "F00->F01", "transition_text": "chopped vegetables accumulate in the bowl.", "review": {"instruction": "Add more chopped vegetables into the bowl.", "edits": ["more chopped vegetables"]}},
                {"pair_from_pos": 1, "pair_to_pos": 2, "pair_key": "F01->F02", "transition_text": "the bowl becomes fuller and the knife is put down."},
                {"pair_from_pos": 2, "pair_to_pos": 3, "pair_key": "F02->F03", "transition_text": "prepared vegetables move from the board area into the pan."},
            ],
            "candidate_pairs": [{"from": 0, "to": 1}, {"from": 1, "to": 2}, {"from": 2, "to": 3}],
            "reviewed_ok_pairs": [{"from": 0, "to": 1}],
        }

    def test_cli_replays_stage_dumps_end_to_end(self) -> None:
        debug_dir = self.root / "logs" / "debug"
        self._write_json(debug_dir / "demo_seg000_T1.json", {"output": {"video_type": "cooking"}})
        self._write_json(debug_dir / "demo_seg000_T2.json", {"output": {"tools": ["ocr"]}})
        self._write_json(debug_dir / "demo_seg000_T3.json", {"output": {"segment_summary": "A cooking segment shows vegetables being chopped and moved into a pan."}})
        self._write_json(
            debug_dir / "demo_seg000_T4.json",
            {
                "evidence": {
                    "video_id": "demo",
                    "segment_index": 0,
                    "segment_summary": "A cooking segment shows vegetables being chopped and moved into a pan.",
                    "frames": [
                        {"pos": 0, "origin_index": 10, "img": "f0.jpg", "caption": "A person prepares vegetables on a cutting board.", "shot_id": "shot_01", "chapter_id": "chapter_01"},
                        {"pos": 1, "origin_index": 11, "img": "f1.jpg", "caption": "More chopped vegetables appear in a bowl beside the board.", "shot_id": "shot_01", "chapter_id": "chapter_01"},
                        {"pos": 2, "origin_index": 12, "img": "f2.jpg", "caption": "The bowl is fuller and the knife is set down near the board.", "shot_id": "shot_01", "chapter_id": "chapter_01"},
                        {"pos": 3, "origin_index": 13, "img": "f3.jpg", "caption": "The vegetables move into a pan on the stove.", "shot_id": "shot_02", "chapter_id": "chapter_01"},
                    ],
                    "evidence_units": [
                        {"evidence_id": "S_SEGEMENT_SUM", "text": "A cooking segment shows vegetables being chopped and moved into a pan."},
                        {"evidence_id": "S_SHOT_00_SUM", "text": "Shot 01 focuses on prep on the cutting board.", "shot_id": "shot_01"},
                        {"evidence_id": "S_CHAPTER_00_SUM", "text": "Chapter 01 covers meal preparation.", "chapter_id": "chapter_01"},
                        {"evidence_id": "F00_CAP", "text": "Frame F00: A person prepares vegetables on a cutting board."},
                        {"evidence_id": "F01_CAP", "text": "Frame F01: More chopped vegetables appear in a bowl beside the board."},
                        {"evidence_id": "F02_CAP", "text": "Frame F02: The bowl is fuller and the knife is set down near the board."},
                        {"evidence_id": "F03_CAP", "text": "Frame F03: The vegetables move into a pan on the stove."},
                        {"evidence_id": "P00_TRANS", "text": "Pair F00->F01: chopped vegetables accumulate in the bowl.", "pair_from_pos": 0, "pair_to_pos": 1},
                        {"evidence_id": "P00_REVIEW", "text": "Reviewed pair P00: a small local prep progression.", "pair_from_pos": 0, "pair_to_pos": 1},
                        {"evidence_id": "P01_TRANS", "text": "Pair F01->F02: the bowl becomes fuller and the knife is put down.", "pair_from_pos": 1, "pair_to_pos": 2},
                        {"evidence_id": "P02_TRANS", "text": "Pair F02->F03: prepared vegetables move from the board area into the pan.", "pair_from_pos": 2, "pair_to_pos": 3},
                    ],
                    "aligned_pairs": [
                        {"pair_from_pos": 0, "pair_to_pos": 1, "pair_key": "F00->F01", "transition_text": "chopped vegetables accumulate in the bowl.", "review": {"instruction": "Add more chopped vegetables into the bowl.", "edits": ["more chopped vegetables"]}},
                        {"pair_from_pos": 1, "pair_to_pos": 2, "pair_key": "F01->F02", "transition_text": "the bowl becomes fuller and the knife is put down."},
                        {"pair_from_pos": 2, "pair_to_pos": 3, "pair_key": "F02->F03", "transition_text": "prepared vegetables move from the board area into the pan."},
                    ],
                    "candidate_pairs": [{"from": 0, "to": 1}, {"from": 1, "to": 2}, {"from": 2, "to": 3}],
                    "reviewed_ok_pairs": [{"from": 0, "to": 1}],
                },
                "decision": {"selected": ["edit"]},
                "runtime": {"segment_index": 0, "video_path": "videos/demo.mp4", "segment_path": "segments/demo_seg000.mp4"},
            },
        )

        out_dir = self.root / "out"
        cmd = [sys.executable, "replay_umm_from_run.py", "--input-root", str(self.root), "--output-dir", str(out_dir)]
        completed = subprocess.run(cmd, cwd=Path.cwd(), check=True, capture_output=True, text=True)
        self.assertIn("Wrote 1 bundles", completed.stdout)

        summary_path = out_dir / "replay_summary.json"
        samples_path = out_dir / "replay_samples.jsonl"
        self.assertTrue(summary_path.exists())
        self.assertTrue(samples_path.exists())

        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(summary["segment_count"], 1)
        self.assertEqual(summary["task_summary"]["unify.seed.l3.gts"]["samples"], 1)
        self.assertEqual(summary["task_summary"]["unify.isg.i2.seq_select"]["samples"], 1)

    def test_replay_bundles_accepts_llm_generation(self) -> None:
        class FakeLLM:
            def chat(self, prompt: str, max_new_tokens: int = 1200) -> str:
                if "TASK TYPE: unify.seed.l3.gts" in prompt:
                    return json.dumps(
                        {
                            "plan": {"goal": "LLM goal", "invariants": ["LLM invariant"], "edits": ["LLM edit"]},
                            "gen": {"summary": "LLM gen"},
                            "select": {"question": "LLM select?", "answer": "target:F01", "rationale": "LLM rationale"},
                        },
                        ensure_ascii=False,
                    )
                return "{}"

        result = replay_bundles([self._bundle()], llm=FakeLLM(), comm_c1_k=3, isg_k=2)
        seed_sample = result["replay_rows"][0]["samples"]["unify.seed.l3.gts"][0]
        self.assertEqual(seed_sample["turns"]["plan"]["goal"], "LLM goal")
        self.assertEqual(seed_sample["turns"]["gen"]["summary"], "LLM gen")
        self.assertEqual(seed_sample["render"]["query"]["messages"][0]["content"], "LLM goal")
        self.assertEqual(seed_sample["render"]["query"]["messages"][1]["content"], "LLM gen")
        self.assertEqual(seed_sample["render"]["answer"]["messages"][0]["content"], "target:F01")

    def test_resolve_deepseek_settings_reads_api_key_file_from_config(self) -> None:
        cfg_dir = self.root / "cfg"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        key_file = cfg_dir / "deepseek_keys.txt"
        key_file.write_text("deepseek-test-key\n", encoding="utf-8")
        config_path = cfg_dir / "config.yaml"
        config_path.write_text(
            "models:\n"
            "  deepseek_replay:\n"
            "    api:\n"
            "      base_url: https://api.deepseek.com/chat/completions\n"
            "      model: deepseek-chat\n"
            "      api_key_file: deepseek_keys.txt\n",
            encoding="utf-8",
        )

        settings = _resolve_deepseek_settings(
            config={
                "models": {
                    "deepseek_replay": {
                        "api": {
                            "base_url": "https://api.deepseek.com",
                            "model": "deepseek-chat",
                            "api_key_file": "deepseek_keys.txt",
                        }
                    }
                }
            },
            config_path=config_path,
            cli_key="",
            cli_model="",
            cli_base_url="",
        )
        self.assertEqual(settings["api_key"], "deepseek-test-key")
        self.assertEqual(settings["model"], "deepseek-chat")
        self.assertEqual(settings["base_url"], "https://api.deepseek.com/chat/completions")

    def test_resolve_deepseek_settings_reads_thinking_config(self) -> None:
        settings = _resolve_deepseek_settings(
            config={
                "models": {
                    "deepseek_replay": {
                        "api": {
                            "base_url": "https://api.deepseek.com",
                            "model": "deepseek-v4-flash",
                            "api_key": "deepseek-test-key",
                            "thinking": {"type": "disabled"},
                            "reasoning_effort": "high",
                        }
                    }
                }
            },
            config_path=self.root / "config.yaml",
            cli_key="",
            cli_model="",
            cli_base_url="",
        )
        self.assertEqual(settings["thinking"], {"type": "disabled"})
        self.assertEqual(settings["reasoning_effort"], "high")

    def test_deepseek_llm_sends_thinking_payload(self) -> None:
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")

        def fake_urlopen(request, timeout=None, context=None):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        llm = DeepSeekLLM(
            api_key="deepseek-test-key",
            model="deepseek-v4-flash",
            thinking={"type": "disabled"},
            reasoning_effort="high",
        )
        with patch("replay_umm_from_run.urllib.request.urlopen", fake_urlopen):
            self.assertEqual(llm.chat("hello", max_new_tokens=32), "ok")

        self.assertEqual(captured["payload"]["thinking"], {"type": "disabled"})
        self.assertEqual(captured["payload"]["reasoning_effort"], "high")


if __name__ == "__main__":
    unittest.main()
