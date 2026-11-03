from __future__ import annotations

import unittest
from unittest.mock import patch

from orchestrator.umm_task_producer import (
    SEGMENT_EVIDENCE_ID,
    TASK_COMM_C1,
    TASK_COMM_C2,
    TASK_ISG,
    TASK_SEED,
    TASK_WEAVE,
    attach_evidence_refs,
    build_comm_c1_candidates,
    build_comm_c2_candidates,
    build_isg_candidates,
    build_seed_candidates,
    build_task_context,
    build_weave_candidates,
    _normalize_transition_text,
    produce_task_samples,
)


class UMMTaskProducerTest(unittest.TestCase):
    def _bundle(self) -> dict:
        frames = [
            {"pos": 0, "origin_index": 10, "img": "f0.jpg", "caption": "A person prepares vegetables on a cutting board.", "shot_id": "shot_01", "chapter_id": "chapter_01", "frame_tool_descs": {"ocr": "A recipe card sits beside the board."}},
            {"pos": 1, "origin_index": 11, "img": "f1.jpg", "caption": "More chopped vegetables appear in a bowl beside the board.", "shot_id": "shot_01", "chapter_id": "chapter_01"},
            {"pos": 2, "origin_index": 12, "img": "f2.jpg", "caption": "The bowl is fuller and the knife is set down near the board.", "shot_id": "shot_01", "chapter_id": "chapter_01"},
            {"pos": 3, "origin_index": 13, "img": "f3.jpg", "caption": "The vegetables move into a pan on the stove.", "shot_id": "shot_02", "chapter_id": "chapter_01"},
        ]
        evidence_units = [
            {"evidence_id": "S_SEGEMENT_SUM", "text": "A cooking segment shows vegetables being chopped and moved into a pan."},
            {"evidence_id": "S_SHOT_00_SUM", "text": "Shot 01 focuses on prep on the cutting board.", "shot_id": "shot_01"},
            {"evidence_id": "S_SHOT_01_SUM", "text": "Shot 02 shows transfer into the pan.", "shot_id": "shot_02"},
            {"evidence_id": "S_CHAPTER_00_SUM", "text": "Chapter 01 covers meal preparation.", "chapter_id": "chapter_01"},
            {"evidence_id": "F00_CAP", "text": "Frame F00: A person prepares vegetables on a cutting board."},
            {"evidence_id": "F01_CAP", "text": "Frame F01: More chopped vegetables appear in a bowl beside the board."},
            {"evidence_id": "F02_CAP", "text": "Frame F02: The bowl is fuller and the knife is set down near the board."},
            {"evidence_id": "F03_CAP", "text": "Frame F03: The vegetables move into a pan on the stove."},
            {"evidence_id": "P00_TRANS", "text": "Pair F00->F01: chopped vegetables accumulate in the bowl.", "pair_from_pos": 0, "pair_to_pos": 1},
            {"evidence_id": "P00_REVIEW", "text": "Reviewed pair P00: a small local prep progression.", "pair_from_pos": 0, "pair_to_pos": 1},
            {"evidence_id": "P01_TRANS", "text": "Pair F01->F02: the bowl becomes fuller and the knife is put down.", "pair_from_pos": 1, "pair_to_pos": 2},
            {"evidence_id": "P02_TRANS", "text": "Pair F02->F03: prepared vegetables move from the board area into the pan.", "pair_from_pos": 2, "pair_to_pos": 3},
        ]
        pair_records = [
            {"pair_from_pos": 0, "pair_to_pos": 1, "pair_key": "F00->F01", "transition_text": "chopped vegetables accumulate in the bowl.", "review": {"instruction": "Add more chopped vegetables into the bowl.", "edits": ["more chopped vegetables"]}},
            {"pair_from_pos": 1, "pair_to_pos": 2, "pair_key": "F01->F02", "transition_text": "the bowl becomes fuller and the knife is put down."},
            {"pair_from_pos": 2, "pair_to_pos": 3, "pair_key": "F02->F03", "transition_text": "prepared vegetables move from the board area into the pan."},
        ]
        return {
            "segment": {"index": 7, "video_id": "demo_video"},
            "t1_out": {},
            "t2_out": {},
            "t3_out": {"segment_summary": "A cooking segment shows vegetables being chopped and moved into a pan."},
            "frames_aligned": frames,
            "evidence": {"video_id": "demo_video", "segment_index": 7, "segment_summary": "A cooking segment shows vegetables being chopped and moved into a pan.", "frames": frames, "evidence_units": evidence_units, "aligned_pairs": pair_records, "candidate_pairs": [{"from": 0, "to": 1}, {"from": 1, "to": 2}, {"from": 2, "to": 3}], "reviewed_ok_pairs": [{"from": 0, "to": 1}]},
            "evidence_units": evidence_units,
            "pair_records": pair_records,
            "candidate_pairs": [{"from": 0, "to": 1}, {"from": 1, "to": 2}, {"from": 2, "to": 3}],
            "reviewed_ok_pairs": [{"from": 0, "to": 1}],
        }

    def _news_bundle(self) -> dict:
        frames = [
            {"pos": 0, "origin_index": 20, "img": "n0.jpg", "caption": "Two anchors sit at a morning news desk with a TODAY logo in the corner.", "shot_id": "shot_news", "chapter_id": "chapter_news"},
            {"pos": 1, "origin_index": 21, "img": "n1.jpg", "caption": "The same anchors remain at the desk, now with a TODAY.com logo and a blurred pedestrian outside the window.", "shot_id": "shot_news", "chapter_id": "chapter_news"},
            {"pos": 2, "origin_index": 22, "img": "n2.jpg", "caption": "The anchors remain at the desk with the TODAY.com logo and the blurred pedestrian, while the camera crops in slightly closer.", "shot_id": "shot_news", "chapter_id": "chapter_news"},
            {"pos": 3, "origin_index": 23, "img": "n3.jpg", "caption": "The tighter crop remains, and one anchor raises a sheet of paper while the outdoor pedestrian is still visible.", "shot_id": "shot_news", "chapter_id": "chapter_news"},
        ]
        evidence_units = [
            {"evidence_id": "S_SEGEMENT_SUM", "text": "A news segment keeps the same studio shot while small grounded updates happen across consecutive frames."},
            {"evidence_id": "S_SHOT_00_SUM", "text": "A single stable shot shows anchors at the desk while on-screen details and pose cues change.", "shot_id": "shot_news"},
            {"evidence_id": "S_CHAPTER_00_SUM", "text": "The chapter covers a stable studio sequence with incremental visual progression.", "chapter_id": "chapter_news"},
            {"evidence_id": "F00_CAP", "text": "Frame F00: Two anchors sit at a morning news desk with a TODAY logo in the corner."},
            {"evidence_id": "F01_CAP", "text": "Frame F01: The same anchors remain at the desk, now with a TODAY.com logo and a blurred pedestrian outside the window."},
            {"evidence_id": "F02_CAP", "text": "Frame F02: The anchors remain at the desk with the TODAY.com logo and the blurred pedestrian, while the camera crops in slightly closer."},
            {"evidence_id": "F03_CAP", "text": "Frame F03: The tighter crop remains, and one anchor raises a sheet of paper while the outdoor pedestrian is still visible."},
            {"evidence_id": "P00_TRANS", "text": "Pair F00->F01: update the logo to TODAY.com and add a blurred pedestrian outside the window.", "pair_from_pos": 0, "pair_to_pos": 1},
            {"evidence_id": "P00_REVIEW", "text": "Reviewed pair P00: a grounded local studio update with logo and background pedestrian.", "pair_from_pos": 0, "pair_to_pos": 1},
            {"evidence_id": "P01_TRANS", "text": "Pair F01->F02: keep the same anchors and background but crop slightly closer.", "pair_from_pos": 1, "pair_to_pos": 2},
            {"evidence_id": "P02_TRANS", "text": "Pair F02->F03: preserve the tighter crop and add a raised sheet of paper in one anchor's hand.", "pair_from_pos": 2, "pair_to_pos": 3},
        ]
        pair_records = [
            {"pair_from_pos": 0, "pair_to_pos": 1, "pair_key": "F00->F01", "transition_text": "update the logo to TODAY.com and add a blurred pedestrian outside the window.", "review": {"instruction": "Change the logo to TODAY.com and add a blurred pedestrian outside the window.", "edits": ["logo update", "background pedestrian"]}},
            {"pair_from_pos": 1, "pair_to_pos": 2, "pair_key": "F01->F02", "transition_text": "keep the same anchors and background but crop slightly closer."},
            {"pair_from_pos": 2, "pair_to_pos": 3, "pair_key": "F02->F03", "transition_text": "preserve the tighter crop and add a raised sheet of paper in one anchor's hand."},
        ]
        return {
            "segment": {"index": 11, "video_id": "news_demo"},
            "t1_out": {},
            "t2_out": {},
            "t3_out": {"segment_summary": "A news segment keeps the same studio shot while small grounded updates happen across consecutive frames."},
            "frames_aligned": frames,
            "evidence": {"video_id": "news_demo", "segment_index": 11, "segment_summary": "A news segment keeps the same studio shot while small grounded updates happen across consecutive frames.", "frames": frames, "evidence_units": evidence_units, "aligned_pairs": pair_records, "candidate_pairs": [{"from": 0, "to": 1}, {"from": 1, "to": 2}, {"from": 2, "to": 3}], "reviewed_ok_pairs": [{"from": 0, "to": 1}]},
            "evidence_units": evidence_units,
            "pair_records": pair_records,
            "candidate_pairs": [{"from": 0, "to": 1}, {"from": 1, "to": 2}, {"from": 2, "to": 3}],
            "reviewed_ok_pairs": [{"from": 0, "to": 1}],
        }

    def _bundle_without_reviewed_pairs(self) -> dict:
        bundle = self._bundle()
        bundle["evidence_units"] = [
            unit for unit in bundle["evidence_units"] if str(unit.get("evidence_id", "")) != "P00_REVIEW"
        ]
        bundle["pair_records"] = [
            {
                **record,
                "review": {},
            }
            for record in bundle["pair_records"]
        ]
        bundle["reviewed_ok_pairs"] = []
        evidence = dict(bundle["evidence"])
        evidence["evidence_units"] = list(bundle["evidence_units"])
        evidence["aligned_pairs"] = list(bundle["pair_records"])
        evidence["reviewed_ok_pairs"] = []
        bundle["evidence"] = evidence
        return bundle

    def test_context_and_candidate_builders(self) -> None:
        ctx = build_task_context(self._bundle())
        self.assertTrue(build_seed_candidates(ctx))
        self.assertTrue(build_weave_candidates(ctx))
        self.assertEqual(build_comm_c1_candidates(ctx, k=3)[0].frame_positions, [0, 1, 2, 3])
        self.assertEqual(build_comm_c2_candidates(ctx)[0].frame_positions, [0, 1, 2])
        isg = build_isg_candidates(ctx, k=2)[0]
        self.assertEqual(isg.context_positions, [0, 1])
        self.assertEqual(isg.future_positions, [2, 3])

    def test_seed_requires_reviewed_pairs_but_non_edit_tasks_keep_context_coverage(self) -> None:
        ctx = build_task_context(self._bundle_without_reviewed_pairs())
        self.assertEqual(build_seed_candidates(ctx), [])
        self.assertEqual(build_weave_candidates(ctx), [])
        self.assertEqual(build_comm_c1_candidates(ctx, k=3), [])
        self.assertEqual(build_comm_c2_candidates(ctx), [])
        self.assertEqual(build_isg_candidates(ctx, k=2), [])

    def test_non_edit_tasks_can_expand_around_reviewed_anchor(self) -> None:
        ctx = build_task_context(self._bundle())
        self.assertEqual(build_weave_candidates(ctx)[0].pair_key, "F00->F01")
        self.assertEqual(build_comm_c1_candidates(ctx, k=3)[0].frame_positions, [0, 1, 2, 3])
        self.assertEqual(build_comm_c2_candidates(ctx)[0].frame_positions, [0, 1, 2])
        isg = build_isg_candidates(ctx, k=2)[0]
        self.assertEqual(isg.context_positions, [0, 1])
        self.assertEqual(isg.future_positions, [2, 3])

    def test_evidence_closure_contains_segment_and_pair_support(self) -> None:
        ctx = build_task_context(self._bundle())
        closure = attach_evidence_refs(ctx, TASK_SEED, build_seed_candidates(ctx)[0])
        self.assertIn(SEGMENT_EVIDENCE_ID, closure.evidence_ids)
        self.assertIn("P00_TRANS", closure.evidence_ids)
        self.assertIn("F00_CAP", closure.evidence_ids)
        self.assertIn("F01_CAP", closure.evidence_ids)

    def test_produce_task_samples_builds_all_requested_task_types(self) -> None:
        result = produce_task_samples(self._bundle(), comm_c1_k=3, isg_k=2)
        samples = result["samples"]
        self.assertEqual(len(samples[TASK_SEED]), 1)
        self.assertEqual(len(samples[TASK_WEAVE]), 1)
        self.assertEqual(len(samples[TASK_COMM_C1]), 1)
        self.assertEqual(len(samples[TASK_COMM_C2]), 1)
        self.assertEqual(len(samples[TASK_ISG]), 1)
        self.assertEqual(samples[TASK_COMM_C1][0]["render"]["frame_positions"], [0, 1, 2, 3])
        self.assertEqual(samples[TASK_COMM_C2][0]["answer"]["target_frame_pos"], 1)
        self.assertEqual(samples[TASK_ISG][0]["answer"]["future_frame_positions"], [2, 3])
        self.assertIn("query", samples[TASK_SEED][0]["render"])
        self.assertIn("answer", samples[TASK_SEED][0]["render"])

    def test_rulebased_render_messages_are_attached_for_unify_samples(self) -> None:
        samples = produce_task_samples(self._bundle(), comm_c1_k=3, isg_k=2)["samples"]

        seed_render = samples[TASK_SEED][0]["render"]
        self.assertEqual(seed_render["query"]["images"], ["f0.jpg"])
        self.assertEqual(seed_render["answer"]["images"], ["f1.jpg"])
        self.assertEqual(seed_render["query"]["messages"][0]["role"], "user")
        self.assertEqual(seed_render["query"]["messages"][1]["role"], "assistant")
        self.assertIn("Which candidate image", seed_render["query"]["messages"][-1]["content"])
        self.assertEqual(seed_render["answer"]["messages"][0]["content"], "target:F01")

        comm_c1_render = samples[TASK_COMM_C1][0]["render"]
        self.assertEqual(comm_c1_render["query"]["images"], ["f0.jpg"])
        self.assertEqual(comm_c1_render["answer"]["images"], ["f1.jpg", "f2.jpg", "f3.jpg"])
        self.assertIn("required intermediate state", comm_c1_render["query"]["messages"][0]["content"])
        self.assertTrue(any(msg["content"].startswith("For step 1") for msg in comm_c1_render["query"]["messages"]))
        self.assertEqual(comm_c1_render["answer"]["messages"][0]["content"], "target:F01")

        isg_render = samples[TASK_ISG][0]["render"]
        self.assertEqual(isg_render["query"]["images"], ["f0.jpg", "f1.jpg"])
        self.assertEqual(isg_render["answer"]["images"], ["f2.jpg", "f3.jpg"])
        self.assertTrue(any("immediate next future step" in msg["content"].lower() for msg in isg_render["query"]["messages"]))
        self.assertEqual(isg_render["answer"]["messages"][0]["content"], "target:F02")

    def test_produce_task_samples_drops_seed_when_no_reviewed_pairs_but_keeps_non_edit_tasks(self) -> None:
        result = produce_task_samples(self._bundle_without_reviewed_pairs(), comm_c1_k=3, isg_k=2)
        samples = result["samples"]
        self.assertEqual(len(samples[TASK_SEED]), 0)
        self.assertEqual(len(samples[TASK_WEAVE]), 0)
        self.assertEqual(len(samples[TASK_COMM_C1]), 0)
        self.assertEqual(len(samples[TASK_COMM_C2]), 0)
        self.assertEqual(len(samples[TASK_ISG]), 0)

    def test_comm_c2_defaults_stay_observed_and_non_speculative(self) -> None:
        sample = produce_task_samples(self._bundle(), comm_c1_k=3, isg_k=2)["samples"][TASK_COMM_C2][0]
        turns = sample["turns"]
        self.assertIn("middle step", turns["explanation"].lower())
        self.assertIn("candidate", turns["select"]["question"].lower())
        self.assertNotIn("not explicitly shown", turns["explanation"].lower())
        self.assertNotIn("might be", turns["explanation"].lower())
        self.assertNotIn("not yet", turns["select"]["rationale"].lower())
        self.assertEqual(turns["select"]["answer"], "target:F01")

    def test_isg_defaults_emphasize_immediate_future_progression(self) -> None:
        sample = produce_task_samples(self._bundle(), comm_c1_k=3, isg_k=2)["samples"][TASK_ISG][0]
        turns = sample["turns"]
        self.assertIn("immediate next future step", turns["overall"].lower())
        self.assertIn("next transition after the context", turns["because"][0].lower())
        self.assertIn("keep the context", turns["check"][0].lower())
        self.assertIn("add the next-step detail", turns["check"][0].lower())
        self.assertIn("immediate next future step", turns["select"][0]["question"].lower())
        self.assertTrue(turns["select"][0]["rationale"].startswith("Choose F02 because it is the next frame after the context"))

    def test_all_task_types_share_benchmark_select_style(self) -> None:
        samples = produce_task_samples(self._news_bundle(), comm_c1_k=3, isg_k=2)["samples"]

        seed_select = samples[TASK_SEED][0]["turns"]["select"]
        weave_select = samples[TASK_WEAVE][0]["turns"]["select"]
        comm_c2_select = samples[TASK_COMM_C2][0]["turns"]["select"]
        for select in (seed_select, weave_select, comm_c2_select):
            self.assertIn("candidate", select["question"].lower())
            self.assertTrue(select["answer"].startswith("target:F"))
            self.assertIn("because", select["rationale"].lower())

        for select in samples[TASK_COMM_C1][0]["turns"]["select"]:
            self.assertIn("candidate", select["question"].lower())
            self.assertIn("step", select["question"].lower())
            self.assertTrue(select["answer"].startswith("target:F"))
            self.assertIn("because", select["rationale"].lower())

        for select in samples[TASK_ISG][0]["turns"]["select"]:
            self.assertIn("candidate", select["question"].lower())
            self.assertIn("future", select["question"].lower())
            self.assertTrue(select["answer"].startswith("target:F"))
            self.assertIn("because", select["rationale"].lower())

    def test_news_simulation_bundle_produces_understanding_grounded_outputs(self) -> None:
        samples = produce_task_samples(self._news_bundle(), comm_c1_k=3, isg_k=2)["samples"]

        self.assertIn("updated scene", samples[TASK_SEED][0]["turns"]["select"]["question"].lower())
        self.assertIn("earlier state", samples[TASK_WEAVE][0]["turns"]["select"]["question"].lower())
        self.assertIn("sequence unfolds step by step", samples[TASK_COMM_C1][0]["turns"]["overall"].lower())
        self.assertIn("middle step", samples[TASK_COMM_C2][0]["turns"]["explanation"].lower())
        self.assertIn("given context frames", samples[TASK_ISG][0]["turns"]["overall"].lower())

    def test_task_defaults_reflect_benchmark_specific_style(self) -> None:
        samples = produce_task_samples(self._news_bundle(), comm_c1_k=3, isg_k=2)["samples"]
        self.assertIn("the correct image shows", samples[TASK_SEED][0]["turns"]["plan"]["goal"].lower())
        self.assertIn("compared with f00", samples[TASK_WEAVE][0]["turns"]["qa_open"]["answer"].lower())
        self.assertIn("from f00 to f01", samples[TASK_COMM_C1][0]["turns"]["steps"][0].lower())
        self.assertIn("middle step", samples[TASK_COMM_C2][0]["turns"]["question"].lower())
        self.assertIn("after f01", samples[TASK_ISG][0]["turns"]["edits"][0].lower())

    def test_defaults_use_natural_constraint_language(self) -> None:
        samples = produce_task_samples(self._news_bundle(), comm_c1_k=3, isg_k=2)["samples"]

        seed_invariants = samples[TASK_SEED][0]["turns"]["plan"]["invariants"]
        self.assertTrue(seed_invariants[0].startswith("Keep the same core scene"))
        self.assertTrue(seed_invariants[1].startswith("Do not introduce"))

        weave_rubric = samples[TASK_WEAVE][0]["turns"]["qa_open"]["rubric"]
        self.assertTrue(any(item.startswith("Stay consistent") for item in weave_rubric))
        self.assertTrue(any(item.startswith("Mention only") for item in weave_rubric))
        self.assertTrue(any(item.startswith("Do not add") for item in weave_rubric))

        isg_check = samples[TASK_ISG][0]["turns"]["check"][0]
        self.assertIn("Keep the context", isg_check)
        self.assertIn("Add the next-step detail", isg_check)
        self.assertIn("Avoid later changes", isg_check)

    def test_compact_phrase_avoids_mid_clause_truncation_noise(self) -> None:
        sample = produce_task_samples(self._news_bundle(), comm_c1_k=3, isg_k=2)["samples"][TASK_COMM_C2][0]
        explanation = sample["turns"]["explanation"]
        self.assertNotIn("whil'", explanation)
        self.assertNotIn("came'", explanation)

    def test_defaults_remove_structured_edits_markers_from_text(self) -> None:
        samples = produce_task_samples(self._news_bundle(), comm_c1_k=3, isg_k=2)["samples"]
        text_chunks = [
            samples[TASK_SEED][0]["turns"]["select"]["rationale"],
            samples[TASK_WEAVE][0]["turns"]["select"]["rationale"],
            samples[TASK_COMM_C1][0]["turns"]["select"][0]["rationale"],
            samples[TASK_COMM_C2][0]["turns"]["explanation"],
            samples[TASK_ISG][0]["turns"]["overall"],
            samples[TASK_ISG][0]["turns"]["select"][0]["rationale"],
        ]
        for text in text_chunks:
            self.assertNotIn("Edits(", text)

    def test_transition_text_is_naturalized_from_structured_edits(self) -> None:
        text = _normalize_transition_text(
            "Edits(2): 1) remove sandwich from woman's hands; 2) add cutting board with bread slices on counter | Shot: minor | View: minor"
        )
        self.assertEqual(
            text,
            "sandwich from woman's hands is removed and cutting board with bread slices on counter appears",
        )

    def test_llm_retry_attempts_can_recover_flaky_comm_c1_generation(self) -> None:
        call_counter = {"count": 0}

        def flaky_comm_c1(ctx, candidate, llm=None):
            call_counter["count"] += 1
            if call_counter["count"] == 1:
                return {
                    "task_type": TASK_COMM_C1,
                    "turns": {
                        "question": "bad",
                        "overall": "bad",
                        "steps": ["bad", "bad"],
                        "select": [{"question": "bad", "answer": "wrong", "rationale": "bad"}],
                    },
                    "answer": {"target_frame_positions": [1, 2]},
                    "render": {"kind": "interleaved_mm", "frame_positions": [0, 1, 2]},
                    "evidence_refs": ["S_SEGEMENT_SUM"],
                    "meta": {"task_type": TASK_COMM_C1},
                }
            return {
                "task_type": TASK_COMM_C1,
                "turns": {
                    "question": "For each grounded step in the coherent sequence, which candidate frame matches the required intermediate state?",
                    "overall": "good",
                    "steps": ["good1", "good2"],
                    "select": [
                        {"question": "Which candidate frame matches step 1?", "answer": "target:F01", "rationale": "Choose F01 because grounded."},
                        {"question": "Which candidate frame matches step 2?", "answer": "target:F02", "rationale": "Choose F02 because grounded."},
                    ],
                },
                "answer": {"target_frame_positions": [1, 2]},
                "render": {"kind": "interleaved_mm", "frame_positions": [0, 1, 2]},
                "evidence_refs": ["S_SEGEMENT_SUM"],
                "meta": {"task_type": TASK_COMM_C1},
            }

        with patch("orchestrator.umm_task_producer.generate_comm_c1_sample", side_effect=flaky_comm_c1):
            result = produce_task_samples(self._bundle(), llm=object(), comm_c1_k=3, isg_k=2, llm_retry_attempts=2)

        self.assertEqual(len(result["samples"][TASK_COMM_C1]), 1)
        self.assertEqual(call_counter["count"], 2)
        self.assertEqual(result["drops"][TASK_COMM_C1], [])


if __name__ == "__main__":
    unittest.main()
