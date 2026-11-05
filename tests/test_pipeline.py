"""Focused regression tests for the entity-resolution pipeline.

Run from the project root:

    python -m unittest discover -s tests -v

The tests use small synthetic tables so they finish in under a second and do
not require the multi-gigabyte challenge dataset.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "student_resource" / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))

import business_entity_resolution as ber  # noqa: E402
import evaluate_holdout  # noqa: E402
import validate_candidate_stream as validator  # noqa: E402

MODEL = json.loads((ROOT / "student_resource" / "model" / "entity_matcher.json").read_text(encoding="utf-8"))


def record(entity_id, name, address, country="US"):
    return {"entity_id": entity_id, "business_name": name, "business_address": address, "country": country}


def write_tsv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class NormalizationTests(unittest.TestCase):
    def test_norm_folds_case_accents_and_punctuation(self):
        self.assertEqual(ber.norm("Café  Müller, LLC."), "cafe muller llc")
        self.assertEqual(ber.norm(None), "")

    def test_legal_suffixes_are_dropped_from_name_tokens(self):
        self.assertEqual(ber.words(ber.norm("North Shore Coffee Roasters LLC"), drop_suffix=True),
                         {"north", "shore", "coffee", "roasters"})

    def test_prep_one_extracts_postal_and_digits(self):
        item = ber.prep_one(record("S1-1", "Acme Corp", "125 W Lake St, Chicago IL 60601"))
        self.assertEqual(item["id"], "S1-1")
        self.assertIn("60601", item["postal"])
        self.assertEqual(item["digits"], "12560601")


class FeatureAndModelTests(unittest.TestCase):
    def test_feature_vector_is_bounded_and_deterministic(self):
        a = ber.prep_one(record("S1-1", "North Shore Coffee Roasters", "125 W Lake St Chicago IL 60601"))
        b = ber.prep_one(record("S2-1", "North Shore Coffee Roasters LLC", "125 West Lake Street Chicago IL 60601"))
        first = ber.features(a, b)
        self.assertEqual(len(first), 10)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in first))
        self.assertAlmostEqual(first[-1], 1.0)  # country match
        self.assertGreater(first[0], 0.5)  # name token overlap
        self.assertEqual(first, ber.features(a, b))

    def test_identical_records_score_as_a_match(self):
        a = ber.prep_one(record("S1-1", "North Shore Coffee Roasters", "125 West Lake St Chicago IL 60601"))
        score = float(ber.probs(MODEL, [ber.features(a, a)])[0])
        self.assertGreaterEqual(score, MODEL["threshold"])

    def test_shared_address_alone_can_inflate_the_raw_score(self):
        # Documented over-prediction: the saved model has no name-contradiction
        # feature, so two unrelated businesses at one address can pass the raw
        # cutoff. The demo's REVIEW/NO MATCH guard exists for exactly this case.
        a = ber.prep_one(record("S1-1", "North Shore Coffee Roasters", "125 West Lake St Chicago IL 60601"))
        b = ber.prep_one(record("S2-1", "Harborline Dental Studio", "125 West Lake St Chicago IL 60601"))
        values = ber.features(a, b)
        self.assertEqual(values[0], 0.0)  # no shared name tokens
        self.assertEqual(values[7], 1.0)  # identical normalized address
        self.assertGreater(float(ber.probs(MODEL, [values])[0]), MODEL["threshold"])

    def test_fit_separates_simple_data_and_returns_probabilities(self):
        X = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]] * 10
        y = [1, 0] * 10
        model = ber.fit(X, y, steps=300, lr=0.1)
        probabilities = ber.probs(model, X)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in probabilities))
        self.assertGreater(probabilities[0], probabilities[1])

    def test_f05_matches_its_definition(self):
        self.assertEqual(ber.f05(set(), set()), 1.0)
        self.assertEqual(ber.f05({"S2-1"}, set()), 0.0)
        self.assertAlmostEqual(ber.f05({"S2-1"}, {"S2-1"}), 1.0)
        # One true positive out of two predictions and two labels.
        self.assertAlmostEqual(ber.f05({"S2-1", "S2-2"}, {"S2-1", "S2-3"}), 0.5)


class BlockingTests(unittest.TestCase):
    def setUp(self):
        self.target = ber.prep_one(record("S1-1", "North Shore Coffee Roasters", "125 West Lake St Chicago IL 60601"))
        self.refs = [
            ber.prep_one(record("S2-1", "North Shore Coffee Roasters LLC", "125 W Lake St Chicago IL 60601")),
            ber.prep_one(record("S2-2", "Harborline Dental Studio", "125 W Lake St Chicago IL 60601")),
            ber.prep_one(record("S2-3", "North Shore Coffee Roasters", "125 West Lake St Chicago IL 60601", "India")),
        ]

    def test_candidates_include_same_country_plausible_matches(self):
        index, cap = ber.build_index(self.refs)
        candidates = ber.candidate_for(self.target, self.refs, index, cap, 10)
        ids = {self.refs[i]["id"] for i in candidates}
        self.assertIn("S2-1", ids)
        self.assertNotIn("S2-3", ids)  # country-aware blocking excludes the India row

    def test_blocks_respects_the_candidate_cap(self):
        results = ber.blocks([self.target], self.refs, limit=1)
        self.assertEqual(len(results), 1)
        self.assertLessEqual(len(results[0]), 1)

    def test_name_signature_is_order_insensitive_and_country_scoped(self):
        keys = set(ber.block_keys(self.target))
        self.assertIn(("us", "s", "coffee north"), keys)
        # A reordered name must hash to the same signature, so that an index
        # built from either spelling can answer the other's query.
        reversed_name = ber.prep_one(record("S2-4", "Coffee North", "9 Elm St Chicago IL 60601"))
        self.assertIn(("us", "s", "coffee north"), set(ber.block_keys(reversed_name)))
        india = ber.prep_one(record("S2-5", "Coffee North", "9 Elm St Chicago", "India"))
        self.assertNotIn(("us", "s", "coffee north"), set(ber.block_keys(india)))

    def test_name_signature_retrieves_an_order_only_variant(self):
        # Shares only a name-word prefix with the target ("north"), so the
        # 4-character name prefix key cannot join the records on its own.
        variant = ber.prep_one(record("S2-6", "Coffee Roasters North", "44 Elm Rd Boston MA 02118"))
        refs = [variant]
        index, cap = ber.build_index(refs)
        self.assertIn(("us", "s", "coffee north"), index)
        self.assertIn(0, ber.candidate_for(self.target, refs, index, cap, 10))

    def test_streamed_index_matches_the_in_memory_index(self):
        rows = [
            record("S2-1", "North Shore Coffee Roasters LLC", "125 W Lake St Chicago IL 60601"),
            record("S2-2", "Harborline Dental Studio", "125 W Lake St Chicago IL 60601"),
            record("S2-3", "Aaa Anvil Works", "8 Mill Road Austin TX 78701"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "refs.tsv"
            write_tsv(path, ["entity_id", "business_name", "business_address", "country"], rows)
            refs = [ber.prep_one(row) for row in rows]
            index, cap = ber.build_index(refs)
            streamed = evaluate_holdout.stream_target_index(path, set(ber.block_keys(self.target)), cap)
            expected = ber.candidate_for(self.target, refs, index, cap, 10)
            actual = ber.candidate_for(self.target, None, streamed, cap, 10)
            self.assertEqual([refs[i]["id"] for i in expected], [refs[i]["id"] for i in actual])
            materialized = evaluate_holdout.materialize_rows(path, set(expected))
        self.assertEqual({row["id"] for row in materialized.values()},
                         {refs[i]["id"] for i in expected})


class StreamingInputTests(unittest.TestCase):
    def test_reference_sampling_bounds_pool_and_keeps_positives(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "refs.tsv"
            rows = [record(f"S2-{i}", f"Business {i}", f"{i} Main St") for i in range(100)]
            write_tsv(path, ["entity_id", "business_name", "business_address", "country"], rows)
            sample = ber.reservoir_sample(path, {"S2-77"}, pool_size=10, seed=1)
        ids = {item["id"] for item in sample}
        self.assertEqual(len(ids), 11)  # 10 reservoir rows + the preserved positive
        self.assertIn("S2-77", ids)

    def test_reference_sampling_with_nonpositive_pool_keeps_everything(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "refs.tsv"
            rows = [record(f"S3-{i}", f"Business {i}", f"{i} Main St") for i in range(25)]
            write_tsv(path, ["entity_id", "business_name", "business_address", "country"], rows)
            sample = ber.reservoir_sample(path, set(), pool_size=0, seed=1)
        self.assertEqual(len(sample), 25)

    def test_id_and_label_readers_filter_to_requested_entities(self):
        with tempfile.TemporaryDirectory() as directory:
            source1 = Path(directory) / "train_source1.tsv"
            truth = Path(directory) / "train_ground_truth.tsv"
            write_tsv(source1, ["entity_id", "business_name", "business_address", "country"],
                      [record("S1-1", "A", "1 Main St"), record("S1-2", "B", "2 Main St")])
            write_tsv(truth, ["source1_entity_id", "matched_entity_ids"],
                      [{"source1_entity_id": "S1-1", "matched_entity_ids": "S2-9,S3-9"},
                       {"source1_entity_id": "S1-2", "matched_entity_ids": ""}])
            self.assertEqual(ber.read_ids(source1), ["S1-1", "S1-2"])
            rows = ber.load_rows_for_ids(source1, {"S1-2"})
            self.assertEqual(set(rows), {"S1-2"})
            labels = ber.load_labels_for_ids(truth, {"S1-1", "S1-2"})
        self.assertEqual(labels["S1-1"], {"S2-9", "S3-9"})
        self.assertEqual(labels["S1-2"], set())


class CandidateValidatorTests(unittest.TestCase):
    HEADER_MATCH = "source1_entity_id\tmatched_entity_ids"
    HEADER_CANDIDATE = "source1_entity_id\tcandidate_entity_ids"

    def run_validator(self, matching_rows, candidate_rows):
        with tempfile.TemporaryDirectory() as directory:
            matching = Path(directory) / "matching.tsv"
            candidates = Path(directory) / "candidates.tsv"
            matching.write_text("\n".join([self.HEADER_MATCH, *matching_rows]) + "\n", encoding="utf-8")
            candidates.write_text("\n".join([self.HEADER_CANDIDATE, *candidate_rows]) + "\n", encoding="utf-8")
            return validator.validate(str(matching), str(candidates))

    def test_aligned_subset_passes(self):
        self.assertEqual(self.run_validator(["S1-1\tS2-1"], ["S1-1\tS2-1,S3-2"]), 0)

    def test_match_absent_from_candidates_fails(self):
        self.assertEqual(self.run_validator(["S1-1\tS2-9"], ["S1-1\tS2-1"]), 1)

    def test_duplicate_candidate_fails(self):
        self.assertEqual(self.run_validator(["S1-1\t"], ["S1-1\tS2-1,S2-1"]), 1)

    def test_invalid_prefix_fails(self):
        self.assertEqual(self.run_validator(["S1-1\tS2-1"], ["S1-1\tX9-1"]), 1)

    def test_row_count_mismatch_fails(self):
        self.assertEqual(self.run_validator(["S1-1\t", "S1-2\t"], ["S1-1\t"]), 1)


class PredictIntegrationTests(unittest.TestCase):
    def test_predict_and_validator_agree_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "test"
            data.mkdir()
            source1 = [record("S1-1", "North Shore Coffee Roasters", "125 W Lake St Chicago IL 60601"),
                       record("S1-2", "Harborline Dental", "9 Oak Ave Boston MA 02108")]
            source2 = [record("S2-1", "North Shore Coffee Roasters LLC", "125 West Lake Street Chicago IL 60601"),
                       record("S2-2", "Harborline Dental Studio", "9 Oak Avenue Boston MA 02108"),
                       record("S2-3", "Unrelated Plumbing", "44 Elm St Austin TX 78701")]
            source3 = [record("S3-1", "North Shore Coffee Roasters", "125 W Lake St Chicago IL 60601"),
                       record("S3-2", "Other Bakery", "1 Bread Ln Austin TX 78701")]
            columns = ["entity_id", "business_name", "business_address", "country"]
            write_tsv(data / "test_source1.tsv", columns, source1)
            write_tsv(data / "test_source2.tsv", columns, source2)
            write_tsv(data / "test_source3.tsv", columns, source3)
            output = Path(directory) / "output"
            result = subprocess.run(
                [sys.executable, str(SRC / "business_entity_resolution.py"), "predict",
                 "--data", str(data), "--model", str(ROOT / "student_resource" / "model" / "entity_matcher.json"),
                 "--output", str(output), "--max-candidates", "10"],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            matching = (output / "matching_results.tsv").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(matching[0], "source1_entity_id\tmatched_entity_ids")
            self.assertIn("S1-1\tS2-1,S3-1", matching)
            self.assertIn("S1-2\tS2-2", matching)
            exit_code = validator.validate(str(output / "matching_results.tsv"),
                                           str(output / "candidate_pairs.tsv"))
            self.assertEqual(exit_code, 0)


class StreamingPredictTests(unittest.TestCase):
    """The bounded-memory predictor must reproduce the in-memory one exactly."""
    ROWS = [
        record("S2-1", "North Shore Coffee Roasters LLC", "125 W Lake St Chicago IL 60601"),
        record("S2-2", "Coffee Roasters North", "44 Elm Rd Boston MA 02118"),
        record("S2-3", "Harborline Dental Studio", "125 W Lake St Chicago IL 60601"),
        record("S2-4", "Aaa Anvil Works", "8 Mill Road Austin TX 78701"),
        record("S2-5", "North Shore Coffee Roasters", "125 West Lake St Chicago IL", "India"),
    ]

    def test_sidecar_postings_match_the_in_memory_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "refs.tsv"
            write_tsv(path, ["entity_id", "business_name", "business_address", "country"], self.ROWS)
            refs = [ber.prep_one(row) for row in self.ROWS]
            index, cap = ber.build_index(refs)
            ber.build_reference_sidecar(path, Path(directory) / "sidecar", log=lambda message: None)
            section = ber.open_sidecar(Path(directory) / "sidecar")
            keys = {key for ref in refs for key in ber.block_keys(ref)}
            streamed = ber.load_sidecar_postings(section, keys)
            sidecar_cap = section["meta"]["cap"]
            ber.close_sidecar(section)
        self.assertEqual(sidecar_cap, cap)
        for key in keys:
            self.assertEqual([int(i) for i in index.get(key, ())], [int(i) for i in streamed.get(key, ())], key)

    def test_streaming_predict_reproduces_the_in_memory_output(self):
        columns = ["entity_id", "business_name", "business_address", "country"]
        targets = [record("S1-1", "North Shore Coffee Roasters", "125 West Lake St Chicago IL 60601"),
                   record("S1-2", "Harborline Dental", "9 Oak Ave Boston MA 02108"),
                   record("S1-3", "Aaa Anvil Works", "8 Mill Road Austin TX 78701")]
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "test"
            data.mkdir()
            write_tsv(data / "test_source1.tsv", columns, targets)
            write_tsv(data / "test_source2.tsv", columns, self.ROWS)
            write_tsv(data / "test_source3.tsv", columns, self.ROWS[:3])
            script = str(SRC / "business_entity_resolution.py")
            base = [sys.executable, script, "predict", "--data", str(data),
                    "--model", str(ROOT / "student_resource" / "model" / "entity_matcher.json"),
                    "--max-candidates", "10"]
            outputs = {}
            for label, extra in (("memory", []), ("stream", ["--streaming-references", "--workers", "2",
                                                             "--shard-rows", "1", "--sort-chunk", "4"])):
                out = Path(directory) / label
                result = subprocess.run(base + ["--output", str(out)] + extra,
                                        capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stderr)
                outputs[label] = out
            for name in ("matching_results.tsv", "candidate_pairs.tsv"):
                self.assertEqual((outputs["memory"] / name).read_bytes(),
                                 (outputs["stream"] / name).read_bytes(), name)


class RootEntryPointTests(unittest.TestCase):
    def test_root_script_delegates_to_the_packaged_implementation(self):
        result = subprocess.run([sys.executable, str(ROOT / "business_entity_resolution.py"), "--help"],
                                cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("train", result.stdout)
        self.assertIn("predict", result.stdout)


if __name__ == "__main__":
    unittest.main()
