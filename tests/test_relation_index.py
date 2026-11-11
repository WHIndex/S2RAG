import unittest

from s2rag.utils.relation_fact_join_utils import (
    build_relation_fact_join_index,
    canonical_schema_predicate_families,
    typed_schema_signatures_v2,
)


class RelationIndexTest(unittest.TestCase):
    def test_predicate_catalog_and_composite_index(self) -> None:
        self.assertEqual(
            canonical_schema_predicate_families("was directed by"),
            ("direction",),
        )
        self.assertEqual(
            canonical_schema_predicate_families("was authored by"),
            ("authorship",),
        )
        signature = typed_schema_signatures_v2(("Alice", "birthplace", "Paris"))[0]
        index = build_relation_fact_join_index(
            [
                {
                    "idx": "p0",
                    "passage": "Alice was born in Paris.",
                    "extracted_triples": [("Alice", "birthplace", "Paris")],
                }
            ],
            {"p0": 0},
        )
        self.assertEqual(signature[1], "birth")
        self.assertEqual(signature[3], "subject_to_object")
        self.assertEqual(index["schema_v2_to_items"][signature], [0])
        self.assertTrue(index["subject_predicate_family_to_items"])


if __name__ == "__main__":
    unittest.main()
