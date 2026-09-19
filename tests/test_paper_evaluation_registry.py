from scripts import quantitative_eval


PAPER_ALIASES = {
    "coco_zeroshot",
    "flickr30k_zeroshot",
    "sugarcrepe_pp",
    "svo_probes",
    "vsr",
    "coco_adapter_tune",
    "flickr30k_adapter_tune",
    "nlvr2_token_interaction_probe",
    "nlvr2_global_raw_probe",
}


def test_paper_core_registry_contains_exactly_manuscript_evaluations():
    assert set(quantitative_eval.GROUPS["paper_core"]) == PAPER_ALIASES
    assert set(quantitative_eval.INTERNAL_ALIASES) == PAPER_ALIASES


def test_paper_core_selection_is_deterministic():
    selected = quantitative_eval.expand_eval_selection(["paper_core"], None)
    assert [spec.alias for spec in selected] == [
        "coco_zeroshot",
        "flickr30k_zeroshot",
        "sugarcrepe_pp",
        "svo_probes",
        "vsr",
        "coco_adapter_tune",
        "flickr30k_adapter_tune",
        "nlvr2_token_interaction_probe",
        "nlvr2_global_raw_probe",
    ]
