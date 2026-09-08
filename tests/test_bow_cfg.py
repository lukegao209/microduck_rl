import math

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_bow_env_cfg import (
    BOW_DESCENT_END,
    BOW_HOLD_END,
    BOW_PERIOD,
    BOW_RISE_END,
    BOW_RUNTIME_END_PHASE,
    BOW_TARGET_PITCH_FINAL,
    BOW_TARGET_PITCH_START,
    MicroduckBowRlCfg,
    make_microduck_bow_env_cfg,
)


def test_bow_replaces_pick_objectives_but_keeps_safety():
    cfg = make_microduck_bow_env_cfg()
    for removed in (
        "mouth_ground_proximity",
        "mouth_perpendicular_to_ground",
        "mouth_payload_force",
        "upright",
    ):
        assert removed not in cfg.rewards
    assert "sample_mouth_payload" not in cfg.events

    for required in (
        "bow_trunk_pitch",
        "bow_head_pose",
        "bow_lateral_tilt",
        "bow_hold_stillness",
        "bow_return_pose",
        "bow_return_upright",
        "feet_grounded",
        "feet_flat",
        "head_impact_penalty",
        "self_collisions",
    ):
        assert required in cfg.rewards, required


def test_bow_phase_finishes_before_runtime_handoff():
    assert BOW_DESCENT_END < BOW_HOLD_END < BOW_RISE_END
    assert BOW_RISE_END < BOW_RUNTIME_END_PHASE
    cfg = make_microduck_bow_env_cfg()
    command = cfg.commands["twist"]
    assert isinstance(command, microduck_mdp.GroundPickPhaseCommandCfg)
    assert command.period == BOW_PERIOD
    assert command.randomize_phase is True
    assert (
        make_microduck_bow_env_cfg(play=True).commands["twist"].randomize_phase is False
    )


def test_bow_pitch_target_has_descent_hold_rise_and_settle():
    phase = torch.tensor(
        [0.0, BOW_DESCENT_END, BOW_HOLD_END, BOW_RISE_END, BOW_RUNTIME_END_PHASE]
    )
    target = microduck_mdp.bow_pitch_target(
        phase,
        BOW_TARGET_PITCH_FINAL,
        BOW_DESCENT_END,
        BOW_HOLD_END,
        BOW_RISE_END,
    )
    assert torch.allclose(
        target,
        torch.tensor([0.0, BOW_TARGET_PITCH_FINAL, BOW_TARGET_PITCH_FINAL, 0.0, 0.0]),
    )


def test_bow_target_curriculum_starts_shallow_and_reaches_thirty_degrees():
    cfg = make_microduck_bow_env_cfg()
    stages = cfg.curriculum["bow_target_pitch"].params["value_stages"]
    assert stages[0]["value"] == BOW_TARGET_PITCH_START
    assert stages[-1]["value"] == BOW_TARGET_PITCH_FINAL
    assert math.isclose(math.degrees(stages[-1]["value"]), 30.0)
    values = [stage["value"] for stage in stages]
    assert values == sorted(values)


def test_bow_observation_layout_matches_ground_pick_exactly():
    from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import (
        make_microduck_ground_pick_env_cfg,
    )

    bow = make_microduck_bow_env_cfg()
    ground_pick = make_microduck_ground_pick_env_cfg()
    for group in ("actor", "critic"):
        assert list(bow.observations[group].terms) == list(
            ground_pick.observations[group].terms
        )
        terms = bow.observations[group].terms
        assert terms["head_command"].params["dim"] == 4
        assert terms["body_command"].params["dim"] == 6


def test_bow_reward_signs_and_runner_identity():
    cfg = make_microduck_bow_env_cfg()
    assert cfg.rewards["bow_trunk_pitch"].weight > 0.0
    assert cfg.rewards["bow_hold_stillness"].weight > 0.0
    assert cfg.rewards["bow_lateral_tilt"].weight < 0.0
    assert MicroduckBowRlCfg.algorithm.symmetry_cfg is None
    assert MicroduckBowRlCfg.experiment_name == "bow"
    assert MicroduckBowRlCfg.max_iterations == 3_000


def test_bow_tasks_are_registered():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401

    tasks = list_tasks()
    assert "Mjlab-Bow-Flat-MicroDuck" in tasks
    assert "Mjlab-Bow-Flat-Backlash-MicroDuck" in tasks
