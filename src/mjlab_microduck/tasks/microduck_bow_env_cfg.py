"""Microduck one-shot bow: stand -> lean forward -> hold -> stand.

The first deployment reuses robotd's ground-pick policy slot and its phase
command ``[cos(2*pi*phase), sin(2*pi*phase), 0]``. The runtime hands that slot
back at phase 0.70, so the learned rise deliberately finishes at 0.65 and has a
short settling margin. This file remains a separate task: proving a bow must
not change or regress the actual ground-pick skill.

The environment inherits GroundPick's full-collision robot, BAM actuator,
sensor delay/noise, and domain randomization. Only task-specific objectives
are replaced.
"""

import math
from copy import deepcopy

from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import (
    MicroduckGroundPickRlCfg,
    make_microduck_ground_pick_env_cfg,
)

BOW_PERIOD = 4.0
BOW_DESCENT_END = 0.30
BOW_HOLD_END = 0.45
BOW_RISE_END = 0.65
BOW_RUNTIME_END_PHASE = 0.70

# The target starts shallow and is raised by curriculum. Thirty degrees is a
# visible bow without committing the first run to GroundPick's much deeper
# reach. It remains a parameter to tune after pose-stability sweeps.
BOW_TARGET_PITCH_START = math.radians(15.0)
BOW_TARGET_PITCH_MID = math.radians(22.5)
BOW_TARGET_PITCH_FINAL = math.radians(30.0)


def make_microduck_bow_env_cfg(play: bool = False):
    """Build the flat-ground bow environment."""
    cfg = make_microduck_ground_pick_env_cfg(play=play, rough=False)

    # Remove objectives that define "pick something from the floor". The
    # collision, feet, smoothness, torque, noise, and DR terms stay inherited.
    for name in (
        "mouth_ground_proximity",
        "mouth_perpendicular_to_ground",
        "ground_pick_return_pose_legs",
        "ground_pick_return_pose_neck",
        "return_upright",
        "neck_vel_descent",
        "mouth_payload_force",
        "upright",  # full upright conflicts with the intentional pitch
    ):
        cfg.rewards.pop(name, None)
    cfg.events.pop("sample_mouth_payload", None)

    phase_params = {
        "command_name": "twist",
        "descent_end": BOW_DESCENT_END,
        "hold_end": BOW_HOLD_END,
        "rise_end": BOW_RISE_END,
    }

    # Task-space objective: the trunk, not merely the neck, must bow. Curriculum
    # starts at 15 degrees and ends at 30 degrees.
    cfg.rewards["bow_trunk_pitch"] = RewardTermCfg(
        func=microduck_mdp.bow_trunk_pitch_track,
        weight=6.0,
        params={
            **phase_params,
            "target_pitch": BOW_TARGET_PITCH_START,
            "std": math.radians(8.0),
        },
    )

    # Keep the head near HOME relative to the body. Together with trunk pitch
    # tracking this produces a whole-body bow and rules out a fast nod.
    cfg.rewards["bow_head_pose"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=1.5,
        params={"joint_indices": [5, 6, 7, 8], "std": 0.25},
    )

    cfg.rewards["bow_lateral_tilt"] = RewardTermCfg(
        func=microduck_mdp.bow_lateral_tilt_l2,
        weight=-2.0,
    )
    cfg.rewards["bow_hold_stillness"] = RewardTermCfg(
        func=microduck_mdp.bow_hold_stillness,
        weight=1.0,
        params={
            "command_name": "twist",
            "hold_start": BOW_DESCENT_END,
            "hold_end": BOW_HOLD_END,
            "ang_vel_std": 0.5,
        },
    )

    # Explicitly reward a clean handoff pose. It ramps in during the rise and
    # remains on through the 0.65..0.70 settling margin.
    cfg.rewards["bow_return_pose"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_pose_phased,
        weight=4.0,
        params={
            "command_name": "twist",
            "std": 0.25,
            "hold_end": BOW_HOLD_END,
            "rise_end": BOW_RISE_END,
        },
    )
    cfg.rewards["bow_return_upright"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_upright_phased,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "command_name": "twist",
            "std": 0.35,
            "hold_end": BOW_HOLD_END,
            "rise_end": BOW_RISE_END,
        },
    )

    command = cfg.commands["twist"]
    assert isinstance(command, microduck_mdp.GroundPickPhaseCommandCfg)
    command.period = BOW_PERIOD
    # Training sees every phase after reset; interactive play mirrors a button
    # press and starts at phase zero from HOME.
    command.randomize_phase = not play

    cfg.curriculum["bow_target_pitch"] = CurriculumTermCfg(
        func=microduck_mdp.reward_param_curriculum,
        params={
            "reward_name": "bow_trunk_pitch",
            "param_name": "target_pitch",
            "value_stages": [
                {"step": 0, "value": BOW_TARGET_PITCH_START},
                {"step": 250 * 24, "value": BOW_TARGET_PITCH_MID},
                {"step": 500 * 24, "value": BOW_TARGET_PITCH_FINAL},
            ],
        },
    )
    return cfg


MicroduckBowRlCfg = deepcopy(MicroduckGroundPickRlCfg)
MicroduckBowRlCfg.experiment_name = "bow"
MicroduckBowRlCfg.run_name = "bow"
# Simple episodic gestures usually converge much earlier than GroundPick's
# 20k ceiling; extend or resume only when measured rollouts justify it.
MicroduckBowRlCfg.max_iterations = 3_000
