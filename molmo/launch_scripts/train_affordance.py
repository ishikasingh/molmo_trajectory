import argparse
import logging
from os.path import join, exists
from typing import cast, List

import omegaconf
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from launch_scripts.utils import get_evaluation, DEBUG_MODEL
from olmo import TrainConfig
from olmo.config import DataConfig, \
    ModelConfig, WandbConfig, OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType, \
    BatchDivisor, SpeedMonitorConfig, ActivationCheckpointingStrategy, FSDPConfig, FSDPWrapStrategy, \
    FSDPPrecision, RootSizeMixture
from olmo.torch_util import get_world_size
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    prepare_cli_environment,
)
from scripts.train import main as train

import os
import torch

# local_rank = int(os.environ.get("LOCAL_RANK", 0))
# print(f"Setting device to {local_rank}")
# torch.cuda.set_device(local_rank)

log = logging.getLogger("train")


AUX = [
    # Supervised datasets we want eval on
    "coco_2014_vqa_multi",
    "text_vqa",
    "okvqa",
    "chart_qa_weighted",
    "doc_qa",
    "info_qa",
    "ai2_diagram_v2_mix_transparent",
    "a_okvqa_mc",
    "a_okvqa_da",
    "android_control",

    # Some other datasets we might want to eval on
    "science_qa_img",
    "tabwmp_da",
    "st_qa",
    "tally_qa",

    # ("clocks", 250000),  # Downsample since it is huge
    "pixmo_docs_charts",
    "pixmo_docs_tables",
    "pixmo_docs_other",
    "pixmo_docs_diagrams",

    # # Other synthetic data, also downsampled since they are huge
    ("dv_qa", 10000),
    ("figure_qa", 10000),
    ("plot_qa", 20000),
]


def get_training_mixture(submixture):
    resolved_weights = {}
    for task_name in submixture:
        mix = {}
        if isinstance(task_name, tuple):
            task_name, size = task_name
        else:
            size = None
        resolved_weights[task_name] = size
    return resolved_weights


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    # Initialize process group.
    dist.init_process_group(backend="nccl")
    log.info("Process group initialized")

    prepare_cli_environment()
    log.info("CLI environment prepared")

    add_cached_path_clients()

    parser = argparse.ArgumentParser(prog="Train a multitask model")
    parser.add_argument("mixture", help="Name of datset mixture to train on")
    parser.add_argument("checkpoint", help="Path to checkpoint to start from")
    parser.add_argument("--seq_len", default=2304, type=int)
    parser.add_argument("--inf_seq_len", default=1792, type=int)
    # parser.add_argument("--max_inf_examples", default=2048, type=int)
    parser.add_argument("--max_inf_examples", default=64, type=int)
    # parser.add_argument("--global_batch_size", default=256, type=int)
    parser.add_argument("--global_batch_size", default=64, type=int)
    # parser.add_argument("--device_eval_batch_size", default=4, type=int)
    parser.add_argument("--device_eval_batch_size", default=2, type=int)
    # parser.add_argument("--device_inf_batch_size", default=4, type=int)
    parser.add_argument("--device_inf_batch_size", default=2, type=int)
    # parser.add_argument("--device_train_batch_size", default=2, type=int)
    parser.add_argument("--device_train_batch_size", default=2, type=int)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--finetune", default=False, action="store_true",
                        help="whether it is a finetuning or post-training run")
    parser.add_argument("--cotrain", default=False, action="store_true",
                        help="whether it is a cotraining run")
    parser.add_argument("--use_transitions", default=False, action="store_true",
                        help="whether to use transitions in the affordance dataset, transitions\
                        means whether the hand is from grasp to release or release to grasp")
    parser.add_argument("--freeze_vlm", default=False, action="store_true",
                        help="whether to freeze the VLM model")
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=30,
        help="Number of timesteps to predict for trajectory actions",
    )
    parser.add_argument(
        "--interpolation_times",
        type=int,
        default=1,
        help="If > 1, load horizon//interpolation_times steps from dataset and interpolate to horizon (e.g. horizon=30, interpolation_times=2 -> load 15, interpolate to 30). Default 1 = no interpolation.",
    )
    parser.add_argument(
        "--exclude_proprio",
        action="store_true",
        default=False,
        help="Exclude proprioceptive information (current finger positions) from trajectory prediction",
    )
    # Per-expert dimension arguments for separate_human_robot mode
    parser.add_argument(
        "--human_action_dim",
        type=int,
        default=30,
        help="Action/trajectory dimension for human expert (EgoDex: 10 joints * 3 coords = 30)",
    )
    parser.add_argument(
        "--human_proprio_dim",
        type=int,
        default=30,
        help="Proprioception dimension for human expert (EgoDex: 10 joints * 3 coords = 30)",
    )
    parser.add_argument(
        "--robot_action_dim",
        type=int,
        default=24,
        help="Action dimension for robot expert (RoboCasa: 24 for bimanual action command)",
    )
    parser.add_argument(
        "--robot_proprio_dim",
        type=int,
        default=30,
        help="Proprioception dimension for robot expert (RoboCasa: robot state dim)",
    ) # Note: at this stage, we are using the fingertip positions as the proprioception, not the actual proprioception. 
    parser.add_argument(
        "--robot_trajectory_dim",
        type=int,
        default=30,
        help="Fingertip trajectory dimension for robot expert (RoboCasa: 10 keypoints * 3 coords = 30)",
    )
    parser.add_argument(
        "--max_crops",
        type=int,
        default=None,
        help="Override max_crops parameter for the VLM model (default: None means no override)",
    )
    parser.add_argument(
        "--action_expert_mode",
        type=str,
        default="shared",
        choices=["disabled", "shared", "separate_human_robot", "sequential"],
        help="Action expert architecture mode: "
             "'disabled' (no action expert), "
             "'shared' (single expert for all trajectories), "
             "'separate_human_robot' (separate experts with per-sample routing), "
             "'sequential' (VLM + Expert A for human, VLM + Expert A + Expert B for robot)",
    )
    parser.add_argument(
        "--flow_matching_prediction_type",
        type=str,
        default="velocity",
        choices=["velocity", "x0"],
        help="Prediction type for flow matching: "
             "'velocity' (predict velocity field v = noise - target), "
             "'x0' (predict clean target x0 directly)",
    )
    parser.add_argument("--slow_warmup", action="store_true", default=False,
                        help="Whether to use slow warmup for the model")
    parser.add_argument("--pad_action_chunk", action="store_true", default=False,
                        help="If True, pad action chunks with repeated last step when near end of trajectory")
    args, other_args = parser.parse_known_args()

    # Default training split
    data_split = "train"

    if args.mixture.startswith("single"):
        task_name = args.mixture.split("_", 1)[1]
        eval_tasks = [task_name,]
        tasks = [["eval", eval_tasks, 1.0]]
    elif args.mixture == "android":
        eval_tasks = ["android_control_ll"]
        tasks = [["eval", ["android_control"], 1.0]]
    elif args.mixture in ["small1", "debug"]:
        eval_tasks = ["chart_qa", "doc_qa"]
        tasks = [["aux", ["chart_qa", "doc_qa"], 1.0]]
    elif args.mixture == "small2":
        eval_tasks = ["chart_qa", "doc_qa", "info_qa"]
        tasks = [["aux", [("chart_qa", 4*4),
                          ("doc_qa", 2*2), ("info_qa", 1)], 1.0]]
    elif args.mixture in ["3.2-synthetic"]:
        aux = list(AUX)
        eval_tasks = [
            "chart_qa",
            "info_qa",
            "doc_qa",
            "ai2_diagram_v2_mix_transparent",
            "coco_2014_vqa_multi",
            # "clocks",
            "android_control_ll",
            "pointing_eval:test",
            "countbench_qa:huggingface"
        ]
        tasks = [
            ["demo", [
                "pixmo_ask_model_anything",
                ("pixmo_cap", 50000),
                "pixmo_cap_qa",
                "pixmo_pointing_explanations"
            ], 0.15],
            ["aux", aux, 0.50],
            ["pointing", [
                "pixmo_points",
                "pixmo_count",
                "pixmo_points_high_freq",
                "pixmo_points_counting",
                "pixmo_points_high_freq_counting",
                "pixmo_count_counting",
            ], 0.35]
        ]
    elif args.mixture == "affordance":
        if args.cotrain:
            eval_tasks = ["affordance_eval", "pointing_eval:test"]
            tasks = [["train", ["affordance"], 0.5],
                     ["pointing", ["pixmo_points"], 0.5]]
        else:
            eval_tasks = []
            tasks = [["train", ["affordance"], 1.0]]
    elif args.mixture == "affordance_new":
        if args.cotrain:
            eval_tasks = ["affordance_eval", "pointing_eval:test"]
            if args.use_transitions:
                tasks = [["train", ["affordance_with_transitions"], 0.5],
                         ["pointing", ["pixmo_points"], 0.5]]
            else:   
                tasks = [["train", ["affordance_new"], 0.5],
                        ["pointing", ["pixmo_points"], 0.5]]
        else:
            eval_tasks = []
            if args.use_transitions:
                tasks = [["train", ["affordance_with_transitions"], 1.0]]
            else:
                tasks = [["train", ["affordance_new"], 1.0]]
    elif args.mixture == "affordance_human_robot":
        # default to use new affordance format
        if args.cotrain:
            eval_tasks = ["affordance_eval", "pointing_eval:test"]
            if args.use_transitions:
                tasks = [["egodex", ["affordance_with_transitions"], 0.35],
                         ["robo_casa", ["robo_casa_affordance"], 0.15],
                    ["demo", [
                        "pixmo_ask_model_anything",
                        ("pixmo_cap", 50000),
                        "pixmo_cap_qa",
                        "pixmo_pointing_explanations"
                    ], 0.15],
                    ["pointing", [
                        "pixmo_points",
                        "pixmo_count",
                        "pixmo_points_high_freq",
                        "pixmo_points_counting",
                        "pixmo_points_high_freq_counting",
                        "pixmo_count_counting",
                    ], 0.35]
                ]
            else:
                tasks = [["egodex", ["affordance_new"], 0.35],
                         ["robo_casa", ["robo_casa_affordance"], 0.15],
                    ["demo", [
                        "pixmo_ask_model_anything",
                        ("pixmo_cap", 50000),
                        "pixmo_cap_qa",
                        "pixmo_pointing_explanations"
                    ], 0.15],
                    ["pointing", [
                        "pixmo_points",
                        "pixmo_count",
                        "pixmo_points_high_freq",
                        "pixmo_points_counting",
                        "pixmo_points_high_freq_counting",
                        "pixmo_count_counting",
                    ], 0.35]
                ]
        else:
            eval_tasks = [] 
            if args.use_transitions:
                tasks = [["egodex", ["affordance_with_transitions"], 0.5],
                         ["robo_casa", ["robo_casa_affordance"], 0.5]]
            else:
                tasks = [["egodex", ["affordance_new"], 0.5],
                         ["robo_casa", ["robo_casa_affordance"], 0.5]]
    elif args.mixture == "trajectory_2d_text":
        eval_tasks = []
        tasks = [["egodex", ["trajectory_2d_text"], 1.0]]
    elif args.mixture == "trajectory_3d_text":
        eval_tasks = []
        tasks = [["egodex", ["trajectory_3d_text"], 1.0]]
    elif args.mixture == "trajectory_3d_fm":
        # Flow matching based 3D trajectory prediction
        eval_tasks = []
        tasks = [["egodex", ["trajectory_3d"], 1.0]]
    elif args.mixture == "trajectory_3d_direct":
        # Direct 3D trajectory prediction (regression) using action expert architecture
        eval_tasks = []
        tasks = [["egodex", ["trajectory_3d"], 1.0]]
    elif args.mixture == "robot_3d_fm":
        eval_tasks = []
        tasks = [["robo_casa", ["robocasa_3d"], 1.0]]
    elif args.mixture == "robot_3d_direct":
        eval_tasks = []
        tasks = [["robo_casa", ["robocasa_3d"], 1.0]]
    elif args.mixture == "robot_action_fm":
        eval_tasks = []
        tasks = [["robo_casa", ["robocasa_action"], 1.0]]
    elif args.mixture == "robot_action_direct":
        eval_tasks = []
        tasks = [["robo_casa", ["robocasa_action"], 1.0]]
    elif args.mixture == "trajectory_3d_human_robot_fm":
        # Human (EgoDex) + Robot (RoboCasa) trajectory prediction with fingertip trajectories
        eval_tasks = []
        tasks = [["egodex", ["trajectory_3d"], 0.5],
                 ["robo_casa", ["robocasa_3d"], 0.5]]
    elif args.mixture == "trajectory_3d_human_robot_action_fm":
        # Human (EgoDex) + Robot (RoboCasa) with robot using joint actions instead of fingertip trajectory
        eval_tasks = []
        tasks = [["egodex", ["trajectory_3d"], 0.5],
                 ["robo_casa", ["robocasa_action"], 0.5]]
    elif args.mixture == "trajectory_3d_human_robot_direct":
        eval_tasks = []
        tasks = [["egodex", ["trajectory_3d"], 0.5],
                ["robo_casa", ["robocasa_3d"], 0.5]]
    elif args.mixture == "trajectory_3d_human_robot_action_direct":
        eval_tasks = []
        tasks = [["egodex", ["trajectory_3d"], 0.5],
                ["robo_casa", ["robocasa_action"], 0.5]]
    elif args.mixture == "trajectory_3d_fm_overfit":
        # Flow matching based 3D trajectory prediction with delta representation
        eval_tasks = []
        tasks = [["egodex", ["trajectory_3d"], 1.0]]
        data_split = "overfit"
    elif args.mixture == "trajectory_3d_robot_fm_overfit":
        eval_tasks = []
        tasks = [["robo_casa", ["robocasa_3d"], 1.0]]
        data_split = "overfit"
    elif args.mixture == "trajecotry_3d_robot_direct_overfit":
        eval_tasks = []
        tasks = [["robo_casa", ["robocasa_3d"], 1.0]]
        data_split = "overfit"
    elif args.mixture == "trajectory_3d_fm_pick_and_place":
        # Flow matching based 3D trajectory prediction with delta representation
        eval_tasks = []
        tasks = [["egodex", ["trajectory_3d"], 1.0]]
        data_split = "train_pick_and_place"
    elif args.mixture == "trajectory_3d_direct_pick_and_place":
        # Direct 3D trajectory prediction (regression) using action expert architecture
        eval_tasks = []
        tasks = [["egodex", ["trajectory_3d"], 1.0]]
        data_split = "train_pick_and_place"
    elif args.mixture == "robo_casa_affordance":
        # eval_tasks = ["robo_casa_affordance"]
        eval_tasks = []
        tasks = [["train", ["robo_casa_affordance"], 1.0]]
    else:
        raise NotImplementedError(args.mixture)

    # debug = args.checkpoint in ["debug", "debug2"] or args.debug
    debug = args.debug
    if debug:
        if exists(join(args.checkpoint, "model.yaml")):
            model_cfg = ModelConfig.load(join(args.checkpoint, "model.yaml"))
        else:
            model_cfg = ModelConfig.load(join(args.checkpoint, "config.yaml"), key="model")
        # model_cfg = DEBUG_MODEL
        # if args.checkpoint == "debug2":
        #     model_cfg.max_crops = 12
        #     model_cfg.crop_mode = "overlap-and-resize-c2"
        #     model_cfg.tokenizer.identifier = "mm:hf-Qwen/Qwen2-7B"
        #     model_cfg.embedding_size = 152064
        #     model_cfg.vocab_size = 152064
        #     model_cfg.pad_tokenizer = True
        global_batch_size = 8
        model_init = None
        inf_eval_interval = 20
        eval_interval = 20
        log_interval = 5
        eval_examples = 8
        max_inf_examples = 16
        duration = 1000
        eval_subset_batches = 4
    else:
        # eval_examples = 2048
        eval_examples = 64
        max_inf_examples = args.max_inf_examples
        log_interval = 20
        global_batch_size = args.global_batch_size
        inf_eval_interval = 2000
        eval_interval = 2000
        if args.finetune:
            duration = 40000
        else:
            duration = 160000
        model_init = args.checkpoint
        if exists(join(args.checkpoint, "model.yaml")):
            model_cfg = ModelConfig.load(join(args.checkpoint, "model.yaml"))
        else:
            model_cfg = ModelConfig.load(join(args.checkpoint, "config.yaml"), key="model")

        eval_subset_batches = eval_examples//(args.device_eval_batch_size*get_world_size())
        logging.info(f"Setting eval subset batches to {eval_subset_batches}")
        assert eval_subset_batches > 0

    # Override max_crops if specified
    if args.max_crops is not None:
        log.info(f"Overriding max_crops from {model_cfg.max_crops} to {args.max_crops}")
        model_cfg.max_crops = args.max_crops

    # Fine-tuning settings
    model_cfg.residual_dropout = 0.1
    model_cfg.response_residual_dropout = 0.0
    model_cfg.prompt_type = "uber_model"
    model_cfg.message_formatting = "role"
    model_cfg.system_prompt_kind = "demo_or_style"
    model_cfg.multi_annotation_weighting = "root_subsegments"

    # Enable flow matching for trajectory prediction tasks
    if "_fm" in args.mixture or "_direct" in args.mixture:
        # Set action expert mode and derive settings
        model_cfg.action_expert_mode = args.action_expert_mode
        
        if args.action_expert_mode == "disabled":
            model_cfg.use_action_expert = False
            model_cfg.num_action_experts = 0
            log.info("Action expert disabled - using VLM text-only output")
        elif args.action_expert_mode == "shared":
            model_cfg.use_action_expert = True
            model_cfg.num_action_experts = 1
            log.info("Using shared action expert for all trajectory types")
        elif args.action_expert_mode == "separate_human_robot":
            model_cfg.use_action_expert = True
            model_cfg.num_action_experts = 2
            log.info("Using separate action experts for human (expert 0) and robot (expert 1) trajectories")
        elif args.action_expert_mode == "sequential":
            model_cfg.use_action_expert = True
            model_cfg.num_action_experts = 2
            log.info("Using sequential action experts: Expert A (fingertip trajectory) + Expert B (robot actions)")
            log.info("  Human data: VLM + Expert A only")
            log.info("  Robot data: VLM + Expert A + Expert B (sequential)")
        else:
            raise ValueError(f"Unknown action_expert_mode: {args.action_expert_mode}")
        
        # Configure action dimensions based on task
        # action_horizon: number of timesteps to predict (e.g., 30) - shared across experts
        # action_dim: flattened coordinate dimension = num_joints * num_coordinates
        #   For 2D trajectories: num_joints * 2
        #   For 3D trajectories: num_joints * 3
        model_cfg.action_horizon = args.action_horizon
        model_cfg.use_adarms_flow_matching = False  # Use Pi0-style MLP conditioning
        model_cfg.flow_matching_loss_weight = 1.0
        model_cfg.flow_matching_num_steps = 10  # ODE integration steps during inference
        model_cfg.flow_matching_prediction_type = args.flow_matching_prediction_type
        log.info(f"Flow matching prediction type: {args.flow_matching_prediction_type}")
        
        # Determine if this mixture uses direct trajectory prediction (regression)
        use_direct_trajectory_prediction = (
            args.mixture == "trajectory_3d_direct" or
            args.mixture == "robot_3d_direct" or
            args.mixture == "robot_action_direct" or
            args.mixture == "trajectory_3d_human_robot_direct" or
            args.mixture == "trajectory_3d_human_robot_action_direct" or
            args.mixture == "trajecotry_3d_robot_direct_overfit" or
            args.mixture == "trajectory_3d_direct_pick_and_place"
        )
        if use_direct_trajectory_prediction:
            model_cfg.use_direct_trajectory_prediction = True
            log.info("Enabled direct trajectory prediction mode (regression)")
        
        # Configure per-expert dimensions for multi-expert mode
        # Human expert (expert_type=0): EgoDex hand trajectories
        model_cfg.human_action_dim = args.human_action_dim
        model_cfg.human_proprio_dim = args.human_proprio_dim
        
        # Robot expert (expert_type=1): RoboCasa robot actions/trajectories
        model_cfg.robot_action_dim = args.robot_action_dim
        model_cfg.robot_proprio_dim = args.robot_proprio_dim
        model_cfg.robot_trajectory_dim = args.robot_trajectory_dim
        
        # Determine robot action mode: check if this mixture uses joint actions
        # Mixtures that use joint actions:
        # - trajectory_3d_human_robot_action_fm
        # - trajectory_3d_human_robot_action_direct
        robot_use_joint_action = (
            args.mixture == "trajectory_3d_human_robot_action_fm" or
            args.mixture == "trajectory_3d_human_robot_action_direct" or
            args.mixture == "robot_action_fm" or
            args.mixture == "robot_action_direct"
        )
        model_cfg.robot_use_trajectory_as_action = not robot_use_joint_action
        
        # Determine effective robot action dim based on mode
        effective_robot_action_dim = (
            args.robot_trajectory_dim if not robot_use_joint_action 
            else args.robot_action_dim
        )
        
        if args.action_expert_mode == "shared":
            # Shared mode: use MAX dimensions across human and robot
            # This ensures the single shared expert can handle both data types
            # The collator will pad smaller samples to max dimensions
            # The loss will mask out padded dimensions using *_dims tensors
            max_action_dim = max(args.human_action_dim, effective_robot_action_dim)
            max_proprio_dim = max(args.human_proprio_dim, args.robot_proprio_dim)
            
            model_cfg.action_dim = max_action_dim
            model_cfg.proprio_dim = max_proprio_dim
            
            log.info(f"Shared expert mode (padded to max dims):")
            log.info(f"  action_dim={max_action_dim} (human={args.human_action_dim}, robot={effective_robot_action_dim})")
            log.info(f"  proprio_dim={max_proprio_dim} (human={args.human_proprio_dim}, robot={args.robot_proprio_dim})")
        elif args.action_expert_mode == "sequential":
            # Sequential mode: Expert A predicts fingertip trajectory (shared), Expert B predicts robot actions
            # Expert A (expert_id=0): uses human_action_dim for fingertip trajectory prediction
            # Expert B (expert_id=1): uses robot_action_dim for robot action prediction
            # Default action_dim/proprio_dim are for Expert A (backward compatible)
            model_cfg.action_dim = args.human_action_dim  # Expert A output dim (fingertip trajectory)
            model_cfg.proprio_dim = args.human_proprio_dim  # Expert A proprio dim
            
            log.info(f"Sequential expert mode dimensions:")
            log.info(f"  Expert A (id=0, fingertip trajectory): action_dim={args.human_action_dim}, proprio_dim={args.human_proprio_dim}")
            log.info(f"  Expert B (id=1, robot actions): action_dim={args.robot_action_dim}, proprio_dim={args.robot_proprio_dim}")
            log.info(f"  Robot data uses both experts sequentially")
        else:
            # separate_human_robot mode: each expert has its own dimensions
            # Set defaults for backward compatibility (used as fallback)
            model_cfg.action_dim = args.human_action_dim
            model_cfg.proprio_dim = args.human_proprio_dim
            
            log.info(f"Separate expert mode dimensions:")
            log.info(f"  Human expert (id=0): action_dim={args.human_action_dim}, proprio_dim={args.human_proprio_dim}")
            log.info(f"  Robot expert (id=1): action_dim={effective_robot_action_dim}, proprio_dim={args.robot_proprio_dim}")
            log.info(f"  Robot uses {'joint actions' if robot_use_joint_action else 'trajectory'}")
        
        # Configure action expert to be smaller than VLM (OpenPI-style)
        # This reduces memory while maintaining head_dim compatibility
        # OpenPI uses gemma_300m (width=1024) with gemma_2b (width=2048)
        vlm_head_dim = model_cfg.d_model // model_cfg.n_heads
        
        if model_cfg.action_expert_d_model is None:
            model_cfg.action_expert_d_model = model_cfg.d_model // 16        
        # Ensure n_heads matches (required for merged attention)
        model_cfg.action_expert_n_heads = model_cfg.n_heads
        
        # Note: head_dim will automatically match VLM's head_dim in ActionExpertBlock
        # (it uses VLM's head_dim directly, not computed from action_expert_d_model)
        # This allows different d_model values while maintaining head_dim compatibility (OpenPI pattern)
        
        log.info("Flow matching trajectory head enabled with action_horizon=%d", model_cfg.action_horizon)
        
        # Proprioception settings
        model_cfg.include_proprio = not args.exclude_proprio
        if not args.exclude_proprio:
            log.info(f"Proprioception enabled")

    else:
        model_cfg.use_action_expert = False
        model_cfg.action_expert_mode = "disabled"
        model_cfg.num_action_experts = 0

    root_size_mixture: List[RootSizeMixture] = []
    for name, submixture, rate in tasks:
        submixture = get_training_mixture(submixture)
        root_size_mixture.append(RootSizeMixture(rate, submixture))

    evaluations = []
    for task in eval_tasks:
        evaluation = get_evaluation(
            task,
            args.inf_seq_len,
            batch_size=get_world_size()*args.device_inf_batch_size,
            max_examples=max_inf_examples,
            num_workers=2
        )
        evaluation.data.persistent_workers = True
        evaluations.append(evaluation)
    save_interval_unsharded = 15000 if not args.finetune else 7500
    # save_interval_unsharded = 1000
    cfg = TrainConfig(
        run_name="affordance_train",
        no_pre_train_checkpoint=True,
        save_folder="debug_run" if debug else omegaconf.MISSING,
        seed=6198,
        dry_run=False,
        wandb=None if debug else WandbConfig(
            name="${run_name}",
            project="${oc.env:WANDB_PROJECT}",
            group=None,
            entity="${oc.env:WANDB_ENTITY}",
            log_interval=log_interval
        ),
        allow_resume=True,
        model=model_cfg,
        save_overwrite=debug,
        save_dataloader_state=False,
        data=DataConfig(
            root_size_mixture=root_size_mixture,
            for_inference=False,
            shuffle=True,
            split=data_split,
            action_chunking_horizon=model_cfg.action_horizon,
            interpolation_times=args.interpolation_times,
            pad_action_chunk=args.pad_action_chunk,
            drop_last=True,
            sequence_length=args.seq_len,
            num_workers=24,  # Increased from 2 to 32 to enable parallel loading
            pad="to_max",
            shuffle_messages=True,
            pin_memory=True,
            seed=50189
        ),
        ft_connector=not args.freeze_vlm,
        ft_llm=not args.freeze_vlm,
        ft_vit=not args.freeze_vlm,
        # ft_connector=False,
        # ft_llm=False,
        # ft_vit=False,
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            connector_learning_rate=5e-6,
            vit_learning_rate=5e-6,
            llm_learning_rate=1e-5 if not args.slow_warmup else 5e-6,
            # flow_matching_learning_rate=5e-4,
            flow_matching_learning_rate=1e-5,
            connector_weight_decay=0.0,
            vit_weight_decay=0.0,
            llm_weight_decay=0.0,
            # flow_matching_weight_decay=1e-10,
            flow_matching_weight_decay=0,
            connector_betas=[0.9, 0.95],
            vit_betas=[0.9, 0.95],
            llm_betas=[0.9, 0.95],
            flow_matching_betas=[0.9, 0.95],
            connector_eps=1e-6,
            vit_eps=1e-6,
            llm_eps=1e-6,
            flow_matching_eps=1e-6,
            metrics_log_interval=20
        ),
        scheduler=SchedulerConfig(
            name=SchedulerType.multimodal,
            connector_t_warmup=200 if not args.slow_warmup else 30000,
            vit_t_warmup=200 if not args.slow_warmup else 30000,
            llm_t_warmup=200 if not args.slow_warmup else 30000,
            flow_matching_t_warmup=200,
            alpha_f=0.1,
            warmup_min_lr=0.0
        ),
        fsdp=FSDPConfig(
            use_orig_params=True,
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float
        ),
        load_path=None,
        initial_model_checkpoint=None if "debug" in args.checkpoint else args.checkpoint,
        save_interval=20000,
        save_num_checkpoints_to_keep=1,
        # save_interval_unsharded="${max_duration}",
        save_interval_unsharded=save_interval_unsharded,
        global_train_batch_size=global_batch_size,
        device_inf_eval_batch_size=args.device_inf_batch_size,
        device_eval_batch_size=args.device_eval_batch_size,
        device_train_microbatch_size=args.device_train_batch_size if not args.debug else 1,
        time_limit=None,
        max_duration=duration,
        stop_at="${max_duration}",
        max_grad_norm=1,
        batch_divisor=BatchDivisor.global_batch,
        precision="amp_bf16",
        console_log_interval=log_interval,
        speed_monitor=SpeedMonitorConfig(window_size=20),
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        activation_checkpointing=ActivationCheckpointingStrategy.whole_layer,
        # activation_checkpointing=ActivationCheckpointingStrategy.fine_grained,
        eval_interval=eval_interval,
        inf_eval_interval=inf_eval_interval,
        inf_evaluators=evaluations,
        eval_subset_num_batches=eval_subset_batches,
        evaluators=[]
    )

    conf = OmegaConf.create(cfg)
    if other_args:
        overrides = [clean_opt(arg) for arg in other_args]
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(overrides))
    cfg = cast(TrainConfig, OmegaConf.to_object(conf))
    train(cfg)
