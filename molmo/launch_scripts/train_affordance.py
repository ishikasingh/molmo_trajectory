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
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=30,
        help="Number of timesteps to predict for trajectory actions",
    )
    parser.add_argument(
        "--exclude_proprio",
        action="store_true",
        default=False,
        help="Exclude proprioceptive information (current finger positions) from trajectory prediction",
    )
    parser.add_argument(
        "--proprio_dim",
        type=int,
        default=30,
        help="Dimension of proprioceptive state vector",
    )
    parser.add_argument(
        "--max_crops",
        type=int,
        default=None,
        help="Override max_crops parameter for the VLM model (default: None means no override)",
    )
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
        tasks = [["egodex", ["trajectory_3d_fm"], 1.0]]
    elif args.mixture == "trajectory_3d_direct":
        # Direct 3D trajectory prediction (regression) using action expert architecture
        eval_tasks = []
        tasks = [["egodex", ["trajectory_3d_fm"], 1.0]]
    elif args.mixture == "trajectory_3d_fm_overfit":
        # Flow matching based 3D trajectory prediction with delta representation
        eval_tasks = []
        tasks = [["egodex", ["trajectory_3d_fm"], 1.0]]
        data_split = "overfit"
    elif args.mixture == "trajectory_3d_fm_pick_and_place":
        # Flow matching based 3D trajectory prediction with delta representation
        eval_tasks = []
        tasks = [["egodex", ["trajectory_3d_fm"], 1.0]]
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
    if "_fm" in args.mixture or "trajectory_3d_direct" in args.mixture:
        model_cfg.use_action_expert = True
        # Configure action dimensions based on task
        # action_horizon: number of timesteps to predict (e.g., 30)
        # action_dim: flattened coordinate dimension = num_joints * num_coordinates
        #   For 2D trajectories: num_joints * 2
        #   For 3D trajectories: num_joints * 3
        # Example: 10 joints * 3 coords = 30 dimensions
        model_cfg.action_horizon = args.action_horizon
        model_cfg.action_dim = 30      # num_joints * coordinates (currently assuming 10 joints * 3 for 3D)
        model_cfg.use_adarms_flow_matching = False  # Use Pi0-style MLP conditioning
        model_cfg.flow_matching_loss_weight = 1.0
        model_cfg.flow_matching_num_steps = 10  # ODE integration steps during inference
        
        if "trajectory_3d_direct" in args.mixture:
            model_cfg.use_direct_trajectory_prediction = True
            log.info("Enabled direct trajectory prediction mode (regression)")
        
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
        
        log.info("Flow matching trajectory head enabled with action_horizon=%d, action_dim=%d", 
                 model_cfg.action_horizon, model_cfg.action_dim)
        
        # Proprioception settings
        model_cfg.include_proprio = not args.exclude_proprio
        model_cfg.proprio_dim = args.proprio_dim
        if not args.exclude_proprio:
            log.info(f"Proprioception enabled with dimension {args.proprio_dim}")

    else:
        model_cfg.use_action_expert = False

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
    save_interval_unsharded = 20000 if not args.finetune else 7500
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
            drop_last=True,
            sequence_length=args.seq_len,
            num_workers=24,  # Increased from 2 to 32 to enable parallel loading
            pad="to_max",
            shuffle_messages=True,
            pin_memory=True,
            seed=50189
        ),
        ft_connector=True,
        ft_llm=True,
        # ft_vit=True,
        # ft_connector=False,
        # ft_llm=False,
        ft_vit=False,
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            connector_learning_rate=5e-6,
            vit_learning_rate=5e-6,
            llm_learning_rate=1e-5,
            flow_matching_learning_rate=5e-4,
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
            connector_t_warmup=200,
            vit_t_warmup=200,
            llm_t_warmup=200,
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
        save_interval=5000,
        save_num_checkpoints_to_keep=1,
        # save_interval_unsharded="${max_duration}",
        save_interval_unsharded=save_interval_unsharded,
        global_train_batch_size=global_batch_size,
        device_inf_eval_batch_size=args.device_inf_batch_size,
        device_eval_batch_size=args.device_eval_batch_size,
        device_train_microbatch_size=args.device_train_batch_size,
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
