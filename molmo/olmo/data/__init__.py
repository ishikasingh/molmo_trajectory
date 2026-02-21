import logging
import os
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

from olmo.config import DataConfig, TrainConfig, ModelConfig
from olmo.data.academic_datasets import ChartQa, ScienceQAImageOnly, TextVqa, OkVqa, DocQa, \
    InfoQa, AOkVqa, Vqa2, PlotQa, FigureQa, DvQa, SceneTextQa, TabWMPDirectAnswer, \
    AndroidControl, TallyQa, AI2D, CountBenchQa, RealWorldQa, MathVista, MMMU, ClockBench
from olmo.data.collator import MMCollator
from olmo.data.data_formatter import DataFormatter
from olmo.data.dataset import DeterministicDataset
from olmo.data.iterable_dataset_mixture import IterableDatasetMixture
from olmo.data.model_preprocessor import Preprocessor, MultiModalPreprocessor
from olmo.data.pixmo_datasets import PixMoPointExplanations as PixMoPointExplanationHF, \
    PixMoDocs, PixMoCount, PixMoPoints, PixMoCapQa, PixMoCap, PixMoPointExplanations, \
    PixMoAskModelAnything, PixMoPointsEval
from olmo.data.affordance_datsets import HandPositioningDataset
from olmo.data.robo_casa_affordance_datasets import RobotCasaHandPositioningDataset
from olmo.torch_util import get_global_rank, get_world_size
from olmo.data.trajectory_datasets import TrajectoryDataset
from olmo.data.robo_casa_affordance_datasets import RoboCasaTrajectoryDataset
from olmo.data.trossen_affordance_datasets import TrossenAffordanceDataset

log = logging.getLogger(__name__)


def build_mm_preprocessor(
    model_config: ModelConfig,
    for_inference=False,
    shuffle_messages=True,
    is_training=False,
    require_image_features=False
):
    v_cfg = model_config.vision_backbone
    h, w = model_config.llm_patches_per_crop()
    if not model_config.image_padding_embed:
        image_padding_mask = None
    elif model_config.fix_image_padding:
        image_padding_mask = 2
    else:
        image_padding_mask = 1

    return Preprocessor(
        DataFormatter(
            prompt_templates=model_config.prompt_type,
            message_format=model_config.message_formatting,
            system_prompt=model_config.system_prompt_kind,
            always_start_with_space=model_config.always_start_with_space,
            default_inference_len=model_config.default_inference_len,
            include_proprio=model_config.include_proprio
        ),
        MultiModalPreprocessor(
            tokenizer=model_config.get_tokenizer(),
            normalize=str(v_cfg.image_model_type),
            crop_mode=model_config.crop_mode,
            max_crops=model_config.max_crops,
            overlap_margins=model_config.overlap_margins,
            resize=v_cfg.resize_mode,
            use_col_tokens=model_config.use_col_tokens,
            base_image_input_size=v_cfg.image_default_input_size,
            image_pooling_w=model_config.image_pooling_w,
            image_pooling_h=model_config.image_pooling_h,
            image_token_length_w=w,
            image_token_length_h=h,
            image_patch_size=v_cfg.image_patch_size,
            image_padding_mask=image_padding_mask,
            pad_value=model_config.pad_value,
            loss_token_weighting=model_config.multi_annotation_weighting,
        ),
        for_inference=for_inference,
        shuffle_messages=shuffle_messages,
        is_training=is_training,
        require_image_features=require_image_features,
    )


def build_torch_mm_eval_dataloader(
    batch_size, seed, model_config, data_config, pad_batches, max_steps=None
):
    preprocessor = build_mm_preprocessor(
        model_config, for_inference=data_config.for_inference, shuffle_messages=data_config.shuffle_messages,
        require_image_features=pad_batches
    )
    logging.info(f"Loading eval dataset: {data_config.dataset}/{data_config.split}")
    dataset = get_dataset_by_name(
        data_config.dataset,
        data_config.split,
        action_chunking_horizon=data_config.action_chunking_horizon,
        pad_action_chunk=data_config.pad_action_chunk,
        interpolation_times=getattr(data_config, 'interpolation_times', None),
    )
    n_pad = 0
    if pad_batches:
        global_batch_size = batch_size*get_world_size()
        n_steps = (len(dataset) + global_batch_size - 1) // global_batch_size
        if max_steps:
            n_steps = min(n_steps, max_steps)
        if n_steps*global_batch_size > len(dataset):
            # Pad the dataset so that it can produce enough batches of `global_batch_size` size
            # to cover the entire dataset without dropping any examples
            # We need this if evaluating FSDP models since they will need all devices to get
            # exactly the same number of batches
            n_pad = (n_steps*global_batch_size) - len(dataset)

    dataset = DeterministicDataset(
        dataset=dataset,
        seed=seed,
        preprocessor=preprocessor,
        n_pad=n_pad
    )

    sampler = DistributedSampler(
        dataset,
        drop_last=data_config.drop_last,
        shuffle=data_config.shuffle,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=MMCollator(
            data_config.sequence_length,
            max_crops=model_config.get_max_crops(),
            include_metadata=True,
            pad=data_config.pad,
        ),
        num_workers=data_config.num_workers,
        sampler=sampler,
        pin_memory=data_config.pin_memory,
        prefetch_factor=None if data_config.num_workers == 0 else data_config.prefetch_factor,
        persistent_workers=False if data_config.num_workers == 0 else data_config.persistent_workers,
        timeout=data_config.timeout,
    )


def build_eval_dataloader(
    train_config: TrainConfig,
    data_config: DataConfig,
    batch_size: int,
    max_steps=None
) -> DataLoader:
    seed = data_config.seed if data_config.seed is not None else train_config.seed
    if data_config.multi_modal in ["torch"]:
        return build_torch_mm_eval_dataloader(
            batch_size, seed, train_config.model, data_config,
            pad_batches=train_config.fsdp is not None and not data_config.drop_last,
            max_steps=max_steps
        )
    else:
        raise NotImplementedError(data_config.multi_modal)


def build_train_dataloader(train_config: TrainConfig, device=None) -> DataLoader:
    if device is None:
        device = "cpu"
    assert train_config.device_train_batch_size is not None
    seed = train_config.data.seed if train_config.data.seed is not None else train_config.seed
    data_config = train_config.data
    if train_config.data.multi_modal in ["torch", "torch_hf"]:
        preprocessor = build_mm_preprocessor(
            train_config.model, shuffle_messages=data_config.shuffle, is_training=True, require_image_features=True)
        if data_config.dataset:
            datasets = [get_dataset_by_name(
                data_config.dataset,
                data_config.split,
                action_chunking_horizon=data_config.action_chunking_horizon,
                pad_action_chunk=data_config.pad_action_chunk,
                interpolation_times=getattr(data_config, 'interpolation_times', None))]
            rates = [1]
        else:
            if data_config.mixture:
                mixture = {}
                for name, rate in data_config.mixture.items():
                    logging.info(f"Loading train dataset {name}/{data_config.split}")
                    mixture[name] = (
                        get_dataset_by_name(
                            name,
                            data_config.split,
                            action_chunking_horizon=data_config.action_chunking_horizon,
                            pad_action_chunk=data_config.pad_action_chunk,
                        ),
                        rate,
                    )
            else:
                mixture = {}
                for root_size_mixture in data_config.root_size_mixture:
                    group_datasets = {}
                    for name, as_size in root_size_mixture.mixture.items():
                        logging.info(f"Loading train dataset {name}/{data_config.split}")
                        dataset = get_dataset_by_name(
                            name,
                            data_config.split,
                            action_chunking_horizon=data_config.action_chunking_horizon,
                            pad_action_chunk=data_config.pad_action_chunk,
                            interpolation_times=getattr(data_config, 'interpolation_times', None),
                        )
                        if as_size is not None:
                            size = as_size
                        else:
                            size = len(dataset)
                        group_datasets[name] = (dataset, np.sqrt(size))
                    total_rate = sum(x[1] for x in group_datasets.values())
                    mixture.update({name: (ds, r/total_rate*root_size_mixture.rate)
                                     for name, (ds, r) in group_datasets.items()})

            total_rate = sum(x[1] for x in mixture.values())
            mixture = sorted(mixture.items(), key=lambda x: x[0])
            rates = [rate/total_rate for (_, (_, rate)) in mixture]
            datasets = [ds for (_, (ds, _)) in mixture]
            logging.info("Sampling rates:")
            names = list(x[0] for x in mixture)
            for ix in np.argsort(rates)[::-1]:
                logging.info(f"{names[ix]}: {100*rates[ix]:0.2f}")
        datasets = [DeterministicDataset(ds, preprocessor, data_config.seed) for ds in datasets]
        assert train_config.epoch == 0 or train_config.epoch is None

        dataset = IterableDatasetMixture(
            datasets=datasets,
            mixture_rates=rates,
            global_batch_size=train_config.global_train_batch_size,
            seed=data_config.seed,
            shuffle=data_config.shuffle,
        )
        return DataLoader(
            dataset,
            batch_size=train_config.device_train_batch_size,
            drop_last=train_config.data.drop_last,
            collate_fn=MMCollator(
                train_config.data.sequence_length, False,
                pad=data_config.pad, max_crops=train_config.model.get_max_crops()),
            num_workers=train_config.data.num_workers,
            pin_memory=train_config.data.pin_memory,
            prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
            persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
            timeout=train_config.data.timeout,
        )
    else:
        raise NotImplementedError(train_config.data.multi_modal)


def get_dataset_by_name(dataset_name, split, action_chunking_horizon=None, pad_action_chunk=False, interpolation_times=None):
    horizon = action_chunking_horizon if action_chunking_horizon is not None else 30
    interp = interpolation_times if interpolation_times is not None else 1
    if dataset_name in ["scifi_document_qa", "pixmo_docs_other"]:
        return PixMoDocs("other", split=split)
    elif dataset_name in ["scifi_table_qa", "pixmo_docs_tables"]:
        return PixMoDocs("tables", split=split)
    elif dataset_name in ["scifi_diagram_qa", "pixmo_docs_diagrams"]:
        return PixMoDocs("diagrams", split=split)
    elif dataset_name in ["scifi_charts_qa", "pixmo_docs_charts"]:
        return PixMoDocs("charts", split=split)

    # PixMo-Pointing
    elif dataset_name in ["pointing_high_freq", "pixmo_points_high_freq"]:
        return PixMoPoints(kind="high_frequency", split=split, counting=False)
    elif dataset_name in ["point_count_high_freq", "pixmo_points_high_freq_counting"]:
        return PixMoPoints(kind="high_frequency", split=split, counting=True)
    elif dataset_name in ["pointing", "pixmo_points"]:
        return PixMoPoints(kind="basic", split=split, counting=False)
    elif dataset_name in ["point_count", "pixmo_points_counting"]:
        return PixMoPoints(kind="basic", split=split, counting=True)

    # PixMo-Point-Explanations
    elif dataset_name in ["point_qa", "pixmo_pointing_explanations"]:
        return PixMoPointExplanations(split=split, split_groups=True)

    # PixMo-Count
    elif dataset_name in ["fast_flickr_count_qa_point_count", "pixmo_count_counting"]:
        return PixMoCount(split=split, counting=True)
    elif dataset_name in ["fast_flickr_count_qa_pointing", "pixmo_count"]:
        return PixMoCount(split=split, counting=False)

    # PixMo-AskModelAnything
    elif dataset_name in ["user_qa", "pixmo_ask_model_anything"]:
        return PixMoAskModelAnything(split=split)

    # PixMo-CapQa
    elif dataset_name in ["synthetic_qa_v3_as_user_qa", "pixmo_cap_qa"]:
        return PixMoCapQa(split=split)

    # PixMo-Cap
    if dataset_name in ["cockatoo_and_transcript_712k_sept6", "pixmo_cap_with_transcripts"]:
        return PixMoCap(split, mode="transcript_and_caption")
    if dataset_name in ["cockatoo_712k_sept6", "pixmo_cap"]:
        return PixMoCap(split, mode="captions")

    if dataset_name == "pointing_eval":
        assert split == "test"
        return PixMoPointsEval()

    # Academic datasets
    if dataset_name == "android_control":
        return AndroidControl(split)
    if dataset_name == "android_control_ll":
        return AndroidControl(split, mode="ll")
    if dataset_name == "chart_qa":
        return ChartQa(split, weighted=False)
    if dataset_name == "real_world_qa_no_instruction":
        assert split == "test"
        return RealWorldQa("no_instruction")
    if dataset_name == "chart_qa_weighted":
        return ChartQa(split, weighted=True)
    if dataset_name == "info_qa":
        return InfoQa(split)
    if dataset_name == "doc_qa":
        return DocQa(split)
    if dataset_name == "science_qa_img":
        return ScienceQAImageOnly(split)
    if dataset_name == "coco_2014_vqa_multi":
        return Vqa2(split, multi_question=True)
    if dataset_name == "coco_2014_vqa":
        return Vqa2(split, multi_question=False)
    if dataset_name == "text_vqa":
        return TextVqa(split)
    if dataset_name == "plot_qa":
        return PlotQa(split, in_memory=False)
    if dataset_name == "figure_qa":
        return FigureQa(dict(train="train", validation="validation1")[split])
    if dataset_name == "dv_qa":
        return DvQa(split, in_memory=False)
    if dataset_name == "okvqa":
        return OkVqa(split)
    if dataset_name in ["mmmu"]:
        return MMMU(split)
    if dataset_name in ["mmmu_test"]:
        return MMMU(split)
    if dataset_name == "a_okvqa_da":
        return AOkVqa(split=split, direct_answer=True)
    if dataset_name == "a_okvqa_mc":
        return AOkVqa(split=split, direct_answer=False)
    if dataset_name == "st_qa":
        return SceneTextQa(split=split)
    if dataset_name == "tabwmp_da":
        return TabWMPDirectAnswer(split=split, include_options=False)
    if dataset_name == "countbench_qa":
        assert split == "huggingface"
        return CountBenchQa()
    if dataset_name == "tally_qa":
        return TallyQa(split=split)
    if dataset_name == "ai2_diagram_v2_mix_transparent":
        return AI2D(split=split, boxes="both")
    if dataset_name == "clock_bench":
        return ClockBench(split=split)
    elif dataset_name == "math_vista_v2":
        if split == "validation":
            split = "testmini"
        return MathVista(split)
    elif dataset_name == "affordance":
        data_path = os.environ.get("AFFORDANCE_DATA_DIR")
        return HandPositioningDataset(data_path=data_path, split=split, use_new_output_format=False, ignore_wrist=False)
    elif dataset_name == "affordance_new":
        data_path = os.environ.get("AFFORDANCE_DATA_DIR")
        return HandPositioningDataset(data_path=data_path, split=split, use_new_output_format=True, ignore_wrist=True)
    elif dataset_name == "affordance_with_transitions":
        data_path = os.environ.get("AFFORDANCE_DATA_DIR")
        return HandPositioningDataset(data_path=data_path, split=split, use_new_output_format=True, ignore_wrist=True, use_transitions=True)
    elif dataset_name == "robo_casa_affordance":
        return RobotCasaHandPositioningDataset(use_new_output_format=True, ignore_wrist=True)
    elif dataset_name == "affordance_eval": # only used in the evaluation after training, not the evaluation during training
        data_path = os.environ.get("AFFORDANCE_EVAL_DATA_DIR")
        return HandPositioningDataset(data_path=data_path, split=split)
    elif dataset_name == "affordance_eval_new":
        data_path = os.environ.get("AFFORDANCE_EVAL_DATA_DIR")
        return HandPositioningDataset(data_path=data_path, split=split, use_new_output_format=True, ignore_wrist=True)

    elif dataset_name == "trajectory_2d_text":
        data_dir = os.environ.get("EGODEX_DATA_DIR")
        return TrajectoryDataset(
            data_dir=data_dir,
            split=split,
            action_chunking_horizon=horizon,
            output_2d_trajectory=True,
            normalize_coordinates=True,
            output_format="text",
            pad_action_chunk=pad_action_chunk,
            interpolation_times=interp
        )
    elif dataset_name == "trajectory_3d_text":
        data_dir = os.environ.get("EGODEX_DATA_DIR")
        return TrajectoryDataset(
            data_dir=data_dir,
            split=split,
            action_chunking_horizon=horizon,
            output_2d_trajectory=False,
            normalize_coordinates=False,
            output_format="text",
            pad_action_chunk=pad_action_chunk,
            interpolation_times=interp
        )
    elif dataset_name == "trajectory_3d":
        data_dir = os.environ.get("EGODEX_DATA_DIR")
        stats_file = os.environ.get("TRAJECTORY_STATS_FILE")
        return TrajectoryDataset(
            data_dir=data_dir,
            split=split,
            action_chunking_horizon=horizon,
            output_2d_trajectory=False,
            normalize_coordinates=bool(stats_file),
            stats_file=stats_file,
            output_format="flow_matching",
            frame_downsampling_ratio=10,
            joint_names=['leftIndexFingerTip', 'rightIndexFingerTip'],
            trajectory_representation="delta", # default to be delta, as it works
            pad_action_chunk=pad_action_chunk,
            interpolation_times=interp
        )
    elif dataset_name == "robocasa_3d" or dataset_name == "robocasa_action":
        # RoboCasa dataset with fingertip trajectory as action target
        data_dir = os.environ.get("ROBOCASA_DATA_DIR")
        stats_file = os.environ.get("ROBOCASA_STATS_FILE")
        return RoboCasaTrajectoryDataset(
            data_dir=data_dir,
            split=split,
            action_chunking_horizon=horizon,
            output_2d_trajectory=False,
            normalize_coordinates=bool(stats_file),
            stats_file=stats_file,
            trajectory_representation="delta", # default to be delta, as it works better for flow matching
            frame_downsampling_ratio=10, # robocasa dataset is recorded at 20 fps, while egodex is recorded at 30 fps
            pad_action_chunk=pad_action_chunk
        )
    elif dataset_name == "trossen_3d" or dataset_name == "trossen_action":
        # Trossen/LeRobot dataset with EE positions (run data/add_trossen_ee_to_dataset.py first)
        data_root = os.environ.get("TROSSEN_DATA_DIR", "/root/sky_workdir/FAR-affordance/trossen_data")
        repo_id = 'ykorkmaz/aloha_play_dataset_part_3' if os.environ.get("TROSSEN_REPO_ID", None) is None else os.environ.get("TROSSEN_REPO_ID", None)
        ee_hdf5 = os.environ.get("TROSSEN_EE_HDF5", "/root/sky_workdir/FAR-affordance/trossen_ee_world.hdf5")
        stats_file = os.environ.get("TROSSEN_STATS_FILE", None)
        return TrossenAffordanceDataset(
            repo_id=repo_id,
            data_root=data_root,
            ee_hdf5_path=ee_hdf5,
            split=split,
            action_chunking_horizon=horizon,
            normalize_coordinates=True,
            stats_file=stats_file,
            trajectory_representation="delta",
            frame_downsampling_ratio=3,
            pad_action_chunk=pad_action_chunk,
        )
    # Delta (velocity) representation variants
    elif dataset_name == "trajectory_2d_delta_text":
        data_dir = os.environ.get("EGODEX_DATA_DIR")
        return TrajectoryDataset(
            data_dir=data_dir,
            split=split,
            action_chunking_horizon=horizon,
            output_2d_trajectory=True,
            normalize_coordinates=True,
            output_format="text",
            trajectory_representation="delta",
            pad_action_chunk=pad_action_chunk,
            interpolation_times=interp
        )
    elif dataset_name == "trajectory_3d_delta_text":
        data_dir = os.environ.get("EGODEX_DATA_DIR")
        return TrajectoryDataset(
            data_dir=data_dir,
            split=split,
            action_chunking_horizon=horizon,
            output_2d_trajectory=False,
            normalize_coordinates=False,
            output_format="text",
            trajectory_representation="delta",
            pad_action_chunk=pad_action_chunk,
            interpolation_times=interp
        )
    raise NotImplementedError(dataset_name, split)