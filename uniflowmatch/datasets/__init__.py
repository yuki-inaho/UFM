from uniflowmatch.datasets.base.flow_postprocessing import (
    PATHWAY_REQUIREMENTS,
    TEXT_QUANTITIES,
    apply_flow_postprocessing_and_merge_batch,
    collate_fn_with_delayed_flow_postprocessing,
    flow_occlusion_post_processing,
)

from uniflowmatch.datasets.blendedmvs import BlendedMVS
from uniflowmatch.datasets.flyingchairs import FlyingChairs
from uniflowmatch.datasets.flyingthings3d import FlyingThings3D
from uniflowmatch.datasets.habitat import Habitat
from uniflowmatch.datasets.hd1k import HD1K
from uniflowmatch.datasets.hypersim import HyperSim
from uniflowmatch.datasets.kubric4d import Kubric4D
from uniflowmatch.datasets.megadepth import MegaDepth
from uniflowmatch.datasets.monkaa import Monkaa
from uniflowmatch.datasets.scannetpp import ScanNetpp
from uniflowmatch.datasets.spring import Spring
from uniflowmatch.datasets.staticthings3d import StaticThings3D
from uniflowmatch.datasets.tartanair_assembled import TartanairAssembled

from uniflowmatch.datasets.rgbd_pairs import RGBDPairDataset

# datasets for evaluation
from uniflowmatch.datasets.dtu import DTU
from uniflowmatch.datasets.eth3d import ETH3D

__all__ = [
    "RGBDPairDataset",
    "BlendedMVS",
    "MegaDepth",
    "ScanNetpp",
    "Habitat",
    "FlyingChairs",
    "FlyingThings3D",
    "Monkaa",
    "TartanairAssembled",
    "StaticThings3D",
    "Spring",
    "HD1K",
    "Kubric4D",
    "DTU",
    "ETH3D",
    "get_data_loader",
    "flow_occlusion_post_processing",
    "collate_fn_with_delayed_flow_postprocessing",
    "apply_flow_postprocessing_and_merge_batch",
    "HyperSim",
]


# The following code is adopted from AnyMap
def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    persistent_workers=False,
    collate_fn=None,
):
    import torch

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    try:
        sampler = dataset.make_sampler(
            batch_size, shuffle=shuffle, world_size=world_size, rank=rank, drop_last=drop_last
        )
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    return data_loader