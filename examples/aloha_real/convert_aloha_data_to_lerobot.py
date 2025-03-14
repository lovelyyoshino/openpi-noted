"""
将Aloha hdf5数据转换为LeRobot数据集v2.0格式的脚本。

示例用法: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True  # 是否使用视频
    tolerance_s: float = 0.0001  # 容忍度（秒）
    image_writer_processes: int = 10  # 图像写入进程数
    image_writer_threads: int = 5  # 图像写入线程数
    video_backend: str | None = None  # 视频后端


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,  # 是否包含速度信息
    has_effort: bool = False,  # 是否包含努力值信息
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    """
    创建一个空的数据集。

    参数：
        repo_id (str): 数据库ID。
        robot_type (str): 机器人类型。
        mode (Literal["video", "image"]): 模式，默认为"video"。
        has_velocity (bool): 是否包含速度信息。
        has_effort (bool): 是否包含努力值信息。
        dataset_config (DatasetConfig): 数据集配置。

    返回：
        LeRobotDataset: 创建的数据集对象。
    """
    motors = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ]# 机械臂关节名称列表
    cameras = [
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
    ]# 相机名称列表

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }# 状态和动作特征，这个格式是固定的，需要包含状态和动作，这个features字典是用于描述数据集的特征，这个shape，只是指定了数据的形状，具体的数据需要在后续的数据填充中添加

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }#如果包含速度信息，则添加速度特征

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }#如果包含力值信息，则添加力值特征

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),  # RGB图像的形状
            "names": [
                "channels",
                "height",
                "width",
            ],
        }#添加图像特征，这里是RGB图像，形状为(3, 480, 640)，数量是定义的cameras列表的长度

    # 如果指定的repo_id目录已存在，则删除该目录
    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,  # 帧率设置为50
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )#创建数据集对象


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    """
    从HDF5文件中获取相机名称。

    参数：
        hdf5_files (list[Path]): HDF5文件列表。

    返回：
        list[str]: 相机名称列表。
    """
    with h5py.File(hdf5_files[0], "r") as ep:
        # 忽略深度通道，目前不处理
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    """
    检查HDF5文件是否包含速度信息。

    参数：
        hdf5_files (list[Path]): HDF5文件列表。

    返回：
        bool: 如果包含速度信息则返回True，否则返回False。
    """
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    """
    检查HDF5文件是否包含努力值信息。

    参数：
        hdf5_files (list[Path]): HDF5文件列表。

    返回：
        bool: 如果包含努力值信息则返回True，否则返回False。
    """
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    """
    加载每个相机的原始图像。

    参数：
        ep (h5py.File): 打开的HDF5文件。
        cameras (list[str]): 相机名称列表。

    返回：
        dict[str, np.ndarray]: 每个相机对应的图像数组字典。
    """
    imgs_per_cam = {}
    for camera in cameras:#遍历相机列表
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4#判断是否为压缩图像

        if uncompressed:
            # 将所有图像加载到内存中
            imgs_array = ep[f"/observations/images/{camera}"][:]#获取图像数据
        else:
            import cv2

            # 一个接一个地加载压缩图像并解压
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))#解压缩图像
            imgs_array = np.array(imgs_array)#转换为numpy数组

        imgs_per_cam[camera] = imgs_array#将图像数据添加到字典中
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    加载原始剧集数据。

    参数：
        ep_path (Path): 剧集路径。

    返回：
        tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        包含图像、状态、动作、速度和努力值的元组。
    """
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])#获取状态数据
        action = torch.from_numpy(ep["/action"][:])

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                "cam_low",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        )

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    """
    填充数据集。

    参数：
        dataset (LeRobotDataset): 要填充的数据集。
        hdf5_files (list[Path]): HDF5文件列表。
        task (str): 任务名称。
        episodes (list[int] | None): 指定的剧集索引，如果为None则处理所有剧集。

    返回：
        LeRobotDataset: 填充后的数据集。
    """
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)#加载原始剧集数据
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }#添加状态和动作数据

            for camera, img_array in imgs_per_cam.items():#遍历图像数据
                frame[f"observation.images.{camera}"] = img_array[i]#添加图像数据

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            dataset.add_frame(frame)#添加帧数据，这个frame是一个字典，包含了状态、动作、速度、努力值和图像数据

        dataset.save_episode(task=task)

    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """
    转换Aloha数据并导入到LeRobot数据集中。

    参数：
        raw_dir (Path): 原始数据目录。
        repo_id (str): 数据库ID。
        raw_repo_id (str | None): 原始数据库ID，如果原始目录不存在则需要提供。
        task (str): 任务名称，默认为"DEBUG"。
        episodes (list[int] | None): 指定的剧集索引，如果为None则处理所有剧集。
        push_to_hub (bool): 是否推送到Hub，默认为True。
        is_mobile (bool): 是否为移动机器人，默认为False。
        mode (Literal["video", "image"]): 模式，默认为"image"。
        dataset_config (DatasetConfig): 数据集配置。

    返回：
        None
    """
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("如果原始目录不存在，则必须提供raw_repo_id")
        download_raw(raw_dir, repo_id=raw_repo_id)

    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))#获取所有的HDF5文件，这个文件是Aloha数据的存储格式

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    dataset.consolidate()#整理数据集

    if push_to_hub:
        dataset.push_to_hub()#将数据集推送到Hub


if __name__ == "__main__":
    tyro.cli(port_aloha)