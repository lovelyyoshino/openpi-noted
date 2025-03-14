import concurrent.futures
import datetime
import getpass
import logging
import os
import pathlib
import re
import shutil
import stat
import time
import urllib.parse

import boto3
import boto3.s3.transfer as s3_transfer
import botocore
import filelock
import fsspec
import fsspec.generic
import s3transfer.futures as s3_transfer_futures
import tqdm_loggable.auto as tqdm
from types_boto3_s3.service_resource import ObjectSummary
# 环境变量，用于控制缓存目录路径，默认将使用 ~/.cache/openpi。
_OPENPI_DATA_HOME = "OPENPI_DATA_HOME"

logger = logging.getLogger(__name__)


def get_cache_dir() -> pathlib.Path:
    """获取缓存目录路径，如果 /mnt/weka 存在，则使用其路径"""

    default_dir = "~/.cache/openpi"  # 默认的缓存目录
    if os.path.exists("/mnt/weka"):  # 如果存在/mnt/weka则更改默认目录
        default_dir = f"/mnt/weka/{getpass.getuser()}/.cache/openpi"

    # 获取环境变量指定的缓存目录，如果未设置则使用默认目录，并扩展为绝对路径
    cache_dir = pathlib.Path(os.getenv(_OPENPI_DATA_HOME, default_dir)).expanduser().resolve()
    # 创建缓存目录及其父目录（如果尚不存在）
    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_folder_permission(cache_dir)  # 设置文件夹权限
    return cache_dir


def maybe_download(url: str, *, force_download: bool = False, **kwargs) -> pathlib.Path:
    """从远程文件系统下载文件或目录到本地缓存，并返回本地路径。

    如果本地文件已经存在，将直接返回它。

    此函数可以安全地在多个进程中并发调用。
    有关缓存目录的更多信息，请参见 `get_cache_dir`。

    Args:
        url: 要下载的文件的 URL。
        force_download: 如果为 True，即使文件已存在于缓存中也会强制下载该文件。
        **kwargs: 传递给 fsspec 的其他参数。

    Returns:
        下载文件或目录的本地路径，保证存在且为绝对路径。
    """
    # 不使用 fsspec 来解析 URL，以避免不必要的连接到远程文件系统。
    parsed = urllib.parse.urlparse(url)

    # 如果是本地路径，则跳过额外处理。
    if parsed.scheme == "":
        path = pathlib.Path(url)
        if not path.exists():  # 如果文件在路径上不存在，抛出异常
            raise FileNotFoundError(f"File not found at {url}")
        return path.resolve()

    cache_dir = get_cache_dir()  # 获取缓存目录

    local_path = cache_dir / parsed.netloc / parsed.path.strip("/")  # 构建本地路径
    local_path = local_path.resolve()

    # 检查缓存是否需要失效。
    invalidate_cache = False
    if local_path.exists():
        if force_download or _should_invalidate_cache(cache_dir, local_path):
            invalidate_cache = True  # 标记为需要失效
        else:
            return local_path  # 文件有效，直接返回

    try:
        lock_path = local_path.with_suffix(".lock")  # 为此操作创建一个锁文件以防止竞争条件
        with filelock.FileLock(lock_path):
            # 确保锁文件的一致权限。
            _ensure_permissions(lock_path)
            # 首先，如果缓存过期，则删除现有缓存。
            if invalidate_cache:
                logger.info(f"Removing expired cached entry: {local_path}")
                if local_path.is_dir():
                    shutil.rmtree(local_path)  # 如果是目录则删除整个目录
                else:
                    local_path.unlink()  # 否则只删除文件

            logger.info(f"Downloading {url} to {local_path}")  # 日志记录下载动作
            scratch_path = local_path.with_suffix(".partial")  # 临时下载路径

            if _is_openpi_url(url):  # 判断是否为 OpenPI 的 URL
                # 无需凭证进行下载
                _download_boto3(
                    url,
                    scratch_path,
                    boto_session=boto3.Session(
                        region_name="us-west-1",  # 指定 AWS 区域
                    ),
                    botocore_config=botocore.config.Config(signature_version=botocore.UNSIGNED),
                )
            elif url.startswith("s3://"):  # 如果 URL 是 S3 的话
                _download_boto3(url, scratch_path)  # 使用 Boto3 下载
            else:
                _download_fsspec(url, scratch_path, **kwargs)  # 使用 FSSPEC 下载数据

            shutil.move(scratch_path, local_path)  # 将临时文件移动到目标位置
            _ensure_permissions(local_path)  # 确保文件权限正确

    except PermissionError as e:
        msg = (
            f"Local file permission error was encountered while downloading {url}. "
            f"Please try again after removing the cached data using: `rm -rf {local_path}*`"
        )  # 如果出现权限错误，输出提示信息
        raise PermissionError(msg) from e

    return local_path


def _download_fsspec(url: str, local_path: pathlib.Path, **kwargs) -> None:
    """从远程文件系统下载文件到本地缓存，并返回本地路径。"""
    fs, _ = fsspec.core.url_to_fs(url, **kwargs)  # 解析 URL 到文件系统
    info = fs.info(url)  # 获取文件信息
    is_dir = (info["type"] == "directory")  # 判断是否为目录类型
    total_size = fs.du(url) if is_dir else info["size"]  # 确定总大小
    with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)  # 定义线程池执行器
        future = executor.submit(fs.get, url, local_path, recursive=is_dir)  # 异步下载文件
        while not future.done():  # 在文件未完成下载时更新进度条
            current_size = sum(f.stat().st_size for f in [*local_path.rglob("*"), local_path] if f.is_file())
            pbar.update(current_size - pbar.n)  # 更新当前大小
            time.sleep(1)
        pbar.update(total_size - pbar.n)  # 完成后确保进度条达到100%


def _download_boto3(
    url: str,
    local_path: pathlib.Path,
    *,
    boto_session: boto3.Session | None = None,
    botocore_config: botocore.config.Config | None = None,
    workers: int = 16,
) -> None:
    """通过 boto3 从 OpenPI S3 bucket 下载文件。这是一个性能更好的下载方式，但仅限于 s3 urls。

    输入:
        url: OpenPI checkpoint 路径的 URL。
        local_path: 本地下载文件的路径。
        boto_session: 可选的boto3会话，如果没有提供，则默认创建。
        botocore_config: 可选的botocore配置。
        workers: 下载的工作线程数。
    """

    def validate_and_parse_url(maybe_s3_url: str) -> tuple[str, str]:
        parsed = urllib.parse.urlparse(maybe_s3_url)  # 解析URL
        if parsed.scheme != "s3":
            raise ValueError(f"URL must be an S3 URL (s3://), got: {maybe_s3_url}")
        bucket_name = parsed.netloc  # 提取桶名称
        prefix = parsed.path.strip("/")  # 提取前缀
        return bucket_name, prefix

    bucket_name, prefix = validate_and_parse_url(url)  # 验证和解析 URL
    session = boto_session or boto3.Session()  # 创建 botosession

    s3api = session.resource("s3", config=botocore_config)  # 根据session获取S3资源
    bucket = s3api.Bucket(bucket_name)

    # 检查前缀是否指向对象，如果不是，则假设它是一个目录并添加尾随斜杠。
    try:
        bucket.Object(prefix).load()
    except botocore.exceptions.ClientError:
        # 确保追加“/”以避免从不同目录下载共享相同前缀的对象。
        if not prefix.endswith("/"):
            prefix += "/"  # 若没有结尾斜杠，添加斜杠

    objects = [x for x in bucket.objects.filter(Prefix=prefix) if not x.key.endswith("/")]  # 获取候选对象，过滤掉目录
    if not objects:
        raise FileNotFoundError(f"No objects found at {url}")

    total_size = sum(obj.size for obj in objects)  # 计算所有对象的大小之和

    s3t = _get_s3_transfer_manager(session, workers, botocore_config=botocore_config)  # 获取 S3 传输管理器

    def transfer(
        s3obj: ObjectSummary, dest_path: pathlib.Path, progress_func
    ) -> s3_transfer_futures.TransferFuture | None:
        if dest_path.exists():  # 如果目标路径存在
            dest_stat = dest_path.stat()  # 获取目标文件状态
            if s3obj.size == dest_stat.st_size:  # 比较大小
                progress_func(s3obj.size)  # 实时更新进度
                return None
        dest_path.parent.mkdir(parents=True, exist_ok=True)  # 确保父目录存在
        return s3t.download(
            bucket_name,
            s3obj.key,
            str(dest_path),
            subscribers=[
                s3_transfer.ProgressCallbackInvoker(progress_func),  # 添加进度回调
            ],
        )

    try:
        with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
            if os.getenv("IS_DOCKER", "false").lower() == "true":  # 特殊情况下绕开 TQDM 的 bug
                def update_progress(size: int) -> None:
                    pbar.update(size)
                    print(pbar)
            else:
                def update_progress(size: int) -> None:
                    pbar.update(size)

            futures = []
            for obj in objects:
                relative_path = pathlib.Path(obj.key).relative_to(prefix)  # 获取相对路径
                dest_path = local_path / relative_path  # 确定目标路径
                
                if future := transfer(obj, dest_path, update_progress):  # 开始传输，获取future
                    futures.append(future)  # 保留未来结果
            
            for future in futures:
                future.result()  # 等待完成的 future 执行
    finally:
        s3t.shutdown()  # 关闭传输管理器


def _get_s3_transfer_manager(
    session: boto3.Session, workers: int, botocore_config: botocore.config.Config | None = None
) -> s3_transfer.TransferManager:
    """获取 S3 传输管理器，支持多线程下载，可增加连接数，防止超出池大小。"""
    config = botocore.config.Config(max_pool_connections=workers + 2)  # 配置最大连接数
    
    if botocore_config is not None:
        config = config.merge(botocore_config)  # 合并自定义配置
        
    s3client = session.client("s3", config=config)  # 获取 S3 客户端
    transfer_config = s3_transfer.TransferConfig(
        use_threads=True,
        max_concurrency=workers,  # 设置并发数量
    )
    
    return s3_transfer.create_transfer_manager(s3client, transfer_config)  # 返回传输管理器


def _set_permission(path: pathlib.Path, target_permission: int):
    """根据目标权限设置文件的权限，如果权限与目标一致则跳过."""
    if path.stat().st_mode & target_permission == target_permission:
        logger.debug(f"Skipping {path} because it already has correct permissions")
        return  # 权限匹配，跳过
    
    path.chmod(target_permission)  # 设置权限
    logger.debug(f"Set {path} to {target_permission}")


def _set_folder_permission(folder_path: pathlib.Path) -> None:
    """设置文件夹的读取、写入和搜索权限。"""
    _set_permission(folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 给用户、组和其他用户都赋予可读写权限


def _ensure_permissions(path: pathlib.Path) -> None:
    """由于缓存目录与容器化运行以及训练脚本共享，需要确保缓存目录拥有正确的权限。"""

    def _setup_folder_permission_between_cache_dir_and_path(path: pathlib.Path) -> None:
        cache_dir = get_cache_dir()  # 获取缓存目录
        relative_path = path.relative_to(cache_dir)  # 计算相对路径
        moving_path = cache_dir
        for part in relative_path.parts:
            _set_folder_permission(moving_path / part)  # 对每个部分设置文件夹权限
            moving_path = moving_path / part

    def _set_file_permission(file_path: pathlib.Path) -> None:
        """设置所有文件为可读可写，如果是脚本，则保持为脚本."""
        file_rw = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
        if file_path.stat().st_mode & 0o100:  # 如果是可执行文件
            _set_permission(file_path, file_rw | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)  # 设置其为可执行
        else:
            _set_permission(file_path, file_rw)  # 否则仅可读写

    _setup_folder_permission_between_cache_dir_and_path(path)  # 调用设置目录权限方法
    for root, dirs, files in os.walk(str(path)):  # 遍历路径中的内容
        root_path = pathlib.Path(root)
        for file in files:  # 对每个文件设置权限
            file_path = root_path / file
            _set_file_permission(file_path)

        for dir in dirs:  # 对每个目录设置权限
            dir_path = root_path / dir
            _set_folder_permission(dir_path)


def _is_openpi_url(url: str) -> bool:
    """检查 URL 是否为 OpenPI S3 Bucket URL。"""
    return url.startswith("s3://openpi-assets/")  # 检查网址前缀


def _get_mtime(year: int, month: int, day: int) -> float:
    """获取给定日期午夜 UTC 的修改时间戳。"""
    date = datetime.datetime(year, month, day, tzinfo=datetime.UTC)  # 获取指定日期的datetime对象
    return time.mktime(date.timetuple())  # 将其转换为时间戳


# 映射相对路径，定义为正则表达式，映射到过期时间戳（mtime格式）。
# 从上到下将使用部分匹配，第一个匹配将被选择。
# 只有当缓存条目新于到期时间戳时，它们才会被保留。
_INVALIDATE_CACHE_DIRS: dict[re.Pattern, float] = {
    re.compile("openpi-assets/checkpoints/pi0_libero"): _get_mtime(2025, 2, 6),
    re.compile("openpi-assets/checkpoints/"): _get_mtime(2025, 2, 3),
}


def _should_invalidate_cache(cache_dir: pathlib.Path, local_path: pathlib.Path) -> bool:
    """如果缓存过期则失效。如果缓存被失效返回 True。"""

    assert local_path.exists(), f"File not found at {local_path}"

    relative_path = str(local_path.relative_to(cache_dir))  # 计算相对路径
    for pattern, expire_time in _INVALIDATE_CACHE_DIRS.items():  # 遍历所有模式匹配缓存失效规则
        if pattern.match(relative_path):  # 如果路劲匹配某个模式
            # 如果未超过到期时间戳，则移除
            return local_path.stat().st_mtime <= expire_time  

    return False  # 未命中任何过期规则，返回False
