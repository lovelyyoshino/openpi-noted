import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.aloha_real import env as _env


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"  # 主机地址，默认为所有可用IP（即0.0.0.0）
    port: int = 8000        # 端口号，默认为8000
    
    action_horizon: int = 25    # 动作预测时间范围，用于控制策略执行的步骤数

    num_episodes: int = 1       # 总共运行的实验次数，即完成多少回合
    max_episode_steps: int = 1000  # 每一轮中最大的步骤数限制


def main(args: Args) -> None:
    # 创建WebSocket客户端策略对象，连接指定主机和端口
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    
    # 输出服务器的元数据到日志
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    # 获取服务端元数据，为环境初始化提供必要的信息
    metadata = ws_client_policy.get_server_metadata()
    
    # 初始化运行时环境，包括仿真环境、智能体以及一些参数设置
    runtime = _runtime.Runtime(
        environment=_env.AlohaRealEnvironment(reset_position=metadata.get("reset_pose")),  # 指定重置位置
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(  # 使用动作块代理作为策略
                policy=ws_client_policy,  # 传入制定的WebSocket客户端策略
                action_horizon=args.action_horizon,  # 设置动作预测时间范围
            )
        ),
        subscribers=[],  # 可以订阅的功能列表，这里为空
        max_hz=50,  # 最大频率设置为50Hz
        num_episodes=args.num_episodes,  # 从命令行参数获取的实验次数
        max_episode_steps=args.max_episode_steps,  # 从命令行参数获取的每个实验的最大步数
    )

    # 启动运行时动态模拟并开始决策过程
    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)  # 配置日志记录等级为INFO
    tyro.cli(main)  # 使用Tyro库解析命令行参数并调用主函数，这个会按照类似uv run main.py --host 1234 --port 8000的方式运行
