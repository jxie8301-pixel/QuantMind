#!/usr/bin/env python3
"""
QuantMind 服务器回测压力测试脚本

测试服务器能同时支持多少并发回测请求。

使用方法:
    python stress_test_backtest.py --host http://localhost:8000 --users 10 --concurrent 5

参数说明:
    --host: 服务器地址 (默认: http://localhost:8000)
    --users: 模拟用户数量 (默认: 10)
    --concurrent: 每用户并发请求数 (默认: 5)
    --duration: 测试持续时间秒数 (默认: 60)
    --ramp-up: 用户启动间隔秒数 (默认: 1)
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import aiohttp


@dataclass
class RequestResult:
    """单次请求结果"""

    user_id: int
    request_id: int
    success: bool
    status_code: int | None = None
    response_time: float = 0.0
    error_message: str | None = None
    annual_return: float | None = None
    sharpe_ratio: float | None = None


@dataclass
class UserStats:
    """用户统计"""

    user_id: int
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    response_times: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests * 100

    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)


@dataclass
class TestConfig:
    """测试配置"""

    host: str
    num_users: int
    concurrent_per_user: int
    duration: int
    ramp_up: float
    internal_secret: str = "dev-internal-call-secret"
    strategy_types: list[str] = field(
        default_factory=lambda: ["TopkDropout", "standard_topk"]
    )

    # 回测参数范围
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    topk_range: tuple[int, int] = (20, 80)
    initial_capital_range: tuple[float, float] = (10_000_000, 100_000_000)


class BacktestStressTest:
    """回测压力测试"""

    def __init__(self, config: TestConfig):
        self.config = config
        self.results: list[RequestResult] = []
        self.user_stats: dict[int, UserStats] = {}
        self.start_time: float = 0
        self.end_time: float = 0
        self._stop_flag = False

    def generate_backtest_request(self, user_id: int) -> dict[str, Any]:
        """生成随机回测请求"""
        topk = random.randint(*self.config.topk_range)
        initial_capital = random.uniform(*self.config.initial_capital_range)
        strategy_type = random.choice(self.config.strategy_types)

        # user_id 必须和 X-User-Id header 一致
        return {
            "strategy_type": strategy_type,
            "strategy_params": {
                "topk": topk,
                "n_drop": topk // 5,
                "signal": "<PRED>",
            },
            "start_date": self.config.start_date,
            "end_date": self.config.end_date,
            "initial_capital": initial_capital,
            "benchmark": "SH000300",
            "universe": "all",
            "user_id": str(user_id),  # 必须和 header 一致
            "tenant_id": "default",
        }

    async def execute_single_request(
        self,
        session: aiohttp.ClientSession,
        user_id: int,
        request_id: int,
    ) -> RequestResult:
        """执行单次回测请求"""
        url = f"{self.config.host}/api/v1/qlib/backtest"
        payload = self.generate_backtest_request(user_id)

        # 内部调用认证头
        headers = {
            "X-Internal-Call": self.config.internal_secret,
            "X-User-Id": str(user_id),
            "Content-Type": "application/json",
        }

        start = time.time()
        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300),  # 5分钟超时
            ) as response:
                response_time = time.time() - start
                status_code = response.status

                if response.status == 200:
                    data = await response.json()
                    return RequestResult(
                        user_id=user_id,
                        request_id=request_id,
                        success=True,
                        status_code=status_code,
                        response_time=response_time,
                        annual_return=data.get("annual_return"),
                        sharpe_ratio=data.get("sharpe_ratio"),
                    )
                else:
                    error_text = await response.text()
                    return RequestResult(
                        user_id=user_id,
                        request_id=request_id,
                        success=False,
                        status_code=status_code,
                        response_time=response_time,
                        error_message=error_text[:200],
                    )

        except asyncio.TimeoutError:
            return RequestResult(
                user_id=user_id,
                request_id=request_id,
                success=False,
                response_time=time.time() - start,
                error_message="Request timeout (>300s)",
            )
        except Exception as e:
            return RequestResult(
                user_id=user_id,
                request_id=request_id,
                success=False,
                response_time=time.time() - start,
                error_message=str(e)[:200],
            )

    async def user_worker(
        self,
        user_id: int,
        session: aiohttp.ClientSession,
    ) -> UserStats:
        """单个用户的工作协程"""
        stats = UserStats(user_id=user_id)
        request_id = 0

        while not self._stop_flag:
            # 创建并发请求
            tasks = []
            for _ in range(self.config.concurrent_per_user):
                if self._stop_flag:
                    break
                request_id += 1
                task = self.execute_single_request(session, user_id, request_id)
                tasks.append(task)

            if not tasks:
                break

            # 等待所有并发请求完成
            results = await asyncio.gather(*tasks)

            for result in results:
                self.results.append(result)
                stats.total_requests += 1
                stats.response_times.append(result.response_time)
                stats.total_response_time += result.response_time

                if result.success:
                    stats.successful_requests += 1
                else:
                    stats.failed_requests += 1

            # 短暂休息，避免过于密集
            await asyncio.sleep(0.1)

        return stats

    async def run_test(self) -> None:
        """运行压力测试"""
        print("\n" + "=" * 60)
        print("QuantMind 回测服务压力测试")
        print("=" * 60)
        print(f"目标服务器: {self.config.host}")
        print(f"模拟用户数: {self.config.num_users}")
        print(f"每用户并发: {self.config.concurrent_per_user}")
        print(f"测试时长: {self.config.duration} 秒")
        print(f"用户启动间隔: {self.config.ramp_up} 秒")
        print("=" * 60 + "\n")

        self.start_time = time.time()

        connector = aiohttp.TCPConnector(
            limit=self.config.num_users * self.config.concurrent_per_user + 10,
            keepalive_timeout=30,
        )

        async with aiohttp.ClientSession(connector=connector) as session:
            # 先检查服务健康状态
            headers = {
                "X-Internal-Call": self.config.internal_secret,
                "X-User-Id": "0",
            }
            try:
                async with session.get(
                    f"{self.config.host}/api/v1/qlib/health",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        print("✓ 服务器健康检查通过\n")
                    else:
                        print(f"⚠ 服务器健康检查异常: {resp.status}\n")
            except Exception as e:
                print(f"✗ 无法连接服务器: {e}\n")
                return

            # 启动用户工作协程
            user_tasks = []
            for user_id in range(self.config.num_users):
                task = asyncio.create_task(self.user_worker(user_id, session))
                user_tasks.append(task)
                # 用户启动间隔
                await asyncio.sleep(self.config.ramp_up)

            # 运行指定时长后停止
            await asyncio.sleep(self.config.duration)
            self._stop_flag = True

            # 等待所有用户完成
            print("\n正在等待所有请求完成...")
            user_stats_list = await asyncio.gather(*user_tasks)

            for stats in user_stats_list:
                self.user_stats[stats.user_id] = stats

        self.end_time = time.time()
        self._print_report()

    def _print_report(self) -> None:
        """打印测试报告"""
        total_time = self.end_time - self.start_time
        total_requests = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total_requests - successful

        # 计算响应时间统计
        response_times = [r.response_time for r in self.results]
        if response_times:
            avg_response = statistics.mean(response_times)
            median_response = statistics.median(response_times)
            min_response = min(response_times)
            max_response = max(response_times)
            if len(response_times) > 1:
                std_response = statistics.stdev(response_times)
            else:
                std_response = 0
        else:
            avg_response = median_response = min_response = max_response = std_response = 0

        # 计算吞吐量
        requests_per_second = total_requests / total_time if total_time > 0 else 0

        # 成功率
        success_rate = (successful / total_requests * 100) if total_requests > 0 else 0

        print("\n" + "=" * 60)
        print("压力测试报告")
        print("=" * 60)

        print("\n【测试概况】")
        print(f"  测试时长: {total_time:.2f} 秒")
        print(f"  总请求数: {total_requests}")
        print(f"  成功请求: {successful}")
        print(f"  失败请求: {failed}")
        print(f"  成功率: {success_rate:.2f}%")

        print("\n【吞吐量】")
        print(f"  请求/秒 (RPS): {requests_per_second:.2f}")
        print(f"  理论最大并发用户: {int(requests_per_second * avg_response):,}")

        print("\n【响应时间】")
        print(f"  平均: {avg_response:.2f} 秒")
        print(f"  中位数: {median_response:.2f} 秒")
        print(f"  最小: {min_response:.2f} 秒")
        print(f"  最大: {max_response:.2f} 秒")
        print(f"  标准差: {std_response:.2f} 秒")

        # 响应时间分布
        if response_times:
            print("\n【响应时间分布】")
            percentiles = [50, 75, 90, 95, 99]
            sorted_times = sorted(response_times)
            for p in percentiles:
                idx = int(len(sorted_times) * p / 100)
                idx = min(idx, len(sorted_times) - 1)
                print(f"  P{p}: {sorted_times[idx]:.2f} 秒")

        # 错误统计
        if failed > 0:
            print("\n【错误统计】")
            error_counts: dict[str, int] = {}
            for r in self.results:
                if not r.success:
                    key = f"{r.status_code or 'N/A'}: {r.error_message or 'Unknown'}"
                    error_counts[key] = error_counts.get(key, 0) + 1

            for error, count in sorted(
                error_counts.items(), key=lambda x: -x[1]
            )[:10]:
                print(f"  [{count}次] {error[:80]}")

        # 每用户统计
        print("\n【用户统计】")
        print(f"{'用户ID':<10} {'总请求':<10} {'成功':<10} {'失败':<10} {'成功率':<12} {'平均响应':<12}")
        print("-" * 64)
        for user_id in sorted(self.user_stats.keys()):
            stats = self.user_stats[user_id]
            print(
                f"{user_id:<10} {stats.total_requests:<10} "
                f"{stats.successful_requests:<10} {stats.failed_requests:<10} "
                f"{stats.success_rate:.1f}%{'':<6} {stats.avg_response_time:.2f}s"
            )

        print("\n" + "=" * 60)

        # 保存详细结果到文件
        self._save_results()

    def _save_results(self) -> None:
        """保存详细结果到JSON文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stress_test_results_{timestamp}.json"

        output = {
            "config": {
                "host": self.config.host,
                "num_users": self.config.num_users,
                "concurrent_per_user": self.config.concurrent_per_user,
                "duration": self.config.duration,
            },
            "summary": {
                "total_time": self.end_time - self.start_time,
                "total_requests": len(self.results),
                "successful": sum(1 for r in self.results if r.success),
                "failed": sum(1 for r in self.results if not r.success),
            },
            "results": [
                {
                    "user_id": r.user_id,
                    "request_id": r.request_id,
                    "success": r.success,
                    "status_code": r.status_code,
                    "response_time": r.response_time,
                    "error_message": r.error_message,
                }
                for r in self.results
            ],
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"详细结果已保存到: {filename}")


async def main():
    parser = argparse.ArgumentParser(description="QuantMind 回测服务压力测试")
    parser.add_argument(
        "--host",
        default="http://localhost:8000",
        help="服务器地址 (默认: http://localhost:8000)",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=10,
        help="模拟用户数量 (默认: 10)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=5,
        help="每用户并发请求数 (默认: 5)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="测试持续时间秒数 (默认: 60)",
    )
    parser.add_argument(
        "--ramp-up",
        type=float,
        default=1.0,
        help="用户启动间隔秒数 (默认: 1.0)",
    )
    parser.add_argument(
        "--secret",
        default="dev-internal-call-secret",
        help="内部调用密钥 (默认: dev-internal-call-secret)",
    )

    args = parser.parse_args()

    config = TestConfig(
        host=args.host,
        num_users=args.users,
        concurrent_per_user=args.concurrent,
        duration=args.duration,
        ramp_up=args.ramp_up,
        internal_secret=args.secret,
    )

    test = BacktestStressTest(config)
    await test.run_test()


if __name__ == "__main__":
    asyncio.run(main())
