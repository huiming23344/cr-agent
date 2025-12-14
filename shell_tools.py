import re
from typing import Any, override
from langchain_community.tools import ShellTool


class SafeShellTool(ShellTool):
    """安全的Shell工具,根据命令危险等级决定是否要求确认"""
    
    # 定义低危命令的正则表达式规则
    LOW_RISK_PATTERNS: list[str] = [
        r'^ls\s',           # 列出目录
        r'^pwd\s*$',        # 显示当前路径
        r'^echo\s',         # 输出文本
        r'^cat\s',          # 查看文件内容
        r'^grep\s',         # 搜索文本
        r'^find\s',         # 查找文件
        r'^ps\s',           # 查看进程
        r'^top\s*$',        # 系统监控
        r'^df\s',           # 磁盘空间
        r'^du\s',           # 目录大小
        r'^whoami\s*$',     # 查看当前用户
        r'^date\s*$',       # 显示日期
        r'^uname\s',        # 系统信息
        r'^which\s',        # 查找命令路径
        r'^head\s',         # 查看文件头部
        r'^tail\s',         # 查看文件尾部
        r'^wc\s',           # 统计文件信息
        r'^sort\s',         # 排序
        r'^uniq\s',         # 去重
        r'^cut\s',          # 提取列
        r'^awk\s',          # 文本处理
        r'^sed\s',          # 文本替换
        r'^tr\s',           # 字符转换
    ]
    
    # 定义高危命令的正则表达式规则
    HIGH_RISK_PATTERNS: list[str] = [
        r'^rm\s',           # 删除文件
        r'^rmdir\s',        # 删除目录
        r'^mv\s',           # 移动/重命名
        r'^cp\s.+\s/\s*$',  # 复制到根目录
        r'^chmod\s',        # 修改权限
        r'^chown\s',        # 修改所有者
        r'^dd\s',           # 磁盘操作
        r'^mkfs\.\w+\s',    # 格式化
        r'^shutdown\s',     # 关机
        r'^reboot\s',       # 重启
        r'^kill\s',         # 终止进程
        r'^pkill\s',        # 批量终止进程
        r'^iptables\s',     # 防火墙
        r'^userdel\s',      # 删除用户
        r'^groupdel\s',     # 删除组
    ]
    
    def __init__(self, **kwargs: Any):
        # 默认开启确认机制
        if 'ask_human_input' not in kwargs:
            kwargs['ask_human_input'] = True
        super().__init__(**kwargs)
    
    def _is_low_risk_command(self, command: str) -> bool:
        """判断命令是否为低危命令"""
        for pattern in self.LOW_RISK_PATTERNS:
            if re.match(pattern, command.strip(), re.IGNORECASE):
                return True
        return False
    
    def _is_high_risk_command(self, command: str) -> bool:
        """判断命令是否为高危命令"""
        for pattern in self.HIGH_RISK_PATTERNS:
            if re.match(pattern, command.strip(), re.IGNORECASE):
                return True
        return False
    
    def _get_command_risk_level(self, command: str) -> str:
        """获取命令的风险等级"""
        if self._is_high_risk_command(command):
            return "high"
        elif self._is_low_risk_command(command):
            return "low"
        else:
            return "medium"
    
    @override
    def _run(
        self,
        commands: str | list[str],
        run_manager: Any | None = None,  # 使用更通用的类型
    ) -> str:
        """执行命令，根据风险等级决定是否要求确认"""
        # 处理命令可能是字符串或列表的情况
        command_str = commands if isinstance(commands, str) else " ".join(commands)
        
        risk_level = self._get_command_risk_level(command_str)
        
        # 低危命令直接执行，不需要确认
        if risk_level == "low":
            self.ask_human_input = False
        else:
            # 中高危命令需要确认
            self.ask_human_input = True
        
        # 如果是高危命令，添加额外警告
        if risk_level == "high":
            print(f"⚠️  警告：检测到高危命令 '{command_str}'")
            print("该命令可能会对系统造成破坏性影响，请谨慎操作！")
        
        return super()._run(command_str, run_manager=run_manager) 


# 创建安全的Shell工具实例
shell_tool = SafeShellTool(
    name="terminal",
    description="Run shell commands with intelligent risk assessment. Low-risk commands execute directly, medium/high-risk commands require confirmation.",
)


