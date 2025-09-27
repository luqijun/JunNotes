参考教程：https://haoxuebing.github.io/%E5%B7%A5%E5%85%B7/winsw.html

# 安装
winsw install [<path-to-config>] [--no-elevate] [--user|--username <username>] [--pass|--password <password>]
# 卸载
winsw uninstall [<path-to-config>] [--no-elevate]
# 启动
winsw start [<path-to-config>] [--no-elevate]
# 停止
winsw stop [<path-to-config>] [--no-elevate] [--no-wait]
# 重启
winsw restart [<path-to-config>] [--no-elevate]
# 状态
winsw status [<path-to-config>]
# 刷新
winsw refresh [<path-to-config>] [--no-elevate]
# 定制
winsw customize -o|--output <output> --manufacturer <manufacturer>
# 绘制与服务关联的进程树
winsw dev ps [<path-to-config>] [-a|--all]
# 如果服务停止响应，则终止服务
winsw dev kill [<path-to-config>] [--no-elevate]
# 列出由当前可执行文件管理的服务
winsw dev list




