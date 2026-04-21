""" Implementation of the command line interface.
"""
from argparse import ArgumentParser

from .__version__ import __version__
from .run_prediction import add_args as pred_args
from .run_prediction import predict
from .run_trainer import add_args as train_args
from .run_trainer import train

__all__ = "main",


# 用于解析命令行参数并执行应用程序的命令行界面(CLI)
def main(argv=None) -> int:
    """ Parse command line arguments. Then execute the application CLI.
    :param argv: argument list to parse
    :return: exit status
    """
    parser = ArgumentParser()
    # 创建了一个ArgumentParser对象，该对象用于解析命令行参数

    parser.add_argument("-v", "--version", action="version",
            version=f"T5Chem {__version__}",
            help="print version and exit")
    # 添加了一个命令行参数，当用户输入 - v或 - -version时，会打印出应用程序的版本号

    subparsers = parser.add_subparsers(title="subcommands")
    # 创建了一个子命令解析器，用于处理不同的子命令

    common = ArgumentParser(add_help=False)
    # 创建了一个不包含帮助信息的ArgumentParser对象，用于定义共享的子命令参数

    _execute(subparsers, common)
    # 调用了一个未定义的函数_execute

    args = parser.parse_args(argv)
    # 解析命令行参数，并将解析结果存储在args变量中

    if not hasattr(args, "command") or not args.command:
        parser.print_help()
        raise SystemExit(1)
    command = args.command
    try:
        command(args)
    except RuntimeError as err:
        return 1
    return 0
 # ''' 检查是否指定了子命令。如果没有指定子命令，打印帮助信息并退出程序
 # 获取用户指定的子命令
 # 执行用户指定的子命令，并捕获可能的RuntimeError异常。如果发生异常，返回退出状态码1
 # 如果程序顺利执行完毕，返回退出状态码0，表示程序成功执行
 # '''

# 用于创建命令行接口（CLI）的解析器
# '''总的来说，这段代码的目的是创建一个命令行接口，其中包含"train"和"predict"两个子命令，每个子命令都有相应的命令行参数，并且定义了默认的操作函数。
# 当用户在命令行中输入"train"或"predict"时，相应的操作函数将会被调用。'''
def _execute(subparsers, common):
    """ CLI adaptor for the api.hello command.
    :param subparsers: subcommand parsers
    :param common: parser for common subcommand arguments
    """
    train_parser = subparsers.add_parser("train", parents=[common])
    train_args(train_parser)
    train_parser.set_defaults(command=train)
    pred_parser = subparsers.add_parser("predict", parents=[common])
    pred_args(pred_parser)
    pred_parser.set_defaults(command=predict)
    return

# '''这个函数用于设置命令行解析器的子命令和相关参数
# 这行代码创建了一个子命令解析器，子命令的名称为"train"。它基于common参数传递的通用解析器进行设置
# 这行代码调用了一个函数train_args()，该函数用于为"train"子命令设置特定的命令行参数
# 这行代码将该子命令的默认操作设置为一个名为train的函数。也就是说，当用户输入"train"子命令时，将会执行train()函数
# '''
# Make the module executable.

if __name__ == "__main__":
    main()
