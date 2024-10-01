import argparse

# 创建解析器
parser = argparse.ArgumentParser(description='Start the model1 and model2 training')

# 添加参数
parser.add_argument('--lr', type=float, required=True, help='Learning Rate')
parser.add_argument('--batch_size', type=int, required=True, help='batch_size')
# parser.add_argument('--verbose', action='store_true', help='是否打印详细信息')

# 解析参数
args = parser.parse_args()

# 使用参数
print(f'Learning rate: {args.lr}')
print(f'Batch Size: {args.batch_size}')
# if args.verbose:
#     print('详细模式已启用')
