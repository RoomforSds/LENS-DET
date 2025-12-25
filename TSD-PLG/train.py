import torch
import torchprofile

from ultralytics import YOLO
from thop import profile

# 安装命令
# python setup.py develop

# 数据集示例百度云链接
# 链接：https://pan.baidu.com/s/19FM7XnKEFC83vpiRdtNA8A?pwd=n93i
# 提取码：n93i

if __name__ == '__main__':
    # 直接使用预训练模型创建模型.
    model = YOLO(r'ultralytics/cfg/models/11/yolo11.yaml')
    model.load('yolo11n.pt')  # 加载预训练的权重文件，可以是 'yolov8n.pt' 或其他预训练模型
    model.train(
        data = "data.yaml",
        imgsz = 640,
        epochs = 300,
        batch = 16,
        close_mosaic = 0,
        workers = 4,
        optimizer = 'SGD',
        project = 'run/train',
        name = 'exp',
    )
    # 查看模型的总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型的总参数量: {total_params}")




    # print('==> Building model..')
    # model = model.cuda()
    # input = torch.randn(1, 3, 640, 640).cuda()
    # flops, params = profile(model, (input,))
    # print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))

    # # 创建输入数据
    # input_tensor = torch.randn(1, 3, 224, 224)
    #
    # # 使用 torchprofile 来计算 FLOPs
    # flops = torchprofile.profile_macs(model, input_tensor)
    # print("模型的总FLOPs为:", flops.items())
    # Evaluate model performance on the validation set
    metrics = model.val()

    # # 使用yaml配置文件来创建模型,并导入预训练权重.
    # model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
    # model.load('yolov8n.pt')
    # # model.train(**{'cfg': 'ultralytics/cfg/exp1.yaml', 'data': 'dataset/data.yaml'})
    #
    # # 模型验证
    # model = YOLO('runs/train/exp46/weights/best.pt')
    # model.val(**{'data': 'dataset/data.yaml'})
    #
    # # 模型推理
    # model = YOLO('runs/train/exp46/weights/best.pt')
    # model.predict(source='dataset/images/test', **{'save': True})