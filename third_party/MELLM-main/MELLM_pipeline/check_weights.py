"""Weight integrity checker for MEFlowNet checkpoint files."""
import os
import sys

def check_file(filepath, description):
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✓ {description}")
        print(f"  路径: {filepath}")
        print(f"  大小: {size:.2f} MB\n")
        return True
    else:
        print(f"✗ {description}")
        print(f"  路径: {filepath}")
        print(f"  状态: 缺失\n")
        return False

def main():
    print("=" * 70)
    print("MELLM Pipeline - 权重文件检查")
    print("=" * 70)
    print()

    all_exists = True

    print("【1】MEFlowNet 模型")
    print("-" * 70)
    meflownet_path = os.path.join("..", "ckpt", "meflownet.pth")
    all_exists &= check_file(meflownet_path, "MEFlowNet 光流模型")

    print("【2】LLM 模型")
    print("-" * 70)
    llm_dir = os.path.join("..", "ckpt", "LLM")
    if os.path.isdir(llm_dir):
        llm_files = [f for f in os.listdir(llm_dir) if f.endswith('.safetensors')]
        if len(llm_files) >= 9:
            total_size = sum(os.path.getsize(os.path.join(llm_dir, f)) for f in llm_files) / (1024 * 1024 * 1024)
            print(f"✓ LLM 模型文件")
            print(f"  路径: {llm_dir}")
            print(f"  文件数: {len(llm_files)} 个 .safetensors 文件")
            print(f"  总大小: {total_size:.2f} GB\n")
        else:
            print(f"✗ LLM 模型文件不完整")
            print(f"  路径: {llm_dir}")
            print(f"  找到: {len(llm_files)} 个文件，需要: 9 个\n")
            all_exists = False
    else:
        print(f"✗ LLM 模型目录不存在")
        print(f"  路径: {llm_dir}\n")
        all_exists = False

    print("【3】DepthAnythingV2 模型")
    print("-" * 70)
    depth_path = os.path.join("thirdparty", "DepthAnythingV2", "depth_anything_v2", "depth_anything_v2_vits.pth")
    all_exists &= check_file(depth_path, "DepthAnythingV2 深度估计模型")

    print("【4】OpenFace 权重文件")
    print("-" * 70)
    weights_dir = "weights"
    openface_files = {
        "mobilenetV1X0.25_pretrain.tar": "MobileNet 预训练模型",
        "Alignment_RetinaFace.pth": "RetinaFace 人脸检测模型",
        "Landmark_98.pkl": "98点人脸关键点模型"
    }

    openface_missing = False
    for filename, description in openface_files.items():
        filepath = os.path.join(weights_dir, filename)
        result = check_file(filepath, description)
        all_exists &= result
        if not result:
            openface_missing = True

    print("【5】测试数据")
    print("-" * 70)
    test_image = os.path.join("data", "test", "test.jpg")
    all_exists &= check_file(test_image, "测试图片")

    print("=" * 70)
    print("检查结果汇总")
    print("=" * 70)
    print()

    if all_exists:
        print("✓ 所有权重文件已就绪！")
        print()
        print("你可以运行以下命令启动 Pipeline：")
        print("    python pipeline.py")
        print()
        return 0
    else:
        print("✗ 有文件缺失，请按照提示下载：")
        print()

        if openface_missing:
            print("【OpenFace 权重文件缺失】")
            print("请按照以下步骤下载：")
            print()
            print("1. 访问网盘链接：")
            print("   https://pan.ustc.edu.cn/share/index/33be577e4e6648bab96a?p=1")
            print("   密码：8888")
            print()
            print("2. 下载以下文件：")
            print("   - mobilenetV1X0.25_pretrain.tar")
            print("   - Alignment_RetinaFace.pth")
            print("   - Landmark_98.pkl")
            print()
            print("3. 将文件放入以下目录：")
            print(f"   {os.path.abspath(weights_dir)}")
            print()
            print("详细说明请查看: OpenFace权重下载指南.md")
            print()

        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)
