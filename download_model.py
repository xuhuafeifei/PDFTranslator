from modelscope import snapshot_download
import os


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 下载pdf解析大模型
    model_dir = os.path.join(script_dir, "weights/Logics-Parsing")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = "Alibaba-DT/Logics-Parsing"
    snapshot_download(repo_id=model_name, local_dir=model_dir)

    print(f"Parsing model downloaded to {model_dir}")

    # 下载翻译大模型
    model_dir = os.path.join(script_dir, "weights/Qwen2.5-7.5B-Instruct")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = "Qwen/Qwen2.5-7.5B-Instruct"
    snapshot_download(repo_id=model_name, local_dir=model_dir)
    
    print(f"Translation model downloaded to {model_dir}")