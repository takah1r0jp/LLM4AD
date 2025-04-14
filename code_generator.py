# 実行時は以下のようにコマンドラインから実行してください
# python code_generator.py prompts/normal_conditions_mvtec_loco/breakfast_box.py
# breakfast_box.pyの部分を変更することで、他のnormal_conditionsも指定できます

import requests
import os
import argparse
import importlib.util
from datetime import datetime


def load_normal_conditions(file_path):
    """指定されたファイルパスからnormal_conditionsを動的にロードする"""
    spec = importlib.util.spec_from_file_location("normal_conditions", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normal_conditions


def generate_code(normal_conditions_path):
    # Concatenate the template prompt and normal conditions to form the text prompt
    from prompts.template_prompt import template_prompt

    # Load normal_conditions dynamically
    normal_conditions = load_normal_conditions(normal_conditions_path)

    text_prompt = template_prompt + normal_conditions
    # OpenAI API Key
    # APIキーを取得する方法（以下のいずれかを選択）
    
    # 方法1: 環境変数から取得（存在しない場合はユーザー入力を求める）
    try:
        api_key = os.environ["OPENAI_API_KEY"]
    except KeyError:
        api_key = input("OpenAI APIキーを入力してください: ")
        # 一時的に環境変数に設定（現在のプロセスのみ有効）
        os.environ["OPENAI_API_KEY"] = api_key

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": [{"type": "text", "text": (text_prompt)}]}],
        "max_tokens": 4000,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    s = response.json()["choices"][0]["message"]["content"]

    # normal_conditionsのファイル名を取得（拡張子なし）
    base_name = os.path.splitext(os.path.basename(normal_conditions_path))[0]

    # 現在の日時を取得してファイル名に追加
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 出力パスを設定
    output_dir = "generated_code"
    output_path = os.path.join(output_dir, f"code_{base_name}_{timestamp}.py")

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    # コードを標準形式で保存
    with open(output_path, "w") as o:
        print('code="""', file=o)
        print(s, file=o)
        print('"""', file=o)

    # print(f"Generated code saved to: {output_path}")
    return s


if __name__ == "__main__":
    # コマンドライン引数を処理
    parser = argparse.ArgumentParser(description="Generate code using OpenAI API.")
    parser.add_argument(
        "normal_conditions_path",
        type=str,
        help="Path to the normal_conditions Python file.",
    )
    args = parser.parse_args()

    # Initialize the OpenAI API key
    try:
        api_key = os.environ["OPENAI_API_KEY"]
    except KeyError:
        api_key = input("OpenAI APIキーを入力してください: ")
        # 一時的に環境変数に設定（現在のプロセスのみ有効）
        os.environ["OPENAI_API_KEY"] = api_key
    
    # APIキーの最初の数文字だけを表示（セキュリティのため）
    masked_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:] if len(api_key) > 8 else "****"
    print(f"Using OpenAI API Key: {masked_key}")

    # Call the generate_code function with the specified normal conditions path
    generated_code = generate_code(args.normal_conditions_path)
    print("Generated Code:\n", generated_code)
