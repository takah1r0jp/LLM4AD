from PIL import Image
import torch
import numpy as np
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoImageProcessor,
    StoppingCriteria,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)
import argparse
import os
import json
from generated_code import code_breakfastbox, code_juicebottle, code_pushpins, code_splicingconnectors, code_screwbag
from datetime import datetime


def get_arguments():
    """コマンドライン引数を解析する関数"""
    parser = argparse.ArgumentParser(description="異常検知の設定を受け取る")

    parser.add_argument("--image_dir", type=str, default="./data/MVTec_LOCO", help="画像のフォルダパス")
    parser.add_argument(
        "--cls",
        type=int,
        default=0,
        help="カテゴリの番号 (0:breakfast_box, 1:juice_bottle, 2:pushpins, 3:screw_bag, 4:splicing_connectors)",
    )
    parser.add_argument(
        "--subcls", type=int, default=0, help="サブカテゴリ (0:good, 1:logical_anomalies, 2:structural_anomalies)"
    )
    parser.add_argument("--begin_of_image", type=int, default=0, help="処理を開始するインデックス")
    parser.add_argument("--end_of_image", type=int, default=0, help="処理を終了するインデックス")
    parser.add_argument("--begin_of_function", type=int, default=1, help="実行を開始する関数")
    parser.add_argument("--end_of_function", type=int, default=1, help="実行を終了する関数")
    parser.add_argument("--output_dir", type=str, default="result", help="結果を保存するフォルダパス")

    return parser.parse_args()


def main():
    """コードを選択し、画像ごとに execute_command を順番に実行"""

    # 使用するコードとカテゴリ
    code_list = [code_breakfastbox, code_juicebottle, code_pushpins, code_screwbag, code_splicingconnectors]
    category_list = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
    anomalies_list = ["good", "logical_anomalies", "structural_anomalies"]

    # 引数の取得
    args = get_arguments()

    num_category = args.cls
    category2 = anomalies_list[args.subcls]
    start = args.begin_of_image
    end = args.end_of_image

    # 使用するコードを選択
    code = code_list[num_category]
    category = category_list[num_category]

    # コードの整形
    code = code.replace("```", "")  # 不要なバッククォートを削除
    function_definitions = code.split("def ")  # 関数ごとに分割

    # 実行する関数の範囲
    func_start = args.begin_of_function
    func_end = args.end_of_function

    # 画像フォルダの選択
    image_dir = args.image_dir
    if not os.path.exists(image_dir):
        print(f"指定された画像フォルダが存在しません: {image_dir}")

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, category)):
        os.makedirs(os.path.join(output_dir, category))

    total_anomaly_score_list = []
    total_anomaly_score_dict = {}
    normalized_anomaly_score_dict = {}

    for j in range(start, end + 1):
        total_anomaly_score = 0
        anomaly_score_list = []

        print(f"Start processing: {category}:{category2}:{j:03d}.png")

        # 画像パスの設定
        image_path = os.path.join(image_dir, category, "test", category2, f"{j:03d}.png")

        # 画像が存在しない場合の処理
        if not os.path.exists(image_path):
            print(f"画像が見つかりません: {image_path}")
            continue

        try:
            image = Image.open(image_path)
            image = image.convert("RGB")  # PNG を JPG 形式に変換
        except Exception as e:
            print(f"画像の読み込みエラー: {e}")
            continue

        for i in range(func_start, func_end + 1):
            func_name = f"execute_command{i}"

            # 関数コードを取得して整形
            if i >= len(function_definitions):
                print(f"関数 execute_command{i} はコード内に存在しません。")
                continue

            final_code = "def " + function_definitions[i]

            # 関数を実行し、異常スコアを取得
            anomaly_score = execute_function_from_code(final_code, func_name, image_path, image)

            # anomaly_scoreがintかfloatじゃなければエラー通知
            if not isinstance(anomaly_score, (int, float)):
                raise TypeError(f"scoreはint型またはfloat型である必要があります。現在の型: {type(anomaly_score).__name__}")

            if not isinstance(anomaly_score, int):  # int型でない場合
                anomaly_score = round(float(anomaly_score), 5)

            if anomaly_score is not None:
                total_anomaly_score += anomaly_score
                anomaly_score_list.append(anomaly_score)
                print(f"{func_name} の異常スコア: {anomaly_score}")

            print("-----------------")

            total_anomaly_score_list.append(total_anomaly_score)
            # print(total_anomaly_score_list)
            total_anomaly_score_dict.setdefault(f"{j:03d}", anomaly_score_list)
            print(total_anomaly_score_dict)
            print("-----------------")
    # total_anomaly_scoreの正規化
    normalized_total_anomaly_score = total_anomaly_score / len(anomaly_score_list)
    normalized_anomaly_score_dict.setdefault(f"{j:03d}", normalized_total_anomaly_score)
    print("normalized anomaly_score")
    print(normalized_anomaly_score_dict)
    print("---------------------")

    # Get today's date
    today_date = datetime.today().strftime("%Y-%m-%d-%H-%M")

    # Save anomaly_score_dict to a JSON file
    output_path = f"{output_dir}/{category}/{today_date}_{category2}_{start}to{end}.json"
    with open(output_path, "w") as json_file:
        json.dump(normalized_anomaly_score_dict, json_file, indent=4)
    print(f"Anomaly scores saved to {output_path}")


def execute_function_from_code(code, func_name, image_path, image):
    """指定された関数をコードから実行し、異常スコアを取得"""
    namespace = {}
    try:
        # `exec` の影響範囲を限定するため `namespace` を使用
        exec(code, globals(), namespace)

        # 実行されたコードの中から `func_name` に対応する関数を取得
        func = namespace.get(func_name)

        # 取得したオブジェクトが実際に関数なら実行する
        if callable(func):
            print(f"excute -> {func_name}")
            return func(image_path, image)
        else:
            print(f"関数 {func_name} が見つかりません。")
            return None
    except Exception as e:
        print(f"関数 {func_name} の実行中にエラーが発生: {e}")
        return None


def check_object_color(image_path, object_name, color):
    question = f"Is the color of {object_name} {color}? Answer True or False."
    answer = vqa(image_path, question).replace(" ", "")
    print(answer)
    return answer


def verify_a_is_b(image_path, object_a, object_b):
    question = f""" Is the {object_a} a {object_b}? Answer Yes or No"""
    answer = vqa(image_path, question).replace(" ", "")
    print(answer)
    return answer


class ModelLoader:
    _model = None

    @classmethod
    def load_model(cls):
        if cls._model is None:
            # モデルがまだロードされていない場合
            print("Loading model...")
            model_name_or_path = "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
            cls._model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, trust_remote_code=True)
            cls._tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False
            )
            cls._image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
            cls._tokenizer = cls._model.update_special_tokens(cls._tokenizer)
            # cls._model = YourModel.load("model_path")  # モデルをロードするコード
            print("finished loading")
        else:
            print("Model already loaded.")
        return cls._model, cls._image_processor, cls._tokenizer


def vqa(img_path, query):
    model, image_processor, tokenizer = ModelLoader.load_model()

    raw_image = Image.open(img_path).convert("RGB")

    model = model.cuda()
    inputs = image_processor([raw_image], return_tensors="pt", image_aspect_ratio="anyres")
    prompt = apply_prompt_template(query)
    language_inputs = tokenizer([prompt], return_tensors="pt")
    inputs.update(language_inputs)
    inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
    generated_text = model.generate(
        **inputs,
        image_size=[raw_image.size],
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        max_new_tokens=768,
        top_p=None,
        num_beams=1,
        stopping_criteria=[EosListStoppingCriteria()],
    )
    prediction = tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0]
    return prediction


# define the prompt template
def apply_prompt_template(prompt):
    s = (
        "<|system|>\nA chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
        f"<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n"
    )
    return s


class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[32007]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence) :].tolist()
        return self.eos_sequence in last_ids


class ImagePatch:
    # A Python class containing a crop of an image centered around a particular object , as well as relevant information .
    # Attributes
    # ----------
    # cropped_image : array_like
    # An array-like of the cropped image taken from the original image.
    # left , lower, right , upper : int
    # An int describing the position of the ( left /lower/ right /upper) border of the crop's bounding box in the original image.
    # Methods
    # -------
    # find (object_name: str )->List[ImagePatch]
    # Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
    # image matching the object_name.
    # visual_question_answering ( question : str =None)->str
    # Returns the answer to a basic question asked about the image. If no question is provided , returns the answer to "What is this ?".
    # image_caption()->str
    # Returns a short description of the image crop.
    # expand_patch_with_surrounding()->ImagePatch
    # Returns a new ImagePatch object that contains the current ImagePatch and its surroundings.
    # overlaps(patch : ImagePatch)->Bool
    # Returns True if the current ImagePatch overlaps with another patch and False otherwise
    # compute_depth()->float
    # Returns the median depth of the image patch . The bigger the depth, the further the patch is from the camera.

    def __init__(self, image, left: int = None, lower: int = None, right: int = None, upper: int = None, score=1.0):
        # Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        # attributes . If no coordinates are provided , the image is left unmodified, and the coordinates are set to the
        # dimensions of the image.
        # Parameters
        # -------
        # image: PIL.Image
        # An array-like of the original image.
        # left , lower, right , upper : int
        # An int describing the position of the ( left /lower/ right /upper) border of the crop's bounding box in the original image.
        # The coordinates (y1,x1,y2,x2) are with respect to the upper left corner the original image.
        # To be closer with human perception, left , lower, right , upper are with respect to the lower left corner of the squared image.
        # Use left , lower, right , upper for downstream tasks.

        self.original_image = image
        size_x, size_y = image.size

        if left is None and right is None and upper is None and lower is None:
            self.x1 = 0
            self.y1 = 0
            self.x2 = size_x
            self.y2 = size_y
        else:
            self.x1 = left
            self.y1 = upper
            self.x2 = right
            self.y2 = lower

        # self .cropped_image = image.crop(( int ( self .x1/1000*self . sz) , int ( self .y1/1000*self . sz) , int ( self .x2/1000*self . sz) , int ( self .y2/1000*self . sz)))

        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1

        # all coordinates use the upper left corner as the origin (0,0) .
        # However, human perception uses the lower left corner as the origin .
        # So, need to revert upper/lower for language model
        self.left = self.x1
        self.right = self.x2
        self.upper = self.y1
        self.lower = self.y2

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

        self.patch_description_string = f"{self.x1} {self.y1} {self.x2} {self.y2}"

        self.detection_score = score
        self.box = [self.x1, self.y1, self.x2, self.y2]

    def __str__(self):
        return self.patch_description_string + f" score: {self.detection_score}"

    def find(self, object_name: str):
        # Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        # The object_name should be as simple as example, including only nouns
        # Otherwise, returns an empty list .
        # Note that the returned patches are not ordered
        # Parameters
        # ----------
        # object_name : str
        # the name of the object to be found

        # Returns
        # -------
        # List [ImagePatch]
        # a list of ImagePatch objects matching object_name contained in the crop

        # Examples
        # --------
        # >>> # find all the kids in the images
        # >>> def execute_command(image) -> List[ImagePatch]:
        # >>> image_patch = ImagePatch(image)
        # >>> kid_patches = image_patch. find ("kid")
        # >>> return kid_patches
        print(f"Calling find function . Detect {object_name}.")
        # return a dict of patches
        det_patches_dict = detect(self.original_image, object_name)
        # print (f"Detection result : {' and '. join ([ str (d) + ' ' + object_name for d in det_patches ])}")
        return det_patches_dict

    def expand_patch_with_surrounding(self):
        # Expand the image patch to include the surroundings. Now done by keeping the center of the patch
        # and returns a patch with double width and height

        # Examples
        # -------
        # >>> # How many kids are not sitting under an umbrella?
        # >>> def execute_command(image):
        # >>> image_patch = ImagePatch(image)
        # >>> kid_patches = image_patch. find ("kid")

        # >>> # Find the kids that are under the umbrella .
        # >>> kids_not_under_umbrella = []

        # >>> for kid_patch in kid_patches:
        # >>> kid_with_surrounding = kid_patch .expand_patch_with_surrounding()
        # >>> if "yes" in kid_with_surrounding . visual_question_answering (" Is the kid under the umbrella?") :
        # >>> print (f"the kid at {kid_patch} is sitting under an umbrella .")
        # >>> else :
        # >>> print (f"the kid at {kid_patch} is not sitting under an umbrella .")
        # >>> kids_not_under_umbrella .append(kid_patch)

        # >>> # Count the number of kids under the umbrella .
        # >>> num_kids_not_under_umbrella = len(kids_not_under_umbrella)

        # >>> return formatting_answer( str (num_kids_not_under_umbrella))

        new_left = max(self.left - self.width / 2, 0)
        new_right = min(self.right + self.width / 2, 999)
        new_lower = max(self.lower - self.height / 2, 0)
        new_upper = min(self.upper + self.height / 2, 999)

        return ImagePatch(self.original_image, new_left, new_lower, new_right, new_upper)

    def overlaps(self, patch) -> bool:
        # check if another image patch overlaps with this image patch
        # if patch overlaps with current patch , return True. Otherwise return False

        if patch.right < self.left or self.right < patch.left:
            return False
        if patch.lower < self.upper or self.lower < patch.upper:
            return False
        return True


def dist(patch_a, patch_b):
    xa = patch_a.horizontal_center
    ya = patch_a.vertical_center
    xb = patch_b.horizontal_center
    yb = patch_b.vertical_center

    d = ((xa - xb) ** 2 + (ya - yb) ** 2) ** (1 / 2)
    return d


def formatting_answer(answer) -> str:
    # Formatting the answer into a string that follows the task 's requirement
    # For example, it changes bool value to "yes" or "no", and clean up long answer into short ones.
    # This function should be used at the end of each program

    final_answer = ""
    if isinstance(answer, str):
        final_answer = answer.strip()

    elif isinstance(answer, bool):
        final_answer = "yes" if answer else "no"

    elif isinstance(answer, list):
        final_answer = " , ".join([str(x) for x in answer])

    elif isinstance(answer, ImagePatch):
        final_answer = answer.image_caption()

    elif isinstance(answer, int or float):
        final_answer = answer

    else:
        final_answer = str(answer)
    print(f"Program output: {final_answer}")
    return final_answer


def get_image_ratio(image):
    size = image.size
    if size[0] > size[1]:  # width > height
        x_ratio = 1
        y_ratio = int(size[0]) / int(size[1])
    else:  # width < height
        x_ratio = int(size[1]) / int(size[0])
        y_ratio = 1
    return x_ratio, y_ratio


def get_list_bbox_score(boxes, scores, labels, x_ratio, y_ratio):
    bboxes_list = []
    scores_list = []
    labels_list = []

    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        box[0], box[2] = round(box[0] * x_ratio, 3), round(box[2] * x_ratio, 3)
        box[1], box[3] = round(box[1] * y_ratio, 3), round(box[3] * y_ratio, 3)
        bboxes_list.append(box)
        scores_list.append(round(score.item(), 3))
        labels_list.append(label)

    return bboxes_list, scores_list, labels_list


# NMS function
def nms(boxes, scores, labels, nms_thresh):
    # numpy配列に変換
    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)

    """
    boxes: np.array([[x1, y1, x2, y2],...])
    """
    keep = []
    if len(boxes) == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    idx = np.argsort(scores, axis=0)

    while len(idx) > 0:
        last = len(idx) - 1
        i = idx[last]  # index of current largest val
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[idx[:last]])
        yy1 = np.maximum(y1[i], y1[idx[:last]])
        xx2 = np.minimum(x2[i], x2[idx[:last]])
        yy2 = np.minimum(y2[i], y2[idx[:last]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h
        iou = inter / (area[idx[:last]] + area[i] - inter)
        idx = np.delete(idx, np.concatenate(([last], np.where(iou > nms_thresh)[0])))

    boxes_list = boxes[keep].tolist()
    scores_list = scores[keep].tolist()
    labels_list = labels[keep].tolist()

    return boxes_list, scores_list, labels_list


def cal_iou(a, b):
    # a, bは矩形を表すリストで、a=[xmin, ymin, xmax, ymax]
    ax_mn, ay_mn, ax_mx, ay_mx = a[0], a[1], a[2], a[3]
    bx_mn, by_mn, bx_mx, by_mx = b[0], b[1], b[2], b[3]

    a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
    b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

    abx_mn = max(ax_mn, bx_mn)
    aby_mn = max(ay_mn, by_mn)
    abx_mx = min(ax_mx, bx_mx)
    aby_mx = min(ay_mx, by_mx)
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w * h

    iou = intersect / (a_area + b_area - intersect)
    return iou


def delete_overlaps(patch_list1, patch_list2):
    for p1 in patch_list1:
        for p2 in patch_list2:
            iou = cal_iou(p1.box, p2.box)
            print("iou", iou)

            if iou > 0.5:
                print("Overlap!")
                if p1.detection_score > p2.detection_score:
                    patch_list2.remove(p2)
                else:
                    patch_list1.remove(p1)

    print(patch_list1, patch_list2)
    return patch_list1, patch_list2


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def delete_large_box(patch_list):
    new_patch_list = []
    for p in patch_list:
        height = p.height
        if height < 800:
            new_patch_list.append(p)
    return new_patch_list


def detect(image, obj_name):  # list(scoreの高い順にbboxを返す)
    # load model
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # orange. peach. のような複数のオブジェクト名を入力とする
    if not obj_name.endswith("."):
        obj_name += "."
    print("obj_name", obj_name)

    inputs = processor(images=image, text=obj_name, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # thresholdの選択
    box_threshold = 0.2
    if obj_name == "pushpin.":
        box_threshold = 0.3

    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=box_threshold, text_threshold=0.3, target_sizes=[image.size[::-1]]
    )

    # 自動でリストから辞書に変換
    if isinstance(results, list) and len(results) == 1:  # リストかつ要素が1つの場合
        results = results[0]  # 辞書型に変換
        print(results)
    else:
        raise ValueError("Results should be a list with one element.")

    boxes_list = []
    scores_list = []
    labels_list = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 1) for i in box.tolist()]
        x0, y0, x1, y1 = (int(coord) for coord in box)
        boxes_list.append([x0, y0, x1, y1])
        scores_list.append(round(score.item(), 2))
        labels_list.append(label)

    if boxes_list == []:
        return []

    # nmsを実行
    if obj_name == "terminal.":
        boxes_list, scores_list, labels_list
    else:
        boxes_list, scores_list, labels_list = nms(boxes_list, scores_list, labels_list, 0.2)
    print(boxes_list, scores_list, labels_list)

    # obj_nameに含まれる要素を.で区切りリストに変換し，空白を削除
    # obj_name = "oatmeal. banana chips. almonds"
    obj_name_list = obj_name.replace(" ", "").split(".")
    obj_name_list = [name for name in obj_name_list if name != ""]
    # obj_name_list = ["oatmeal", "bananachips", "almonds"]

    # もしobj_name1つの場合，patch_listを返す
    if len(obj_name_list) == 1:
        patch_list = []
        for box, score, label in zip(boxes_list, scores_list, labels_list):
            cropped_image = image.crop(box)
            left = int(box[0])
            lower = int(box[3])
            right = int(box[2])
            upper = int(box[1])
            patch_list.append(ImagePatch(cropped_image, left, lower, right, upper, score))
        return patch_list

    else:
        # keyが物体名，valueがpatchを含むリストの辞書を作成
        patch_dict = {label: [] for label in obj_name_list}
        # patch_dict = {"oatmeal": [], "bananachips": [], "almonds": []}

        # labels_list =  ['oatmeal', 'banana chips almonds']
        for box, score, label in zip(boxes_list, scores_list, labels_list):
            # labelが""の場合，スキップ
            if label == "":
                continue
            # label = 'banana chips almonds'
            label = label.replace(" ", "")  #  labelの空白を削除
            # label = 'bananachipsalmonds'

            # patch_dictのkeyに含まれる物体名がlabelに含まれる場合，patch_dict[label]に追加
            for obj_name in obj_name_list:
                print("obj_name", obj_name)
                print("label", label)
                if obj_name in label:
                    cropped_image = image.crop(box)
                    left = int(box[0])
                    lower = int(box[3])
                    right = int(box[2])
                    upper = int(box[1])
                    patch_dict[obj_name].append(ImagePatch(cropped_image, left, lower, right, upper, score))
            print(patch_dict)

        return patch_dict


if __name__ == "__main__":
    main()

    # 1. Select start number of image and end number of image, category
    # 2. Select to generate code or to use generated code
    # 3. Select to excute function number
    # 4. Select place to save result
