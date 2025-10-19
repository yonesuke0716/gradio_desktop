import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from rfdetr import RFDETRBase

COCO_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

# クラスごとの色を定義
CLASS_COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "pink",
    "cyan",
    "magenta",
    "lime",
    "navy",
    "maroon",
    "olive",
    "teal",
    "silver",
    "gray",
    "darkred",
    "darkblue",
    "darkgreen",
    "gold",
    "indigo",
    "coral",
    "hotpink",
    "lightblue",
    "lightgreen",
    "lightcoral",
    "lightgray",
    "mediumblue",
    "mediumgreen",
    "mediumpurple",
    "mediumseagreen",
    "mediumslateblue",
    "mediumturquoise",
    "mediumvioletred",
    "midnightblue",
    "mistyrose",
    "moccasin",
    "navajowhite",
    "oldlace",
    "olivedrab",
    "orangered",
    "orchid",
    "palegoldenrod",
    "palegreen",
    "paleturquoise",
    "palevioletred",
    "papayawhip",
    "peachpuff",
    "peru",
    "plum",
    "powderblue",
    "rosybrown",
    "royalblue",
    "saddlebrown",
    "salmon",
    "sandybrown",
    "seagreen",
    "seashell",
    "sienna",
    "skyblue",
    "slateblue",
    "slategray",
    "snow",
    "springgreen",
    "steelblue",
    "tan",
    "thistle",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
    "whitesmoke",
    "yellowgreen",
]

model = RFDETRBase()


def detect_objects(image, confidence_threshold):
    """
    画像から物体を検出し、結果を可視化した画像を返す
    """
    if image is None:
        return None, "画像がアップロードされていません。"

    try:
        # 画像をPIL形式に変換
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # 物体検出を実行
        predictions = model.predict(pil_image, threshold=confidence_threshold)

        # 検出結果を可視化した画像を作成
        annotated_image = pil_image.copy()
        draw = ImageDraw.Draw(annotated_image)

        # フォントの設定（システムフォントを使用）
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except Exception as e:
            print(f"フォントの読み込みに失敗しました: {e}")
            font = ImageFont.load_default()

        # 検出結果のサマリーを作成
        newline = "\n"
        summary = f"検出された物体数: {len(predictions)}{newline}"
        if len(predictions) > 0:
            summary += f"検出された物体:{newline}"

            # Detectionsオブジェクトの場合の処理
            if (
                hasattr(predictions, "xyxy")
                and hasattr(predictions, "confidence")
                and hasattr(predictions, "class_id")
            ):
                # バウンディングボックス、信頼度、クラスIDを取得
                bboxes = predictions.xyxy
                confidences = predictions.confidence
                class_ids = predictions.class_id

                for i in range(len(bboxes)):
                    bbox = bboxes[i]
                    confidence = float(confidences[i])
                    class_id = int(class_ids[i])

                    # バウンディングボックスの座標を取得
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox[:4]
                        # 座標を整数に変換
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # クラス名を決定（COCO2017クラス名を使用）
                        class_name = COCO_CLASSES.get(class_id, f"unknown_{class_id}")

                        # 色を決定（クラスIDに基づいて色を選択）
                        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]

                        # バウンディングボックスを描画
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                        # ラベルを描画
                        label = f"{class_name}: {confidence:.2f}"

                        # ラベルの背景を描画
                        bbox_text = draw.textbbox((x1, y1 - 25), label, font=font)
                        draw.rectangle(bbox_text, fill=color)
                        draw.text((x1, y1 - 25), label, fill="white", font=font)

                        summary += (
                            f"- {class_name}: {confidence:.2f} "
                            f"(座標: [{x1}, {y1}, {x2}, {y2}], 色: {color}){newline}"
                        )
                    else:
                        summary += (
                            f"- Object {i + 1}: {confidence:.2f} "
                            f"(座標データ異常){newline}"
                        )
            else:
                # 従来のタプル形式の場合の処理
                for i, prediction in enumerate(predictions):
                    # タプルの場合の処理
                    if isinstance(prediction, tuple):
                        if len(prediction) >= 2:
                            # バウンディングボックスの座標を取得
                            bbox = prediction[0]
                            confidence = (
                                float(prediction[1])
                                if prediction[1] is not None
                                else 0.0
                            )

                            # バウンディングボックスを描画
                            if (
                                isinstance(bbox, (list, tuple, np.ndarray))
                                and len(bbox) >= 4
                            ):
                                x1, y1, x2, y2 = bbox[:4]
                                # 座標を整数に変換
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                                # クラス名と色を決定
                                # RF-DETRの予測結果からクラス情報を取得を試行
                                class_id = i  # 仮のクラスID（実際のクラス情報が取得できない場合）
                                if len(prediction) >= 3:  # クラス情報がある場合
                                    class_id = (
                                        int(prediction[2])
                                        if prediction[2] is not None
                                        else i
                                    )

                                # クラス名を決定（COCO2017クラス名を使用）
                                class_name = COCO_CLASSES.get(
                                    class_id, f"unknown_{class_id}"
                                )

                                # 色を決定（クラスIDに基づいて色を選択）
                                color = CLASS_COLORS[class_id % len(CLASS_COLORS)]

                                # バウンディングボックスを描画
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                                # ラベルを描画
                                label = f"{class_name}: {confidence:.2f}"

                                # ラベルの背景を描画
                                bbox_text = draw.textbbox(
                                    (x1, y1 - 25), label, font=font
                                )
                                draw.rectangle(bbox_text, fill=color)
                                draw.text((x1, y1 - 25), label, fill="white", font=font)

                                summary += (
                                    f"- {class_name}: {confidence:.2f} "
                                    f"(座標: [{x1}, {y1}, {x2}, {y2}], "
                                    f"色: {color}){newline}"
                                )
                            else:
                                summary += (
                                    f"- Object {i + 1}: {confidence:.2f} "
                                    f"(座標データ異常){newline}"
                                )
                        else:
                            summary += f"- Object {i + 1}: データ形式異常{newline}"
                    # 辞書の場合の処理
                    elif isinstance(prediction, dict):
                        class_name = prediction.get("class", "Unknown")
                        confidence = prediction.get("confidence", 0.0)
                        class_id = hash(class_name) % len(CLASS_COLORS)
                        color = CLASS_COLORS[class_id]
                        summary += (
                            f"- {class_name}: {confidence:.2f} (色: {color}){newline}"
                        )
                    else:
                        summary += f"- Object {i + 1}: {str(prediction)}{newline}"
        else:
            summary += "物体が検出されませんでした。"

        return annotated_image, summary

    except Exception as e:
        return None, f"エラーが発生しました: {str(e)}"


# Gradioインターフェースを作成
demo = gr.Interface(
    fn=detect_objects,
    inputs=[
        gr.Image(label="画像をアップロードしてください", type="pil"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.5,
            step=0.1,
            label="信頼度閾値",
            info="検出の信頼度の最小値を設定します",
        ),
    ],
    outputs=[gr.Image(label="検出結果"), gr.Textbox(label="検出サマリー", lines=10)],
    title="物体検出アプリ",
    description="画像をアップロードして物体検出を実行します。RF-DETRモデルを使用して物体を検出し、バウンディングボックスで表示します。",
    examples=[["https://media.roboflow.com/dog.jpeg", 0.5]],
)


def main():
    demo.launch(server_name="0.0.0.0", inbrowser=True)


if __name__ == "__main__":
    main()
