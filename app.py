import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
import numpy as np
import cv2
from streamlit_image_coordinates import streamlit_image_coordinates
import io
import zipfile
from datetime import datetime

# -------------------------------------------------
# 1. 初期設定とモデルロード (OpenVINO対応)
# -------------------------------------------------
st.set_page_config(page_title="Pipe Counter Pro", layout="wide")

@st.cache_resource
def load_model():
    # ご提示いただいたOpenVINOフォルダパスを指定
    model_path = "last_openvino_model" 
    model = YOLO(model_path)
    return model

model = load_model()
TARGET_SIZE = 960

# -------------------------------------------------
# 2. Session State の初期化
# -------------------------------------------------
if "phase" not in st.session_state: st.session_state.phase = "upload"
if "mask_rects" not in st.session_state: st.session_state.mask_rects = []
if "temp_point" not in st.session_state: st.session_state.temp_point = None
if "last_processed_click" not in st.session_state: st.session_state.last_processed_click = None
if "points" not in st.session_state: st.session_state.points = [] 
if "auto_boxes" not in st.session_state: st.session_state.auto_boxes = None

# -------------------------------------------------
# 3. 便利関数
# -------------------------------------------------
def apply_masks(img, rects):
    """画像に黒塗りのマスクを適用する"""
    draw = ImageDraw.Draw(img)
    for r in rects:
        draw.rectangle([r[0], r[1], r[2], r[3]], fill="black")
    return img

def resize_with_aspect_ratio(img, target_max_side):
    """アスペクト比を維持してリサイズする"""
    w, h = img.size
    if w > h:
        new_w = target_max_side
        new_h = int(h * (target_max_side / w))
    else:
        new_h = target_max_side
        new_w = int(w * (target_max_side / h))
    return img.resize((new_w, new_h), Image.LANCZOS)

# -------------------------------------------------
# 4. メイン UI フロー
# -------------------------------------------------
st.title("PINEAPPLE 🍍")

uploaded_file = st.file_uploader("パイプの画像を選択してください", type=["jpg", "jpeg", "png"])

if uploaded_file:
    original_image = Image.open(uploaded_file).convert("RGB")
    base_image = resize_with_aspect_ratio(original_image, TARGET_SIZE)
    display_width, display_height = base_image.size

    # --- PHASE: UPLOAD ---
    if st.session_state.phase == "upload":
        st.image(base_image, caption="アップロード画像", width=display_width)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✂️ 検出エリアを編集（除外設定）", use_container_width=True):
                st.session_state.phase = "edit"
                st.rerun()
        with col2:
            if st.button("🚀 パイプを検出開始", type="primary", use_container_width=True):
                st.session_state.phase = "detect"
                st.rerun()

    # --- PHASE: EDIT (除外エリア設定) ---
    elif st.session_state.phase == "edit":
        st.subheader("除外エリアの設定 (2点クリックで四角形を作成)")
        edit_img = base_image.copy()
        edit_img = apply_masks(edit_img, st.session_state.mask_rects)
        
        if st.session_state.temp_point:
            draw = ImageDraw.Draw(edit_img)
            tx, ty = st.session_state.temp_point
            draw.ellipse([tx-8, ty-8, tx+8, ty+8], fill="red")

        coords = streamlit_image_coordinates(edit_img, key="mask_editor", width=display_width)

        if coords:
            curr_c = (coords["x"], coords["y"])
            if curr_c != st.session_state.last_processed_click:
                st.session_state.last_processed_click = curr_c
                if st.session_state.temp_point is None:
                    st.session_state.temp_point = curr_c
                else:
                    p1 = st.session_state.temp_point
                    p2 = curr_c
                    rect = (min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1]))
                    st.session_state.mask_rects.append(rect)
                    st.session_state.temp_point = None
                st.rerun()

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ 確定して検出へ", type="primary"):
                st.session_state.phase = "detect"
                st.rerun()
        with col2:
            if st.button("↩️ 1つ戻す"):
                if st.session_state.mask_rects: st.session_state.mask_rects.pop()
                st.rerun()
        with col3:
            if st.button("❌ キャンセル"):
                st.session_state.phase = "upload"
                st.rerun()

    # --- PHASE: DETECT (検出・直接消去・保存) ---
    elif st.session_state.phase == "detect":
        st.subheader("検出結果の確認と補正")
        
        # 初回推論（OpenVINOモデル使用）
        if st.session_state.auto_boxes is None:
            inference_img = base_image.copy()
            inference_img = apply_masks(inference_img, st.session_state.mask_rects)
            with st.spinner("AIがパイプを検出中..."):
                # OpenVINOで推論を実行
                results = model.predict(inference_img, conf=0.5, max_det=2000, verbose=False)
                boxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    # 後の削除処理のためにリスト形式で保存
                    boxes.append([x1, y1, x2, y2])
                st.session_state.auto_boxes = boxes

        # 描画用画像の作成
        display_np = np.array(base_image.copy())
        # マスクエリアを視覚的に暗くする（任意）
        for r in st.session_state.mask_rects:
            cv2.rectangle(display_np, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (40, 40, 40), -1)

        # AI検出点の描画 (水色)
        for (x1, y1, x2, y2) in st.session_state.auto_boxes:
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            cv2.circle(display_np, (cx, cy), 13, (255, 255, 0), -1)
            cv2.circle(display_np, (cx, cy), 14, (0, 0, 0), 1)

        # 手動追加点の描画 (赤色)
        for (px, py) in st.session_state.points:
            cv2.circle(display_np, (px, py), 13, (255, 0, 0), -1)
            cv2.circle(display_np, (px, py), 14, (255, 255, 255), 1)

        # --- カウントと操作設定 ---
        total_pipes = len(st.session_state.auto_boxes) + len(st.session_state.points)
        
        st.success(f"合計： **{total_pipes}** 本 (AI自動: {len(st.session_state.auto_boxes)} / 手動追加: {len(st.session_state.points)})")
        
        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        with col_ctrl1:
            mode = st.radio("クリック操作を選択:", ["📍 ポイント追加", "🧽 消しゴム (クリックで削除)"])
            if st.button("🔄 全リセット (手動分のみ)"):
                st.session_state.points = []
                st.rerun()

        # クリックによる動的な修正
        fix_coords = streamlit_image_coordinates(Image.fromarray(display_np), key="manual_fix", width=display_width)

        if fix_coords:
            curr_p = (fix_coords["x"], fix_coords["y"])
            if curr_p != st.session_state.last_processed_click:
                st.session_state.last_processed_click = curr_p
                
                if "追加" in mode:
                    st.session_state.points.append(curr_p)
                else:
                    # 消しゴム機能：25ピクセル以内の「点」を削除対象にする
                    threshold = 25
                    hit_detected = False
                    
                    # 1. AI検出ボックスのリストから削除
                    new_auto = []
                    for box in st.session_state.auto_boxes:
                        bx, by = (box[0]+box[2])/2, (box[1]+box[3])/2
                        dist = ((curr_p[0]-bx)**2 + (curr_p[1]-by)**2)**0.5
                        if dist < threshold and not hit_detected:
                            hit_detected = True # 1クリック1つ消去
                            continue
                        new_auto.append(box)
                    st.session_state.auto_boxes = new_auto
                    
                    # 2. AIが消えなかった場合、手動追加点から削除
                    if not hit_detected:
                        st.session_state.points = [p for p in st.session_state.points 
                                                 if ((curr_p[0]-p[0])**2 + (curr_p[1]-p[1])**2)**0.5 > threshold]
                st.rerun()

        # --- 保存と出力 ---
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1: group_name = st.selectbox("グループ名", ["A", "B", "C", "D"])
        with c2: user_comment = st.text_area("備考")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as z:
            # 画像保存
            res_img = io.BytesIO()
            Image.fromarray(display_np).save(res_img, format="PNG")
            z.writestr(f"result_{timestamp}.png", res_img.getvalue())
            # CSV保存
            csv_data = f"日時,グループ,AI検出数,手動追加数,最終合計,備考\n"
            csv_data += f"{timestamp},{group_name},{len(st.session_state.auto_boxes)},{len(st.session_state.points)},{total_pipes},\"{user_comment}\""
            z.writestr(f"result_{timestamp}.csv", csv_data.encode("utf-8-sig"))

        st.download_button("⬇️ 判定結果を保存 (ZIP)", data=zip_buffer.getvalue(), 
                           file_name=f"pipe_report_{timestamp}.zip", mime="application/zip", type="primary")

        if st.button("⏮️ 最初に戻る"):
            st.session_state.phase = "upload"
            st.session_state.auto_boxes = None
            st.session_state.points = []
            st.rerun()