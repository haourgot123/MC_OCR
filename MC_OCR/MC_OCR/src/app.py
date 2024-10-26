from module.img_segmentation.img_seg import DetectronSegmentation
from module.text_detect.text_det import TextDetection
from module.text_recog.text_recognition import VietOCRPrediction
from module.img_cls.image_classification import EfficientNetClassification
from module.text_cls.PhoBert_prediction import PhoBertPrediction
from module.text_cls.svm_cls import SVMClassifier
import streamlit as st
from PIL import Image, ImageDraw
import cv2
import numpy as np
import pandas as pd
from utils import remove_background, detection_and_rotate, text_recog_vietocr, text_classification, visualize_image1

st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50; /* Màu xanh */
        color: white; /* Màu chữ trắng */
    }
    </style>
    """, unsafe_allow_html=True)



# Tiêu đề của ứng dụng
st.title("🎯Trích xuất thông tin từ hoá đơn")
st.sidebar.header("📤Tùy chọn tải hình ảnh lên")

uploaded_file = st.sidebar.file_uploader("Chọn hình ảnh để tải lên:", type=["jpg", "jpeg", "png"])

# Hiển thị nội dung dựa trên việc tải lên hình ảnh
if uploaded_file is not None:
    # Đọc hình ảnh và hiển thị
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="Hình ảnh đã tải lên.", use_column_width=True)

# Tao buton send
send_flag = False
if st.sidebar.button('Send'):
    send_flag = True

# Tạo các tab
tabs = st.tabs(["🔥Phase 1", "💧Phase 2", "⚡Phase 3"])

# Nội dung của Tab 1
with tabs[0]:
    done_task1 = False
    detectron_cfg_path = './module/img_segmentation/config.yml'
    segment = DetectronSegmentation(config_path=detectron_cfg_path)
    st.header("Xoá background ảnh 🔨")
    with st.spinner("⏳ Đang xử lý..."):
        if send_flag:
            img = np.array(image)
            img = remove_background(segment, img)
            st.image(img, caption="Đã xoá background ảnh.", use_column_width=True,  width=50)
            done_task1 = True
            st.success("Xử lý hoàn tất!")
# Nội dung của Tab 2
with tabs[1]:
    st.header("Xoay ảnh và phát hiện các box 🎨")
    text_det = TextDetection(device='cuda')
    img_clf_weight = 'weights/weightinvoiced_weight.pth'
    efficientnet_model = EfficientNetClassification()
    efficientnet_model.load_weight(img_clf_weight)
    with st.spinner("⏳ Đang xử lý..."):
        done_task2 = False
        if done_task1:
            img_draw, boxes = detection_and_rotate(text_det, efficientnet_model, img)
            img = img_draw
            draw = ImageDraw.Draw(img_draw)
            for box in boxes:
                x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]
                draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=1)
            st.image(img_draw, caption="Xoay ảnh và phát hiện các box.", use_column_width=True,  width=50)
            done_task2 = True
            st.success("Xử lý hoàn tất!")

# Nội dung của Tab 3
with tabs[2]:
    st.header("Phân loại và hiển thị 🚩")
    
    with st.spinner("⏳ Đang xử lý..."):
        ocr_model  = VietOCRPrediction(
            config_name= 'vgg_transformer',
            weight_path= './weights/transformerocr.pth', 
        )
        # Load model PhoBert
        bert_model = PhoBertPrediction(3)
        bert_model.load_weight( weight_path= 'weights/model_best_valoss.pt')

        # Load model SVM Classification
        svm_clf = SVMClassifier(
            tfidf_path='weights/model_vectorizer_tfidf.pkl',
            svm_path= 'weights/svm_classifier_model.pkl'
        )
        if done_task2:
            text = text_recog_vietocr(ocr_model, boxes, img)
            seller, address, timestamp, totalcost = text_classification(
                phobert_model=bert_model,
                svm_model=svm_clf,
                text = text,
                boxes=boxes
            )
            text_seller, text_address, text_timestamp, text_totalcost = '', '', '', ''
            for seller_box in seller:
                text_seller += seller_box['text'] +  ' '
            for address_box in address:
                text_address += address_box['text'] + ' '
            for timestamp_box in timestamp:
                text_timestamp += timestamp_box['text'] + ' '
            for totalcost_box in totalcost:
                text_totalcost += totalcost_box['text'] + ' '
            data = {
                'SELLER': [text_seller], 
                'ADDRESS': [text_address],
                'TIMESTAMP': [text_timestamp],
                'TOTALCOST': [text_totalcost]
            }
            df = pd.DataFrame(data)
            st.table(df)
            img = visualize_image1(seller, address, timestamp, totalcost, img)
            st.image(img_draw, caption="Phân loại và hiển thị.", use_column_width=True,  width=50)
            st.success("Xử lý hoàn tất!")
