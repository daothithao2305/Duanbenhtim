import streamlit as st
import numpy as np
import joblib

# 1. TẢI MÔ HÌNH VÀ BỘ CHUẨN HÓA
try:
    mo_hinh = joblib.load('best_svm_model.pkl')
    bo_loc = joblib.load('scaler.pkl')
except:
    st.warning("⚠️ Chưa tìm thấy file. Hãy đảm bảo 2 file 'svm_model.pkl' và 'scaler.pkl' nằm cùng thư mục với file app.py!")

# 2. CẤU HÌNH GIAO DIỆN
st.set_page_config(page_title="Dự Đoán Bệnh Tim", page_icon="❤️", layout="centered")

# 3. CSS TÙY CHỈNH CỰC MẠNH (BACKGROUND, NÚT BẤM, HIỆU ỨNG KÍNH)
st.markdown("""
    <style>
    /* Chỉnh màu nền toàn bộ trang web (Gradient hồng nhạt sang xanh nhạt) */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #fdf4f7 0%, #e1e7fa 100%);
    }
    
    /* Làm trong suốt thanh header mặc định của Streamlit */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }

    /* Hiệu ứng kính mờ (Glassmorphism) cho khu vực nội dung chính */
    .block-container {
        background: rgba(255, 255, 255, 0.75);
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        backdrop-filter: blur(8px);
        margin-top: 30px;
        margin-bottom: 30px;
    }

    /* Tiêu đề chính */
    .main-title { text-align: center; color: #ff4b2b; font-size: 3rem; font-weight: 900; margin-bottom: 0px; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);}
    .sub-title { text-align: center; color: #6c757d; font-size: 1.1rem; margin-bottom: 30px; font-weight: 500;}
    
    /* Box khuyến nghị và lưu ý */
    .khuyen-nghi-box { background-color: white; padding: 25px; border-radius: 15px; border-left: 6px solid #4CAF50; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-top: 20px;}
    .khuyen-nghi-box.loi { border-left: 6px solid #ff416c; }
    .luu-y-box { background-color: transparent; padding: 15px; border-top: 1px dashed #ccc; margin-top: 30px; font-size: 0.85rem; color: #888; text-align: center;}

    /* Tùy chỉnh Nút bấm (Button) xịn xò */
    div.stButton > button:first-child {
        background: linear-gradient(to right, #ff416c, #ff4b2b) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 15px 0 rgba(255, 65, 108, 0.5) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    /* Hiệu ứng nảy lên khi di chuột vào nút */
    div.stButton > button:first-child:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px 0 rgba(255, 65, 108, 0.6) !important;
    }
    </style>
""", unsafe_allow_html=True)

# 4. HEADER
st.markdown('<p class="main-title">❤️ DỰ ĐOÁN BỆNH TIM</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Công cụ AI đánh giá sức khỏe tim mạch nhanh chóng & chính xác</p>', unsafe_allow_html=True)

# 5. FORM NHẬP LIỆU 
st.markdown("### 📋 Thông tin sức khỏe")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Tuổi", min_value=1, max_value=120, value=50)
    cp = st.selectbox("Loại đau ngực", [0, 1, 2, 3], format_func=lambda x: "Không điển hình" if x==1 else f"Loại {x}")
    chol = st.number_input("Cholesterol (mg/dL)", value=200)
    restecg = st.selectbox("Điện tâm đồ nghỉ", [0, 1, 2], format_func=lambda x: "Bình thường" if x==0 else "Có bất thường")
    exang = st.selectbox("Đau thắt ngực khi vận động", [0, 1], format_func=lambda x: "Không" if x==0 else "Có")
    slope = st.selectbox("Độ dốc ST", [0, 1, 2])
    
with col2:
    sex = st.selectbox("Giới tính", [1, 0], format_func=lambda x: "Nam" if x==1 else "Nữ")
    trestbps = st.number_input("Huyết áp nghỉ (mm Hg)", value=120)
    fbs = st.selectbox("Đường huyết lúc đói", [0, 1], format_func=lambda x: "< 120 mg/dL" if x==0 else "> 120 mg/dL")
    thalach = st.number_input("Nhịp tim tối đa (bpm)", value=150)
    oldpeak = st.number_input("Chỉ số Oldpeak (ST depression)", value=0.0, step=0.1)
    ca = st.selectbox("Số mạch máu chính", [0, 1, 2, 3, 4])
    
thal = st.selectbox("Khiếm khuyết (Thal)", [0, 1, 2, 3])

st.write("") # Dòng trống cho thoáng

# 6. NÚT XỬ LÝ VÀ HIỂN THỊ KẾT QUẢ
if st.button("✨ XEM KẾT QUẢ DỰ ĐOÁN"):
    
    # Chuẩn bị dữ liệu
    vector_benh_nhan = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    vector_chuan_hoa = bo_loc.transform(vector_benh_nhan)
    ket_qua = mo_hinh.predict(vector_chuan_hoa)[0]
    
    st.markdown("---")
    
    # HIỂN THỊ KẾT QUẢ
    if ket_qua == 1:
        # Giao diện KHI CÓ BỆNH 
        st.error("### ⚠️ CẢNH BÁO: NGUY CƠ CAO!")
        st.write("Dựa trên thuật toán, chúng tôi phát hiện những dấu hiệu bất thường về tim mạch của bạn.")
        
        st.markdown("""
        <div class="khuyen-nghi-box loi">
            <h4 style="margin-top:0px; color:#ff416c;">💡 Lời khuyên Y tế:</h4>
            <ul>
                <li style="margin-bottom: 10px;"><b>Khám chuyên khoa ngay:</b> Cần đặt lịch siêu âm tim và điện tâm đồ tại bệnh viện.</li>
                <li style="margin-bottom: 10px;"><b>Chế độ vận động:</b> Tuyệt đối không tập thể thao mạnh, không làm việc gắng sức.</li>
                <li style="margin-bottom: 10px;"><b>Dinh dưỡng:</b> Bỏ hẳn rượu bia, thuốc lá; ăn nhạt và giảm mỡ động vật.</li>
                <li><b>Tâm lý:</b> Giữ tinh thần bình tĩnh, nhờ người thân hỗ trợ đi khám.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Giao diện KHI KHỎE MẠNH
        st.success("### 🎉 TUYỆT VỜI: KHÔNG CÓ NGUY CƠ!")
        st.write("Các chỉ số của bạn đều nằm trong vùng an toàn. Trái tim của bạn đang rất khỏe mạnh!")
        
        st.markdown("""
        <div class="khuyen-nghi-box">
            <h4 style="margin-top:0px; color:#4CAF50;">🌿 Lời khuyên Sức khỏe:</h4>
            <ul>
                <li style="margin-bottom: 10px;"><b>Duy trì phong độ:</b> Tiếp tục ăn nhiều rau xanh, quả mọng và ngũ cốc.</li>
                <li style="margin-bottom: 10px;"><b>Tập luyện:</b> Duy trì tập thể dục 30 - 45 phút mỗi ngày (đi bộ, đạp xe, bơi lội).</li>
                <li style="margin-bottom: 10px;"><b>Giấc ngủ:</b> Đảm bảo ngủ đủ 7-8 tiếng/ngày để tim được nghỉ ngơi.</li>
                <li><b>Khám định kỳ:</b> Đừng quên kiểm tra sức khỏe tổng quát 6 tháng/lần nhé.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Dòng lưu ý chung ở cuối
    st.markdown("""
    <div class="luu-y-box">
        Disclaimer: Hệ thống AI này phục vụ mục đích học tập và tham khảo. Kết quả dự đoán không thể thay thế phác đồ chẩn đoán của bác sĩ chuyên khoa.
    </div>
    """, unsafe_allow_html=True)