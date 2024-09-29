import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures

# ส่วนของอินพุตจากผู้ใช้
st.markdown("<h1 style='text-align: center; font-size: 30px;'>การทำนายราคาบ้าน</h1>", unsafe_allow_html=True)

bedrooms = st.number_input("จำนวนห้องนอนในบ้าน:", min_value=0, max_value=5)
bathrooms = st.number_input("จำนวนห้องน้ำในบ้าน:", min_value=0, max_value=5)
sqft_living = st.number_input("พื้นที่ใช้สอยภายในบ้าน(เป็นตารางฟุต):", min_value=0, max_value=400000)
sqft_lot = st.number_input("พื้นที่ทั้งหมดของบ้านและที่ดินรวมกัน(เป็นตารางฟุต):", min_value=0, max_value=300000)
floors = st.number_input("จำนวนชั้นของบ้าน:", min_value=0, max_value=5)
waterfront = st.number_input("บ้านมีติดกับริมน้ำหรือไม่ มี =1 ไม่ติด = 0:", min_value=0, max_value=1)
view = st.number_input("คะแนนการมองเห็นวิวจากบ้าน 1-5:", min_value=0, max_value=5)
condition = st.number_input("สภาพของบ้าน 1-5:", min_value=1, max_value=5)
grade = st.number_input("ระดับคุณภาพโดยรวมของการก่อสร้างและการออกแบบ 1-10:", min_value=1, max_value=10)
sqft_above = st.number_input("พื้นที่ใช้สอยเหนือพื้นดิน(เป็นตารางฟุต):", min_value=0, max_value=40000)
sqft_basement = st.number_input("พื้นที่ใช้สอยในชั้นใต้ดิน(เป็นตารางฟุต):", min_value=0, max_value=20000)
yr_built = st.number_input("ปีที่สร้างบ้าน:", min_value=0, max_value=2020)
yr_renovated = st.number_input("ปีที่ทำการปรับปรุงบ้านครั้งล่าสุด (ไม่มีใส่ 0):", min_value=0, max_value=2020)
zipcode = st.number_input("รหัสไปรษณีย์ของบ้าน:", min_value=0, max_value=100000)
lat = st.number_input("พิกัดละติจูดของบ้าน:", min_value=0.0000, max_value=50.0000,step=0.0001)
long = st.number_input("พิกัดลองจิจูดของบ้าน:", min_value=-125.0000, max_value=0.0000, value=0.0 ,step=0.0001)
sqft_living15 = st.number_input("พื้นที่ใช้สอยภายในบ้านเฉลี่ยใน 15 หลังใกล้เคียง(เป็นตารางฟุต):", min_value=0, max_value=40000)
sqft_lot15 = st.number_input("พื้นที่รวมของบ้านและที่ดินเฉลี่ยใน 15 หลังใกล้เคียง(เป็นตารางฟุต):", min_value=0)

# กดปุ่มเพื่อทำการทำนาย
if st.button('ทำนายราคาบ้าน'):
    # สร้าง DataFrame จากข้อมูลที่ได้รับจากผู้ใช้
    data = {
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft_living': [sqft_living],
        'sqft_lot': [sqft_lot],
        'floors': [floors],
        'waterfront': [waterfront],
        'view': [view],
        'condition': [condition],
        'grade': [grade],
        'sqft_above': [sqft_above],
        'sqft_basement': [sqft_basement],
        'yr_built': [yr_built],
        'yr_renovated': [yr_renovated],
        'zipcode': [zipcode],
        'lat': [lat],
        'long': [long],
        'sqft_living15': [sqft_living15],
        'sqft_lot15': [sqft_lot15]
    }

    df = pd.DataFrame(data)

    # โหลด PolynomialFeatures ที่ถูกบันทึกไว้แล้ว
    poly = joblib.load('Model/polynomial_features.pkl')

    # แปลงข้อมูลด้วย PolynomialFeatures
    X_test = poly.transform(df)

    # โหลดโมเดลที่ถูกฝึกไว้แล้ว
    best_model = joblib.load('Model/linear_regression_model.pkl')

    # ทำการพยากรณ์ราคาบ้าน
    y_pred = best_model.predict(X_test)

    # แสดงผลการทำนายเป็น popup-like modal
    st.success(f"ผลการทำนายราคาบ้าน: {y_pred[0]:,.2f} บาท")

    #st.success(f"Test: {X_test[0]} ")
