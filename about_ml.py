import streamlit as st

def show():
    st.title("📖 การพัฒนาโมเดล Machine Learning (Pokedex)")

    # 📌 ระบุที่มาของ Dataset
    st.header("📌 แหล่งที่มาของ Dataset")
    st.write("""
    ข้อมูลที่ใช้มาจากเว็บไซต์ **www.kaggle.com**  
    - Dataset: [Pokedex Dataset]
    - Dataset นี้รวบรวมข้อมูลโปเกมอน เช่น ชื่อ (`Name`), ค่าพลัง (`HP`, `Attack`, `Defense`), และประเภท (`Type`)
    """)

    # 📌 อธิบาย Feature ของ Dataset
    st.header("🔹 Feature ของ Dataset")
    st.write("""
    - **# (ID)**: หมายเลขโปเกมอนใน Pokedex  
    - **Name**: ชื่อของโปเกมอน เช่น "Pikachu", "Charizard"  
    - **Type 1, Type 2**: ประเภทของโปเกมอน (Type 2 เป็นประเภทเสริม อาจมีหรือไม่มี)  
    - **HP**: ค่าพลังชีวิตของโปเกมอน  
    - **Attack**: ค่าพลังโจมตี  
    - **Defense**: ค่าพลังป้องกัน  
    - **Speed**: ค่าความเร็วของโปเกมอน  
    """)

    # การเตรียมข้อมูล
    st.header("🔹 การเตรียมข้อมูล")
    st.write("""
    ข้อมูลที่ใช้คือ **pokedex.csv** ซึ่งมีข้อมูลโปเกมอนรวมถึงค่าพลัง เช่น HP, Attack, Defense เป็นต้น  
    **ขั้นตอนการเตรียมข้อมูล:**
    - ลบค่าที่ผิดปกติ เช่น น้ำหนักที่สูงเกินไป (`weight > 5000`)
    - แยกประเภท (`type_1`, `type_2`) ออกจากคอลัมน์เดียว
    - ทำ Standardization ให้ค่าต่างๆ อยู่ในช่วงที่เหมาะสม
    """)

    # ทฤษฎีของอัลกอริทึมที่ใช้
    st.header("🔹 ทฤษฎีของอัลกอริทึมที่ใช้ (Random Forest)")
    st.write("""
    อัลกอริทึมที่ใช้ในการพัฒนาโมเดลคือ **Random Forest Classifier** ซึ่งเป็น Ensemble Learning  
    - ใช้หลายต้นไม้ตัดสินใจ (Decision Trees) มารวมกันเพื่อเพิ่มความแม่นยำ  
    - ลด Overfitting เพราะใช้การสุ่มตัวอย่างข้อมูลและ Feature  
    - ปรับ `n_estimators=200` และ `max_depth=10` เพื่อให้โมเดลมีความแม่นยำขึ้น  
    """)

    # ขั้นตอนการพัฒนาโมเดล
    st.header("🔹 ขั้นตอนการพัฒนาโมเดล")
    st.write("""
    1. โหลดข้อมูล `pokedex.csv` และทำความสะอาด  
    2. เลือก Feature ที่เหมาะสม เช่น `attack`, `defense`, `speed`, `hp`  
    3. ใช้ **Random Forest Classifier** ในการ Train โมเดล  
    4. บันทึกโมเดลเป็นไฟล์ `pokedex_model.pkl` เพื่อนำไปใช้ใน Web App  
    """)

    # แสดงโค้ดที่ใช้ Train โมเดล
    st.subheader("🔹 โค้ดสำหรับการ Train โมเดล")
    st.code("""
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
    model.fit(X_train_selected, y_train)

    # บันทึกโมเดล
    joblib.dump(model, "pokedex_model.pkl")
    """, language="python")

    st.success("✨ โมเดลพร้อมใช้งาน และสามารถนำไปใช้ใน Web Application ได้แล้ว!")
