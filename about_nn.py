import streamlit as st

def show():
    st.title("📖 การพัฒนาโมเดล Neural Network (Ramen)")

    # 📌 ระบุที่มาของ Dataset
    st.header("📌 แหล่งที่มาของ Dataset")
    st.write("""
    ข้อมูลที่ใช้มาจากเว็บไซต์ **www.kaggle.com**  
    - Dataset: [Ramen Ratings]  
    - Dataset นี้รวบรวมรีวิวราเมนจากทั่วโลก พร้อมคะแนนรีวิว (`Stars`)
    """)

    # 📌 อธิบาย Feature ของ Dataset
    st.header("🔹 Feature ของ Dataset")
    st.write("""
    - **Review #**: หมายเลขรีวิวของราเมนแต่ละรายการ
    - **Brand**: แบรนด์ของราเมน เช่น "Nissin", "Maruchan"
    - **Variety**: ชื่อของราเมน เช่น "Chicken Flavor", "Tonkotsu"
    - **Style**: รูปแบบของราเมน เช่น "Cup", "Pack", "Bowl"
    - **Country**: ประเทศที่ผลิตราเมน เช่น "Japan", "USA"
    - **Stars**: คะแนนรีวิวราเมน (เป้าหมายของโมเดล)
    """)

    # การเตรียมข้อมูล
    st.header("🔹 การเตรียมข้อมูล")
    st.write("""
    ข้อมูลราเมนถูกทำความสะอาดโดย:
    - ลบแถวที่ไม่มีค่า `Stars`
    - แปลง `Stars` เป็นตัวเลข
    - ใช้ **Label Encoding** แปลงข้อความ (`Brand`, `Variety`, `Style`, `Country`) เป็นตัวเลข
    - ใช้ **StandardScaler** ปรับค่าข้อมูลให้อยู่ในช่วงมาตรฐาน
    """)

    # ทฤษฎีของ Neural Network ที่ใช้
    st.header("🔹 ทฤษฎีของ Neural Network ที่ใช้")
    st.write("""
    โมเดลใช้โครงสร้าง **Fully Connected Neural Network (Dense Layers)** เพื่อทำนายค่าคะแนนราเมน (`Stars`)  
    - ใช้ **4 Features** ได้แก่ `Brand`, `Variety`, `Style`, `Country`
    - ใช้ **3 Hidden Layers** พร้อม `ReLU Activation`
    - มี **Dropout Layers** เพื่อลด Overfitting
    - ใช้ **Huber Loss** เพื่อลดผลกระทบของ Outlier
    """)

    # โครงสร้างโมเดล
    st.subheader("🔹 โครงสร้างของโมเดล")
    st.code("""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(4,)),  
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)  # Output Layer
    ])
    """, language="python")

    # ขั้นตอนการพัฒนาโมเดล
    st.header("🔹 ขั้นตอนการพัฒนาโมเดล")
    st.write("""
    1. โหลดข้อมูล `ramen.csv` และทำความสะอาด
    2. ใช้ **Label Encoding** แปลงข้อความเป็นตัวเลข
    3. ใช้ **StandardScaler** ปรับค่าข้อมูล
    4. สร้างโมเดล Neural Network ด้วย **Keras Sequential API**
    5. ใช้ **Adam Optimizer** และ **Huber Loss**
    6. ใช้ **Early Stopping** เพื่อลด Overfitting
    7. บันทึกโมเดลเป็นไฟล์ `ramen_model.keras`
    """)

    # แสดงโค้ดการ Train โมเดล
    st.subheader("🔹 โค้ดการ Train โมเดล")
    st.code("""
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                  loss=tf.keras.losses.Huber(), 
                  metrics=['mae'])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, 
              validation_data=(X_test_scaled, y_test), callbacks=[early_stop])
    """, language="python")

    st.success("✨ โมเดลพร้อมใช้งาน และสามารถนำไปใช้ใน Web Application ได้แล้ว!")
