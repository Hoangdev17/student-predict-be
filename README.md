# Student Performance Prediction - FastAPI Backend

Backend dùng **FastAPI** để dự đoán điểm cuối cùng (G3) dựa trên dữ liệu sinh viên.  
Model được train bằng **LinearRegression** và dữ liệu scale bằng **StandardScaler**.

---

## 1. Yêu cầu

- Python >= 3.9
- pip
- virtualenv (khuyến nghị)
- Model và scaler đã được lưu (`student_model.pkl`, `scaler.pkl`)

---

## 2. Cài đặt môi trường

1. Tạo virtual environment:

```bash
python -m venv venv
```

2.Kích hoạt virtualenv:

- Window:

```bash
venv\Scripts\activate
```

- Linux:

```bash
source venv/bin/activate
```

3. Cài dependencies:
   pip install fastapi uvicorn scikit-learn numpy pandas joblib
4. Chạy BE

```bash
   uvicorn main:app --reload
```
