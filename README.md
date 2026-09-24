# Voice to Video Studio V11.0

Ứng dụng Streamlit tạo video minh họa từ voice, giữ các chế độ Comic/Horror, khóa nhân vật, bốn kiểu diễn hoạt và các provider của bản gốc.

Bản này sửa cache nhầm voice, cảnh lặp, timeline batch và xử lý Groq 429; thêm xem/sửa kịch bản, checkpoint theo phiên, trộn âm thanh ưu tiên voice và phụ đề.

- [Hướng dẫn cập nhật Streamlit và sử dụng](HUONG_DAN_STREAMLIT.md)
- [Báo cáo rà soát logic, bản sửa và giới hạn](REVIEW_LOGIC.md)

Cần đặt `app.py` và `studio_core.py` cùng cấp. Giữ `hand.png`, `requirements.txt`, `packages.txt`. Khóa API nhập trong sidebar hoặc Streamlit Secrets; không commit khóa.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Kiểm thử (cần FFmpeg và các dependency):

```bash
python -m unittest discover -s tests -v
```

API được giả lập trong kiểm thử; render, mux và giao diện Streamlit được chạy thật. Chưa xác minh bằng khóa dịch vụ của người dùng.
