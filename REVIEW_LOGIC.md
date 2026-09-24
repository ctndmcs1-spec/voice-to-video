# Rà soát app.py và bản sửa V11.0

Bản gốc được lấy từ repo ctndmcs1-spec/voice-to-video; các kết luận dưới đây dựa trên mã, không phải log của một lần chạy cụ thể của người dùng.

| Lỗi / rủi ro xác nhận trong mã gốc | Hệ quả | Xử lý |
|---|---|---|
| Cache transcript theo batch_000/batch_001 trong thư mục dùng chung | Voice mới đọc transcript voice cũ | SHA-256 nội dung audio + model + ngôn ngữ; tách dự án theo phiên |
| Cộng offset toàn video cho transcript nhưng planner/render dùng thời gian batch | Các batch sau có thể mất cảnh hoặc rơi vào fallback | Dùng thời gian cục bộ; offset tích lũy chỉ dùng cho SRT và hiển thị |
| Tách cảnh dài bằng sao chép cùng visual_prompt, thêm chữ TIẾP | Gọi API tạo lại cùng bố cục | Giữ một cảnh, không tự nhân đôi prompt |
| Planner lỗi -> xoay vòng hai prompt chung chung | Cảnh lặp, không bám voice | Retry sửa kịch bản có giới hạn rồi dừng rõ lỗi |
| max_tokens tối thiểu 5.000, tối đa 14.000 | Có thể vượt OTPM; gán mọi lỗi là Qwen fail | Token budget, batch nhỏ hơn; xử lý 429 và JSON riêng |
| Mỗi lần bấm Start tạo thư mục temp mới, xóa cảnh sau batch | Không chạy tiếp khi lỗi | Checkpoint JSON + checksum ảnh/clip/batch; khóa job trong phiên |
| Nút test nằm trong điều kiện của một nút khác | Streamlit rerun làm mất nhánh thực thi nút test | Giữ trạng thái mở panel trong session_state |
| Download/video chỉ hiện trong nhánh Start | Mất giao diện kết quả khi rerun | Render lại từ checkpoint; download không yêu cầu rerun |
| Bỏ mọi batch cuối dưới 5 giây | Mất đoạn cuối audio | Giữ tất cả các đoạn có trong audio |
| Chia script theo dòng, một đoạn dài không xuống dòng | Batch sau thiếu text | Chia theo tỉ lệ thời gian trên danh sách từ |
| Không sort/validate đầy đủ timeline; tự merge mất cảnh | Sai thứ tự / lặp nội dung / thời lượng | Sort, chặn overlap/NaN/prompt trùng; không tự gộp mất nội dung |
| Làm tròn duration từng cảnh riêng | Tích lũy lệch hình/tiếng | Làm tròn ranh giới khung hình tích lũy |
| Encoder stderr PIPE không được drain; một số style không check exit | Nguy cơ treo hoặc chấp nhận clip hỏng | RawVideoWriter có log trên đĩa, kiểm tra exit, cleanup và xuất atomic |
| Cached ảnh không cập nhật bộ đếm; Future lỗi không được gọi result | UI sai tiến độ / lỗi worker bị bỏ qua | Bộ đếm cached và propagate lỗi worker |
| Fallback tuần tự vượt retry/circuit cap | Gửi quá nhiều lần khi provider lỗi | Giới hạn retry thống nhất và không bypass circuit |
| amix mặc định normalize | Voice nhỏ hơn khi thêm nhạc/SFX | Giữ gain khi mix, duck nhạc theo voice, limiter; loudnorm hai pass cuối |
| Cache nhạc dùng int(duration) | Đoạn cùng số giây nguyên nhưng dài khác bị dùng sai mẫu | Cache theo số sample |
| Horror title tối trên nền tối | Khó đọc | Title sáng và tương phản |
| Camera crop làm tròn tọa độ nguyên | Di chuyển có thể nhảy pixel | Crop bằng affine subpixel |
| Draw animation tối thiểu 1,5s cho cảnh ngắn hơn | Cảnh kết thúc khi chưa lộ hình | Giới hạn phase theo số frame có thật |

## Kiểm chứng

- Bộ test offline cho cache, timeline, duplicate prompt, quota retry, checkpoint, đổi một cảnh, ảnh hỏng và khóa job.
- Render thật bốn kiểu trên clip ngắn, kiểm tra duration và frame cuối bằng FFprobe/OpenCV.
- Ghép thật audio + SFX + nhạc, xuất MP4 1080p upscale với phụ đề tiếng Việt; chạy lại và xác nhận không gọi thêm provider cho cảnh đã xong.
- AppTest Streamlit: tải voice giả lập, lập kịch bản, render bằng provider giả lập, rerun sidebar và vẫn còn kết quả tải xuống.
- Trong kiểm thử audio im lặng đã tìm và sửa lỗi loudnorm sinh gain vô hạn: đo trước và bỏ normalization nếu kết quả không hữu hạn.

Chưa chạy thử API thật, voice thật của người dùng hoặc triển khai lên app Streamlit đang hoạt động. Không khẳng định đã đạt chất lượng nghệ thuật tương đương một kênh YouTube cụ thể.

## Bổ sung V11.2

V11.0/V11.1 mới sửa text một dòng bằng cách chia theo tỉ lệ thời lượng; chưa xử lý tốc độ đọc thay đổi. V11.2 thay đường combined bằng đối chiếu toàn bài theo từ, giữ timestamp Whisper, chặn mismatch lớn. Phép đối chiếu này không phải forced alignment trên tín hiệu âm thanh và không bảo đảm sửa được transcript sai nghiêm trọng. Gallery hiển thị ảnh/clip có checkpoint khớp phiên bản cảnh hiện tại.
