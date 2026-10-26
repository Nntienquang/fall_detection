# Lời thoại video giới thiệu — từng thành viên

Dùng kèm `KICH_BAN_VIDEO.md` (cách quay, phân cảnh) và bộ slide 16 trang. Tổng khoảng **9–10 phút**.

**Cách dùng:**
- Mỗi người **đọc to phần mình 3–4 lần trước khi quay** cho quen, rồi nói tự nhiên theo ý; không cần thuộc từng chữ. Không nhìn chằm chằm vào giấy khi nói.
- Chữ trong ngoặc vuông `[…]` thay bằng thông tin thật. Dòng *nghiêng* là chỉ dẫn hành động, không đọc.
- Xưng "em", gọi "cô". Nói chậm, rõ; khoảng 150 chữ mỗi phút.
- Ai đang nói thì **bật camera**. Khi chia sẻ màn hình, camera vẫn hiện ở góc.

| Cảnh | Người nói | Thời lượng | Slide / màn hình |
|---|---|---|---|
| 1. Mở đầu | Cả 4 | 0:45 | Mặt cả nhóm → slide 1–2 |
| 2. Bài toán & kiến trúc | TV4 | 1:15 | Slide 3–4 |
| 3. Phát hiện & theo dõi người | TV1 | 1:30 | Slide 4 → VS Code `pose_tracker.py`, `fall_detector.py` |
| 4. Đặc trưng hình học | TV2 | 1:30 | Slide 5 → VS Code `features.py` |
| 5. Phân tích theo thời gian | TV3 | 1:30 | Slide 5–7 → VS Code `_step` |
| 6. Demo | Cả 4 | 2:00 | Webcam trực tiếp → thư mục `outputs/` |
| 7. Kết quả & hạn chế | TV4 | 1:15 | Slide 8–13 |
| 8. Kết luận | Cả 4 | 0:30 | Slide 15–16, mặt cả nhóm |

---

## Cảnh 1 — Mở đầu (cả 4, ~45 giây)

*Cả 4 người cùng hiện trong khung hình (ngồi cạnh nhau, hoặc 4 camera trong cuộc họp).*

**TV4:** "Em chào cô ạ. Chúng em là nhóm [__], lớp [mã lớp]. Đề tài của nhóm em là **phát hiện người té ngã qua camera**, chạy thời gian thực trên máy tính thường, không cần card đồ hoạ. Sau đây từng bạn trong nhóm xin tự giới thiệu."

**TV1:** "Em là [tên TV1], em phụ trách phần **phát hiện và theo dõi người**: dùng YOLO Pose để lấy khung xương và ByteTrack để giữ ID cho từng người."

**TV2:** "Em là [tên TV2], em phụ trách phần **đặc trưng hình học**: từ các khớp trên cơ thể tính ra góc thân người và các đặc trưng để biết người đang đứng hay nằm."

**TV3:** "Em là [tên TV3], em phụ trách phần **phân tích theo thời gian**: máy trạng thái quyết định khi nào là một cú ngã thật."

**TV4:** "Còn em là [tên TV4], em phụ trách **luồng chạy chính, đánh giá kết quả và phần demo**. Em xin bắt đầu với bài toán."

---

## Cảnh 2 — Bài toán và kiến trúc (TV4, ~1 phút 15)

*Chia sẻ màn hình, slide 3.*

"Theo Tổ chức Y tế Thế giới, té ngã là nguyên nhân thứ hai gây tử vong do chấn thương không chủ ý, và phần lớn nạn nhân là người cao tuổi. Nguy hiểm nhất là khi người già ở nhà một mình, ngã xuống mà không ai biết.

Nhóm em đặt ra ba yêu cầu cho hệ thống: một là chạy được trên CPU của máy tính thường; hai là nhận được cả webcam lẫn video có sẵn; và ba, quan trọng nhất, là ít báo nhầm, tức là phải phân biệt được ngã với ngồi xuống, cúi người hay chủ động nằm."

*Chuyển sang slide 4.*

"Đây là kiến trúc hệ thống. Mỗi khung hình đi qua năm bước. Bước một, lấy ảnh từ camera hoặc video. Bước hai, mô hình YOLO11n-Pose tìm người và 17 khớp trên cơ thể. Bước ba, ByteTrack gán cho mỗi người một ID để theo dõi qua thời gian. Bước bốn, tính các đặc trưng như góc thân người. Và bước năm, máy trạng thái quyết định có phát cảnh báo hay không.

Bước hai và ba nhóm dùng mô hình có sẵn của thư viện Ultralytics, không huấn luyện lại. **Phần nhóm tự xây dựng là bước bốn, bước năm và toàn bộ phần đánh giá.** Sau đây bạn [tên TV1] sẽ trình bày bước hai và ba."

---

## Cảnh 3 — Phát hiện và theo dõi người (TV1, ~1 phút 30)

*Slide 4, chỉ vào ô 02 và 03.*

"Em xin trình bày phần phát hiện và theo dõi người. YOLO Pose khác YOLO thường ở chỗ: ngoài khung bao quanh người, nó dự đoán luôn 17 khớp theo chuẩn COCO, gồm mũi, mắt, tai, vai, khuỷu tay, cổ tay, hông, đầu gối và cổ chân, mỗi khớp có toạ độ và độ tin cậy. Tất cả chỉ trong một lần chạy mạng, nên rất nhanh."

*Chuyển sang VS Code, mở `src/pose_tracker.py`, hàm `__call__`.*

"Đây là đoạn code chính. Hàm `model.track` vừa phát hiện vừa theo dõi. Tham số `persist=True` để ByteTrack nhớ các người ở khung trước, nhờ vậy ID ổn định. `classes=[0]` nghĩa là chỉ lấy lớp người. Khớp nào có độ tin cậy dưới 0,35 thì nhóm coi như không nhìn thấy.

Vì sao cần theo dõi chứ không chỉ phát hiện? Vì muốn biết một người có ngã hay không thì phải xem **lịch sử của chính người đó** qua nhiều khung hình."

*Mở `src/fall_detector.py`, hàm `_overlap` và `_drop_duplicates`.*

"Khi chạy thử trên video fall-08, nhóm gặp một lỗi thật: lúc người nằm xuống, YOLO trả về **hai khung cho cùng một người**, và khung thứ hai bị gán ID mới, làm mất toàn bộ lịch sử. Hai khung này lồng vào nhau nên IoU thấp, NMS không loại được. Nhóm em tính phần giao chia cho diện tích **khung nhỏ hơn**: nếu từ 60% trở lên thì coi là cùng một người và bỏ khung trùng. Ngoài ra, khi ByteTrack đổi ID giữa lúc ngã, hàm `_get_track` sẽ nối ID mới vào lịch sử cũ nếu vị trí gần nhau. Lỗi này đã có unit test để không tái diễn. Tiếp theo là phần của bạn [tên TV2]."

---

## Cảnh 4 — Đặc trưng hình học (TV2, ~1 phút 30)

*Slide 5, phần ba đặc trưng phía dưới.*

"Em xin trình bày phần đặc trưng hình học, tức là từ 17 khớp làm sao biết người đang đứng hay đang nằm.

Đặc trưng quan trọng nhất là **góc thân người**: góc giữa đường nối trung điểm hai vai với trung điểm hai hông, so với phương thẳng đứng. Người đứng thẳng thì góc gần 0 độ, nằm ngang thì gần 90 độ."

*Chuyển sang VS Code, mở `src/features.py`, hàm `torso_angle`.*

"Trong code, em lấy trung điểm vai và trung điểm hông, tính độ lệch dx, dy, rồi dùng `atan2` của trị tuyệt đối dx và dy. Có một điểm cần chú ý: trong ảnh, **trục y hướng xuống**. Dùng `atan2` thì không bị lỗi chia cho 0 khi người nằm ngang hẳn, và lấy trị tuyệt đối để ngã sang trái hay phải đều ra cùng một góc.

Đặc trưng thứ hai là tỷ lệ chiều rộng trên chiều cao của khung người, đứng thì khoảng 0,3 đến 0,5, nằm thì lớn hơn 1. Nhưng tỷ lệ này dễ sai khi người dang tay, nên phương pháp dùng góc thân cho kết quả tốt hơn.

Thứ ba, người ở xa camera thì trông nhỏ hơn. Để một ngưỡng dùng được ở mọi khoảng cách, em **chuẩn hoá theo cạnh dài nhất của khung người**, vì cạnh này gần như không đổi khi người chuyển từ đứng sang nằm.

Khi thử webcam, nhóm còn thêm luật nhận biết người **đang cúi**: thân nghiêng nhưng chân vẫn dựng đứng và hông chưa hạ xuống thì không tính là nằm. Tiếp theo là bạn [tên TV3]."

---

## Cảnh 5 — Phân tích theo thời gian (TV3, ~1 phút 30)

*Slide 5, sơ đồ ba trạng thái.*

"Em xin trình bày phần phân tích theo thời gian. Chỉ nhìn một khung hình thì không phân biệt được người đang ngã với người đang nằm nghỉ hay đang cúi. Vì vậy mỗi người có một **máy trạng thái** với ba trạng thái.

Bình thường là NORMAL, khung màu xanh. Khi thân người nằm ngang, và người đó trước đó đã từng đứng, thì chuyển sang SUSPECTED, màu vàng, nghĩa là đang nghi ngờ. Nếu người đứng dậy trong nửa giây thì huỷ. Nếu nằm đủ 0,5 giây và ít nhất 70% số khung là nằm ngang thì xác nhận FALL, màu đỏ, phát cảnh báo. Khi người đứng thẳng lại đủ 2 giây thì ghi nhận đã dậy."

*Chuyển sang VS Code, mở `src/fall_detector.py`, hàm `_step`, kéo đến nhánh `SUSPECTED`.*

"Đây là đoạn code tương ứng. Điều kiện 'đã từng đứng' giúp không báo nhầm người nằm sẵn từ đầu. Tỷ lệ 70% cho phép một vài khung bị nhiễu khớp."

*Chuyển sang slide 6.*

"Nhóm so sánh ba phương pháp, chỉ khác nhau ở cách quyết định một khung là đang nằm: A dùng tỷ lệ khung người, B dùng góc thân, còn C thêm điều kiện phải có cú hạ hông nhanh trước đó, vì ngã thì rơi nhanh còn chủ động nằm thì chậm.

Ngoài ra, bản demo webcam còn chờ người **nằm yên** khoảng một giây mới báo. Người ngã thường nằm im, còn chống đẩy thì vai lên xuống liên tục, nên không bị báo nhầm."

*Chuyển sang slide 7.*

"Đây là kết quả thật trên một video: đi bình thường màu xanh, chúi xuống màu vàng, nằm trên sàn thì đỏ và cảnh báo. Bây giờ nhóm em xin demo trực tiếp."

---

## Cảnh 6 — Demo trực tiếp (cả 4, ~2 phút)

*Quay tại phòng, camera thấy trọn người. Chạy `python main.py --source 0 --config config_demo.yaml --imgsz 416`, nhấn F để toàn màn hình. Làm theo `KICH_BAN_DEMO.md`.*

**TV4 (đứng cạnh máy, giới thiệu):** "Đây là hệ thống đang chạy với webcam của laptop. Góc trên hiển thị số khung hình mỗi giây và thời gian chạy mô hình. Mỗi người có khung, ID và góc thân."

*TV1 đi lại trước camera.*

**TV1:** "Em đang đi lại bình thường. Hệ thống nhận ra em, vẽ 17 khớp, khung màu xanh, góc thân gần 0 độ."

*TV2 ngồi xuống ghế rồi đứng lên.*

**TV2:** "Em ngồi xuống ghế. Thân em vẫn thẳng nên góc nhỏ, hệ thống vẫn để xanh, không báo ngã."

*TV3 cúi nhặt đồ, giữ vài giây, rồi đứng lên.*

**TV3:** "Em cúi xuống nhặt đồ và giữ một lúc. Thân em nghiêng nhưng chân vẫn đứng, hông chưa hạ xuống, nên hệ thống hiểu là đang cúi, không báo."

*TV4 ngã có kiểm soát xuống đệm, ngang qua khung hình, nằm yên 3 giây. Một bạn khác đứng cạnh đệm.*

**TV1 (giải thích trong lúc TV4 ngã):** "Bây giờ bạn [tên TV4] mô phỏng một cú ngã xuống đệm. Khung chuyển sang vàng, khi bạn nằm yên thì chuyển đỏ, màn hình hiện cảnh báo và hệ thống tự lưu ảnh."

*TV4 đứng dậy, chờ 2 giây.*

**TV4:** "Khi em đứng dậy được 2 giây, hệ thống ghi nhận em đã dậy và trở về màu xanh."

*Nhấn Q để thoát, mở thư mục `outputs/` mới nhất.*

**TV4:** "Mỗi lần chạy, hệ thống lưu một thư mục gồm file `events.csv` ghi thời điểm và góc thân của từng sự kiện, cùng ảnh chụp lúc phát hiện ngã, để người chăm sóc kiểm tra lại."

---

## Cảnh 7 — Kết quả và hạn chế (TV4, ~1 phút 15)

*Chia sẻ màn hình, slide 8.*

"Để đánh giá khách quan, nhóm dùng **40 video người thật** của bộ dữ liệu UR Fall, chia ba tập: tập val để chọn ngưỡng, và tập test 20 video **chưa từng dùng để chỉnh** để báo cáo kết quả. Nhóm chấm theo từng cú ngã, tính Precision, Recall và F1."

*Slide 9.*

"Kết quả: phương pháp B, góc thân người, tốt nhất với **F1 bằng 0,86**, bắt được 9 trên 10 cú ngã, báo nhầm 2 lần trên 10 video sinh hoạt, độ trễ trung bình khoảng 0,7 giây. Ban đầu nhóm dự đoán C sẽ tốt nhất, nhưng kết quả không xác nhận điều đó, và nhóm báo cáo đúng như vậy."

*Slide 10–11.*

"Khi phân tích từng ca sai, nhóm thấy nhiều cú ngã bị sót vì YOLO không nhìn thấy người đang nằm ở xa. Tăng kích thước ảnh vào từ 320 lên 640 đã cứu được hai ca đó, đổi lại tốc độ giảm còn khoảng 15 khung hình mỗi giây."

*Slide 13.*

"Trên webcam thật, hệ thống chạy 13 đến 21 khung hình mỗi giây. Nhóm đã gặp và xử lý bốn kiểu báo nhầm: ngồi sát webcam, người bị cắt ở mép ảnh, cúi người lâu, và chống đẩy. Hạn chế còn lại là tư thế quỳ chống tay giữ yên và ngã thẳng về phía camera, vì trên ảnh hai chiều chúng trông giống tư thế bình thường. Nhóm đã thử chặn nhưng sẽ làm sót cú ngã thật, mà bỏ sót thì nguy hiểm hơn báo nhầm."

---

## Cảnh 8 — Kết luận (cả 4, ~30 giây)

*Slide 15, rồi cả 4 người hiện lại trong khung hình.*

**TV2:** "Tóm lại, nhóm em đã xây dựng được hệ thống phát hiện té ngã chạy thời gian thực trên CPU, có đánh giá trên dữ liệu người thật và thử nghiệm trực tiếp bằng webcam."

**TV3:** "Hướng phát triển là quay thêm dữ liệu, học phân loại trực tiếp từ chuỗi khung xương, và gửi thông báo về điện thoại người thân."

**TV1:** "Toàn bộ mã nguồn, báo cáo và slide nhóm em đã nộp kèm theo."

**Cả nhóm:** "Chúng em cảm ơn cô đã theo dõi ạ!"

---

## Nếu cô hỏi thêm trong buổi báo cáo

Câu hỏi và gợi ý trả lời nằm trong `HOC_CODE_LOI.md` (từng phần TV1–TV4 và bảng "Câu hỏi chung").
