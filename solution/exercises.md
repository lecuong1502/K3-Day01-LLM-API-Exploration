# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng  bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Khi temperature tăng từ 0.0 đến 1.5, các câu trả lời sẽ chuyển từ trạng thái an toàn, thực tế, có tính dự đoán cao sang trạng thái đa dạng và sáng tạo hơn. Tuy vậy khi temperature = 1.5, văn bản sinh ra trở nên thiếu mạch lạc, ngữ pháp lộn xộn, có thể gây hallucinate những thông tin hoàn toàn sai lệch.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Em đặt ở mức khoảng 0.2-0.3, vì ưu tiên hàng đầu của chatbot khách hàng là tính chính xác, độ tin cậy, tuân thủ chặt chẽ các chính sách. Mức này đủ để giảm thiểu tối đa hiện tượng ảo giác, nhưng vẫn tạo được văn phong giao tiếp tự nhiên, lịch sự,có cảm xúc hơn.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> *Ước tính: GPT-4o đắt hơn GPT-4o-mini khoảng 16.67 lần, vì tổng giá token đầu vào và đầu ra của GPT-4o đều cao hơn 16.67 lần so với GPT-4o-mini. Trường hợp GPT-4o xứng đáng với chi phí: Khi xử lý các tác vụ đòi hỏi suy luận logic nhiều bước, hiểu ngữ cảnh phức tạp hoặc phân tích chuyên sâu, ví dụ như chatbot y tế hoặc trợ lý AI phân tích rủi ro pháp lý. Trường hợp dùng GPT-4o-mini: Dành cho các task khối lượng lớn, tính lặp lại cao, ít yêu cầu suy luận phức tạp, ví dụ như tóm tắt nội dung chuỗi email, trích xuất dữ liệu thô thành định dạng JSON.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Phản hồi của "giáo viên tiểu học" thường ngắn gọn, sử dụng từ ngữ đời thường. dùng các phép ẩn dụ để mô tả tính minh bạch. Còn "chuyên gia tài chính" sử dụng câu dài, thuật ngữ chuyên ngành, tập trung vào tính ứng dụng tài chính. System prompt ảnh hướng đến hành vi mô hình bằng cách thiết lập trọng số ngữ cảnh toàn cục, ép bộ tạo văn bản phải giới hạn không gian từ vựng và văn phong suy luận sao cho khớp chính xác với nhân vật.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> *Với một đoạn văn 100 âm tiết, ước lượng theo công thức tiếng Anh sẽ cho ra khoảng 133 token. Còn khi dùng tiktoken, đoạn văn tiêu tốn 195 token. Nguyên nhân tiếng Việt tốn nhiều token hơn: Tokenizer được huấn luyện tối ưu hóa chủ yếu trên kho dữ liệu tiếng Anh, giúp các từ tiếng Anh nguyên vẹn thường xuyên được gộp thành 1 token duy nhất. Trong khi đó, tiếng Việt ít phổ biến hơn trong dữ liệu huấn luyện và sử dụng hệ thống dấu câu phức tạp trong chuẩn Unicode. Do đó, thuật toán mã hóa không tìm được chuỗi khớp dài, buộc phải chia nhỏ 1 âm tiết tiếng Việt ra thành 2–3 token ngắn, hoặc thậm chí mã hóa từng byte riêng lẻ đối với các ký tự có dấu.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming quan trọng nhất trong các ứng dụng tương tác trực tiếp với người dùng cuối (như chatbot, trợ lý ảo, công cụ viết code), nơi 1 phản hồi dài có thể mất nhiều giây. Việc nhìn thấy từng từ hiện ra giúp giảm đáng kể độ trễ cảm nhận (perceived latency), cho phép họ đọc ngay lập tức thay vì phải nhìn màn hình chờ. Ngược lại, non-streaming phù hợp hơn cho các luồng xử lý nền hoặc pipeline tự động hóa, ví dụ: trích xuất dữ liệu hàng loạt, phân tích cú pháp JSON, tóm tắt tài liệu — nơi hệ thống chỉ cần nhận được toàn bộ kết quả cuối cùng để xử lý bước tiếp theo, giúp giảm thiểu độ phức tạp của code quản lý luồng dữ liệu.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Lợi thế lớn nhất của exponential backoff là khả năng tự động giãn áp lực lên hệ thống. Khi API quá tải, việc tăng dần thời gian chờ sau mỗi lần thất bại (ví dụ: 1s, 2s, 4s, 8s) sẽ cho máy chủ một khoảng nghỉ cần thiết để phục hồi. Nếu hàng nghìn client cùng sử dụng delay cố định (ví dụ 1 giây), hiện tượng "thundering herd" sẽ xảy ra. Hàng nghìn request bị từ chối sẽ đồng loạt tấn công lại máy chủ chính xác vào 1 giây sau đó. Sự dồn ứ cục bộ này tạo ra các spike lặp đi lặp lại, khiến máy chủ không thể phục hồi và có thể dẫn đến sập toàn bộ hệ thống.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *Persona đã chọn: Trợ lý Lập trình viên Senior. System Prompt: "Bạn là một Kỹ sư Phần mềm Senior. Hãy phân tích vấn đề và giải thích code thật ngắn gọn, đi thẳng vào trọng tâm. Khi người dùng gặp lỗi, luôn cung cấp đoạn code sửa lỗi và chỉ rõ nguyên nhân gốc rễ . Luôn giao tiếp bằng tiếng Việt, nhưng giữ nguyên tiếng Anh cho các thuật ngữ kỹ thuật và tên biến." Giải thích lựa chọn từ ngữ dựa trên code hiện tại: "ngắn gọn, đi thẳng vào trọng tâm": Việc ép mô hình trả lời ngắn gọn không chỉ giúp tiết kiệm thời gian chờ (đặc biệt khi đang dùng stream) mà còn tối ưu trực tiếp số lượng output_tokens, giúp giảm chi phí cho mỗi lượt num_turns. "giữ nguyên tiếng Anh cho thuật ngữ kỹ thuật": Tránh tình trạng dịch gượng ép, đồng thời giúp luồng chữ chảy trên terminal (qua print(delta, end="", flush=True)) hiển thị tự nhiên, chuyên nghiệp và dễ đọc hơn đối với lập trình viên.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Hạn chế lớn nhất: Thể hiện ở dòng code history = history[-6:]. Trợ lý này đang bị ép áp dụng cơ chế "trí nhớ ngắn hạn" (Sliding Window) nghiêm ngặt — nó chỉ nhớ được tối đa 3 lượt hội thoại gần nhất (3 tin nhắn của user và 3 tin nhắn trả lời). Bất kỳ ngữ cảnh, quy ước code, hay cấu trúc hàm nào người dùng định nghĩa ở lượt thứ 4 trở về trước sẽ bị cắt bỏ và mô hình sẽ hoàn toàn quên mất. Đề xuất cải thiện: Thay thế việc cắt tỉa thô bạo (history[-6:]) bằng cơ chế Tóm tắt ngữ cảnh (Context Summarization) hoặc Bộ nhớ Vector (Vector Memory). Cách triển khai cụ thể: Cách 1 (Tóm tắt tự động - Dễ tích hợp): Thay vì đếm số lượng tin nhắn (bằng [-6:]), tính tổng số token của list history. Khi history vượt quá một ngưỡng nhất định, gọi một API ẩn (không hiển thị ra màn hình) yêu cầu mô hình: "Hãy tóm tắt ngắn gọn các thông tin chính của cuộc hội thoại này". Sau đó, thay thế toàn bộ các tin nhắn cũ bằng một tin nhắn {"role": "system", "content": "Ngữ cảnh cũ: <đoạn tóm tắt>"} và chỉ giữ lại 1-2 lượt chat gần nhất. Cách 2 (Memory bằng Vector DB - Nâng cao): Lưu toàn bộ lịch sử trò chuyện vào một cơ sở dữ liệu Vector (như ChromaDB/FAISS). Mỗi khi user_msg được nhập vào, hệ thống sẽ thực hiện tìm kiếm ngữ nghĩa (semantic search) để trích xuất 2-3 đoạn hội thoại cũ có liên quan nhất tới câu hỏi hiện tại, sau đó nhúng chúng vào mảng messages ngay bên dưới persona trước khi gọi stream = retry_with_backoff(...)*

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
