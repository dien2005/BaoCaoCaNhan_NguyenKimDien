# AI Pathfinding Algorithms - 8 Puzzle Solver

## 🎯 Mục tiêu

Báo cáo này được thực hiện nhằm:

- Nghiên cứu và đánh giá một số thuật toán trí tuệ nhân tạo thường dùng trong các bài toán tìm đường và ra quyết định.
- Triển khai 6 thuật toán:  
  - Tìm kiếm theo chiều sâu (DFS)  
  - A*  
  - Leo đồi (Hill Climbing)  
  - Làm lạnh mô phỏng (Simulated Annealing)  
  - Beam Search  
  - Quay lui (Backtracking)  
  - Q-Learning
- Làm rõ cách thức hoạt động và cơ chế nội tại của từng thuật toán.
- Phân tích ưu điểm, điểm yếu và khả năng ứng dụng thực tế của từng phương pháp.
- So sánh hiệu quả giữa các thuật toán theo nhiều tiêu chí:
  - Độ chính xác
  - Tốc độ xử lý
  - Khả năng thích nghi khi môi trường thay đổi
- Xác định thuật toán phù hợp nhất cho từng loại bài toán cụ thể, đặc biệt trong lĩnh vực trò chơi và các ứng dụng AI tương tác.

---

## 📌 Nội dung

### Không gian trạng thái

- Là tập hợp tất cả các cấu hình của ma trận 3x3, trong đó mỗi vị trí chứa một số từ 1 tới 8 và ô trống (ký hiệu là `0` trong thuật toán).
- Mỗi trạng thái là một cách sắp xếp khác nhau của các ô.

### Các hành động và chi phí

- **Hành động**: di chuyển ô trống theo 4 hướng (lên, xuống, trái, phải).
- **Chi phí**: mỗi hành động có chi phí đơn vị hoặc được đánh giá thông qua hàm heuristic.

### Trạng thái

- **Trạng thái ban đầu**: cấu hình người chơi nhận được lúc bắt đầu.
- **Trạng thái kết thúc**: ma trận 3x3 có dạng:  
  1 2 3
  4 5 6
  7 8 0

---

## 🔍 Thuật toán tìm kiếm không thông tin

### Thành phần chính:

1. Không gian trạng thái
2. Trạng thái đầu
3. Trạng thái đích
4. Tập hành động
5. Hàm hành động
6. Kiểm tra trạng thái đích
7. Chi phí nước đi

### a) BFS - Breadth-First Search

1. Khởi tạo hàng đợi (queue) chứa trạng thái ban đầu.  
2. Lặp cho đến khi hàng đợi rỗng:  
 - Lấy trạng thái đầu ra.  
 - Nếu là trạng thái đích → trả về lời giải.  
 - Nếu không → sinh các trạng thái kề và thêm vào cuối hàng đợi (nếu chưa từng duyệt).
3. Nếu hàng đợi rỗng mà chưa đến đích → không có lời giải.

### b) DFS - Depth-First Search

1. Khởi tạo ngăn xếp (stack) chứa trạng thái ban đầu.  
2. Lặp cho đến khi ngăn xếp rỗng:  
 - Lấy trạng thái trên đỉnh ngăn xếp.  
 - Nếu là trạng thái đích → trả về lời giải.  
 - Nếu không → sinh các trạng thái kề và thêm vào ngăn xếp (ưu tiên đẩy sau để xử lý trước).
3. Nếu ngăn xếp rỗng mà chưa tới đích → không có lời giải.

### c) UCS - Uniform Cost Search

1. Khởi tạo hàng đợi ưu tiên với `g(n) = 0`.  
2. Lặp:
 - Lấy nút có `g(n)` nhỏ nhất.  
 - Nếu là đích → trả về lời giải.  
 - Sinh các trạng thái kề, cập nhật `g(n)` và thêm vào hàng đợi nếu chưa duyệt hoặc có chi phí tốt hơn.

### d) IDDFS - Iterative Deepening DFS

1. Giống DFS nhưng giới hạn độ sâu.  
2. Nếu tìm thấy trạng thái đích trong giới hạn → trả về lời giải.  
3. Nếu không → thử lại với độ sâu lớn hơn.

---

## 🎯 Thuật toán tìm kiếm có thông tin

### Thành phần chính:

- Tương tự như nhóm không thông tin, **có thêm hàm đánh giá `h(n)` hoặc `f(n)`**

### a) Greedy Best-First Search

1. Khởi tạo hàng đợi ưu tiên với `f(n) = h(n)`.  
2. Lặp:
 - Lấy trạng thái có `h(n)` nhỏ nhất.  
 - Nếu là đích → trả về lời giải.  
 - Nếu không → sinh trạng thái kề và thêm vào hàng đợi.

### b) A* Search

1. Khởi tạo hàng đợi ưu tiên với `g(n)=0`, `f(n)=h(n)`.  
2. Lặp:
 - Lấy trạng thái có `f(n)` nhỏ nhất.  
 - Nếu là đích → trả về lời giải.  
 - Sinh các trạng thái kề:
   - Cập nhật `g(n)`
   - Tính `f(n) = g(n) + h(n)`
   - Nếu chưa duyệt hoặc tốt hơn → thêm vào hàng đợi

### c) IDA* - Iterative Deepening A*

1. Khởi tạo trạng thái ban đầu, `f_limit = h(start)`.  
2. Lặp:
 - Gọi hàm tìm kiếm theo chiều sâu giới hạn `f_limit`.  
 - Nếu tìm thấy → trả lời giải.  
 - Nếu không, cập nhật `f_limit = f_min` để lặp tiếp.
3. Nếu không còn trạng thái mở rộng → không có lời giải.

---

## 🔄 Thuật toán tìm kiếm cục bộ

- Hill Climbing  
- Simulated Annealing  
- Beam Search  

---

## ❓ Thuật toán trong môi trường không xác định

- Q-Learning  
- Reinforcement Learning  

---

## 🔐 Thuật toán có ràng buộc

- Backtracking  
- CSP (Constraint Satisfaction Problem)  

---

## ✅ Kết luận

Việc triển khai và đánh giá các thuật toán giúp làm rõ đặc điểm, điểm mạnh và hạn chế của từng phương pháp. Kết quả từ nghiên cứu này có thể giúp chọn thuật toán phù hợp cho từng tình huống cụ thể, đặc biệt trong các ứng dụng trò chơi và AI tương tác thông minh.

---
