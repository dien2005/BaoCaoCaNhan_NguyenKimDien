# AI Pathfinding Algorithms - 8 Puzzle Solver

## Mục tiêu

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

## Nội dung

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

## Thuật toán tìm kiếm không thông tin

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

## Thuật toán tìm kiếm có thông tin

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

## Thuật toán tìm kiếm cục bộ
### Các thành phần chính:

1. Trạng thái hiện tại (Current state)
2. Hàm kế cận (Successor function) — sinh trạng thái lân cận
3. Hàm đánh giá (Evaluation function, thường là h(n))
4. Tiêu chí dừng (goal state hoặc local optimum)
5. (Tuỳ chọn) Lịch giảm nhiệt (T) với Simulated Annealing

### a) Hill Climbing

1. Khởi tạo tại trạng thái ban đầu.
2. Lặp:
 - Sinh các trạng thái kề.
 - Nếu tồn tại trạng thái có giá trị heuristic tốt hơn → chuyển sang nó.
 - Nếu không → dừng (đạt cực trị cục bộ).

### b) Steepest-Ascent Hill Climbing

1. Khởi tạo tại trạng thái ban đầu.
2. Lặp:
 - Sinh tất cả các trạng thái kề.
 - Chọn trạng thái tốt nhất theo heuristic.
 - Nếu tốt hơn trạng thái hiện tại → chuyển sang đó.
 - Nếu không → dừng.

### c) Stochastic Hill Climbing

1. Khởi tạo trạng thái ban đầu.
2. Lặp lại:
 - Sinh ngẫu nhiên một vài trạng thái kề.
 - Nếu có trạng thái kề tốt hơn hiện tại → chọn ngẫu nhiên một trong số đó để di chuyển.
 - Nếu không có trạng thái tốt hơn → dừng.

### d) Simulated Annealing

1. Khởi tạo trạng thái ban đầu và nhiệt độ T.
2. Lặp đến khi T giảm về 0:
 - Chọn ngẫu nhiên một trạng thái kề.
 - Tính ΔE = giá trị trạng thái mới - hiện tại.
 - Nếu ΔE < 0 → chấp nhận trạng thái mới (tốt hơn).
 - Nếu ΔE > 0 → chấp nhận với xác suất e^(-ΔE / T).
 - Giảm nhiệt độ T theo thời gian (theo lịch làm lạnh).

### e) Local Beam Search

1. Khởi tạo K trạng thái ngẫu nhiên.
2. Lặp:
 - Từ tất cả K trạng thái, sinh tất cả trạng thái kề.
 - Chọn K trạng thái tốt nhất từ tập tất cả trạng thái con.
 - Nếu một trạng thái là đích → trả về lời giải.

### f) Genetic Algorithm

1. Khởi tạo quần thể cá thể ngẫu nhiên.
2. Lặp đến khi hội tụ hoặc đủ thế hệ:
 - Đánh giá độ thích nghi (fitness) của từng cá thể.
 - Chọn các cặp cá thể (theo fitness) để **lai ghép**.
 - Áp dụng **đột biến** ngẫu nhiên trên cá thể con.
 - Hình thành quần thể mới từ cá thể con.
 - Nếu có cá thể đạt mục tiêu → trả về.

---

## Thuật toán tìm kiếm trong môi trường không xác định
### Các thành phần chính:

1. Môi trường (Environment)
2. Trạng thái (State)
3. Hành động (Action)


### a) And-Or Search

1. Khởi tạo nút gốc (initial state).
2. Nếu nút là goal hoặc dừng, trả về thành công.
3. Nếu không, tạo các nút con (successors) theo các hành động có thể thực hiện.
4. Với mỗi nút con, gọi đệ quy And-Or Search.
5. Tạo node AND nếu cần tất cả con phải thành công, hoặc node OR nếu chỉ cần một con thành công.
6. Trả về cây tìm kiếm kết hợp các node AND-OR.

### b) Belief-State Search

1. Khởi tạo belief state ban đầu (tập trạng thái có thể xảy ra).
2. Lặp lại:
 - Xác định các hành động có thể.
 - Dựa vào hành động và quan sát, cập nhật belief state mới.
 - Kiểm tra nếu belief state mới thỏa điều kiện goal.

3. Tìm đường đi qua các belief state tới goal.

---

## Thuật toán tìm kiếm có rạng buộc
### Các thành phần chính:
1. Không gian trạng thái
2. Tập biến (Variables)
3. Tập giá trị (Domains)
4. Tập ràng buộc (Constraints)

### a) Thuật toán AC-3 (Arc Consistency 3)

1. Khởi tạo hàng đợi Q chứa tất cả cung (arc) (Xi, Xj).
2. Lặp cho đến khi Q rỗng:
 - Lấy cung (Xi, Xj) ra khỏi Q.
 - Gọi hàm REVISE(Xi, Xj) để loại bỏ giá trị không phù hợp trong miền Xi.
 - Nếu miền Xi thay đổi:
 - Nếu miền Xi rỗng, trả về thất bại.
 - Thêm tất cả cung (Xk, Xi) vào Q, k là các biến liên quan đến Xi.
3. Trả về thành công nếu không có miền biến rỗng.

### b) Thuật toán Kiểm thử (Testing Algorithms)

1. Chọn biến và giá trị để gán.
2. Kiểm tra các ràng buộc liên quan đến biến đó.
3. Nếu ràng buộc thỏa mãn, tiếp tục với biến kế tiếp.
4. Nếu không, quay lui hoặc thay đổi giá trị.

### Thuật toán BackTracking

1. Bắt đầu với biến đầu tiên.
2. Gán giá trị hợp lệ cho biến hiện tại.
3. Kiểm tra ràng buộc với các biến đã gán.
4. Nếu thỏa, tiếp tục với biến kế tiếp.
5. Nếu không thỏa, quay lui (backtrack) để thay đổi giá trị biến trước đó.
6. Lặp lại đến khi tìm được nghiệm hoặc hết khả năng gán.

---

## Thuật tìm kiếm học củng cố
### Các thành phần chính:
1. Môi trường (Environment)
2. Trạng thái (State)
3. Hành động (Action)
4. Chính sách (Policy)
5. Hàm phần thưởng (Reward Function)
6. Hàm giá trị (Value Function)
6. Bảng Q (Q-Table)
7. Công thức cập nhật Q-learning
8. Chiến lược khám phá

### a) Thuật toán Reinforce

1. Khởi tạo chính sách π với tham số θ.
2. Thực hiện một tập episode (lượt chơi) với chính sách π(θ).
3. Thu thập trạng thái, hành động, và phần thưởng.
4. Tính toán gradient của hàm mục tiêu (expected reward) dựa trên dữ liệu thu thập.
5. Cập nhật tham số θ theo hướng gradient tăng.
6. Lặp lại các bước trên cho đến khi hội tụ.

### b) Thuật toán Q-learning

1. Khởi tạo bảng Q(s,a) tùy ý (ví dụ 0).
2. Lặp cho mỗi bước trong tập episode:
 - Chọn hành động a từ trạng thái s theo chính sách (thí dụ ε-greedy).
 - Thực hiện hành động, nhận phần thưởng r và chuyển đến trạng thái s'.
 - Cập nhật Q(s,a)
 - Cập nhật trạng thái s = s'.
3. Lặp lại cho đến khi bảng Q hội tụ.


---
## Kết luận

Việc triển khai và đánh giá các thuật toán giúp làm rõ đặc điểm, điểm mạnh và hạn chế của từng phương pháp. Kết quả từ nghiên cứu này có thể giúp chọn thuật toán phù hợp cho từng tình huống cụ thể, đặc biệt trong các ứng dụng trò chơi và AI tương tác thông minh.

---
