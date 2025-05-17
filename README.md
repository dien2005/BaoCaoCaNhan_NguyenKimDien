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
