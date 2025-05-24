# license-plate-recognition-alpr

# License Plate Recognition System with ALPR Dashboard Integration 🚗📸

This project implements an advanced License Plate Recognition (LPR) pipeline using a combination of:
- YOLOv8 object detection for vehicle localization,
- OCR engines (PaddleOCR + Tesseract) for plate reading,
- Machine learning–based correction for noisy OCR,
- DeepSORT vehicle tracking, and
- Optional integration with an ALPR Web Dashboard.

---

## 🔍 Features

✅ Real-time vehicle detection and tracking  
✅ License plate detection and enhancement  
✅ Multi-OCR inference with ML-based post-processing  
✅ Consensus-based final plate prediction  
✅ ALPR Dashboard integration (optional)  
✅ Output annotated video and plate logs  
✅ Saves per-vehicle images for further analysis

---

## 📦 Dependencies

```bash
pip install -q opencv-python-headless ultralytics deep_sort_realtime paddleocr pytesseract scikit-learn joblib
```

#### ROS Node Python Dependencies (if running from source)
The `lpr_ros_node.py` relies on specific Python libraries for its enhanced functionality. If you are setting it up in a Python environment that doesn't automatically provide these (e.g., outside a pre-built Docker container or a ROS installation that includes them), you might need to install them via pip:
```bash
pip install numpy filterpy scikit-learn ultralytics
```
*   `numpy` is fundamental for numerical operations.
*   `filterpy` is used by the SORT tracker for Kalman filtering.
*   `scikit-learn` is used by SORT for the Hungarian algorithm (linear assignment) if `lap` is not available. `lap` (e.g. `lapjv`) is preferred if installable.
*   `ultralytics` is used for the YOLOv8 object detection. Note: The node uses an ONNX model (`yolov8n.onnx`), which can be generated using the `ultralytics` library. For runtime, OpenCV's DNN module is used, which doesn't strictly require the full `ultralytics` package if the ONNX model is already available.
Ensure OpenCV (cv2) is also available, typically provided by `cv_bridge` in a ROS environment or installed separately (e.g., `opencv-python`).

---

##  chạy dưới dạng mô-đun ROS

Project này bao gồm một nút ROS để xử lý hình ảnh từ một chủ đề ROS và xuất bản dữ liệu biển số xe đã nhận dạng cũng như các hình ảnh đã được chú thích.

### Điều kiện tiên quyết của ROS
*   ROS Noetic Ninjemys (hoặc bản phân phối ROS tương thích nơi `cv_bridge`, `rospy`, `sensor_msgs`, `std_msgs` có sẵn).
*   Một không gian làm việc catkin đã được thiết lập.
*   Một camera ROS đang chạy hoặc một tệp rosbag đang phát hành hình ảnh trên chủ đề `/camera/image_raw` (hoặc chủ đề đầu vào được cấu hình).

### Thiết lập gói ROS
1.  Tạo một gói ROS mới (ví dụ: `lpr_package`) nếu bạn chưa có, đảm bảo bao gồm các phần phụ thuộc cần thiết:
    ```bash
    catkin_create_pkg lpr_package rospy sensor_msgs std_msgs cv_bridge
    ```
2.  Sao chép nút ROS (`lpr_ros_node.py`) và thư mục `launch` vào gói ROS của bạn. Ví dụ:
    ```bash
    # Giả sử không gian làm việc catkin của bạn là ~/catkin_ws và gói là lpr_package
    cp <repository_root>/lpr_ros_node.py ~/catkin_ws/src/lpr_package/scripts/
    cp -r <repository_root>/launch ~/catkin_ws/src/lpr_package/
    # Sao chép thư mục sort_tracker và mô hình ONNX
    # (Giả sử lpr_package/scripts/sort_tracker tồn tại trong thư mục gốc của kho lưu trữ này)
    mkdir -p ~/catkin_ws/src/lpr_package/scripts/
    cp -r <repository_root>/lpr_package/scripts/sort_tracker ~/catkin_ws/src/lpr_package/scripts/
    mkdir -p ~/catkin_ws/src/lpr_package/models
    # (Giả sử yolov8n.onnx tồn tại trong thư mục gốc của kho lưu trữ này)
    cp <repository_root>/yolov8n.onnx ~/catkin_ws/src/lpr_package/models/ 
    ```
    Đảm bảo rằng `lpr_ros_node.py` có thể thực thi được:
    ```bash
    chmod +x ~/catkin_ws/src/lpr_package/scripts/lpr_ros_node.py
    ```
3.  Xây dựng không gian làm việc catkin của bạn:
    ```bash
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash
    ```

### Chạy nút ROS
Sử dụng tệp launch được cung cấp để khởi động nút. Tệp launch cho phép cấu hình các chủ đề đầu vào/đầu ra và các thông số khác.
```bash
roslaunch lpr_package lpr_ros.launch
```
Bạn có thể sửa đổi các đối số trong `lpr_ros.launch` hoặc ghi đè chúng từ dòng lệnh. Ví dụ, để sử dụng một chủ đề hình ảnh đầu vào khác:
```bash
roslaunch lpr_package lpr_ros.launch input_image_topic_arg:=/my_camera/image_topic
```
Nút sẽ:
*   Đăng ký vào chủ đề hình ảnh đầu vào (mặc định là `/camera/image_raw`).
*   Sử dụng YOLOv8n để phát hiện các phương tiện trong mỗi khung hình.
*   Sử dụng thuật toán SORT để theo dõi các phương tiện này qua các khung hình liên tiếp.
*   Đối với mỗi phương tiện được theo dõi, nó sẽ cố gắng phát hiện và đọc biển số xe bằng Tesseract OCR.
*   Áp dụng logic đồng thuận cho nhiều lượt đọc OCR từ cùng một phương tiện được theo dõi để cải thiện độ chính xác.
*   Xuất bản dữ liệu biển số xe đã được đồng thuận (dưới dạng chuỗi JSON) trên chủ đề `/lpr/license_plate_data`.
*   Xuất bản hình ảnh đã được chú thích (hiển thị các hộp giới hạn xe, ID theo dõi và biển số được nhận dạng) trên `/lpr/annotated_image`.

#### Cấu hình thông số ROS nâng cao
Các thông số sau có thể được điều chỉnh trong tệp `lpr_ros.launch` hoặc ghi đè từ dòng lệnh để tinh chỉnh hành vi của nút:

*   **Phát hiện YOLOv8:**
    *   `yolo_model_path` (string, mặc định: `yolov8n.onnx`): Đường dẫn đến tệp mô hình YOLOv8 ONNX.
    *   `yolo_conf_threshold` (float, mặc định: `0.4`): Ngưỡng tin cậy tối thiểu để phát hiện đối tượng.
    *   `yolo_nms_threshold` (float, mặc định: `0.5`): Ngưỡng Non-Maximum Suppression.
    *   `yolo_input_width` (int, mặc định: `640`): Chiều rộng đầu vào cho mô hình YOLO.
    *   `yolo_input_height` (int, mặc định: `640`): Chiều cao đầu vào cho mô hình YOLO.

*   **Theo dõi SORT:**
    *   `sort_max_age` (int, mặc định: `20`): Số khung hình tối đa mà một track sẽ được giữ lại nếu không có phát hiện nào được liên kết.
    *   `sort_min_hits` (int, mặc định: `3`): Số lượt phát hiện tối thiểu cần thiết để khởi tạo một track.
    *   `sort_iou_threshold` (float, mặc định: `0.3`): Ngưỡng IoU (Intersection over Union) để khớp các phát hiện với các track.

*   **Logic đồng thuận:**
    *   `consensus_min_readings` (int, mặc định: `3`): Số lượt đọc OCR tối thiểu cần thiết cho một track để thử áp dụng logic đồng thuận.
    *   `consensus_max_track_staleness_frames` (int, mặc định: `10`): Số khung hình sau đó một track không hoạt động sẽ được coi là cũ và kích hoạt xử lý đồng thuận.

Ví dụ: để thay đổi ngưỡng tin cậy YOLO và tuổi tối đa của SORT:
```bash
roslaunch lpr_package lpr_ros.launch yolo_conf_threshold_arg:=0.5 sort_max_age_arg:=30
```

---

## Ứng dụng Web để hiển thị

Project cũng bao gồm một ứng dụng web đơn giản để hiển thị dữ liệu biển số xe và luồng hình ảnh đã được chú thích từ nút ROS. Ứng dụng này được đóng gói bằng Docker.

### Điều kiện tiên quyết của ứng dụng Web
*   Docker đã được cài đặt.
*   Nút LPR ROS đang chạy và xuất bản trên các chủ đề dự kiến (ví dụ: `/lpr/license_plate_data` và `/lpr/annotated_image`).

### Xây dựng và Chạy ứng dụng Web
1.  Xây dựng hình ảnh Docker từ thư mục gốc của kho lưu trữ (nơi chứa `Dockerfile`):
    ```bash
    docker build -t lpr-webapp .
    ```
2.  Chạy bộ chứa Docker:
    ```bash
    docker run -p 5000:5000 --network="host" lpr-webapp
    ```
    *   `-p 5000:5000`: Ánh xạ cổng 5000 trên máy chủ của bạn tới cổng 5000 trong bộ chứa (nơi ứng dụng Flask chạy).
    *   `--network="host"`: Cho phép bộ chứa truy cập vào ngăn xếp mạng của máy chủ. Điều này là cần thiết để ứng dụng web bên trong bộ chứa (hoặc cụ thể hơn là nút ROS bên trong nó) có thể kết nối với ROS master đang chạy trên máy chủ và đăng ký các chủ đề. Hãy thận trọng khi sử dụng `--network="host"` và đảm bảo bạn hiểu ý nghĩa bảo mật của nó. Một giải pháp thay thế có thể là thiết lập một cầu nối mạng tùy chỉnh nếu ROS master của bạn đang chạy trong một bộ chứa khác.

3.  Mở trình duyệt web của bạn và điều hướng đến `http://localhost:5000` để xem ứng dụng.

Ứng dụng web sẽ tìm nạp dữ liệu từ nút ROS thông qua backend của nó và hiển thị thông tin biển số xe mới nhất cùng với luồng hình ảnh đã được chú thích.
