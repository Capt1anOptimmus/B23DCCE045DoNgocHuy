import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score # Mặc dù không sử dụng silhouette_score trong code này, giữ lại import
import matplotlib.pyplot as plt
import os

# Thư mục gốc nơi chứa các thư mục con csv và png
base_dir = r"C:\Users\84353\OneDrive\Desktop\BTL1_Python"

# Định nghĩa đường dẫn đến thư mục csv
csv_dir = os.path.join(base_dir, "csv")
# Định nghĩa đường dẫn đầy đủ đến file result.csv trong thư mục csv
result_path = os.path.join(csv_dir, "result.csv")

# Định nghĩa đường dẫn đến thư mục png để lưu ảnh
png_dir = os.path.join(base_dir, "png")

# Đảm bảo thư mục csv và png tồn tại
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True) # Tạo thư mục png nếu chưa có

# Tải dữ liệu
# Thêm xử lý lỗi trong trường hợp không tìm thấy file
try:
    df = pd.read_csv(result_path, encoding="utf-8-sig")
    print(f"Đã tải dữ liệu thành công từ {result_path}")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy result.csv tại {result_path}")
    # Thoát hoặc xử lý lỗi phù hợp nếu file là cần thiết
    exit()
except Exception as e:
    print(f"Đã xảy ra lỗi khi tải dữ liệu: {e}")
    exit()


# Chọn các cột số cho phân cụm (loại trừ các cột không phải số và cột định danh)
# Thêm kiểm tra để đảm bảo các cột số tồn tại trước khi tiếp tục
numeric_columns = [col for col in df.columns if col not in ["Player", "Nation", "Team", "Position"]]
if not numeric_columns:
    print("Lỗi: Không tìm thấy cột số nào để phân cụm sau khi loại trừ các cột định danh.")
    exit()
X = df[numeric_columns]

# Xử lý giá trị thiếu bằng cách điền giá trị trung bình
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Xác định số lượng cụm tối ưu bằng phương pháp Elbow và điểm Silhouette
wcss = []
max_clusters = 10
# Đảm bảo có đủ mẫu cho ít nhất 2 cụm
if X_scaled.shape[0] < 2:
    print("Lỗi: Không đủ điểm dữ liệu để thực hiện phân cụm.")
    exit()

# Đảm bảo max_clusters không lớn hơn số lượng mẫu
max_clusters = min(max_clusters, X_scaled.shape[0])
if max_clusters < 2:
     print("Lỗi: Không thể xác định số cụm tối ưu với ít hơn 2 điểm dữ liệu.")
     exit()

for i in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10) # n_init=10 để chạy thuật toán 10 lần với các hạt giống khác nhau
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Vẽ biểu đồ Elbow
optimal_k = 4  # Giả định ban đầu; điều chỉnh sau khi kiểm tra biểu đồ elbow
plt.figure(figsize=(8, 5))
plt.plot(range(2, max_clusters + 1), wcss, marker='o', linestyle='-', color='b')
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Số cụm tối ưu = {optimal_k}') # Chú thích số cụm tối ưu
plt.title('Phương pháp Elbow để xác định số lượng cụm tối ưu')
plt.xlabel('Số lượng cụm (K)')
plt.ylabel('WCSS (Tổng bình phương trong cụm)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
# Lưu biểu đồ elbow vào thư mục png
elbow_plot_path = os.path.join(png_dir, "elbow_plot.png")
plt.savefig(elbow_plot_path, format='png', dpi=300, bbox_inches='tight')
plt.close() # Đóng biểu đồ để giải phóng bộ nhớ
print(f"Biểu đồ elbow đã được lưu vào {elbow_plot_path}")

# Áp dụng K-means với số lượng cụm tối ưu
# Đảm bảo optimal_k hợp lệ
if optimal_k < 2 or optimal_k > X_scaled.shape[0]:
     print(f"Lỗi: Giá trị optimal_k không hợp lệ ({optimal_k}). Vui lòng chọn giá trị từ 2 đến {X_scaled.shape[0]}.")
     exit()

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10) # n_init=10
cluster_labels = kmeans.fit_predict(X_scaled)

# Thêm nhãn cụm vào dataframe gốc
df['Cluster'] = cluster_labels

# PCA để trực quan hóa 2D
# Đảm bảo đủ mẫu cho PCA
if X_scaled.shape[0] < 2:
     print("Lỗi: Không đủ điểm dữ liệu cho PCA.")
     exit()
# Đảm bảo số thành phần nhỏ hơn hoặc bằng số lượng đặc trưng
n_components_pca = min(2, X_scaled.shape[1])
pca = PCA(n_components=n_components_pca)
X_pca = pca.fit_transform(X_scaled)

# Phương sai giải thích cho nhãn trục
explained_variance = pca.explained_variance_ratio_
total_variance = sum(explained_variance)

# Vẽ biểu đồ các cụm 2D với chú thích chi tiết
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, optimal_k))
handles = [] # Danh sách để lưu các đối tượng handle cho chú giải

# Vẽ biểu đồ phân tán cho từng cụm
for cluster in range(optimal_k):
    mask = cluster_labels == cluster
    # Đảm bảo có điểm trong cụm trước khi vẽ
    if np.any(mask):
        scatter = plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            s=100, alpha=0.7, color=colors[cluster], label=f'Cụm {cluster}' # Nhãn cho chú giải
        )
        handles.append(scatter)
    else:
        print(f"Cảnh báo: Cụm {cluster} trống và sẽ không được vẽ.")


# Thêm chú giải
plt.legend(handles=handles, title='Các cụm', loc='best', fontsize=10)

# Chú thích 5 cầu thủ ghi bàn hàng đầu
# Thêm kiểm tra để đảm bảo cột 'Gls' tồn tại
if 'Gls' in df.columns:
    # Đảm bảo có đủ cầu thủ để chọn top 5
    num_players_to_annotate = min(5, len(df))
    top_players = df.nlargest(num_players_to_annotate, 'Gls').index
    for idx in top_players:
        plt.annotate(
            df.loc[idx, 'Player'], # Tên cầu thủ
            (X_pca[idx, 0], X_pca[idx, 1]), # Vị trí trên biểu đồ PCA
            fontsize=8, xytext=(5, 5), textcoords='offset points', # Offset cho văn bản
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8) # Hộp văn bản
        )
else:
    print("Cảnh báo: Không tìm thấy cột 'Gls'. Không thể chú thích các cầu thủ hàng đầu theo bàn thắng.")


# Tiêu đề và nhãn trục với phương sai giải thích
plt.title(f'Trực quan hóa PCA các cụm cầu thủ (K={optimal_k}, {total_variance:.1%} Phương sai được giải thích)', fontsize=12)
plt.xlabel(f'Thành phần PCA 1 ({explained_variance[0]:.1%} Phương sai)', fontsize=10)
plt.ylabel(f'Thành phần PCA 2 ({explained_variance[1]:.1%} Phương sai)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Lưu biểu đồ cụm PCA vào thư mục png
pca_plot_path = os.path.join(png_dir, "pca_cluster_plot.png")
plt.savefig(pca_plot_path, format='png', dpi=300, bbox_inches='tight')
plt.close() # Đóng biểu đồ
print(f"Biểu đồ cụm PCA đã được lưu vào {pca_plot_path}")

# Phân tích các cụm
print("\nKích thước các cụm:")
print(df['Cluster'].value_counts())

# Thống kê tóm tắt cụm
# Thêm kiểm tra để đảm bảo các cột cho tóm tắt tồn tại
summary_cols = ['Gls', 'Ast', 'Tkl', 'Save%']
available_summary_cols = [col for col in summary_cols if col in df.columns]

if available_summary_cols:
    cluster_summary = df.groupby('Cluster')[available_summary_cols].mean().round(2)
    print("\nTóm tắt cụm (Giá trị trung bình):")
    print(cluster_summary)
else:
    print("\nCảnh báo: Không tìm thấy cột nào trong số các cột tóm tắt được yêu cầu (Gls, Ast, Tkl, Save%) trong dataframe.")


# Phương sai được giải thích bởi các thành phần PCA
print(f"\nTỷ lệ phương sai được giải thích bởi PCA: {explained_variance}")
print(f"Tổng phương sai được giải thích: {total_variance:.2%}")
