import time
import pandas as pd
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from io import StringIO
import matplotlib.pyplot as plt
import os

# Định nghĩa thư mục gốc
base_dir = r"C:\Users\84353\OneDrive\Desktop\BTL1_Python"

# Định nghĩa thư mục cho các tệp CSV đầu vào và đầu ra
csv_dir = os.path.join(base_dir, "csv")

# Định nghĩa đường dẫn đến tệp CSV đầu vào trong thư mục csv
input_csv_path = os.path.join(csv_dir, "result.csv") # Cập nhật đường dẫn

# Tạo thư mục 'csv' nếu nó chưa tồn tại (hữu ích khi chạy lần đầu)
os.makedirs(csv_dir, exist_ok=True)
print(f"Đảm bảo thư mục {csv_dir} tồn tại.")


# Đọc tệp CSV vào DataFrame của pandas
try:
    # na_values=["N/A"] để đảm bảo các giá trị "N/A" được đọc là NaN (Not a Number)
    df = pd.read_csv(input_csv_path, na_values=["N/A"])
    print(f"✅ Tải dữ liệu thành công từ {input_csv_path}")
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy tệp đầu vào tại {input_csv_path}")
    exit() # Thoát nếu không tìm thấy tệp đầu vào
except Exception as e:
    print(f"❌ Lỗi khi tải tệp CSV: {e}")
    exit() # Thoát nếu có lỗi khác khi tải

# Tạo một bản sao của DataFrame để thực hiện các tính toán, chuyển NaN thành 0 ở các cột số
df_calc = df.copy()

# Định nghĩa các cột cần loại trừ (không phải là số)
exclude_columns = ["Player", "Nation", "Team", "Position"]

# Chuyển NaN thành 0 trong các cột số để tính toán
numeric_columns = [col for col in df_calc.columns if col not in exclude_columns]
for col in numeric_columns:
    # Chuyển sang dạng số, đảm bảo NaN cho các giá trị không phải số, sau đó điền 0 vào NaN
    # Sử dụng errors='coerce' để biến các giá trị không phải số thành NaN trước khi điền
    df_calc[col] = pd.to_numeric(df_calc[col], errors="coerce").fillna(0)

print("Dữ liệu đã được làm sạch và các cột số đã được xử lý.")

# 1. Tạo tệp top_3.txt
rankings = {}
for col in numeric_columns:
    # Xử lý các trường hợp mà một cột có thể hoàn toàn là 0 hoặc không phải số sau khi chuyển đổi
    # Kiểm tra nếu tổng tất cả các giá trị là 0 nhưng vẫn có dữ liệu trong cột
    if df_calc[col].sum() == 0 and df_calc[col].count() > 0:
          print(f"Bỏ qua xếp hạng cho '{col}' vì tất cả các giá trị đều là 0.")
          continue # Bỏ qua cột này và chuyển sang cột tiếp theo nếu tất cả giá trị đều là 0

    # Top 3 Cao nhất
    # Sử dụng copy() để tránh SettingWithCopyWarning
    top_3_high = df_calc[["Player", "Team", col]].sort_values(by=col, ascending=False).head(3).copy()
    top_3_high = top_3_high.rename(columns={col: "Value"})
    top_3_high["Rank"] = ["1st", "2nd", "3rd"]

    # Top 3 Thấp nhất (chỉ xem xét các giá trị khác 0 nếu tồn tại)
    # Sử dụng copy()
    non_zero_df = df_calc[df_calc[col] > 0].copy()
    if not non_zero_df.empty:
        # Sử dụng copy()
        top_3_low = non_zero_df[["Player", "Team", col]].sort_values(by=col, ascending=True).head(3).copy()
    else:
        # Nếu không có giá trị khác 0, lấy 3 giá trị thấp nhất từ dữ liệu gốc (sẽ là 0)
        # Sử dụng copy()
        top_3_low = df_calc[["Player", "Team", col]].sort_values(by=col, ascending=True).head(3).copy()


    top_3_low = top_3_low.rename(columns={col: "Value"})
    top_3_low["Rank"] = ["1st", "2nd", "3rd"]

    rankings[col] = {
        "Highest": top_3_high,
        "Lowest": top_3_low
    }

# Lưu kết quả vào tệp top_3.txt trong base_dir
top_3_path = os.path.join(base_dir, "top_3.txt")
with open(top_3_path, "w", encoding="utf-8") as f:
    for stat, data in rankings.items():
        f.write(f"\nThống kê: {stat}\n")
        f.write("\nTop 3 Cao nhất:\n")
        # Đảm bảo các cột tồn tại trước khi cố gắng in
        if not data["Highest"].empty:
             f.write(data["Highest"][["Rank", "Player", "Team", "Value"]].to_string(index=False))
        else:
             f.write("Không có dữ liệu.\n")

        f.write("\n\nTop 3 Thấp nhất:\n")
        if not data["Lowest"].empty:
            f.write(data["Lowest"][["Rank", "Player", "Team", "Value"]].to_string(index=False))
        else:
             f.write("Không có dữ liệu.\n")

        f.write("\n" + "-" * 50 + "\n")
print(f"✅ Đã lưu xếp hạng top 3 vào {top_3_path}")

# 2. Tính toán trung vị (median), trung bình (mean) và độ lệch chuẩn (standard deviation) cho tệp results2.csv
rows = []
# Thêm hàng tổng thể trước
all_stats = {"": "all"}
for col in numeric_columns:
    # Đảm bảo cột là số trước khi tính toán thống kê
    if pd.api.types.is_numeric_dtype(df_calc[col]):
        all_stats[f"Trung vị của {col}"] = df_calc[col].median()
        all_stats[f"Trung bình của {col}"] = df_calc[col].mean()
        all_stats[f"Độ lệch chuẩn của {col}"] = df_calc[col].std()
    else:
        all_stats[f"Trung vị của {col}"] = None # Hoặc một chỉ báo nào đó
        all_stats[f"Trung bình của {col}"] = None
        all_stats[f"Độ lệch chuẩn của {col}"] = None

rows.append(all_stats)

# Tính toán thống kê cho từng đội
teams = sorted(df_calc["Team"].unique()) # Lấy danh sách các đội duy nhất và sắp xếp
for team in teams:
    # Sử dụng copy()
    team_df = df_calc[df_calc["Team"] == team].copy()
    team_stats = {"": team}
    for col in numeric_columns:
          if pd.api.types.is_numeric_dtype(team_df[col]):
              team_stats[f"Trung vị của {col}"] = team_df[col].median()
              team_stats[f"Trung bình của {col}"] = team_df[col].mean()
              team_stats[f"Độ lệch chuẩn của {col}"] = team_df[col].std()
          else:
              team_stats[f"Trung vị của {col}"] = None
              team_stats[f"Trung bình của {col}"] = None
              team_stats[f"Độ lệch chuẩn của {col}"] = None
    rows.append(team_stats)

# Tạo DataFrame từ các hàng thống kê
results_df = pd.DataFrame(rows)
# Đổi tên cột đầu tiên
results_df = results_df.rename(columns={"": "Đội/Tổng thể"})
for col in results_df.columns:
    if col != "Đội/Tổng thể":
        # Chỉ làm tròn các cột số
        if pd.api.types.is_numeric_dtype(results_df[col]):
            results_df[col] = results_df[col].round(2)

# Lưu kết quả vào tệp results2.csv trong thư mục 'csv'
results2_path = os.path.join(csv_dir, "results2.csv")
results_df.to_csv(results2_path, index=False, encoding="utf-8-sig")
print(f"✅ Đã lưu thống kê thành công vào {results2_path} với {results_df.shape[0]} hàng và {results_df.shape[1]} cột.")

# 3. Vẽ biểu đồ histogram cho các thống kê đã chọn
selected_stats = ["Gls per 90", "xG per 90", "SCA90", "GA90", "TklW", "Blocks"]
histograms_dir = os.path.join(base_dir, "histograms")
league_dir = os.path.join(histograms_dir, "league")
teams_dir = os.path.join(histograms_dir, "teams")

# Tạo các thư mục lưu histogram
os.makedirs(league_dir, exist_ok=True)
os.makedirs(teams_dir, exist_ok=True)
print(f"Đảm bảo các thư mục {league_dir} và {teams_dir} tồn tại.")

# Lấy danh sách các đội đã sắp xếp
teams = sorted(df_calc["Team"].unique())
for stat in selected_stats:
    # Kiểm tra xem thống kê có tồn tại và là kiểu số hay không
    if stat not in df_calc.columns or not pd.api.types.is_numeric_dtype(df_calc[stat]):
        print(f"⚠️ Thống kê '{stat}' không tìm thấy hoặc không phải là số trong DataFrame. Bỏ qua việc tạo histogram.")
        continue

    # Histogram toàn giải đấu
    plt.figure(figsize=(10, 6))
    plt.hist(df_calc[stat], bins=20, color="skyblue", edgecolor="black")
    plt.title(f"Phân phối toàn giải đấu của {stat}")
    plt.xlabel(stat)
    plt.ylabel("Số lượng cầu thủ")
    plt.grid(True, alpha=0.3)
    # Lưu biểu đồ
    plt.savefig(os.path.join(league_dir, f"{stat}_league.png"), bbox_inches="tight")
    plt.close() # Đóng biểu đồ để giải phóng bộ nhớ
    print(f"📊 Đã lưu histogram toàn giải đấu cho {stat}")

    # Histogram cho từng đội
    for team in teams:
        # Sử dụng copy()
        team_data = df_calc[df_calc["Team"] == team].copy()
        # Kiểm tra nếu dữ liệu đội rỗng hoặc cột thống kê không phải số
        if team_data.empty or not pd.api.types.is_numeric_dtype(team_data[stat]):
             print(f"Bỏ qua histogram cho '{team}' - '{stat}' do dữ liệu rỗng hoặc cột không phải số.")
             continue

        plt.figure(figsize=(8, 6))
        # Sử dụng màu khác nhau cho các thống kê phòng ngự
        color = "lightgreen" if stat in ["GA90", "TklW", "Blocks"] else "skyblue"
        plt.hist(team_data[stat], bins=10, color=color,
                 edgecolor="black", alpha=0.7)
        plt.title(f"{team} - Phân phối của {stat}")
        plt.xlabel(stat)
        plt.ylabel("Số lượng cầu thủ")
        plt.grid(True, alpha=0.3)
        # Thay thế khoảng trắng và dấu gạch chéo cho tên tệp
        stat_filename = stat.replace(" ", "_").replace("/", "_")
        # Lưu biểu đồ cho từng đội
        plt.savefig(os.path.join(teams_dir, f"{team}_{stat_filename}.png"), bbox_inches="tight")
        plt.close() # Đóng biểu đồ
        print(f"📊 Đã lưu histogram cho {team} - {stat}")

print("✅ Tất cả các histogram cho các thống kê đã chọn đã được tạo và lưu trong thư mục 'histograms'.")

# 4. Xác định đội có giá trị trung bình cao nhất cho mỗi thống kê
# Đảm bảo chỉ các cột số được bao gồm trong tính toán trung bình theo nhóm
numeric_cols_for_mean = [col for col in numeric_columns if pd.api.types.is_numeric_dtype(df_calc[col])]

if not numeric_cols_for_mean:
    print("⚠️ Không có cột số nào khả dụng để tính toán trung bình của đội.")
    highest_teams_df = pd.DataFrame() # Tạo DataFrame rỗng
else:
    # Tính trung bình cho từng đội theo các cột số
    team_means = df_calc.groupby("Team")[numeric_cols_for_mean].mean().reset_index()

    highest_teams = []
    for stat in numeric_cols_for_mean:
        # Kiểm tra xem cột có tồn tại và có dữ liệu trước khi tìm giá trị lớn nhất
        if stat in team_means.columns and not team_means[stat].isnull().all():
            # Tìm hàng có giá trị trung bình lớn nhất cho thống kê hiện tại
            max_row = team_means.loc[team_means[stat].idxmax()]
            highest_teams.append({
                "Thống kê": stat,
                "Đội": max_row["Team"],
                "Giá trị Trung bình": round(max_row[stat], 2)
            })
        else:
             print(f"Bỏ qua tính toán trung bình cao nhất cho '{stat}' do thiếu dữ liệu hoặc tất cả là NaN.")

    # Tạo DataFrame từ kết quả
    highest_teams_df = pd.DataFrame(highest_teams)

# Lưu thống kê đội có giá trị cao nhất vào tệp highest_team_stats.csv trong thư mục 'csv'
highest_team_stats_path = os.path.join(csv_dir, "highest_team_stats.csv")
highest_teams_df.to_csv(highest_team_stats_path, index=False, encoding="utf-8-sig")
print(f"✅ Đã lưu thống kê đội có giá trị cao nhất vào {highest_team_stats_path} với {highest_teams_df.shape[0]} hàng.")

# 5. Xác định đội có thành tích tốt nhất
# Định nghĩa các thống kê mà giá trị thấp hơn là tốt hơn (ví dụ: số bàn thua, thẻ phạt, mất bóng)
negative_stats = [
    "GA90", "crdY", "crdR", "Lost", "Mis", "Dis", "Fls", "Off", "Aerl Lost"
]

# Đảm bảo highest_teams_df không rỗng trước khi tiếp tục
if not highest_teams_df.empty:
    # Lọc ra các thống kê "tích cực" (không nằm trong danh sách negative_stats) thực sự có trong DataFrame
    # Sử dụng copy()
    positive_stats_df = highest_teams_df[~highest_teams_df["Thống kê"].isin(negative_stats)].copy()

    if not positive_stats_df.empty:
        # Đếm số lần mỗi đội đứng đầu trong các thống kê tích cực
        team_wins = positive_stats_df["Đội"].value_counts()

        if not team_wins.empty:
            # Xác định đội có số lần đứng đầu nhiều nhất
            best_team = team_wins.idxmax()
            win_count = team_wins.max()

            print(f"\nĐội có thành tích tốt nhất mùa giải Premier League 2024-2025 (dựa trên việc dẫn đầu nhiều thống kê tích cực nhất) là: {best_team}")
            print(f"Họ dẫn đầu trong {win_count} trên tổng số {len(positive_stats_df)} thống kê tích cực.")
        else:
            print("\nKhông thể xác định đội có thành tích tốt nhất vì không có đội nào dẫn đầu trong các thống kê tích cực.")
    else:
        print("\nKhông tìm thấy thống kê tích cực nào để xác định đội có thành tích tốt nhất.")
else:
    print("\nKhông thể xác định đội có thành tích tốt nhất vì dữ liệu thống kê đội có giá trị cao nhất không khả dụng.")