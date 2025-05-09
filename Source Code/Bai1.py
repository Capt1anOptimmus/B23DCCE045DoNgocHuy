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

# Thư mục gốc nơi mọi thứ sẽ được lưu
base_dir = r"C:\Users\84353\OneDrive\Desktop\BTL1_Python"

# Hàm chuyển đổi tuổi sang định dạng số thập phân
def convert_age_to_decimal(age_str):
    try:
        if pd.isna(age_str) or age_str == "N/A":
            return "N/A"
        age_str = str(age_str).strip()
        if "-" in age_str:
            years, days = map(int, age_str.split("-"))
            decimal_age = years + (days / 365)
            return round(decimal_age, 2)
        if "." in age_str:
            return round(float(age_str), 2)
        if age_str.isdigit():
            return round(float(age_str), 2)
        return "N/A"
    except (ValueError, AttributeError) as e:
        print(f"⚠️ Lỗi chuyển đổi tuổi cho '{age_str}': {e}")
        return "N/A"

# Hàm trích xuất mã quốc gia từ cột "Nation"
def extract_country_code(nation_str):
    try:
        if pd.isna(nation_str) or nation_str == "N/A":
            return "N/A"
        return nation_str.split()[-1]
    except (AttributeError, IndexError):
        return "N/A"

# Hàm làm sạch tên cầu thủ
def clean_player_name(name):
    try:
        if pd.isna(name) or name == "N/A":
            return "N/A"
        if "," in name:
            parts = [part.strip() for part in name.split(",")]
            if len(parts) >= 2:
                return " ".join(parts[::-1])
        return " ".join(name.split()).strip()
    except (AttributeError, TypeError):
        return "N/A"

# Thiết lập Selenium WebDriver
options = Options()
options.add_argument("--headless") # Chạy trình duyệt ẩn
options.add_argument("--disable-gpu") # Vô hiệu hóa GPU (đôi khi hữu ích trong môi trường không có GPU)
options.add_argument("--no-sandbox") # Vô hiệu hóa sandbox (đôi khi cần thiết trong một số môi trường)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options) # Khởi tạo WebDriver

# Định nghĩa các URL và ID bảng
urls = [
    "https://fbref.com/en/comps/9/2024-2025/stats/2024-2025-Premier-League-Stats", # URL thống kê chung
    "https://fbref.com/en/comps/9/2024-2025/keepers/2024-2025-Premier-League-Stats", # URL thống kê thủ môn
    "https://fbref.com/en/comps/9/2024-2025/shooting/2024-2025-Premier-League-Stats", # URL thống kê sút bóng
    "https://fbref.com/en/comps/9/2024-2025/passing/2024-2025-Premier-League-Stats", # URL thống kê chuyền bóng
    "https://fbref.com/en/comps/9/2024-2025/gca/2024-2025-Premier-League-Stats", # URL thống kê kiến tạo và tạo cơ hội ghi bàn
    "https://fbref.com/en/comps/9/2024-2025/defense/2024-2025-Premier-League-Stats", # URL thống kê phòng ngự
    "https://fbref.com/en/comps/9/2024-2025/possession/2024-2025-Premier-League-Stats", # URL thống kê kiểm soát bóng
    "https://fbref.com/en/comps/9/2024-2025/misc/2024-2025-Premier-League-Stats", # URL thống kê khác
]

table_ids = [
    "stats_standard", # ID bảng thống kê chung
    "stats_keeper", # ID bảng thống kê thủ môn
    "stats_shooting", # ID bảng thống kê sút bóng
    "stats_passing", # ID bảng thống kê chuyền bóng
    "stats_gca", # ID bảng thống kê kiến tạo và tạo cơ hội ghi bàn
    "stats_defense", # ID bảng thống kê phòng ngự
    "stats_possession", # ID bảng thống kê kiểm soát bóng
    "stats_misc", # ID bảng thống kê khác
]

# Định nghĩa các cột cần thiết theo đúng thứ tự
required_columns = [
    "Player", "Nation", "Team", "Position", "Age",
    "Matches Played", "Starts", "Minutes",
    "Gls", "Ast", "crdY", "crdR",
    "xG", "xAG",
    "PrgC", "PrgP", "PrgR",
    "Gls per 90", "Ast per 90", "xG per 90", "xAG per 90",
    "GA90", "Save%", "CS%", "PK Save%",
    "SoT%", "SoT per 90", "G per Sh", "Dist",
    "Cmp", "Cmp%", "TotDist", "ShortCmp%", "MedCmp%", "LongCmp%", "KP", "Pass into 1_3", "PPA", "CrsPA",
    "SCA", "SCA90", "GCA", "GCA90",
    "Tkl", "TklW",
    "Deff Att", "Lost",
    "Blocks", "Sh", "Pass", "Int",
    "Touches", "Def Pen", "Def 3rd", "Mid 3rd", "Att 3rd", "Att Pen",
    "Take-Ons Att", "Succ%", "Tkld%",
    "Carries", "ProDist", "Carries 1_3", "CPA", "Mis", "Dis",
    "Rec", "Rec PrgR",
    "Fls", "Fld", "Off", "Crs", "Recov",
    "Aerl Won", "Aerl Lost", "Aerl Won%"
]

# Định nghĩa các từ điển đổi tên cột cho từng bảng
column_rename_dict = {
    "stats_standard": {
        "Unnamed: 1": "Player",
        "Unnamed: 2": "Nation",
        "Unnamed: 3": "Position",
        "Unnamed: 4": "Team",
        "Unnamed: 5": "Age",
        "Playing Time": "Matches Played",
        "Playing Time.1": "Starts",
        "Playing Time.2": "Minutes",
        "Performance": "Gls",
        "Performance.1": "Ast",
        "Performance.6": "crdY",
        "Performance.7": "crdR",
        "Expected": "xG",
        "Expected.2": "xAG",
        "Progression": "PrgC",
        "Progression.1": "PrgP",
        "Progression.2": "PrgR",
        "Per 90 Minutes": "Gls per 90",
        "Per 90 Minutes.1": "Ast per 90",
        "Per 90 Minutes.5": "xG per 90",
        "Per 90 Minutes.6": "xAG per 90"
    },
    "stats_keeper": {
        "Unnamed: 1": "Player",
        "Performance.1": "GA90",
        "Performance.4": "Save%",
        "Performance.9": "CS%",
        "Penalty Kicks.4": "PK Save%"
    },
    "stats_shooting": {
        "Unnamed: 1": "Player",
        "Standard.3": "SoT%",
        "Standard.5": "SoT per 90",
        "Standard.6": "G per Sh",
        "Standard.8": "Dist"
    },
    "stats_passing": {
        "Unnamed: 1": "Player",
        "Total": "Cmp",
        "Total.2": "Cmp%",
        "Total.3": "TotDist",
        "Short.2": "ShortCmp%",
        "Medium.2": "MedCmp%",
        "Long.2": "LongCmp%",
        "Unnamed: 26": "KP",
        "Unnamed: 27": "Pass into 1_3",
        "Unnamed: 28": "PPA",
        "Unnamed: 29": "CrsPA",
    },
    "stats_gca": {
        "Unnamed: 1": "Player",
        "SCA.1": "SCA90",
        "GCA.1": "GCA90",
    },
    "stats_defense": {
        "Unnamed: 1": "Player",
        "Tackles": "Tkl", "Tackles.1": "TklW",
        "Challenges.1": "Deff Att",
        "Challenges.3": "Lost",
        "Blocks": "Blocks",
        "Blocks.1": "Sh",
        "Blocks.2": "Pass",
        "Unnamed: 20": "Int",
    },
    "stats_possession": {
        "Unnamed: 1": "Player",
        "Touches": "Touches",
        "Touches.1": "Def Pen",
        "Touches.2": "Def 3rd",
        "Touches.3": "Mid 3rd",
        "Touches.4": "Att 3rd",
        "Touches.5": "Att Pen",
        "Touches.6": "Live",
        "Take-Ons": "Take-Ons Att",
        "Take-Ons.2": "Succ%",
        "Take-Ons.4": "Tkld%",
        "Carries": "Carries",
        "Carries.2": "ProDist",
        "Carries.4": "Carries 1_3",
        "Carries.5": "CPA",
        "Carries.6": "Mis",
        "Carries.7": "Dis",
        "Receiving": "Rec",
        "Receiving.1": "Rec PrgR",
    },
    "stats_misc": {
        "Unnamed: 1": "Player",
        "Performance.3": "Fls",
        "Performance.4": "Fld",
        "Performance.5": "Off",
        "Performance.6": "Crs",
        "Performance.12": "Recov",
        "Aerial Duels": "Aerl Won",
        "Aerial Duels.1": "Aerl Lost",
        "Aerial Duels.2": "Aerl Won%"
    }
}

# Khởi tạo từ điển để lưu trữ tất cả các bảng
all_tables = {}

# Thu thập và xử lý từng bảng
for url, table_id in zip(urls, table_ids):
    print(f"🔍 Đang xử lý {table_id} từ {url}")
    driver.get(url)
    time.sleep(3) # Đợi trang tải xong

    soup = BeautifulSoup(driver.page_source, "html.parser")
    # Tìm kiếm các comment trong HTML
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    table = None
    # Duyệt qua các comment để tìm bảng có ID phù hợp
    for comment in comments:
        if table_id in comment:
            comment_soup = BeautifulSoup(comment, "html.parser")
            table = comment_soup.find("table", {"id": table_id})
            if table:
                break # Tìm thấy bảng, thoát vòng lặp

    if not table:
        print(f"⚠️ Không tìm thấy bảng {table_id}!")
        continue # Bỏ qua bảng này và chuyển sang bảng tiếp theo

    try:
        # Đọc bảng HTML vào DataFrame
        df = pd.read_html(StringIO(str(table)), header=0)[0]
    except Exception as e:
        print(f"❌ Lỗi khi đọc bảng {table_id}: {e}")
        continue # Xử lý lỗi và bỏ qua bảng này

    print(f"📋 Các cột gốc trong {table_id}:", df.columns.tolist())
    # Đổi tên các cột theo từ điển đã định nghĩa
    df = df.rename(columns=column_rename_dict.get(table_id, {}))
    # Xóa các cột trùng lặp (do header nhiều dòng)
    df = df.loc[:, ~df.columns.duplicated()]

    # Làm sạch và xử lý cột "Player"
    if "Player" in df.columns:
        df["Player"] = df["Player"].apply(clean_player_name)
        print(f"Tên cầu thủ mẫu trong {table_id}:", df["Player"].head(5).tolist())

    # Chuyển đổi và xử lý cột "Age"
    if "Age" in df.columns:
        print(f"Giá trị Age thô trong {table_id} (trước khi chuyển đổi):", df["Age"].head(5).tolist())
        df["Age"] = df["Age"].apply(convert_age_to_decimal)
        print(f"Giá trị Age đã xử lý trong {table_id} (sau khi chuyển đổi):", df["Age"].head(5).tolist())

    print(f"📝 Các cột đã đổi tên và làm sạch trong {table_id}:", df.columns.tolist())
    # Lưu DataFrame vào từ điển
    all_tables[table_id] = df

# Gộp tất cả các DataFrame dựa trên cột "Player"
merged_df = None

for table_id, df in all_tables.items():
    # Chỉ giữ lại các cột cần thiết
    df = df[[col for col in df.columns if col in required_columns]]
    # Xóa các hàng trùng lặp dựa trên "Player"
    df = df.drop_duplicates(subset=["Player"], keep="first")

    if merged_df is None:
        merged_df = df # Bảng đầu tiên được gán làm merged_df
    else:
        try:
            # Gộp các DataFrame bằng cách sử dụng cột "Player"
            merged_df = pd.merge(merged_df, df, on="Player", how="outer", validate="1:1")
        except Exception as e:
            print(f"❌ Lỗi khi gộp bảng {table_id}: {e}")
            continue # Xử lý lỗi gộp và bỏ qua bảng này

# Sắp xếp lại các cột theo thứ tự của required_columns
merged_df = merged_df.loc[:, [col for col in required_columns if col in merged_df.columns]]

# Chuyển đổi cột "Minutes" sang dạng số, xử lý các giá trị không hợp lệ
merged_df["Minutes"] = pd.to_numeric(merged_df["Minutes"], errors="coerce")

# Định nghĩa các cột theo kiểu dữ liệu
int_columns = ["Matches Played", "Starts", "Minutes", "Gls", "Ast", "crdY", "crdR", "PrgC", "PrgP", "PrgR",
               "Cmp", "TotDist", "Tkl", "TklW", "Deff Att", "Lost", "Blocks", "Sh", "Pass", "Int",
               "Touches", "Def Pen", "Def 3rd", "Mid 3rd", "Att 3rd", "Att Pen", "Take-Ons Att",
               "Carries", "Carries 1_3", "CPA", "Mis", "Dis", "Rec", "Rec PrgR",
               "Fls", "Fld", "Off", "Crs", "Recov", "Aerl Won", "Aerl Lost"]
float_columns = ["Age", "xG", "xAG", "Gls per 90", "Ast per 90", "xG per 90", "xAG per 90", "GA90", "Save%", "CS%", "PK Save%",
                 "SoT%", "SoT per 90", "G per Sh", "Dist", "Cmp%", "ShortCmp%", "MedCmp%", "LongCmp%", "KP", "Pass into 1_3", "PPA",
                 "CrsPA", "SCA", "SCA90", "GCA", "GCA90", "Succ%", "Tkld%", "ProDist", "Aerl Won%"]
string_columns = ["Player", "Nation", "Team", "Position"]

# Chuyển đổi các cột kiểu số nguyên, xử lý các giá trị không hợp lệ thành NaN
for col in int_columns:
    if col in merged_df.columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").astype("Int64")

# Chuyển đổi các cột kiểu số thực, giữ nguyên NaN
for col in float_columns:
    if col in merged_df.columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").round(2)

# Lọc ra các cầu thủ có hơn 90 phút thi đấu
merged_df = merged_df[merged_df["Minutes"].notna() & (merged_df["Minutes"] > 90)]

# Chuyển đổi cột "Nation" chỉ còn mã quốc gia
if "Nation" in merged_df.columns:
    merged_df["Nation"] = merged_df["Nation"].apply(extract_country_code)

# Làm sạch lại cột "Player" sau khi gộp
if "Player" in merged_df.columns:
    merged_df["Player"] = merged_df["Player"].apply(clean_player_name)

# Điền giá trị NaN trong các cột chuỗi bằng "N/A"
for col in string_columns:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna("N/A")

# In vài dòng đầu để kiểm tra
print("\n📊 Xem trước DataFrame cuối cùng (5 dòng đầu) trước khi lưu vào result.csv:")
print(merged_df.head(5).to_string())

# Tạo thư mục 'csv' bên trong base_dir nếu nó chưa tồn tại
csv_dir = os.path.join(base_dir, "csv")
os.makedirs(csv_dir, exist_ok=True)

# Lưu DataFrame đã gộp vào tệp CSV trong thư mục 'csv', giữ nguyên các giá trị NaN
result_path = os.path.join(csv_dir, "result.csv")
merged_df.to_csv(result_path, index=False, encoding="utf-8-sig", na_rep="N/A") # na_rep="N/A" để biểu diễn NaN bằng "N/A" trong CSV
print(f"✅ Đã lưu dữ liệu đã gộp thành công vào {result_path} với {merged_df.shape[0]} hàng và {merged_df.shape[1]} cột.")

# Đóng WebDriver
driver.quit()