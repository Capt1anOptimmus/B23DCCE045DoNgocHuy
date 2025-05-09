# Import các thư viện cần thiết
from fuzzywuzzy import fuzz, process # Thư viện xử lý chuỗi mờ (fuzzy string matching)
# fuzz: Cung cấp các hàm so sánh mức độ tương đồng giữa hai chuỗi (ví dụ: fuzz.token_sort_ratio)
# process: Cung cấp các công cụ để tìm kiếm chuỗi phù hợp nhất trong một danh sách hoặc tập dữ liệu (ví dụ: process.extractOne)

from selenium import webdriver # Thư viện tự động hóa trình duyệt web
# Selenium dùng để điều khiển trình duyệt (như Chrome, Firefox) để tương tác với các trang web động (chạy JavaScript).
from selenium.webdriver.chrome.service import Service # Dùng để tạo một dịch vụ điều khiển trình điều khiển ChromeDriver.
from selenium.webdriver.common.by import By # Dùng để xác định vị trí các phần tử trên trang web (ví dụ: theo CLASS_NAME, TAG_NAME).
from selenium.webdriver.chrome.options import Options # Dùng để cấu hình tùy chọn cho Chrome, ví dụ chạy ẩn (headless), không hiển thị ảnh, tắt thông báo.
from selenium.webdriver.support.ui import WebDriverWait # Cung cấp các phương thức để đợi một điều kiện cụ thể xảy ra trên trang web.
from selenium.webdriver.support import expected_conditions as EC # Cung cấp các điều kiện chờ đợi sẵn có (ví dụ: chờ phần tử xuất hiện).
from webdriver_manager.chrome import ChromeDriverManager # Tự động tải và cấu hình ChromeDriver phù hợp với phiên bản trình duyệt Chrome trên máy tính của bạn.

import pandas as pd # Thư viện xử lý dữ liệu dạng bảng (DataFrame).
import os # Thư viện tương tác với hệ điều hành, dùng để xử lý đường dẫn file/thư mục.

# --- Cấu hình đường dẫn file/thư mục ---

# Thư mục gốc nơi các file sẽ được lưu vào
base_dir = r"C:\Users\84353\OneDrive\Desktop\BTL1_Python"

# Đường dẫn đến directory con 'csv' nơi chứa file result.csv và các file csv đầu ra
csv_dir = os.path.join(base_dir, "csv")

# Đường dẫn đầy đủ đến file result.csv
result_path = os.path.join(csv_dir, "result.csv")

# --- Tải và xử lý dữ liệu ban đầu ---

# Mở file result.csv để tính toán
# na_values=["N/A"] chỉ định rằng các giá trị "N/A" trong file sẽ được đọc là NaN (Not a Number)
try:
    df = pd.read_csv(result_path, na_values=["N/A"], encoding="utf-8-sig")
    print(f"Đã tải dữ liệu thành công từ {result_path}")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy result.csv tại {result_path}")
    # Thoát chương trình nếu không tìm thấy file dữ liệu gốc
    exit()
except Exception as e:
    print(f"Đã xảy ra lỗi khi tải dữ liệu: {e}")
    exit()

# Tạo ra bản copy của df để tính toán, tránh làm ảnh hưởng đến dataframe gốc
df_calc = df.copy()

# Lọc số cầu thủ có trên 900 phút thi đấu và lưu vào DataFrame mới
minutes_threshold = 900
df_calc_filtered = df_calc[df_calc['Minutes'] > minutes_threshold].copy()
print(f"Số cầu thủ có trên {minutes_threshold} phút thi đấu: {len(df_calc_filtered)}")

# Ghi danh sách cầu thủ đủ điều kiện ra file CSV mới trong thư mục csv
filtered_players_filename = "players_over_900_minutes.csv"
filtered_path = os.path.join(csv_dir, filtered_players_filename)
df_calc_filtered.to_csv(filtered_path, index=False, encoding='utf-8-sig')
print(f"Đã lưu danh sách cầu thủ đủ điều kiện vào {filtered_path} với {df_calc_filtered.shape[0]} dòng và {df_calc_filtered.shape[1]} cột.")

# --- Chuẩn bị dữ liệu cho Fuzzy Matching ---

# Hàm cắt ngắn tên cầu thủ thành 2 từ đầu tiên (để tăng độ chính xác khi so khớp tên)
def shorten_name(name):
    parts = name.strip().split()
    # Trả về 2 từ đầu tiên nếu tên có ít nhất 2 từ, ngược lại trả về toàn bộ tên
    return " ".join(parts[:2]) if len(parts) >= 2 else name

# Đọc lại danh sách cầu thủ đã lọc từ file CSV vừa tạo
# Điều này đảm bảo chúng ta làm việc với cùng một tập dữ liệu đã lọc
csv_file_filtered = os.path.join(csv_dir, filtered_players_filename)
try:
    df_players = pd.read_csv(csv_file_filtered)
except FileNotFoundError:
     print(f"Lỗi: Không tìm thấy file cầu thủ đã lọc tại {csv_file_filtered}")
     exit()


# Tạo ra danh sách tên rút gọn từ cột 'Player' để sử dụng cho fuzzy matching
# .str.strip() loại bỏ khoảng trắng ở đầu và cuối tên
player_names_shortened_list = [shorten_name(name) for name in df_players['Player'].str.strip()]

# Tạo ra dictionary để tra cứu phút thi đấu theo tên cầu thủ (tên đầy đủ)
player_minutes_dict = dict(zip(df_players['Player'].str.strip(), df_players['Minutes']))

# --- Cấu hình và khởi tạo Selenium WebDriver ---

# Định cấu hình trình duyệt Chrome chạy ở chế độ ẩn (headless)
options = Options() # Tạo một đối tượng Options để cấu hình trình duyệt Chrome
options.add_argument("--headless") # Chạy trình duyệt ở chế độ ẩn (không mở cửa sổ trình duyệt thật)
options.add_argument("--no-sandbox") # Tắt chế độ sandbox (bảo vệ) của Chrome.
options.add_argument("--disable-dev-shm-usage") # Yêu cầu Chrome không dùng /dev/shm (shared memory) làm nơi lưu trữ tạm thời.
options.add_argument("--disable-gpu") # Tắt tăng tốc phần cứng GPU (đôi khi cần thiết trong môi trường headless)


# Khởi tạo trình điều khiển Chrome (webdriver.Chrome)
# ChromeDriverManager().install() tự động tải và chỉ định đúng phiên bản ChromeDriver phù hợp với trình duyệt của bạn.
# options=options áp dụng tất cả cấu hình vừa khai báo.
try:
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    print("Đã khởi tạo Selenium WebDriver thành công.")
except Exception as e:
    print(f"Lỗi khi khởi tạo WebDriver: {e}")
    print("Vui lòng kiểm tra cài đặt Chrome và kết nối internet.")
    exit()


# --- Crawl dữ liệu chuyển nhượng ---

# Tạo danh sách các URL từ trang 1 đến 14 của danh sách chuyển nhượng Premier League mùa 2024-2025.
base_url = "https://www.footballtransfers.com/us/transfers/confirmed/2024-2025/uk-premier-league/"
urls_to_scrape = [f"{base_url}{i}" for i in range(1, 15)] # Crawl từ trang 1 đến trang 14

# Tạo danh sách rỗng để lưu thông tin cầu thủ khớp (Tên cầu thủ, Giá trị chuyển nhượng)
transfer_data = []

# Duyệt qua từng URL trong danh sách cần crawl
for url in urls_to_scrape:
    try:
        driver.get(url) # Dùng Selenium WebDriver để mở trang web
        print(f"Đang crawl dữ liệu từ: {url}")

        # Đợi bảng chuyển nhượng xuất hiện (tối đa 10 giây).
        # Điều này đảm bảo trang web đã tải xong dữ liệu cần thiết trước khi tìm kiếm phần tử.
        wait = WebDriverWait(driver, 10)
        table = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "transfer-table"))
        )

        # Tìm tất cả các dòng (<tr>) trong bảng
        rows = table.find_elements(By.TAG_NAME, "tr")

        # Lặp qua từng dòng trong bảng để lấy dữ liệu
        for row in rows:
            # Tìm tất cả các ô dữ liệu (<td>) trong dòng hiện tại
            cols = row.find_elements(By.TAG_NAME, "td")

            # Kiểm tra xem dòng có đủ cột dữ liệu cần thiết không
            if cols and len(cols) >= 2: # Cần ít nhất 2 cột (Tên cầu thủ, Giá trị chuyển nhượng)

                # Lấy tên cầu thủ từ cột đầu tiên (index 0)
                # .text lấy nội dung văn bản, .strip() loại bỏ khoảng trắng dư thừa
                # .split("\n")[0] xử lý trường hợp tên có xuống dòng
                player_name_scraped = cols[0].text.strip().split("\n")[0].strip()

                # Rút gọn tên cầu thủ vừa crawl được để so sánh với danh sách tên rút gọn
                shortened_player_name_scraped = shorten_name(player_name_scraped)

                # Lấy giá trị chuyển nhượng từ cột cuối cùng (index -1)
                # Kiểm tra độ dài cột để tránh lỗi index out of range
                transfer_value_scraped = cols[-1].text.strip() if len(cols) >= 3 else "N/A"

                # --- Thực hiện Fuzzy Matching ---

                # Tìm tên cầu thủ trong danh sách player_names_shortened_list khớp nhất với tên cầu thủ vừa crawl được
                # scorer=fuzz.token_sort_ratio: sử dụng phương pháp so sánh dựa trên sắp xếp token
                # process.extractOne trả về (tên khớp nhất, điểm tương đồng) hoặc None
                best_match = process.extractOne(shortened_player_name_scraped, player_names_shortened_list, scorer=fuzz.token_sort_ratio)

                # Kiểm tra xem có kết quả khớp và điểm tương đồng có đủ cao không (ví dụ >= 85)
                similarity_threshold = 85
                if best_match and best_match[1] >= similarity_threshold:
                    # Lấy tên cầu thủ khớp nhất từ danh sách của chúng ta
                    matched_name_shortened = best_match[0]

                    # Tìm tên đầy đủ tương ứng với tên rút gọn đã khớp (tùy chọn, nếu cần tên đầy đủ)
                    # Trong trường hợp này, chúng ta lưu tên cầu thủ đã crawl được (player_name_scraped)
                    # và giá trị chuyển nhượng (transfer_value_scraped)

                    # Thêm thông tin cầu thủ khớp vào danh sách transfer_data
                    transfer_data.append([player_name_scraped, transfer_value_scraped])
                    # print(f"Đã khớp: {player_name_scraped} (Điểm: {best_match[1]})") # Có thể bỏ ghi log này nếu không cần

    except Exception as e:
        print(f"Đã xảy ra lỗi khi crawl hoặc xử lý URL {url}: {e}")
        # Tiếp tục vòng lặp để thử URL tiếp theo

# Đóng WebDriver sau khi hoàn thành crawl
driver.quit()
print("Đã đóng Selenium WebDriver.")

# --- Lưu kết quả vào file CSV ---

# Kiểm tra xem có dữ liệu chuyển nhượng nào được tìm thấy không
if transfer_data:
    # Tạo DataFrame từ danh sách transfer_data
    df_tv = pd.DataFrame(transfer_data, columns=['Player', 'Price'])

    # Lưu DataFrame vào file player_transfer_fee.csv trong thư mục csv
    transfer_fee_filename = "player_transfer_fee.csv"
    transfer_fee_path = os.path.join(csv_dir, transfer_fee_filename)
    df_tv.to_csv(transfer_fee_path, index=False, encoding='utf-8-sig')
    print(f"Kết quả giá trị chuyển nhượng đã được lưu vào '{transfer_fee_path}'")
else:
    print("Không tìm thấy cầu thủ nào khớp với danh sách chuyển nhượng.")

# --- Kết thúc ---
print("\nQuá trình xử lý hoàn tất.")
