import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from fuzzywuzzy import fuzz, process
import sys # Import sys để thoát chương trình một cách an toàn khi gặp lỗi nghiêm trọng

# --- Hằng số và Cấu hình ---
# Thư mục gốc nơi các file sẽ được lưu vào
BASE_DIR = r"C:\Users\84353\OneDrive\Desktop\BTL1_Python"

# Đường dẫn đến directory csv
CSV_DIR = os.path.join(BASE_DIR, "csv")

# Đường dẫn đến file result.csv (chứa thông tin cầu thủ từ nguồn khác)
RESULT_PATH = os.path.join(CSV_DIR, "result.csv")

# Base URL cần lấy dữ liệu Estimated Transfer Value (ETV)
BASE_URL = "https://www.footballtransfers.com/us/players/uk-premier-league/"
# Danh sách các URL cần crawl (từ trang 1 đến trang 22)
PAGE_NUMBERS = range(1, 23)
URLS = [f"{BASE_URL}{i}" for i in PAGE_NUMBERS]

# Ngưỡng so khớp mờ (fuzzy matching) cho tên cầu thủ
FUZZY_MATCH_THRESHOLD = 80

# --- Hàm Hỗ trợ ---
# Hàm để thu ngắn tên nhằm tăng độ chính xác của thư viện fuzzywuzzy
# Các trường hợp cụ thể được xử lý trước, sau đó là quy tắc chung.
def shorten_name(name):
    """Thu ngắn tên cầu thủ để so khớp mờ tốt hơn."""
    name = name.strip()
    # Xử lý các trường hợp tên đặc biệt
    if name == "Manuel Ugarte Ribeiro": return "Manuel Ugarte"
    elif name == "Igor Júlio": return "Igor"
    elif name == "Igor Thiago": return "Thiago"
    elif name == "Felipe Morato": return "Morato"
    elif name == "Nathan Wood-Gordon": return "Nathan Wood"
    elif name == "Bobby Reid": return "Bobby Cordova-Reid"
    elif name == "J. Philogene": return "Jaden Philogene Bidace"

    # Nếu tên dài quá 3 từ thì chỉ lấy từ đầu tiên và từ cuối cùng
    parts = name.split(" ")
    return parts[0] + " " + parts[-1] if len(parts) >= 3 else name

# --- Hàm Chuẩn bị Dữ liệu ---
def load_player_data(file_path):
    """Tải dữ liệu cầu thủ từ CSV và chuẩn bị các từ điển để so khớp."""
    print(f"Đang tải dữ liệu cầu thủ từ {file_path}")
    try:
        df_players = pd.read_csv(file_path)
        # Làm sạch tên cầu thủ từ result.csv
        df_players['Player'] = df_players['Player'].astype(str).str.strip()
        df_players['Position'] = df_players['Position'].astype(str).str.strip()

        # Áp dụng hàm thu ngắn tên và tạo các từ điển tra cứu
        shortened_names = df_players['Player'].apply(shorten_name)

        player_positions = dict(zip(shortened_names, df_players['Position']))
        player_original_names = dict(zip(shortened_names, df_players['Player']))
        player_names_shortened_list = list(player_positions.keys())

        print(f"Đã tải {len(df_players)} cầu thủ.")
        return player_positions, player_original_names, player_names_shortened_list

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp đầu vào tại {file_path}")
        sys.exit(1) # Thoát nếu không tìm thấy tệp đầu vào
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu cầu thủ: {str(e)}")
        sys.exit(1) # Thoát nếu có lỗi khác khi tải

# --- Hàm Thiết lập Selenium ---
def setup_webdriver():
    """Thiết lập và trả về Selenium Chrome WebDriver đã cấu hình."""
    print("Đang thiết lập WebDriver...")
    options = Options()
    options.add_argument("--headless")           # Chạy ở chế độ ẩn (không hiển thị cửa sổ trình duyệt)
    options.add_argument("--no-sandbox")         # Bỏ qua mô hình bảo mật của hệ điều hành (đôi khi cần thiết)
    options.add_argument("--disable-dev-shm-usage") # Khắc phục vấn đề tài nguyên hạn chế
    options.add_argument("--disable-gpu")        # Áp dụng cho hệ điều hành Windows đôi khi

    try:
        # Tải xuống và cài đặt ChromeDriver nếu cần, sau đó khởi tạo WebDriver
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        print("Thiết lập WebDriver hoàn tất.")
        return driver
    except Exception as e:
        print(f"Lỗi khi khởi tạo WebDriver: {str(e)}")
        sys.exit(1) # Thoát nếu WebDriver không khởi tạo được

# --- Hàm Thu thập và Xử lý Dữ liệu cho Một Trang ---
def scrape_and_process_page(driver, url, player_positions, player_original_names, player_names_shortened_list):
    """Thu thập dữ liệu từ một URL và so khớp cầu thủ."""
    # Khởi tạo danh sách để lưu dữ liệu cho từng vị trí trên trang hiện tại
    page_data_gk = []
    page_data_df = []
    page_data_mf = []
    page_data_fw = []

    try:
        driver.get(url)
        print(f"Đang xử lý: {url}")

        # Chờ bảng cầu thủ tải xong (chờ phần tử có class "similar-players-table" xuất hiện)
        wait = WebDriverWait(driver, 15) # Tăng thời gian chờ lên một chút
        table = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "similar-players-table"))
        )

        # Tìm tất cả các hàng (tr) trong bảng
        rows = table.find_elements(By.TAG_NAME, "tr")

        # Duyệt qua từng hàng
        for row in rows:
            # Tìm tất cả các ô dữ liệu (td) trong hàng
            cols = row.find_elements(By.TAG_NAME, "td")

            # Đảm bảo có đủ cột (ít nhất là tên cầu thủ và ETV)
            if cols and len(cols) >= 3:
                # Trích xuất tên cầu thủ và ETV từ dữ liệu thu thập được
                # Lấy phần đầu tiên trước dấu xuống dòng và loại bỏ khoảng trắng
                player_name_scraped = cols[1].text.strip().split("\n")[0].strip()
                etv = cols[-1].text.strip() # ETV thường nằm ở cột cuối cùng

                # Thu ngắn tên cầu thủ đã thu thập để so khớp
                shortened_player_name_scraped = shorten_name(player_name_scraped)

                # So khớp mờ với danh sách tên cầu thủ đã thu ngắn từ result.csv
                best_match = process.extractOne(
                    shortened_player_name_scraped,
                    player_names_shortened_list,
                    scorer=fuzz.token_sort_ratio # Sử dụng tỷ lệ sắp xếp token để so khớp
                )

                # Nếu tìm thấy kết quả so khớp tốt (điểm >= ngưỡng)
                if best_match and best_match[1] >= FUZZY_MATCH_THRESHOLD:
                    matched_shortened_name = best_match[0] # Lấy tên đã thu ngắn được so khớp

                    # Lấy tên gốc và vị trí từ các từ điển tra cứu
                    original_name = player_original_names.get(matched_shortened_name, matched_shortened_name)
                    position = player_positions.get(matched_shortened_name, "Unknown")

                    # Thêm dữ liệu vào danh sách phù hợp dựa trên vị trí
                    if "GK" in position:
                        page_data_gk.append([original_name, position, etv])
                    elif position.startswith("DF"): # Hậu vệ
                        page_data_df.append([original_name, position, etv])
                    elif position.startswith("MF"): # Tiền vệ
                        page_data_mf.append([original_name, position, etv])
                    elif position.startswith("FW"): # Tiền đạo
                        page_data_fw.append([original_name, position, etv])
                    # else: # Tùy chọn: xử lý các vị trí không khớp nếu cần
                    #     print(f"Cảnh báo: Vị trí '{position}' cho cầu thủ '{original_name}' không được nhận dạng.")

    except Exception as e:
        print(f"Lỗi khi xử lý trang {url}: {str(e)}")
        # Tiếp tục sang trang tiếp theo ngay cả khi một trang bị lỗi

    # Trả về dữ liệu đã thu thập cho trang này, giữ nguyên thứ tự nhóm vị trí
    return page_data_gk, page_data_df, page_data_mf, page_data_fw

# --- Luồng Thực thi Chính ---
if __name__ == "__main__":
    # 1. Tải và chuẩn bị dữ liệu cầu thủ ban đầu
    player_positions, player_original_names, player_names_shortened_list = load_player_data(RESULT_PATH)

    # Kiểm tra xem có dữ liệu cầu thủ để so khớp hay không
    if not player_names_shortened_list:
        print("Không có dữ liệu cầu thủ nào được tải để so khớp. Đang thoát.")
        sys.exit(0) # Thoát một cách an toàn nếu không tìm thấy cầu thủ

    # 2. Thiết lập WebDriver
    driver = setup_webdriver()

    # Khởi tạo danh sách để tích lũy dữ liệu từ tất cả các trang
    all_data_gk = []
    all_data_df = []
    all_data_mf = []
    all_data_fw = []

    try:
        # 3. Lặp qua các URL và thu thập dữ liệu
        for url in URLS:
            # Thu thập và xử lý dữ liệu cho trang hiện tại
            gk, df, mf, fw = scrape_and_process_page(
                driver,
                url,
                player_positions,
                player_original_names,
                player_names_shortened_list
            )
            # Mở rộng danh sách chính với dữ liệu từ trang hiện tại
            all_data_gk.extend(gk)
            all_data_df.extend(df)
            all_data_mf.extend(mf)
            all_data_fw.extend(fw)

    finally:
        # Đảm bảo đóng driver ngay cả khi có lỗi xảy ra trong quá trình thu thập
        if driver:
            driver.quit()
            print("WebDriver đã đóng.")

    # 4. Kết hợp và Lưu Dữ liệu
    # Kết hợp dữ liệu giữ nguyên thứ tự nhóm ban đầu (GK, DF, MF, FW)
    combined_all_data = all_data_gk + all_data_df + all_data_mf + all_data_fw

    if combined_all_data:
        # Tạo DataFrame từ dữ liệu đã kết hợp
        df_all = pd.DataFrame(combined_all_data, columns=['Cầu thủ', 'Vị trí', 'Giá trị'])

        # Tạo thư mục CSV nếu nó chưa tồn tại
        os.makedirs(CSV_DIR, exist_ok=True)

        # Định nghĩa đường dẫn tệp đầu ra
        combined_path = os.path.join(CSV_DIR, "all_estimate_transfer_fee.csv")

        # Lưu vào tệp CSV
        df_all.to_csv(combined_path, index=False, encoding='utf-8')

        print(f"Đã lưu tất cả kết quả thành công vào '{combined_path}'")
    else:
        print("Không có dữ liệu nào được thu thập cho bất kỳ cầu thủ nào.")
