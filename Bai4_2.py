import pandas as pd
import numpy as np
import os
import re
from fuzzywuzzy import process, fuzz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score # Mặc dù không thực sự cần thiết cho tệp CSV đầu ra cuối cùng, giữ lại để đầy đủ

# --- Cấu hình ---
# Thư mục gốc nơi các file sẽ được lưu vào
base_dir = r"C:\Users\84353\OneDrive\Desktop\BTL1_Python"
# Đường dẫn đến thư mục csv
csv_dir = os.path.join(base_dir, "csv")
# Đường dẫn đến file result.csv (chứa thông tin cầu thủ từ nguồn khác)
result_path = os.path.join(csv_dir, "result.csv")
# Đường dẫn đến file all_estimate_transfer_fee.csv (chứa ETV đã thu thập)
etv_path = os.path.join(csv_dir, 'all_estimate_transfer_fee.csv')
# Đường dẫn đến file CSV đầu ra cuối cùng
output_path = os.path.join(csv_dir, 'ml_transfer_values_linear.csv')

# Định nghĩa các cột chuẩn cho tệp CSV đầu ra
standard_output_columns = [
    'Player', 'Team', 'Nation', 'Position', 'Actual_Transfer_Value_M', 'Predicted_Transfer_Value_M'
]

# Cấu hình đặc trưng và bộ lọc cho từng vị trí
positions_config = {
    'GK': { # Thủ môn
        'position_filter': 'GK', # Bộ lọc vị trí chính
        'features': [ # Các đặc trưng được sử dụng cho mô hình
            'Save%', 'CS%', 'GA90', 'Minutes', 'Age', 'PK Save%', 'Team', 'Nation'
        ],
        'important_features': ['Save%', 'CS%', 'PK Save%'] # Các đặc trưng quan trọng hơn (sẽ được tăng trọng số)
    },
    'DF': { # Hậu vệ
        'position_filter': 'DF',
        'features': [
            'Tkl', 'TklW', 'Int', 'Blocks', 'Recov', 'Minutes', 'Team', 'Age', 'Nation', 'Aerl Won%',
            'Aerl Won', 'Cmp', 'Cmp%', 'PrgP', 'LongCmp%', 'Carries', 'Touches', 'Dis', 'Mis'
        ],
        'important_features': ['Tkl', 'TklW', 'Int', 'Blocks', 'Aerl Won%', 'Aerl Won', 'Recov']
    },
    'MF': { # Tiền vệ
        'position_filter': 'MF',
        'features': [
            'Cmp%', 'KP', 'PPA', 'PrgP', 'Tkl', 'Ast', 'SCA', 'Touches', 'Minutes', 'Team', 'Age', 'Nation',
            'Pass into 1_3', 'xAG', 'Carries 1_3', 'ProDist', 'Rec', 'Mis', 'Dis'
        ],
        'important_features': ['KP', 'PPA', 'PrgP', 'SCA', 'xAG', 'Pass into 1_3', 'Carries 1_3']
    },
    'FW': { # Tiền đạo
        'position_filter': 'FW',
        'features': [
            'Gls', 'Ast', 'Gls per 90', 'xG per 90', 'SoT%', 'G per Sh', 'SCA90', 'GCA90',
            'PrgC', 'Carries 1_3', 'Aerl Won%', 'Team', 'Age', 'Minutes'
        ],
        'important_features': ['Gls', 'Ast', 'Gls per 90', 'xG per 90', 'SCA90', 'GCA90']
    }
}

# --- Hàm Hỗ trợ (Giống với mã gốc để đảm bảo logic/đầu ra tương tự) ---
def shorten_name(name):
    """Thu ngắn tên cầu thủ (lấy tối đa 2 từ đầu) để so khớp mờ."""
    if not isinstance(name, str):
        return ""
    parts = name.strip().split()
    return " ".join(parts[:2]) if len(parts) >= 2 else name

def parse_etv(etv_text):
    """Phân tích chuỗi ETV (ví dụ: '€20M', '£500K') thành giá trị số (USD)."""
    if pd.isna(etv_text) or etv_text in ["N/A", ""]:
        return np.nan
    try:
        # Loại bỏ ký hiệu tiền tệ và khoảng trắng, chuyển thành chữ hoa
        etv_text = re.sub(r'[€£]', '', etv_text).strip().upper()
        # Xác định hệ số nhân (triệu, nghìn)
        multiplier = 1000000 if 'M' in etv_text else 1000 if 'K' in etv_text else 1
        # Loại bỏ ký hiệu 'M', 'K' và chuyển thành số thực, sau đó nhân với hệ số
        value = float(re.sub(r'[MK]', '', etv_text)) * multiplier
        return value
    except (ValueError, TypeError):
        # Trả về NaN nếu không thể phân tích
        return np.nan

def fuzzy_match_name(name, choices, score_threshold=90):
    """Thực hiện so khớp mờ tên cầu thủ với danh sách lựa chọn."""
    if not isinstance(name, str):
        return None, None

    # Thu ngắn tên và chuyển sang chữ thường để so khớp
    shortened_name = shorten_name(name).lower()
    shortened_choices = [shorten_name(c).lower() for c in choices if isinstance(c, str)]

    # Sử dụng fuzzywuzzy để tìm kết quả so khớp tốt nhất
    match = process.extractOne(
        shortened_name,
        shortened_choices,
        scorer=fuzz.token_sort_ratio, # Sử dụng tỷ lệ sắp xếp token
        score_cutoff=score_threshold # Chỉ trả về kết quả nếu điểm đạt ngưỡng
    )

    if match is not None:
        # Tìm chỉ số của tên đã so khớp trong danh sách gốc (không thu ngắn)
        try:
            matched_idx = shortened_choices.index(match[0])
            # Trả về tên gốc và điểm so khớp
            return choices[matched_idx], match[1]
        except ValueError:
             # Xử lý trường hợp hiếm khi tên thu ngắn khớp nhưng tên gốc không tìm thấy
             return None, None


    # Trả về None nếu không tìm thấy kết khớp tốt
    return None, None

# --- Tải Dữ liệu ---
try:
    # Tải dữ liệu cầu thủ từ result.csv
    df_result_all = pd.read_csv(result_path)
    # Tải dữ liệu ETV từ all_estimate_transfer_fee.csv
    df_etv_all = pd.read_csv(etv_path)
except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy tệp dữ liệu - {e}")
    # Thoát chương trình nếu không tìm thấy tệp cần thiết
    exit()

# --- Vòng lặp xử lý chính ---
all_results_list = [] # Danh sách để lưu kết quả dự đoán cho tất cả các vị trí
all_unmatched_players = [] # Danh sách để lưu các cầu thủ không khớp

# Lấy danh sách tên cầu thủ duy nhất từ dữ liệu ETV để so khớp mờ
etv_player_names = df_etv_all['Cầu thủ'].dropna().unique().tolist() # Sử dụng cột 'Cầu thủ' từ file ETV

# Duyệt qua từng vị trí và cấu hình tương ứng
for position, config in positions_config.items():
    print(f"\nĐang xử lý {position}...")

    # Lọc dữ liệu cầu thủ theo vị trí chính
    # Đảm bảo cột 'Position' tồn tại trước khi tách
    if 'Position' not in df_result_all.columns:
        print(f"Lỗi: Cột 'Position' không tìm thấy trong {result_path}.")
        continue # Bỏ qua vị trí này nếu cột 'Position' không có

    df_position_data = df_result_all.copy()
    # Chuyển cột 'Position' sang chuỗi trước khi tách để xử lý các kiểu dữ liệu khác
    df_position_data['Primary_Position'] = df_position_data['Position'].astype(str).str.split(r'[,/]').str[0].str.strip()
    # Lọc theo vị trí chính đã cấu hình
    df_position_data = df_position_data[
        df_position_data['Primary_Position'].str.upper() == config['position_filter'].upper()
    ].copy() # Sử dụng copy() để tránh SettingWithCopyWarning

    if df_position_data.empty:
        print(f"Không tìm thấy cầu thủ {position} trong dữ liệu kết quả.")
        continue # Bỏ qua vị trí này nếu không có cầu thủ

    # --- So khớp mờ và Phân tích ETV ---
    # Thêm các cột mới để lưu kết quả so khớp và ETV
    df_position_data['Matched_Name'] = None
    df_position_data['Match_Score'] = None
    df_position_data['ETV'] = np.nan # Lưu ETV thô (số)

    # Áp dụng so khớp mờ và phân tích ETV cho từng hàng
    for idx, row in df_position_data.iterrows():
        # Thực hiện so khớp mờ cho tên cầu thủ trong hàng hiện tại
        matched_name, score = fuzzy_match_name(row['Player'], etv_player_names)
        if matched_name:
            # Nếu tìm thấy tên khớp, lưu tên và điểm so khớp
            df_position_data.at[idx, 'Matched_Name'] = matched_name
            df_position_data.at[idx, 'Match_Score'] = score
            # Tìm hàng ETV tương ứng với tên đã khớp
            matched_etv_row = df_etv_all[df_etv_all['Cầu thủ'] == matched_name]
            if not matched_etv_row.empty:
                # Nếu tìm thấy ETV, phân tích giá trị và lưu
                etv_value = parse_etv(matched_etv_row['Giá trị'].iloc[0]) # Sử dụng cột 'Giá trị' từ file ETV
                df_position_data.at[idx, 'ETV'] = etv_value

    # Lọc ra các cầu thủ đã được so khớp thành công
    df_matched = df_position_data[df_position_data['Matched_Name'].notna()].copy()

    # Loại bỏ các bản sao dựa trên tên đã so khớp (giữ lại lần xuất hiện đầu tiên trong result.csv)
    df_matched = df_matched.drop_duplicates(subset='Matched_Name')

    # Xác định các cầu thủ không khớp từ danh sách ban đầu đã lọc theo vị trí (trước khi loại bỏ trùng lặp)
    unmatched_players_list = df_position_data[df_position_data['Matched_Name'].isna()]['Player'].dropna().tolist()
    if unmatched_players_list:
          print(f"Cầu thủ {position} không khớp: {len(unmatched_players_list)} cầu thủ không được khớp.")
          # print(unmatched_players_list) # Tùy chọn: in danh sách cầu thủ không khớp
          # Thêm các cầu thủ không khớp vào danh sách tổng
          all_unmatched_players.extend([(position, player) for player in unmatched_players_list])


    # --- Tiền xử lý và Kỹ thuật đặc trưng ---
    features = config['features'] # Lấy danh sách các đặc trưng cho vị trí này
    target = 'ETV' # Cột mục tiêu là ETV

    # Đảm bảo tất cả các đặc trưng cần thiết tồn tại trong df_matched; thêm vào với giá trị mặc định nếu thiếu
    for col in features:
          if col not in df_matched.columns:
              if col in ['Team', 'Nation']:
                   df_matched[col] = 'Unknown' # Điền 'Unknown' cho đặc trưng phân loại
              else:
                   df_matched[col] = np.nan # Điền NaN cho đặc trưng số (sẽ được điền sau)

    # Xử lý giá trị thiếu trong các đặc trưng (điền 'Unknown' cho phân loại, trung vị/0 cho số)
    numeric_features = [col for col in features if col not in ['Team', 'Nation']]
    categorical_features = [col for col in features if col in ['Team', 'Nation']] # Định nghĩa lại cho rõ ràng

    for col in numeric_features:
        # Chuyển sang dạng số, xử lý lỗi thành NaN, sau đó điền giá trị thiếu
        df_matched[col] = pd.to_numeric(df_matched[col], errors='coerce')
        median_value = df_matched[col].median()
        # Điền NaN bằng trung vị nếu có, nếu không thì điền 0
        df_matched[col] = df_matched[col].fillna(median_value if not pd.isna(median_value) else 0)

    for col in categorical_features:
        # Điền NaN bằng 'Unknown' cho đặc trưng phân loại
        df_matched[col] = df_matched[col].fillna('Unknown')

    # Áp dụng phép biến đổi log1p cho các đặc trưng số
    for col in numeric_features:
          # Cắt giá trị trước khi áp dụng log1p (như mã gốc)
          df_matched[col] = np.log1p(df_matched[col].clip(lower=0))

    # Áp dụng trọng số đặc trưng
    for col in config['important_features']:
        if col in df_matched.columns: # Kiểm tra lại để đảm bảo cột tồn tại
             df_matched[col] = df_matched[col] * 2.0 # Tăng trọng số cho các đặc trưng quan trọng
    if 'Minutes' in df_matched.columns:
        df_matched['Minutes'] = df_matched['Minutes'] * 1.5 # Tăng trọng số cho Minutes
    if 'Age' in df_matched.columns:
        df_matched['Age'] = df_matched['Age'] * 0.5 # Giảm trọng số cho Age

    # --- Chuẩn bị Dữ liệu cho Huấn luyện Mô hình ML ---
    # Tạo DataFrame được sử dụng để huấn luyện mô hình (phải có giá trị mục tiêu không rỗng)
    df_ml_train = df_matched.dropna(subset=[target]).copy()

    if df_ml_train.empty:
        print(f"Lỗi: Không có dữ liệu ETV hợp lệ cho {position} để huấn luyện mô hình.")
        # Không có dữ liệu hợp lệ để huấn luyện cho vị trí này, thêm các cầu thủ không khớp và tiếp tục
        continue

    # Chọn đặc trưng (X) và biến mục tiêu (y) cho tập huấn luyện
    X_train_full = df_ml_train[features]
    y_train_full = df_ml_train[target]

    # Chia tập dữ liệu huấn luyện (nếu kích thước cho phép)
    if len(df_ml_train) > 5: # Cần ít nhất vài mẫu để chia
        X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
    else:
        print(f"Cảnh báo: Không đủ dữ liệu cho {position} để chia tập huấn luyện/kiểm tra. Sử dụng toàn bộ dữ liệu có ETV để huấn luyện.")
        X_train, y_train = X_train_full, y_train_full
        X_test, y_test = pd.DataFrame(), pd.Series() # Tạo tập kiểm tra rỗng nếu không chia

    # --- Xây dựng và Huấn luyện Pipeline ---
    # Định nghĩa bộ tiền xử lý (giống mã gốc)
    # Bao gồm chuẩn hóa cho đặc trưng số và mã hóa one-hot cho đặc trưng phân loại
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features), # Chuẩn hóa đặc trưng số
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) # Mã hóa One-Hot cho đặc trưng phân loại
        ],
        remainder='passthrough' # Đảm bảo các cột khác được giữ nguyên (nếu có)
    )

    # Tạo pipeline (chuỗi các bước tiền xử lý và mô hình) (giống mã gốc)
    pipeline = Pipeline([
        ('preprocessor', preprocessor), # Bước tiền xử lý
        ('regressor', LinearRegression()) # Bước mô hình hồi quy tuyến tính
    ])

    # Huấn luyện pipeline trên tập huấn luyện
    pipeline.fit(X_train, y_train)

    # --- Dự đoán ---
    # Dự đoán trên *toàn bộ* tập dữ liệu df_matched (bao gồm cả những cầu thủ không có ETV ban đầu)
    # Mục đích là để có giá trị dự đoán cho tất cả các cầu thủ đã so khớp thành công,
    # ngay cả khi họ không có ETV trong tệp 'all_estimate_transfer_fee.csv'.
    # Cấu trúc của mã gốc ngụ ý rằng việc dự đoán xảy ra trên df_filtered,
    # tương ứng với df_matched trước khi loại bỏ các giá trị NaN ETV cho df_ml_train.
    # Kiểm tra lại mã gốc: df_filtered = df_result[df_result['Matched_Name'].notna()].copy()
    #                       df_ml = df_filtered.dropna(subset=[target]).copy()
    #                       X = df_ml[features] # được sử dụng cho train/test split
    #                       df_filtered['Predicted_Transfer_Value'] = pipeline.predict(df_filtered[features]) # Dự đoán trên df_filtered

    # Vì vậy, việc dự đoán thực sự là trên df_matched (tương ứng với df_filtered trong mã gốc)
    df_matched['Predicted_Transfer_Value'] = pipeline.predict(df_matched[features])

    # --- Xử lý sau dự đoán ---
    # Áp dụng cắt giá trị (clipping) để giới hạn giá trị dự đoán trong một khoảng hợp lý
    df_matched['Predicted_Transfer_Value'] = df_matched['Predicted_Transfer_Value'].clip(lower=100_000, upper=200_000_000)

    # Tính toán và làm tròn các cột đầu ra cuối cùng sang đơn vị triệu
    df_matched['Predicted_Transfer_Value_M'] = (df_matched['Predicted_Transfer_Value'] / 1_000_000).round(2)
    # Giá trị ETV thực tế (sẽ là NaN cho các cầu thủ không có ETV ban đầu)
    df_matched['Actual_Transfer_Value_M'] = (df_matched['ETV'] / 1_000_000).round(2)

    # Đảm bảo cột Vị trí chính xác trong phần kết quả
    df_matched['Position'] = position

    # Chỉ chọn các cột chuẩn cho đầu ra và giữ nguyên thứ tự của chúng
    result_df_position = df_matched[standard_output_columns].copy()

    # --- Lưu ý về bước expm1 trong mã gốc ---
    # Mã gốc có một phần áp dụng np.expm1 cho các đặc trưng số *sau khi* chọn standard_output_columns.
    # Nếu standard_output_columns không bao gồm các đặc trưng gốc (mà thực tế là không), bước này
    # sẽ không ảnh hưởng đến các cột trong tệp CSV đầu ra cuối cùng. Để đảm bảo đầu ra giống hệt,
    # bước này (có khả năng không hiệu quả) được bỏ qua ở đây vì nó không làm thay đổi tệp CSV cuối cùng.
    # Nếu mã gốc *có* bao gồm các đặc trưng đã biến đổi này trong đầu ra cuối cùng, phần này sẽ cần được thêm vào,
    # nhưng dựa trên `standard_output_columns`, dường như nó không được dự định cho tệp CSV cuối cùng.


    # Thêm DataFrame đã xử lý cho vị trí này vào danh sách
    all_results_list.append(result_df_position)

# --- Kết hợp và Lưu Kết quả Cuối cùng ---
if all_results_list:
    # Nối tất cả các DataFrame của từng vị trí lại với nhau
    combined_results_df = pd.concat(all_results_list, ignore_index=True)

    # Sắp xếp kết quả cuối cùng theo Predicted_Transfer_Value_M giảm dần
    combined_results_df = combined_results_df.sort_values(by='Predicted_Transfer_Value_M', ascending=False)

    # Lưu DataFrame cuối cùng vào tệp CSV
    combined_results_df.to_csv(output_path, index=False)
    print(f"\nGiá trị ước tính của các cầu thủ đã được lưu vào '{output_path}'")
else:
    print("\nKhông có dữ liệu hợp lệ để tạo file kết quả.")


# Tùy chọn: In danh sách tất cả các cầu thủ không khớp từ tất cả các vị trí
if all_unmatched_players:
    print("\nDanh sách cầu thủ không khớp trên tất cả các vị trí:")
    # Loại bỏ các bản sao khỏi danh sách cầu thủ không khớp dựa trên tên và vị trí
    unique_unmatched = list(set(all_unmatched_players))
    for pos, player in unique_unmatched:
        print(f"- {player} ({pos})")
