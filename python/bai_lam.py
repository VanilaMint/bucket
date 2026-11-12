import os
import pandas as pd
import numpy as np


def main():
    # Hỗ trợ đọc từ thư mục du_lieu/ (theo đề bài) hoặc thư mục hiện tại
    data_dir = "du_lieu" if os.path.exists("du_lieu/khach_hang.csv") else "."

    # =========================================================================
    # Bước 1: Đọc 6 file dữ liệu với dtype=str, keep_default_na=False; in shape
    # =========================================================================
    khach_hang = pd.read_csv(f"{data_dir}/khach_hang.csv", dtype=str, keep_default_na=False)
    chi_nhanh = pd.read_csv(f"{data_dir}/chi_nhanh.csv", dtype=str, keep_default_na=False)
    tai_khoan = pd.read_csv(f"{data_dir}/tai_khoan.csv", dtype=str, keep_default_na=False)
    giao_dich = pd.read_csv(f"{data_dir}/giao_dich.csv", dtype=str, keep_default_na=False)
    the = pd.read_csv(f"{data_dir}/the.csv", dtype=str, keep_default_na=False)
    ty_gia = pd.read_csv(f"{data_dir}/ty_gia.csv", dtype=str, keep_default_na=False)

    print("Bước 1 - Kích thước ban đầu:")
    for name, df in [
        ("khach_hang", khach_hang),
        ("chi_nhanh", chi_nhanh),
        ("tai_khoan", tai_khoan),
        ("giao_dich", giao_dich),
        ("the", the),
        ("ty_gia", ty_gia),
    ]:
        print(f"  {name}: {df.shape}")

    # =========================================================================
    # Bước 2: Làm sạch văn bản (Q1)
    # - Bỏ khoảng trắng đầu/cuối mọi cột chữ ở tất cả các bảng
    # - ma_kh (ở mọi file) viết HOA
    # - trang_thai_kh, trang_thai (giao dịch, thẻ) viết HOA
    # =========================================================================
    for df in [khach_hang, chi_nhanh, tai_khoan, giao_dich, the, ty_gia]:
        for col in df.columns:
            df[col] = df[col].str.strip()

    for df in [khach_hang, tai_khoan, the]:
        df["ma_kh"] = df["ma_kh"].str.upper()

    khach_hang["trang_thai_kh"] = khach_hang["trang_thai_kh"].str.upper()
    giao_dich["trang_thai"] = giao_dich["trang_thai"].str.upper()
    the["trang_thai"] = the["trang_thai"].str.upper()

    # =========================================================================
    # Bước 3: Khách hàng: lọc HOAT_DONG, nối chi nhánh, tính tuổi (Q2 - Q4)
    # =========================================================================
    # Q2: Chỉ giữ trang_thai_kh = HOAT_DONG
    kh_active = khach_hang[khach_hang["trang_thai_kh"] == "HOAT_DONG"].drop_duplicates(subset=["ma_kh"]).copy()
    print(f"Bước 3 - Khách hàng đang hoạt động: {kh_active.shape}")

    # Q3: Lấy ten_cn, vung theo ma_cn; không có trong danh mục -> UNMAPPED
    kh_active = kh_active.merge(chi_nhanh[["ma_cn", "ten_cn", "vung"]], on="ma_cn", how="left")
    kh_active["ten_cn"] = kh_active["ten_cn"].fillna("UNMAPPED")
    kh_active["vung"] = kh_active["vung"].fillna("UNMAPPED")

    # Q4: TUOI = 2026 - năm sinh. Ngày sinh trống/sai -> để trống
    nam_sinh = pd.to_datetime(kh_active["ngay_sinh"], format="%Y-%m-%d", errors="coerce").dt.year
    kh_active["TUOI"] = np.where(
        nam_sinh.notna(),
        (2026 - nam_sinh).fillna(0).astype(int).astype(str),
        "",
    )

    # =========================================================================
    # Bước 4: Tài khoản × tỷ giá, gom theo khách hàng (Q5)
    # =========================================================================
    tk_tg = tai_khoan.merge(ty_gia, on="loai_tien", how="left")
    tk_tg["so_du_vnd"] = tk_tg["so_du"].astype(float) * tk_tg["ty_gia"].astype(float)

    tk_agg = tk_tg.groupby("ma_kh", as_index=False).agg(
        SO_TAI_KHOAN=("so_tk", "count"),
        TONG_SO_DU_VND=("so_du_vnd", "sum"),
    )
    print(f"Bước 4 - Gom tài khoản theo khách hàng: {tk_agg.shape}")

    # =========================================================================
    # Bước 5: Giao dịch hợp lệ (Q6)
    # - Không trùng ma_gd (giữ 1 dòng)
    # - so_tien đổi được sang số (bỏ dấu phẩy; ô trống bị loại)
    # - Ngày thuộc 2024 (đọc cả 2 định dạng YYYY-MM-DD và DD/MM/YYYY)
    # - trang_thai = OK
    # - so_tk có trong tai_khoan (tài khoản lạ bị loại)
    # =========================================================================
    gd_dedup = giao_dich.drop_duplicates(subset=["ma_gd"], keep="first").copy()
    print(f"Bước 5 - Giao dịch sau khi bỏ trùng ma_gd: {gd_dedup.shape}")

    gd_dedup["so_tien_num"] = pd.to_numeric(
        gd_dedup["so_tien"].str.replace(",", "", regex=False), errors="coerce"
    )

    ngay_1 = pd.to_datetime(gd_dedup["ngay"], format="%Y-%m-%d", errors="coerce")
    ngay_2 = pd.to_datetime(gd_dedup["ngay"], format="%d/%m/%Y", errors="coerce")
    gd_dedup["ngay_parsed"] = ngay_1.fillna(ngay_2)

    gd_valid = gd_dedup[
        gd_dedup["so_tien_num"].notna()
        & (gd_dedup["ngay_parsed"].dt.year == 2024)
        & (gd_dedup["trang_thai"] == "OK")
    ].copy()

    # Nối inner với tài khoản để loại tài khoản lạ và lấy ma_kh, ty_gia
    gd_valid = gd_valid.merge(
        tk_tg[["so_tk", "ma_kh", "ty_gia"]], on="so_tk", how="inner"
    )
    gd_valid["so_tien_vnd"] = gd_valid["so_tien_num"] * gd_valid["ty_gia"].astype(float)
    print(f"Bước 5 - Giao dịch hợp lệ năm 2024: {gd_valid.shape}")

    # =========================================================================
    # Bước 6: Gom giao dịch theo khách hàng (Q7); tìm kênh nhiều nhất
    # =========================================================================
    gd_agg = gd_valid.groupby("ma_kh", as_index=False).agg(
        SO_GD_OK=("ma_gd", "count"),
        TONG_TIEN_GD_VND=("so_tien_vnd", "sum"),
        GD_LON_NHAT_VND=("so_tien_vnd", "max"),
    )

    kenh_counts = (
        gd_valid.groupby(["ma_kh", "kenh"])
        .size()
        .reset_index(name="so_luong")
        .sort_values(by=["ma_kh", "so_luong", "kenh"], ascending=[True, False, True])
    )
    kenh_max = (
        kenh_counts.drop_duplicates(subset=["ma_kh"], keep="first")[["ma_kh", "kenh"]]
        .rename(columns={"kenh": "KENH_NHIEU_NHAT"})
    )
    print(f"Bước 6 - Gom giao dịch theo khách hàng: {gd_agg.shape}")

    # =========================================================================
    # Bước 7: Đếm thẻ ACTIVE theo khách hàng (Q8)
    # =========================================================================
    the_agg = (
        the[the["trang_thai"] == "ACTIVE"]
        .groupby("ma_kh", as_index=False)
        .agg(SO_THE_ACTIVE=("so_the", "count"))
    )
    print(f"Bước 7 - Gom thẻ ACTIVE theo khách hàng: {the_agg.shape}")

    # =========================================================================
    # Bước 8: Lắp báo cáo, điền mặc định 0, xếp loại (Q9)
    # =========================================================================
    ket_qua = kh_active.rename(
        columns={
            "ma_kh": "MA_KH",
            "ten_kh": "TEN_KH",
            "gioi_tinh": "GIOI_TINH",
            "tinh_thanh": "TINH_THANH",
            "ma_cn": "MA_CN",
            "ten_cn": "TEN_CN",
            "vung": "VUNG",
        }
    )[["MA_KH", "TEN_KH", "GIOI_TINH", "TUOI", "TINH_THANH", "MA_CN", "TEN_CN", "VUNG"]].copy()

    ket_qua = ket_qua.merge(tk_agg.rename(columns={"ma_kh": "MA_KH"}), on="MA_KH", how="left", validate="1:1")
    ket_qua = ket_qua.merge(gd_agg.rename(columns={"ma_kh": "MA_KH"}), on="MA_KH", how="left", validate="1:1")
    ket_qua = ket_qua.merge(kenh_max.rename(columns={"ma_kh": "MA_KH"}), on="MA_KH", how="left", validate="1:1")
    ket_qua = ket_qua.merge(the_agg.rename(columns={"ma_kh": "MA_KH"}), on="MA_KH", how="left", validate="1:1")

    ket_qua["SO_TAI_KHOAN"] = ket_qua["SO_TAI_KHOAN"].fillna(0).astype(int)
    ket_qua["TONG_SO_DU_VND"] = ket_qua["TONG_SO_DU_VND"].fillna(0.0).round(2)
    ket_qua["SO_GD_OK"] = ket_qua["SO_GD_OK"].fillna(0).astype(int)
    ket_qua["TONG_TIEN_GD_VND"] = ket_qua["TONG_TIEN_GD_VND"].fillna(0).round(0).astype(np.int64)
    ket_qua["GD_LON_NHAT_VND"] = ket_qua["GD_LON_NHAT_VND"].fillna(0).round(0).astype(np.int64)
    ket_qua["KENH_NHIEU_NHAT"] = ket_qua["KENH_NHIEU_NHAT"].fillna("")
    ket_qua["SO_THE_ACTIVE"] = ket_qua["SO_THE_ACTIVE"].fillna(0).astype(int)

    # Q9: Xếp loại theo TONG_SO_DU_VND
    conditions = [
        ket_qua["TONG_SO_DU_VND"] >= 1_000_000_000,
        ket_qua["TONG_SO_DU_VND"] >= 100_000_000,
        ket_qua["TONG_SO_DU_VND"] > 0,
    ]
    choices = ["PLATINUM", "GOLD", "STANDARD"]
    ket_qua["XEP_LOAI"] = np.select(conditions, choices, default="NO_BALANCE")

    # =========================================================================
    # Bước 9: Sắp xếp, định dạng cột, ghi ket_qua.csv (Q10)
    # =========================================================================
    cols_order = [
        "MA_KH",
        "TEN_KH",
        "GIOI_TINH",
        "TUOI",
        "TINH_THANH",
        "MA_CN",
        "TEN_CN",
        "VUNG",
        "SO_TAI_KHOAN",
        "TONG_SO_DU_VND",
        "SO_GD_OK",
        "TONG_TIEN_GD_VND",
        "GD_LON_NHAT_VND",
        "KENH_NHIEU_NHAT",
        "SO_THE_ACTIVE",
        "XEP_LOAI",
    ]
    ket_qua = (
        ket_qua[cols_order]
        .sort_values(by=["TONG_SO_DU_VND", "MA_KH"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ket_qua.to_csv("ket_qua.csv", index=False, encoding="utf-8", float_format="%.2f")
    print(f"Bước 9 - Đã ghi ket_qua.csv: {ket_qua.shape}")

    # =========================================================================
    # Bước 10: Tổng hợp theo vùng, ghi tong_hop_vung.csv
    # =========================================================================
    tong_hop_vung = (
        ket_qua.groupby("VUNG", as_index=False)
        .agg(
            SO_KH=("MA_KH", "count"),
            TONG_SO_DU_VND=("TONG_SO_DU_VND", "sum"),
            TONG_TIEN_GD_VND=("TONG_TIEN_GD_VND", "sum"),
        )
        .sort_values(by="VUNG", ascending=True)
        .reset_index(drop=True)
    )
    tong_hop_vung["TONG_SO_DU_VND"] = tong_hop_vung["TONG_SO_DU_VND"].round(2)
    tong_hop_vung["TB_SO_DU_MOI_KH"] = (
        tong_hop_vung["TONG_SO_DU_VND"] / tong_hop_vung["SO_KH"]
    ).round(2)

    tong_hop_vung.to_csv("tong_hop_vung.csv", index=False, encoding="utf-8", float_format="%.2f")
    print(f"Bước 10 - Đã ghi tong_hop_vung.csv: {tong_hop_vung.shape}")


if __name__ == "__main__":
    main()
