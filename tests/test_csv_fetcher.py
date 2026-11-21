"""CSV importer: headers, date formats, number formats and the VND-unit rule (no internet, tmp files)."""
import pandas as pd
import pytest

from stocklens.ingest.csv_files import CsvFetcher, _to_number
from stocklens.ingest.update import get_fetcher
from stocklens.ingest.vietnam import VietnamFetcher


def write(tmp_path, name, text):
    (tmp_path / name).write_text(text, encoding="utf-8")
    return CsvFetcher(tmp_path)


def test_english_headers_iso_dates(tmp_path):
    f = write(tmp_path, "FPT.csv", "date,open,high,low,close,volume\n2024-01-02,100.0,101,99,100.5,1000\n2024-01-03,100.5,102,100,101.5,2000\n")
    out = f.fetch_prices("FPT", "2024-01-01")
    assert list(out["close"]) == [100.5, 101.5]
    assert list(out["adj_close"]) == [100.5, 101.5]  # no adj column -> same as close
    assert out["date"].iloc[0] == pd.Timestamp("2024-01-02")


def test_vietnamese_headers_day_first_dates_and_full_vnd_converted(tmp_path):
    text = ("Ngày,Giá mở cửa,Cao nhất,Thấp nhất,Giá đóng cửa,Khối lượng\n"
            "05/03/2024,\"46,000\",\"46,500\",\"45,800\",\"46,200\",\"1,250,000\"\n"
            "06/03/2024,\"46,200\",\"47,000\",\"46,000\",\"46,900\",\"900,000\"\n")
    out = write(tmp_path, "fpt.csv", text).fetch_prices("FPT", "2024-01-01")  # lower-case file name works
    assert out["date"].iloc[0] == pd.Timestamp("2024-03-05")  # 5 March, not 3 May
    assert out["close"].iloc[0] == pytest.approx(46.2)  # stored in thousands of VND
    assert out["volume"].iloc[0] == 1_250_000


def test_prices_already_in_thousands_are_left_alone(tmp_path):
    f = write(tmp_path, "SAB.csv", "date,close\n2024-01-02,42.8\n2024-01-03,43.1\n")
    assert list(f.fetch_prices("SAB", "2024-01-01")["close"]) == [42.8, 43.1]


def test_investing_style_volume_suffix_and_start_filter(tmp_path):
    f = write(tmp_path, "VNM.csv", "Date,Price,Open,High,Low,Vol.,Change %\n01/02/2024,50.0,49,51,48,1.2M,1%\n01/03/2024,51.0,50,52,49,350K,2%\n")
    out = f.fetch_prices("VNM", "2024-03-01")
    assert len(out) == 1 and out["volume"].iloc[0] == 350_000


def test_number_parser_cases():
    s = _to_number(pd.Series(["46,000", "45,5", "1.234.567", "1.2M", "--", "7.52"]))
    assert s.iloc[0] == 46000 and s.iloc[1] == 45.5 and s.iloc[2] == 1234567
    assert s.iloc[3] == 1_200_000 and pd.isna(s.iloc[4]) and s.iloc[5] == 7.52


def test_missing_file_and_missing_close_column_fail_clearly(tmp_path):
    f = CsvFetcher(tmp_path)
    with pytest.raises(FileNotFoundError, match="FPT.csv"):
        f.fetch_prices("FPT", "2024-01-01")
    write(tmp_path, "BAD.csv", "date,something\n2024-01-02,1\n")
    with pytest.raises(ValueError, match="close"):
        f.fetch_prices("BAD", "2024-01-01")


def test_vn_routing_prefers_csv_and_otherwise_explains_vnstock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "vn_csv").mkdir(parents=True)
    (tmp_path / "data" / "vn_csv" / "FPT.csv").write_text("date,close\n2024-01-02,100\n")
    assert isinstance(get_fetcher("VN", "FPT"), CsvFetcher)
    assert isinstance(get_fetcher("VN", "SAB"), VietnamFetcher)
