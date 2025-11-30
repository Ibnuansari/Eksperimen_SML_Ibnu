import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

COMMON_ENCODINGS = ["utf-8", "latin-1"]

def read_csv_with_fallback(path, encoding=None, **kwargs):
    if encoding:
        try:
            df = pd.read_csv(path, encoding=encoding, **kwargs)
            return df, encoding
        except Exception as e:
            raise RuntimeError(f"Gagal membaca {path} dengan encoding {encoding}: {e}")

    last_exc = None
    for enc in COMMON_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            return df, enc
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"Gagal membaca {path} dengan encodings {COMMON_ENCODINGS}. "
                    f"Terakhir error: {last_exc}. "
                    "Coba periksa file atau gunakan tool deteksi encoding (chardet/charset-normalizer).")

def simple_preprocess(df):
    df_p = df.copy()
    for col in df_p.select_dtypes(include=[np.number]).columns:
        df_p[col].fillna(df_p[col].median(), inplace=True)
    for col in df_p.select_dtypes(exclude=[np.number]).columns:
        df_p[col].fillna("unknown", inplace=True)
    le = LabelEncoder()
    for col in df_p.select_dtypes(include=['object']).columns:
        if df_p[col].nunique() <= 20:
            df_p[col] = le.fit_transform(df_p[col].astype(str))
        else:
            freq = df_p[col].value_counts() / len(df_p)
            df_p[col] = df_p[col].map(freq).fillna(0)
    df_p.drop_duplicates(inplace=True)
    return df_p

def run_preprocessing(input_path, output_dir, target, encoding=None, test_size=0.2, random_state=42):
    os.makedirs(output_dir, exist_ok=True)
    df, used_encoding = read_csv_with_fallback(input_path, encoding=encoding, low_memory=False)
    print(f"[INFO] Berhasil membaca {input_path} menggunakan encoding: {used_encoding}")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' tidak ditemukan di {input_path}")
    df_p = simple_preprocess(df)
    X = df_p.drop(columns=[target])
    y = df_p[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    base = os.path.splitext(os.path.basename(input_path))[0]
    train_path = os.path.join(output_dir, f"{base}_train.csv")
    test_path  = os.path.join(output_dir, f"{base}_test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return {'train': train_path, 'test': test_path, 'used_encoding': used_encoding}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automate preprocessing - Ibnu (with encoding fallback)")
    parser.add_argument("--input", required=True, help="Path ke file raw CSV")
    parser.add_argument("--output", default="namadataset_preprocessing", help="Direktori hasil")
    parser.add_argument("--target", required=True, help="Nama kolom target")
    parser.add_argument("--encoding", required=False, help="(opsional) paksa encoding, mis. cp1252 atau latin-1")
    args = parser.parse_args()
    res = run_preprocessing(args.input, args.output, args.target, encoding=args.encoding)
    print("Preprocessing selesai. Files:", res)