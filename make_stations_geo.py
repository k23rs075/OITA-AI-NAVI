# make_stations_geo.py
# これは1回だけ実行する「データ作成用」のスクリプトです

import pandas as pd
import requests
import time
import os

# ファイルの場所設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, 'gtfs_data', 'stations.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'gtfs_data', 'stations_with_geo.csv')

def get_station_coords_free(station_name):
    """ HeartRails Geo API (無料) を使って駅の座標を取得 """
    url = "http://express.heartrails.com/api/json"
    params = {
        "method": "getStations",
        "name": station_name
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if 'response' in data and 'station' in data['response']:
            # 複数の候補がある場合は、大分県に近いものなどを選びたいが、
            # とりあえず先頭(0番目)を取得する
            station = data['response']['station'][0]
            return station['y'], station['x'] # yが緯度(lat), xが経度(lon)
    except Exception as e:
        print(f"Error: {e}")
    return 0, 0

def main():
    if not os.path.exists(INPUT_CSV):
        print("エラー: stations.csv が見つかりません")
        return

    print("--- 駅座標データの作成を開始します ---")
    df = pd.read_csv(INPUT_CSV)
    
    # 新しい列を作る
    lats = []
    lons = []

    for index, row in df.iterrows():
        name = row['station_name']
        print(f"[{index+1}/{len(df)}] 検索中: {name} ...", end=" ")
        
        lat, lon = get_station_coords_free(name)
        
        if lat != 0:
            print(f"OK ({lat}, {lon})")
        else:
            print("見つかりませんでした")
        
        lats.append(lat)
        lons.append(lon)
        
        # 無料APIへの負荷軽減のため少し待つ
        time.sleep(0.5)

    df['lat'] = lats
    df['lon'] = lons
    
    # 保存
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n完了！ '{OUTPUT_CSV}' に保存しました。")
    print("app.py ではこの新しいCSVを読み込むようにしてください。")

if __name__ == "__main__":
    main()