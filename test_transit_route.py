# -*- coding: utf-8 -*-
import os
import requests
import json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# .encファイルからキーを読み込む
load_dotenv(dotenv_path='.enc') 
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

def test_transit_route():
    if not GOOGLE_MAPS_API_KEY:
        print("エラー: APIキーが設定されていません。")
        return

    # テスト用: 東京駅 -> 新宿駅 (確実に存在するルート)
    origin = "35.681236,139.767125" # 東京駅
    destination = "35.689634,139.700567" # 新宿駅
    
    # 出発時刻: 現在時刻の1時間後 (深夜などを避けるため)
    departure_time = int((datetime.now(timezone(timedelta(hours=+9))) + timedelta(hours=1)).timestamp())

    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        'origin': origin,
        'destination': destination,
        'mode': 'transit', # 公共交通機関
        'key': GOOGLE_MAPS_API_KEY.strip(),
        'language': 'ja',
        'departure_time': departure_time
    }

    print(f"--- Google Maps Transit Route Test ---")
    print(f"From: 東京駅 ({origin})")
    print(f"To:   新宿駅 ({destination})")
    print(f"Time: {departure_time} (Unix Timestamp)")
    print("----------------------------------------")

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if data['status'] == 'OK':
            print("✅ 成功: 公共交通機関のルートが見つかりました！")
            route = data['routes'][0]
            leg = route['legs'][0]
            print(f"所要時間: {leg['duration']['text']}")
            print(f"距離: {leg['distance']['text']}")
            print("概要: " + route.get('summary', '詳細なし'))
            
            # ステップの詳細を表示
            print("\n[経路詳細]")
            for step in leg['steps']:
                print(f"- {step['travel_mode']} ({step['duration']['text']}): {step['html_instructions']}")
                
        else:
            print(f"❌ 失敗: ルートが見つかりませんでした。")
            print(f"Status: {data['status']}")
            if 'error_message' in data:
                print(f"Error Message: {data['error_message']}")
                
    except Exception as e:
        print(f"エラー発生: {e}")

if __name__ == "__main__":
    test_transit_route()
