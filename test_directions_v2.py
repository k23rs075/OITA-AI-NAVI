# -*- coding: utf-8 -*-
import os
import requests
import json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# .encファイルからキーを読み込む
load_dotenv(dotenv_path='.enc') 
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

def test_api_call(mode, origin, destination):
    if not GOOGLE_MAPS_API_KEY:
        print("❌ エラー: APIキーが設定されていません。")
        return

    url = "https://maps.googleapis.com/maps/api/directions/json"
    
    # 出発時刻: 現在時刻 (タイムスタンプなし = Now)
    # transitの場合は departure_time が必須に近いが、省略すると「現在」になる
    params = {
        'origin': origin,
        'destination': destination,
        'mode': mode,
        'key': GOOGLE_MAPS_API_KEY.strip(),
        'language': 'ja'
    }

    print(f"\n--- テスト: {mode}モード ---")
    print(f"From: {origin}")
    print(f"To:   {destination}")

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if data['status'] == 'OK':
            route = data['routes'][0]
            leg = route['legs'][0]
            print(f"✅ 成功!")
            print(f"  所要時間: {leg['duration']['text']}")
            print(f"  距離: {leg['distance']['text']}")
            print(f"  概要: {route.get('summary', '詳細なし')}")
        else:
            print(f"❌ 失敗")
            print(f"  Status: {data['status']}")
            if 'error_message' in data:
                print(f"  Error Message: {data['error_message']}")
                
    except Exception as e:
        print(f"  通信エラー: {e}")

if __name__ == "__main__":
    # 1. 自動車でのルート検索 (これが失敗するなら API自体が使えていない)
    test_api_call("driving", "Tokyo Station", "Shinjuku Station")
    
    # 2. 公共交通機関でのルート検索 (地名で指定)
    test_api_call("transit", "Tokyo Station", "Shinjuku Station")

    # 3. 公共交通機関でのルート検索 (座標で指定)
    test_api_call("transit", "35.681236,139.767125", "35.689634,139.700567")

