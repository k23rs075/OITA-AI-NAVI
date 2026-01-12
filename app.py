import os
import json
import re
import time
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
# AI Libraries
from google import genai
from google.genai import types
from groq import Groq
# Standard Libraries
import requests
import traceback
import math 
from datetime import datetime, timedelta, timezone 
import pandas as pd
import numpy as np 

# .encファイルから環境変数を読み込む
load_dotenv(dotenv_path='.enc') 

app = Flask('AINavigationApp')
CORS(app) 

# 環境変数からAPIキーを読み込み
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY") 

client_gemini = None
client_groq = None

if GEMINI_API_KEY:
    try:
        client_gemini = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini client initialized.")
    except Exception as e: print(f"⚠️ Gemini Init Error: {e}")

if GROQ_API_KEY:
    try:
        client_groq = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq client initialized.")
    except Exception as e: print(f"⚠️ Groq Init Error: {e}")

if not GOOGLE_MAPS_API_KEY:
    print("FATAL: GOOGLE_MAPS_API_KEY not found. Navigation will fail.")


# 2点間の距離を計算する関数
def haversine_distance_vector(lat1, lon1, lat2_series, lon2_series):
    R = 6371000  # 地球の半径(m)
    phi1 = math.radians(lat1)
    phi2 = np.radians(lat2_series)
    delta_phi = np.radians(lat2_series - lat1)
    delta_lambda = np.radians(lon2_series - lon1)
    a = np.sin(delta_phi / 2)**2 + math.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000 
    lat1_rad = math.radians(lat1); lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2); lon2_rad = math.radians(lon2)
    dlon = lon2_rad - lon1_rad; dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c



GTFS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gtfs_data')
LOADED_GTFS = {}
LOADED_CSV_POIS = None 
LOADED_SHARE_CYCLE_POIS = None 
LOADED_RENTAL_CYCLE_POIS = None
LOADED_TRAIN_DATA = {} 

def load_all_data():
    global LOADED_GTFS, LOADED_CSV_POIS, LOADED_SHARE_CYCLE_POIS, LOADED_RENTAL_CYCLE_POIS, LOADED_TRAIN_DATA
    print("\n--- Loading Data (Full Version) ---")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_dir, 'gtfs_data')
    
    # 1. GTFS (バスデータ読み込み)
    if os.path.exists(target_dir):
        DATA_DIR = target_dir
        # サブフォルダ（バス会社ごと）を走査
        for agency_name in os.listdir(DATA_DIR):
            agency_path = os.path.join(DATA_DIR, agency_name)
            if os.path.isdir(agency_path):
                try:
                    # 必要な4つのファイルを読み込む
                    dfs = {}
                    required_files = ['stops.txt', 'routes.txt', 'trips.txt', 'stop_times.txt']
                    loaded_count = 0
                    
                    for fname in required_files:
                        fpath = os.path.join(agency_path, fname)
                        if os.path.exists(fpath):
                            # データ型を指定して読み込み（IDの0落ち防止など）
                            dfs[fname.replace('.txt', '')] = pd.read_csv(fpath, dtype=str)
                            loaded_count += 1
                    
                    # stops.txtの座標を数値変換
                    if 'stops' in dfs:
                        dfs['stops']['stop_lat'] = pd.to_numeric(dfs['stops']['stop_lat'], errors='coerce')
                        dfs['stops']['stop_lon'] = pd.to_numeric(dfs['stops']['stop_lon'], errors='coerce')

                    # 4つ揃っていれば登録
                    if loaded_count >= 4:
                        LOADED_GTFS[agency_name] = dfs
                        print(f"✅ Loaded GTFS Bus: {agency_name}")
                    elif 'stops' in dfs:
                        # stopsだけでもあればバス停検索用に登録
                        LOADED_GTFS[agency_name] = {'stops': dfs['stops']}
                        print(f"⚠️ Partial GTFS: {agency_name} (Stops only)")
                        
                except Exception as e:
                    print(f"❌ GTFS Load Error ({agency_name}): {e}")

    # 2. POI / Cycle
    try: LOADED_CSV_POIS = pd.read_csv(os.path.join(DATA_DIR, 'poi_list.csv'))
    except: pass
    try: LOADED_SHARE_CYCLE_POIS = pd.read_csv(os.path.join(DATA_DIR, 'oita_share cycle.csv'))
    except: pass
    try: LOADED_RENTAL_CYCLE_POIS = pd.read_csv(os.path.join(DATA_DIR, 'oita_rental cycle.csv'))
    except: pass

    # 3. Custom Train
    try:
        stations_path = os.path.join(DATA_DIR, 'stations.csv')
        stop_times_path = os.path.join(DATA_DIR, 'stop_times.csv')
        
        stations_data = []
        if os.path.exists(stations_path):
            try:
                stations_data = pd.read_csv(stations_path).to_dict('records')
                print(f"✅ Stations: {len(stations_data)} 件")
            except: pass

        st_data = []
        if os.path.exists(stop_times_path):
            trip_counter = 0; last_seq = 9999
            with open(stop_times_path, 'r', encoding='utf-8') as f:
                next(f, None)
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 2: continue 
                    try:
                        seq = int(parts[-1]) if parts[-1].isdigit() else 0
                        if seq <= last_seq: trip_counter += 1
                        last_seq = seq
                        times = [p for p in parts if ':' in p and len(p) <= 5]
                        arr = times[0] if times else None
                        dep = times[-1] if times else None
                        if arr and dep:
                            st_data.append({
                                'trip_id': trip_counter, 'train_type': parts[0], 'station_id': parts[1],
                                'station_name': parts[2] if len(parts)>2 else "",
                                'arrival_time': arr, 'departure_time': dep, 'stop_sequence': seq
                            })
                    except: continue
            print(f"✅ Stop Times: {len(st_data)} 件")

        if stations_data and st_data:
            LOADED_TRAIN_DATA = {'stations': pd.DataFrame(stations_data), 'stop_times': pd.DataFrame(st_data)}
            print("🎉 自作鉄道データ読み込み完了！")
    except Exception as e: print(f"⚠️ Custom Data Load Error: {e}")

# API関連関数群
def parse_time_spec(time_spec):
    # None, None を返すようにしてエラーを防ぐ
    if not time_spec or str(time_spec).lower() in ['none', 'null', 'すぐ']: 
        return None, None
    try:
        now = datetime.now(timezone(timedelta(hours=+9)))
        if '-' in time_spec and ':' in time_spec:
             dt = datetime.strptime(time_spec, '%Y-%m-%d %H:%M').replace(tzinfo=timezone(timedelta(hours=+9)))
             return int(dt.timestamp()), dt
        import re; match = re.search(r'(\d{1,2}):(\d{2})', time_spec)
        if match:
            h, m = map(int, match.groups())
            dt = datetime.combine(now.date(), datetime.min.time().replace(hour=h, minute=m), timezone(timedelta(hours=+9)))
            if dt < now: dt += timedelta(days=1)
            return int(dt.timestamp()), dt
        return None, None
    except: 
        return None, None

def geocode_location(location_name):
    if not location_name or location_name in ["None", "不明"]: return None
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {'address': location_name, 'key': GOOGLE_MAPS_API_KEY.strip(), 'language': 'ja'}
    try:
        res = requests.get(url, params=params); res.raise_for_status(); data = res.json()
        if data['status'] == 'OK' and data['results']:
            r = data['results'][0]; l = r['geometry']['location']
            print(f"DEBUG: Found '{r['formatted_address']}'")
            return {'lat': l['lat'], 'lon': l['lng'], 'display_name': r['formatted_address'], 'place_id': r.get('place_id')}
        return None
    except: return None

def search_nearby_poi(lat, lon, radius=1000, poi_type="transit", keyword=None):
    all_pois = []
    
    # CSV Search (Cycle)
    if poi_type == "cycle":
        if LOADED_SHARE_CYCLE_POIS is not None:
            try:
                df = LOADED_SHARE_CYCLE_POIS.copy(); df['dist'] = haversine_distance_vector(lat, lon, df['lat'], df['lon'])
                for _, r in df[df['dist'] <= radius].iterrows(): all_pois.append({'name': r['name_ja'], 'lat': r['lat'], 'lon': r['lon'], 'type': 'share_cycle', 'distance': r['dist']})
            except: pass
        if LOADED_RENTAL_CYCLE_POIS is not None:
            try:
                df = LOADED_RENTAL_CYCLE_POIS.copy(); df['dist'] = haversine_distance_vector(lat, lon, df['lat'], df['lon'])
                for _, r in df[df['dist'] <= radius].iterrows(): all_pois.append({'name': r['name_ja'], 'lat': r['lat'], 'lon': r['lon'], 'type': 'rental_cycle', 'distance': r['dist']})
            except: pass

    # GTFS Search
    if poi_type == "transit" and LOADED_GTFS:
        for _, d in LOADED_GTFS.items():
            if 'stops' in d:
                df = d['stops'].copy(); df['dist'] = haversine_distance_vector(lat, lon, df['stop_lat'], df['stop_lon'])
                for _, r in df[df['dist'] <= radius].iterrows(): all_pois.append({'name': f"[GTFS] {r['stop_name']}", 'lat': r['stop_lat'], 'lon': r['stop_lon'], 'type': 'bus_station'})
    
    # Places API
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {'location': f"{lat},{lon}", 'radius': radius, 'key': GOOGLE_MAPS_API_KEY.strip(), 'language': 'ja'}
    if poi_type == "transit": params['keyword'] = "駅 OR バス停 OR station"
    elif poi_type == "detour" and keyword: params['keyword'] = keyword
    elif poi_type == "cycle": params['keyword'] = "cycle"
    
    try:
        data = requests.get(url, params=params).json()
        if data['status'] == 'OK':
            for r in data['results']:
                t = 'other_transit'
                if poi_type == "cycle": t = 'rental_cycle'
                else:
                    ts = r.get('types', [])
                    if 'train_station' in ts or 'subway_station' in ts: t = 'train_station'
                    elif 'bus_station' in ts: t = 'bus_station_gmp'
                    elif poi_type == 'detour': t = 'detour'
                all_pois.append({'name': r.get('name'), 'lat': r['geometry']['location']['lat'], 'lon': r['geometry']['location']['lng'], 'type': t, 'place_id': r.get('place_id')})
    except: pass

    for p in all_pois: 
        if 'distance' not in p: p['distance'] = haversine_distance(lat, lon, p['lat'], p['lon'])
    all_pois.sort(key=lambda x: x['distance'])
    
    if poi_type == "transit":
        tr = [p for p in all_pois if p['type']=='train_station']; bu = [p for p in all_pois if 'bus' in p['type']]; ot = [p for p in all_pois if p not in tr and p not in bu]
        return tr[:5] + bu[:10] + ot[:5]
    if poi_type == "cycle":
        u = []; s = set()
        for p in all_pois:
            if p['name'] not in s: u.append(p); s.add(p['name'])
        return u
    return all_pois 

def get_ors_route(start_coords, end_coords, profile="walking", departure_time=None):
    if not GOOGLE_MAPS_API_KEY: return None
    mode = "driving" if profile == "driving-car" else "walking"
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {'origin': f"{start_coords[0]},{start_coords[1]}", 'destination': f"{end_coords[0]},{end_coords[1]}", 'mode': mode, 'key': GOOGLE_MAPS_API_KEY.strip(), 'language': 'ja'}
    if departure_time: params['departure_time'] = departure_time
    try:
        res = requests.get(url, params=params).json()
        if res['status'] == 'OK': 
            l = res["routes"][0]["legs"][0]
            return {"distance": l["distance"]["value"], "duration": l["duration"]["value"], "geometry": res["routes"][0]["overview_polyline"]["points"], "profile": mode}
    except: pass
    return None

def get_gmp_waypoint_route(origin, dest, wps, mode, time=None):
    if not GOOGLE_MAPS_API_KEY: return None
    params = {'origin': origin, 'destination': dest, 'waypoints': '|'.join(wps), 'mode': mode, 'key': GOOGLE_MAPS_API_KEY.strip(), 'language': 'ja'}
    if time: params['departure_time'] = time
    try:
        res = requests.get("https://maps.googleapis.com/maps/api/directions/json", params=params).json()
        if res['status'] == 'OK':
            r = res["routes"][0]
            return {"distance": sum(l["distance"]["value"] for l in r["legs"]), "duration": sum(l["duration"]["value"] for l in r["legs"]), "geometry": r["overview_polyline"]["points"], "profile": mode}
    except: pass
    return None

def get_gmp_public_transit_route(origin, dest, time=None, mode=None):
    url = "https://maps.googleapis.com/maps/api/directions/json"
    o = f"place_id:{origin['place_id']}" if isinstance(origin, dict) and origin.get('place_id') else origin
    d = f"place_id:{dest['place_id']}" if isinstance(dest, dict) and dest.get('place_id') else dest
    if not time: time = int(datetime.now(timezone(timedelta(hours=+9))).timestamp())
    params = {'origin': o, 'destination': d, 'mode': 'transit', 'key': GOOGLE_MAPS_API_KEY.strip(), 'language': 'ja', 'departure_time': time}
    if mode: params['transit_mode'] = mode
    try:
        res = requests.get(url, params=params).json()
        if res['status'] == 'OK':
            l = res["routes"][0]["legs"][0]
            return {"distance": l["distance"]["value"], "duration": l["duration"]["value"], "geometry": res["routes"][0]["overview_polyline"]["points"], "profile": "transit"}
    except: pass
    return None

# app.py

# 1. 自作鉄道検索 (座標のNone判定修正版)
def search_custom_train_routes(start_name, dest_name, departure_dt, start_lat, start_lon, dest_lat, dest_lon):
    print(f"\n🔍 [DEBUG] 鉄道検索 (修正版): '{start_name}' -> '{dest_name}'")
    candidates = []
    if not LOADED_TRAIN_DATA: return candidates
    
    stations = LOADED_TRAIN_DATA['stations']
    times = LOADED_TRAIN_DATA['stop_times']
    
    def find_nearest_station(lat, lon):
        if lat is None or lon is None: return None
        try:
            df = stations.copy()
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
            df = df.dropna(subset=['lat', 'lon'])
            df['dist'] = ((df['lat'] - lat)**2 + (df['lon'] - lon)**2)
            return df.sort_values('dist').iloc[0]
        except: return None

    def get_station_by_name(name):
        key = name.replace("駅", "")
        exact = stations[stations['station_name'] == key]
        if not exact.empty: return exact.iloc[0]
        match = stations[stations['station_name'].str.contains(key, na=False)]
        if not match.empty: return match.iloc[0]
        return None

    s_info = get_station_by_name(start_name)
    e_info = get_station_by_name(dest_name)
    if s_info is None and start_lat: s_info = find_nearest_station(start_lat, start_lon)
    if e_info is None and dest_lat: e_info = find_nearest_station(dest_lat, dest_lon)
    if s_info is None or e_info is None: return candidates

    s_id = s_info['station_id']; e_id = e_info['station_id']
    s_lat_st = float(s_info.get('lat', 0)); s_lon_st = float(s_info.get('lon', 0))
    e_lat_st = float(e_info.get('lat', 0)); e_lon_st = float(e_info.get('lon', 0))

    if s_id == e_id: return candidates

    possible_starts = times[times['station_id'] == s_id].copy()
    base_time = departure_dt if departure_dt else datetime.now(timezone(timedelta(hours=+9)))
    target_time_str = base_time.strftime('%H:%M')
    possible_starts = possible_starts[possible_starts['departure_time'] >= target_time_str]

    for _, s_row in possible_starts.iterrows():
        trip_rows = times[times['trip_id'] == s_row['trip_id']]
        e_row = trip_rows[trip_rows['station_id'] == e_id]
        
        if not e_row.empty:
            e_data = e_row.iloc[0]
            if int(s_row['stop_sequence']) < int(e_data['stop_sequence']):
                try:
                    train_dep_dt = datetime.strptime(s_row['departure_time'], '%H:%M')
                    train_arr_dt = datetime.strptime(e_data['arrival_time'], '%H:%M')
                    if train_arr_dt < train_dep_dt: train_arr_dt += timedelta(days=1)
                    
                    segments = []
                    full_geometry = []
                    summary_parts = []
                    total_dur = 0
                    
                    pre_bus_route = None
                    final_first_stop = {'lat': s_lat_st, 'lon': s_lon_st}
                    
                    dist_to_start = haversine_distance(start_lat, start_lon, s_lat_st, s_lon_st)
                    if dist_to_start > 500:
                        bus_candidates = search_gtfs_bus_routes(start_lat, start_lon, s_lat_st, s_lon_st, base_time)
                        limit_time = train_dep_dt - timedelta(minutes=5)
                        for bus in bus_candidates:
                            # 簡易判定
                            try:
                                bus_start_hm = bus['summary'].split(' ')[-2].replace('発', '')
                                bus_st_dt = datetime.strptime(bus_start_hm, '%H:%M')
                                bus_arr_dt = bus_st_dt + timedelta(seconds=bus['total_duration_s'])
                                if bus_arr_dt.time() <= limit_time.time():
                                    pre_bus_route = bus
                                    break
                            except: pass

                    if pre_bus_route:
                        segments.append({"type": "bus", "geometry": pre_bus_route['geometry']})
                        full_geometry += pre_bus_route['geometry']
                        summary_parts.append(f"🚌{pre_bus_route['summary'].split(' ')[0]}")
                        total_dur += pre_bus_route['total_duration_s'] + 300
                        
                        last_bus_pt = pre_bus_route['geometry'][-1]
                        walk_to_station = [last_bus_pt, [s_lat_st, s_lon_st]]
                        segments.append({"type": "walk", "geometry": walk_to_station})
                        full_geometry += walk_to_station
                        final_first_stop = {'lat': pre_bus_route['geometry'][0][0], 'lon': pre_bus_route['geometry'][0][1]}

                    train_geometry = []
                    segment = trip_rows[
                        (trip_rows['stop_sequence'].astype(int) >= int(s_row['stop_sequence'])) & 
                        (trip_rows['stop_sequence'].astype(int) <= int(e_data['stop_sequence']))
                    ].sort_values(by='stop_sequence', key=lambda col: col.astype(int))
                    
                    for _, row in segment.iterrows():
                        st = stations[stations['station_id'] == row['station_id']]
                        if not st.empty:
                            lat = float(st.iloc[0].get('lat', 0)); lon = float(st.iloc[0].get('lon', 0))
                            if lat != 0: train_geometry.append([lat, lon])

                    segments.append({"type": "train", "geometry": train_geometry})
                    full_geometry += train_geometry
                    summary_parts.append(f"🚃{s_row['train_type']}")
                    total_dur += (train_arr_dt - train_dep_dt).seconds

                    post_bus_route = None
                    final_last_stop = {'lat': e_lat_st, 'lon': e_lon_st}
                    
                    transfer_start_time = train_arr_dt + timedelta(minutes=5)
                    bus_candidates = search_gtfs_bus_routes(e_lat_st, e_lon_st, dest_lat, dest_lon, transfer_start_time)
                    if bus_candidates:
                        post_bus_route = bus_candidates[0]

                    if post_bus_route:
                        first_bus_pt = post_bus_route['geometry'][0]
                        walk_from_station = [[e_lat_st, e_lon_st], first_bus_pt]
                        segments.append({"type": "walk", "geometry": walk_from_station})
                        full_geometry += walk_from_station

                        segments.append({"type": "bus", "geometry": post_bus_route['geometry']})
                        full_geometry += post_bus_route['geometry']
                        summary_parts.append(f"🚌{post_bus_route['summary'].split(' ')[0]}")
                        total_dur += post_bus_route['total_duration_s'] + 300
                        final_last_stop = post_bus_route['last_stop_coords']

                    full_summary = " ➡ ".join(summary_parts)
                    if not pre_bus_route and not post_bus_route:
                         full_summary = f"電車 ({s_row['train_type']} {s_row['departure_time']}発)"

                    # ★修正: start_coords を None かどうか厳密に判定
                    start_coords_obj = {'lat': start_lat, 'lon': start_lon} if start_lat else None
                    if pre_bus_route:
                        start_coords_obj = {'lat': start_lat, 'lon': start_lon} if start_lat else None
                    
                    candidates.append({
                        "summary": full_summary,
                        "profile": "transit_train",
                        "total_duration_s": total_dur,
                        "total_distance_m": 0, 
                        "geometry": full_geometry,
                        "segments": segments,
                        "is_raw_path": True, 
                        "is_manual": True,
                        "start_coords": start_coords_obj,
                        "first_stop_coords": final_first_stop,
                        "last_stop_coords": final_last_stop,
                        "final_dest_coords": { 'lat': dest_lat, 'lon': dest_lon } if dest_lat else None
                    })
                    if len(candidates) >= 5: break 
                except Exception as e: pass
    
    return candidates


# SYSTEM_INSTRUCTION
SYSTEM_INSTRUCTION = """
あなたは、ユーザーからのチャットに基づき、経路案内を計画するAIアシスタントです。
以下のJSON形式のみで出力してください。Markdownのバッククォートは不要です。

{
  "departure": "出発地（'現在地' または具体的な場所・駅名）",
  "destination": "目的地（具体的な施設名・駅名。例: '海たまご', '別府駅'。 ※'〇〇の観光地'のような曖昧な指定の場合は、そのエリアの代表的な駅名を設定すること）",
  "waypoints": ["経由地1", "経由地2"],
  "time_spec": "時刻指定（YYYY-MM-DD HH:MM形式。日付が不明ならHH:MM、指定がなければ 'None'）",
  "transport": "交通手段",
  "detour": "寄り道の種類（カテゴリー名、なければ 'None'）"
}
"""

def generate_ai_response_robust(user_message, lat, lon):
    prompt_content = f"Lat:{lat},Lon:{lon}\n{user_message}"
    
    # 1. Gemini
    if client_gemini:
        try:
            print("🚀 Calling Gemini...")
            res = client_gemini.models.generate_content(model='gemini-1.5-flash', contents=prompt_content, config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION))
            return json.loads(res.text.strip().replace("```json", "").replace("```", "").strip())
        except Exception as e: print(f"⚠️ Gemini Failed: {e}")

    # 2. Groq
    if client_groq:
        try:
            print("🔄 Switching to Groq...")
            res = client_groq.chat.completions.create(
                messages=[{"role": "system", "content": SYSTEM_INSTRUCTION}, {"role": "user", "content": prompt_content}],
                model="llama-3.3-70b-versatile", temperature=0, response_format={"type": "json_object"}
            )
            return json.loads(res.choices[0].message.content)
        except Exception as e: print(f"⚠️ Groq Failed: {e}")

    # 3. Mock
    print("🚨 Using Mock.")
    m = {"departure": "Current Location", "destination": "大分駅", "waypoints": [], "time_spec": "None", "transport": "driving", "detour": "None"}
    if "別府" in user_message: m["destination"] = "別府駅"
    elif "空港" in user_message: m["destination"] = "大分空港"
    return m



@app.route('/api/get_spot_details', methods=['POST'])
def get_spot_details():
    d = request.json
    pid = d.get('place_id')
    nm = d.get('name')
    clean_name = re.sub(r'\[.*?\]\s*', '', nm).strip() if nm else nm
    res = {"name": clean_name, "description": "詳細なし", "payment": "不明", "photo_url": None, "rating": None}

    # Place IDがない場合、名前で検索してIDを取得する
    if not pid and clean_name and GOOGLE_MAPS_API_KEY:
        try:
            find_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
            find_params = {
                'input': clean_name,
                'inputtype': 'textquery',
                'fields': 'place_id',
                'key': GOOGLE_MAPS_API_KEY.strip(),
                'language': 'ja'
            }
            # 緯度経度があれば優先検索 (バイアス)
            if d.get('lat') and d.get('lon'):
                find_params['locationbias'] = f"circle:2000@{d['lat']},{d['lon']}"
            
            resp = requests.get(find_url, params=find_params).json()
            if resp['status'] == 'OK' and resp['candidates']:
                pid = resp['candidates'][0]['place_id']
                print(f"DEBUG: Found Place ID for '{clean_name}': {pid}")
        except Exception as e:
            print(f"⚠️ Find Place Error: {e}")

    if pid:
        try:
            u = "https://maps.googleapis.com/maps/api/place/details/json"
            r = requests.get(u, params={'place_id': pid, 'fields': 'name,rating,photos', 'key': GOOGLE_MAPS_API_KEY.strip(), 'language': 'ja'}).json()
            if r['status'] == 'OK':
                result = r['result']
                res['rating'] = result.get('rating')
                if result.get('photos'):
                    photo_ref = result['photos'][0]['photo_reference']
                    res['photo_url'] = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photo_reference={photo_ref}&key={GOOGLE_MAPS_API_KEY.strip()}"
        except: pass
        
    return jsonify(res)

@app.route('/api/get_spot_ai_description', methods=['POST'])
def get_spot_ai_description():
    name = request.json.get('name')
    desc = "解説を取得できませんでした。"
    if name:
        try:
            if client_gemini:
                try:
                    res = client_gemini.models.generate_content(model='gemini-1.5-flash', contents=f"「{name}」の観光ガイドを50文字で。")
                    return jsonify({"description": res.text.strip()})
                except: pass
            if client_groq:
                try:
                    res = client_groq.chat.completions.create(messages=[{"role":"user", "content":f"「{name}」の観光ガイドを50文字で。"}], model="llama-3.3-70b-versatile")
                    return jsonify({"description": res.choices[0].message.content.strip()})
                except: pass
        except: pass
    return jsonify({"description": desc})

@app.route('/api/search_nearby', methods=['POST'])
def search_nearby():
    try:
        data = request.json
        lat = float(data.get('latitude'))
        lon = float(data.get('longitude'))
        radius = int(data.get('radius', 500))
        req_type = data.get('type', 'transit')
        
        # リクエスト種別に応じて検索モードを決定
        search_mode = 'transit'
        if req_type == 'cycle': 
            search_mode = 'cycle'
        elif req_type in ['train', 'bus']: 
            search_mode = 'transit'
        
        pois = search_nearby_poi(lat, lon, radius, search_mode)
        
        # リクエスト種別に応じてPOIをフィルタリング
        if req_type == 'train':
            # 駅のみを抽出
            pois = [p for p in pois if p['type'] in ['train_station', 'subway_station']]
        elif req_type == 'bus':
            # バス停のみを抽出
            pois = [p for p in pois if p['type'] in ['bus_station', 'bus_station_gmp', 'other_transit']]
        
        return jsonify({"pois": pois, "center": {"lat": lat, "lon": lon}, "count": len(pois)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# 2. GTFSバス検索 
def search_gtfs_bus_routes(start_lat, start_lon, dest_lat, dest_lon, departure_dt=None):
    candidates = []
    if not LOADED_GTFS: return candidates
    
    print(f"🔍 バスルート検索開始: ({start_lat}, {start_lon}) -> ({dest_lat}, {dest_lon})")
    
    SEARCH_RADIUS_M = 3000 
    target_time_str = "00:00:00"
    if departure_dt:
        target_time_str = departure_dt.strftime('%H:%M:%S')

    for agency, data in LOADED_GTFS.items():
        if not all(k in data for k in ['stops', 'routes', 'trips', 'stop_times']): continue
        
        stops = data['stops'].copy()
        stop_times = data['stop_times'].copy()
        trips = data['trips']
        routes = data['routes']

        stops['dist_start'] = haversine_distance_vector(start_lat, start_lon, stops['stop_lat'], stops['stop_lon'])
        stops['dist_dest'] = haversine_distance_vector(dest_lat, dest_lon, stops['stop_lat'], stops['stop_lon'])
        
        start_stops = stops[stops['dist_start'] <= SEARCH_RADIUS_M].copy()
        dest_stops = stops[stops['dist_dest'] <= SEARCH_RADIUS_M].copy()
        
        if start_stops.empty or dest_stops.empty: continue
        
        start_dist_map = start_stops.set_index('stop_id')['dist_start'].to_dict()
        dest_dist_map = dest_stops.set_index('stop_id')['dist_dest'].to_dict()
            
        potential_trips = stop_times[stop_times['stop_id'].isin(start_stops['stop_id'])]
        potential_trips = potential_trips[potential_trips['departure_time'] >= target_time_str]
        if potential_trips.empty: continue

        dest_trip_rows = stop_times[stop_times['stop_id'].isin(dest_stops['stop_id'])]
        
        start_merged = potential_trips[['trip_id', 'departure_time', 'stop_sequence', 'stop_id']].rename(columns={'stop_id': 'start_stop_id', 'departure_time': 'start_time', 'stop_sequence': 'start_seq'})
        dest_merged = dest_trip_rows[['trip_id', 'arrival_time', 'stop_sequence', 'stop_id']].rename(columns={'stop_id': 'end_stop_id', 'arrival_time': 'end_time', 'stop_sequence': 'end_seq'})
        
        merged = pd.merge(start_merged, dest_merged, on='trip_id')
        merged['start_seq'] = pd.to_numeric(merged['start_seq'], errors='coerce')
        merged['end_seq'] = pd.to_numeric(merged['end_seq'], errors='coerce')
        
        valid_trips = merged[merged['start_seq'] < merged['end_seq']].copy()
        if valid_trips.empty: continue

        valid_trips['walk_start'] = valid_trips['start_stop_id'].map(start_dist_map)
        valid_trips['walk_end'] = valid_trips['end_stop_id'].map(dest_dist_map)
        valid_trips['total_walk'] = valid_trips['walk_start'] + valid_trips['walk_end']
        valid_trips = valid_trips.sort_values(by=['total_walk', 'start_time'])
        
        for _, row in valid_trips.head(3).iterrows():
            trip_id = row['trip_id']
            trip_info = trips[trips['trip_id'] == trip_id].iloc[0]
            route_info = routes[routes['route_id'] == trip_info['route_id']].iloc[0]
            
            bus_name = f"{route_info.get('route_short_name', '')} {route_info.get('route_long_name', '')}".strip()
            headsign = trip_info.get('trip_headsign', '')

            geometry_points = []
            try:
                this_trip = stop_times[stop_times['trip_id'] == trip_id].copy()
                this_trip['stop_sequence'] = pd.to_numeric(this_trip['stop_sequence'], errors='coerce')
                this_trip = this_trip.sort_values('stop_sequence')
                
                start_seq = row['start_seq']; end_seq = row['end_seq']
                segment = this_trip[(this_trip['stop_sequence'] >= start_seq) & (this_trip['stop_sequence'] <= end_seq)]
                segment_geo = pd.merge(segment, stops[['stop_id', 'stop_lat', 'stop_lon']], on='stop_id', how='left')
                segment_geo = segment_geo.sort_values('stop_sequence')
                
                for _, stop_row in segment_geo.iterrows():
                    if pd.notnull(stop_row['stop_lat']):
                        geometry_points.append([float(stop_row['stop_lat']), float(stop_row['stop_lon'])])
            except: geometry_points = []

            if not geometry_points: continue

            try:
                t1 = datetime.strptime(row['start_time'][:5], '%H:%M')
                t2 = datetime.strptime(row['end_time'][:5], '%H:%M')
                if t2 < t1: t2 += timedelta(days=1)
                duration = (t2 - t1).seconds
            except: duration = 0

            walk_min = int(row['total_walk'] / 80)
            # 行き先(headsign)が空の場合は "(行)" を付けない
            if headsign and str(headsign).strip():
                summary = f"{bus_name} ({headsign}行) {row['start_time'][:5]}発 (歩{walk_min}分)"
            else:
                summary = f"{bus_name} {row['start_time'][:5]}発 (歩{walk_min}分)"
            
            segments = [{"type": "bus", "geometry": geometry_points}]

            # 距離を geometry_points から算出
            total_m = 0
            try:
                for i in range(1, len(geometry_points)):
                    total_m += haversine_distance(geometry_points[i-1][0], geometry_points[i-1][1], geometry_points[i][0], geometry_points[i][1])
            except:
                total_m = 0

            candidates.append({
                "summary": summary,
                "profile": "transit_bus",
                "total_duration_s": duration,
                "total_distance_m": int(total_m),
                "geometry": geometry_points,
                "segments": segments,
                "is_raw_path": True,
                "is_manual": True,
            
                "start_coords": {'lat': start_lat, 'lon': start_lon} if start_lat else None,
                "first_stop_coords": {'lat': geometry_points[0][0], 'lon': geometry_points[0][1]},
                "last_stop_coords": {'lat': geometry_points[-1][0], 'lon': geometry_points[-1][1]},
                "final_dest_coords": {'lat': dest_lat, 'lon': dest_lon} if dest_lat else None
            })
            
    print(f"✅ バスルート候補: {len(candidates)}件")
    return candidates

@app.route('/api/process_chat', methods=['POST'])
def process_chat():
    data = request.json
    msg = data.get('message', '')
    lat = float(data.get('latitude')) if data.get('latitude') else None
    lon = float(data.get('longitude')) if data.get('longitude') else None
    
    print(f"\n--- Chat Received: {msg} ---")
    if not GOOGLE_MAPS_API_KEY: return jsonify({"response": "APIキー設定エラー", "is_system": True})

    try:
        # AI解析
        ai_json = generate_ai_response_robust(msg, lat, lon)
        print(json.dumps(ai_json, indent=2, ensure_ascii=False))
        
        # 時刻解析
        dep_timestamp, dep_dt = parse_time_spec(ai_json.get('time_spec'))
        
        # 出発地特定
        start_geo = None
        dep_raw = ai_json.get('departure', 'Current Location')
        start_name = "現在地"
        
        if dep_raw in ["Current Location", "ユーザーの現在地", "現在地", "ここ", "私"]:
            if lat and lon:
                curr = (lat, lon)
            else:
                start_geo = geocode_location("大分駅")
                curr = (start_geo['lat'], start_geo['lon']) if start_geo else None
                start_name = "大分駅"
        else:
            # 出発地にも「大分県」補正をかける
            search_dep = dep_raw
            if dep_raw and "大分" not in dep_raw and "県" not in dep_raw and "駅" not in dep_raw:
                 search_dep = f"大分県 {dep_raw}"
            
            start_geo = geocode_location(search_dep)
            curr = (start_geo['lat'], start_geo['lon']) if start_geo else None
            start_name = dep_raw

        # 目的地特定
        dest_name = ai_json.get('destination')
        
  
        search_dest_name = dest_name
        # 「大分」が含まれず、かつ座標入力でもない場合に「大分県」を頭につける
        if dest_name and "大分" not in dest_name and "県" not in dest_name:
             # ただし「東京」とか明らかに県外の文字がある場合は避けるロジックも入れられますが、
             # 基本は大分アプリとのことなので強制付与します
             search_dest_name = f"大分県 {dest_name}"
             print(f"🔍 大分限定検索: '{dest_name}' -> '{search_dest_name}'")

        dest_geo = geocode_location(search_dest_name)
        
        # 経由地
        wps = []; wps_raw = ai_json.get('waypoints')
        if wps_raw and isinstance(wps_raw, list):
            for w in wps_raw:
            
                search_w = w
                if w and "大分" not in w:
                    search_w = f"大分県 {w}"
                
                g = geocode_location(search_w)
                if g: wps.append({'name': w, 'lat': g['lat'], 'lon': g['lon']})
        ai_json['waypoints_data'] = wps
        
        nearby_pois = []; route_candidates = []
        
        # 観光スポット検索
        if dest_geo and LOADED_CSV_POIS is not None:
            try:
                print(f"🔍 周辺スポット検索: {dest_name}")
                csv_df = LOADED_CSV_POIS.copy()
                
                csv_df['lat'] = pd.to_numeric(csv_df['lat'], errors='coerce')
                csv_df['lon'] = pd.to_numeric(csv_df['lon'], errors='coerce')
                csv_df = csv_df.dropna(subset=['lat', 'lon'])

                csv_df['dist'] = haversine_distance_vector(dest_geo['lat'], dest_geo['lon'], csv_df['lat'], csv_df['lon'])
                
                nearby_df = csv_df[csv_df['dist'] <= 5000].sort_values('dist').head(10)
                
                csv_recs = []
                for _, r in nearby_df.iterrows():
                    csv_recs.append({
                        'name': f"[おすすめ] {r['name_ja']}", 
                        'lat': r['lat'], 'lon': r['lon'], 
                        'type': 'detour', 
                        'category': r['category'], 
                        'distance': r['dist']
                    })
                
                if csv_recs:
                    if 'detour_pois' not in ai_json: ai_json['detour_pois'] = []
                    ai_json['detour_pois'].extend(csv_recs)
                    print(f"✅ おすすめスポット {len(csv_recs)} 件を追加")
            except Exception as e:
                print(f"⚠️ Spot Search Error: {e}")
                traceback.print_exc()

        

        # ルート検索
        if curr and dest_geo:
            dest_lat = dest_geo['lat']; dest_lon = dest_geo['lon']
            t_origin = start_geo if start_geo else f"{curr[0]},{curr[1]}"
            
            # ---------------------------------------------------------
            # 1. 自動車ルート
            # ---------------------------------------------------------
            if wps:
                wps_str = [f"{w['lat']},{w['lon']}" for w in wps]
                cr = get_gmp_waypoint_route(t_origin, f"{dest_lat},{dest_lon}", wps_str, "driving", dep_timestamp)
                if cr: route_candidates.append({"summary": "自動車 (経由地込)", "profile": "driving-car_waypoint", "total_duration_s": cr["duration"], "total_distance_m": cr["distance"], "geometry": cr["geometry"]})
            else:
                cr = get_ors_route(curr, (dest_lat, dest_lon), "driving-car", dep_timestamp)
                if cr: route_candidates.append({"summary": "自動車 (直行)", "profile": "driving-car", "total_duration_s": cr["duration"], "total_distance_m": cr["distance"], "geometry": cr["geometry"]})
            
            # ---------------------------------------------------------
            # 2. 電車ルート (自作データ優先 -> なければGoogle)
            # ---------------------------------------------------------
            # まず自作データ(CSV)で検索
            custom_trains = search_custom_train_routes(
                start_name, 
                dest_name, 
                dep_dt, 
                curr[0] if curr else None, 
                curr[1] if curr else None, 
                dest_geo['lat'], 
                dest_geo['lon']
            )

            if custom_trains:
                # 自作データが見つかった場合
                print("✅ 自作鉄道データを使用します")
                route_candidates.extend(custom_trains)
            else:
                # 自作データがない場合、Googleに頼る
                print("⚠️ 自作鉄道データなし -> Googleマップ検索(電車)を実行")
                tr_train = get_gmp_public_transit_route(t_origin, dest_geo, dep_timestamp, "train|subway|rail")
                if tr_train: 
                    route_candidates.append({"summary": "電車 (Google)", "profile": "transit_train", "total_duration_s": tr_train["duration"], "total_distance_m": tr_train["distance"], "geometry": tr_train["geometry"]})

            # ---------------------------------------------------------
            # 3. バスルート (GTFS優先 -> なければGoogle)
            # ---------------------------------------------------------
            gtfs_buses = []
            # まずGTFSで検索
            if curr and dest_geo:
                bus_time = dep_dt if dep_dt else datetime.now(timezone(timedelta(hours=+9)))
                gtfs_buses = search_gtfs_bus_routes(curr[0], curr[1], dest_geo['lat'], dest_geo['lon'], bus_time)
            
            if gtfs_buses:
                # GTFSデータが見つかった場合
                print("✅ GTFSバスデータを使用します")
                route_candidates.extend(gtfs_buses)
            else:
                # GTFSデータがない場合、Googleに頼る
                print("⚠️ GTFSバスデータなし -> Googleマップ検索(バス)を実行")
                tr_bus = get_gmp_public_transit_route(t_origin, dest_geo, dep_timestamp, "bus")
                if tr_bus: 
                    route_candidates.append({"summary": "バス (Google)", "profile": "transit_bus", "total_duration_s": tr_bus["duration"], "total_distance_m": tr_bus["distance"], "geometry": tr_bus["geometry"]})

        ai_json['nearby_pois'] = nearby_pois
        ai_json['route_candidates'] = route_candidates
        
    
        
        info = f"✅ **案内プラン**\n  - 出発: {start_name}\n  - 到着: {dest_name}\n"
        if dep_dt: info += f"  - 日時: {dep_dt.strftime('%m/%d %H:%M')} 以降\n"
        
        if 'detour_pois' in ai_json and ai_json['detour_pois']:
             info += f"  - 周辺スポット: {len(ai_json['detour_pois'])} 件\n"

        if not route_candidates: info += "\n⚠️ ルートが見つかりませんでした。"
        else: info += f"\n{len(route_candidates)} 件のルートが見つかりました。"
        
        if dest_geo: ai_json['destination_coords'] = {'lat': dest_geo['lat'], 'lon': dest_geo['lon']}

        return jsonify({"response": info, "is_system": True, "parsed_data": ai_json})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"response": f"処理エラー: {e}", "is_system": True})
    
@app.route('/')
def index():
    cleaned_key = GOOGLE_MAPS_API_KEY.strip() if GOOGLE_MAPS_API_KEY else None
    return render_template('index.html', google_maps_api_key=cleaned_key)

load_all_data()
if __name__ == '__main__':
    load_all_data()  
    app.run(debug=True, port=5000)
