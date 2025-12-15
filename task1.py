from dronekit import connect, VehicleMode, LocationGlobalRelative
import time, math, sys

CONNECTION_STRING = "tcp:127.0.0.1:5762"

# 起飛高度
TAKEOFF_ALT = 10.0

# 固定航點經緯度
WP_A_LAT = 24.146149072846978
WP_A_LON = 120.66119631695354

WP_B_LAT = 24.147336364095334
WP_B_LON = 120.6612421706531


# 判定到達航點的半徑（單位：米），當飛機距離航點小於等於這個值時認為到達
ARRIVAL_RADIUS = 8.0

# 通用的等待函數,接收一個函數 cond_fn（返回 True/False）並輪詢直到返回 True 或超時
def wait_for(cond_fn, timeout, poll=0.5, desc="condition"):
    t0 = time.time()                        
    while time.time() - t0 < timeout:      
        try:
            if cond_fn():                   
                return True                
        except Exception:
            pass                           
        time.sleep(poll)                   
    print("等待超時：", desc)             
    return False                         

# 將以米為單位的北向/東向偏移，轉換成經緯度增量並返回新的經緯度
# original_location 是 dronekit 返回的 location 對象（有 lat/lon 屬性）
def get_location_metres(original_location, dNorth, dEast):
    R = 6378137.0  # 地球半徑（WGS-84 赤道半徑），單位：米。用於米->經緯度近似換算
    # 緯度的弧度增量約等於北向米數除以地球半徑
    dLat = dNorth / R
    # 經度的弧度增量需要除以地球半徑再除以 cos(緯度)，因為經度線在高緯度處收縮
    dLon = dEast / (R * math.cos(math.radians(original_location.lat)))
    # 把弧度轉換為度並加回原始經緯度，得到新的經緯度
    return original_location.lat + math.degrees(dLat), original_location.lon + math.degrees(dLon)

# 用 haversine 公式計算兩點之間的球面距離（米）
# 傳入的是兩點的經緯度（十進制度）
def haversine_distance_m(lat1, lon1, lat2, lon2):
    R = 6378137.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def ensure_params_for_vtol(vehicle):
    print("確保 Q_GUIDED_MODE=1, Q_RTL_MODE=1")
    try:
        # 讀取當前 ARMING_CHECK 參數（有些固件上可能不存在或讀取失敗）
        orig_arm_check = vehicle.parameters.get('ARMING_CHECK', None)
    except Exception:
        orig_arm_check = None

    try:
        # 把 Q_GUIDED_MODE 和 Q_RTL_MODE 設置為 1，
        vehicle.parameters['Q_GUIDED_MODE'] = 1
        vehicle.parameters['Q_RTL_MODE'] = 1
        time.sleep(0.5)
    except Exception as e:
        # 如果設置失敗，打印錯誤但繼續執行（SITL 中偶爾會有短暫失敗）
        print("設置 VTOL 參數失敗（繼續，但可能出現行為異常）:", e)
    return orig_arm_check

# 安全切換模式並等待切換完成
def safe_set_mode(vehicle, mode_name, timeout=5):
    try:
        vehicle.mode = VehicleMode(mode_name)
        return wait_for(lambda: vehicle.mode.name == mode_name, timeout, desc=f"mode {mode_name}")
    except Exception as e:
        print("切換模式出錯:", e)
        return False

# 解鎖（arm）流程，包含必要的前置檢查
def arm_with_checks(vehicle, arm_timeout=30, allow_disable_checks=True):
    # 如果當前在 QRTL/RTL 模式，先回到 STABILIZE，避免 PreArm 校驗拒絕解鎖
    if vehicle.mode and vehicle.mode.name in ("QRTL", "RTL"):
        print("當前處於 RTL/QRTL，先切成 STABILIZE 以允許解鎖")
        safe_set_mode(vehicle, "STABILIZE", timeout=3)

    # 等待飛控 reports is_armable 為 True（表示自檢通過）
    print("等待 vehicle.is_armable（飛控自檢）...")
    ok = wait_for(lambda: getattr(vehicle, 'is_armable', False), 10, desc="is_armable")
    if not ok:
        print("飛控仍未準備好（is_armable=False）。")
        if allow_disable_checks:
            try:
                orig = vehicle.parameters.get('ARMING_CHECK', None)
                print("嘗試臨時將 ARMING_CHECK 置 0 以允許解鎖（僅調試/ SITL）")
                if orig is not None:
                    vehicle.parameters['ARMING_CHECK'] = 0
                    time.sleep(0.5)  # 等待參數傳播
                    ok2 = wait_for(lambda: getattr(vehicle, 'is_armable', False), 6, desc="is_armable after disabling checks")
                    if ok2:
                        # 返回原始值以便後續恢復
                        return orig
                else:
                    print("無法讀取 ARMING_CHECK 參數，不能臨時禁用檢查")
            except Exception as e:
                print("設置 ARMING_CHECK 失敗:", e)
        # 如果不能變為 armable，就報錯退出
        raise RuntimeError("飛控未準備好且無法通過臨時禁用檢查解決（請檢查 GPS/傳感器/模擬環境）")

    # 切到 GUIDED 模式準備解鎖
    print("切換到 GUIDED 並嘗試解鎖")
    safe_set_mode(vehicle, "GUIDED", timeout=5)
    vehicle.armed = True  # 發送解鎖命令
    t0 = time.time()
    while not vehicle.armed:
        if time.time() - t0 > arm_timeout:
            raise RuntimeError("解鎖超時")
        print(" 等待解鎖...")
        time.sleep(0.5)
    print("已解鎖")
    return None

# 使用 DroneKit 的 simple_takeoff 功能並等待達到目標高度
def simple_takeoff_wait(vehicle, target_alt, timeout=60):
    print("調用 simple_takeoff")
    try:
        vehicle.simple_takeoff(target_alt)  # 讓飛控開始起飛到目標高度
    except Exception as e:
        print("simple_takeoff 調用異常:", e)
    # 等待達到目標高度的條件，使用 wait_for 輪詢高度
    return wait_for(lambda: (vehicle.location.global_relative_frame.alt is not None and vehicle.location.global_relative_frame.alt >= target_alt*0.9),
                    timeout, poll=1, desc=f"reach alt {target_alt}")

# 使用simple_goto導航到目標經緯
def goto_wait(vehicle, lat, lon, radius=ARRIVAL_RADIUS, timeout=180):
    loc = LocationGlobalRelative(lat, lon, TAKEOFF_ALT)  # 構造目標位置對象
    try:
        vehicle.simple_goto(loc)  # 發送導航命令
    except Exception as e:
        print("simple_goto 調用失敗:", e)
    def reached():
        try:
            cur = vehicle.location.global_relative_frame
            if cur.lat is None:
                return False
            d = haversine_distance_m(cur.lat, cur.lon, lat, lon)
            return d <= radius
        except Exception:
            return False
    return wait_for(reached, timeout, poll=1, desc="reach waypoint")

# 觸發 QRTL(QuadPlane的返回策略)
def do_qrtl_return_and_wait(vehicle, land_timeout=300):
    print("切換到QRTL")
    safe_set_mode(vehicle, "QRTL", timeout=3)  
    # landed 函數判斷是否已經著陸並斷開油門（armed == False）
    def landed():
        try:
            alt = vehicle.location.global_relative_frame.alt
            if alt is None:
                return False
            return alt <= 0.9 and not vehicle.armed
        except Exception:
            return False
    # 等待著陸，超時則返回 False
    return wait_for(landed, land_timeout, poll=2, desc="landing via QRTL")

# 任務結束後的清理工作
def cleanup_after_mission(vehicle, orig_arm_check):
    try:
        if vehicle.armed:
            print("嘗試 disarm...")
            vehicle.armed = False
            wait_for(lambda: not vehicle.armed, 8, desc="disarm")
    except Exception as e:
        print("disarm 出錯:", e)
    if orig_arm_check is not None:
        try:
            vehicle.parameters['ARMING_CHECK'] = orig_arm_check
            print("恢復 ARMING_CHECK =", orig_arm_check)
        except Exception as e:
            print("恢復 ARMING_CHECK 失敗:", e)
    # 把模式切回 STABILIZE 作為安全默認狀態
    try:
        safe_set_mode(vehicle, "STABILIZE", timeout=3)
    except Exception:
        pass

# 主函數
def main():
    print("連接到", CONNECTION_STRING)
    try:
        vehicle = connect(CONNECTION_STRING, wait_ready=True, timeout=60)
    except Exception as e:
        print("連接失敗:", e); sys.exit(1)

    # 在開始之前把Q_GUIDED_MODE和Q_RTL_MODE設置好
    orig_arm_check = ensure_params_for_vtol(vehicle)

    # 等待飛控提供 home（位置）信息，最多等待 15 秒（SITL 有時會短暫延遲）
    t0 = time.time()
    while not vehicle.location.global_frame.lat:
        if time.time() - t0 > 15:
            print("等待 home 超時")
            break
        print("等待 home...")
        time.sleep(0.5)
    # 讀取 home 的經緯度
    home = vehicle.location.global_frame
    print("Home:", home.lat, home.lon)

    # 根據 home 計算航點 A 與航點 B 的經緯度
    wpA_lat, wpA_lon = WP_A_LAT, WP_A_LON
    wpB_lat, wpB_lon = WP_B_LAT, WP_B_LON
    print("航點A:", wpA_lat, wpA_lon)
    print("航點B:", wpB_lat, wpB_lon)

    # 如果當前模式處於 QRTL 或 RTL，先切回 STABILIZE 再繼續，以免 PreArm 阻止解鎖
    if vehicle.mode and vehicle.mode.name in ("QRTL", "RTL"):
        print("檢測到當前模式為RTL/QRTL,切換為STABILIZE")
        safe_set_mode(vehicle, "STABILIZE", timeout=3)
        time.sleep(0.5)

    # 執行解鎖流程（可能會臨時修改 ARMING_CHECK，返回值為原始 ARMING_CHECK）
    modified_arm_check = None
    try:
        modified_arm_check = arm_with_checks(vehicle, arm_timeout=30, allow_disable_checks=True)
    except Exception as e:
        print("arm 失敗:", e)
        cleanup_after_mission(vehicle, modified_arm_check if modified_arm_check is not None else orig_arm_check)
        vehicle.close()
        return

    # 起飛到 TAKEOFF_ALT（等待爬升完成或超時）
    ok = simple_takeoff_wait(vehicle, TAKEOFF_ALT, timeout=60)
    if not ok:
        print("爬升到目標高度失敗或超時,進入RTL保護")
        do_qrtl_return_and_wait(vehicle)
        cleanup_after_mission(vehicle, modified_arm_check if modified_arm_check is not None else orig_arm_check)
        vehicle.close()
        return

    # 飛向航點A
    ok_wp = goto_wait(vehicle, wpA_lat, wpA_lon, radius=ARRIVAL_RADIUS, timeout=180)
    if not ok_wp:
        print("到達航點A超時,直接返航")
        do_qrtl_return_and_wait(vehicle)
        cleanup_after_mission(vehicle, modified_arm_check if modified_arm_check is not None else orig_arm_check)
        vehicle.close()
        return

    # 到達航點A後在該點停留10秒
    print("到達航點A,懸停10s")
    time.sleep(10)

    # 飛向航點B
    ok_wp2 = goto_wait(vehicle, wpB_lat, wpB_lon, radius=ARRIVAL_RADIUS, timeout=180)
    if not ok_wp2:
        print("到達航點B超時,直接返航")
        do_qrtl_return_and_wait(vehicle)
        cleanup_after_mission(vehicle, modified_arm_check if modified_arm_check is not None else orig_arm_check)
        vehicle.close()
        return

    # 到達航點B後停留10秒
    print("到達航點B,懸停10s")
    time.sleep(10)

    # 發起QRTL返回並等待降落完成
    landed = do_qrtl_return_and_wait(vehicle, land_timeout=300)
    if landed:
        print("檢測到降落並上鎖")
    else:
        print("QRTL未在超時內完成")

    cleanup_after_mission(vehicle, modified_arm_check if modified_arm_check is not None else orig_arm_check)

    print("關閉連接")
    vehicle.close()


if __name__ == "__main__":
    main()
