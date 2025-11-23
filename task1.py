from dronekit import connect, VehicleMode, LocationGlobalRelative
import time, math, sys

CONNECTION_STRING = "tcp:127.0.0.1:5762"

# 起飞高度
TAKEOFF_ALT = 10.0

# 固定航點經緯度
WP_A_LAT = 24.146149072846978
WP_A_LON = 120.66119631695354

WP_B_LAT = 24.147336364095334
WP_B_LON = 120.6612421706531


# 判定到达航点的半径（单位：米），当飞机距离航点小于等于这个值时认为到达
ARRIVAL_RADIUS = 8.0

# 通用的等待函数,接收一个函数 cond_fn（返回 True/False）并轮询直到返回 True 或超时
def wait_for(cond_fn, timeout, poll=0.5, desc="condition"):
    t0 = time.time()                        
    while time.time() - t0 < timeout:      
        try:
            if cond_fn():                   
                return True                
        except Exception:
            pass                           
        time.sleep(poll)                   
    print("等待超时：", desc)             
    return False                         

# 将以米为单位的北向/东向偏移，转换成经纬度增量并返回新的经纬度
# original_location 是 dronekit 返回的 location 对象（有 lat/lon 属性）
def get_location_metres(original_location, dNorth, dEast):
    R = 6378137.0  # 地球半径（WGS-84 赤道半径），单位：米。用于米->经纬度近似换算
    # 纬度的弧度增量约等于北向米数除以地球半径
    dLat = dNorth / R
    # 经度的弧度增量需要除以地球半径再除以 cos(纬度)，因为经度线在高纬度处收缩
    dLon = dEast / (R * math.cos(math.radians(original_location.lat)))
    # 把弧度转换为度并加回原始经纬度，得到新的经纬度
    return original_location.lat + math.degrees(dLat), original_location.lon + math.degrees(dLon)

# 用 haversine 公式计算两点之间的球面距离（米）
# 传入的是两点的经纬度（十进制度）
def haversine_distance_m(lat1, lon1, lat2, lon2):
    R = 6378137.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def ensure_params_for_vtol(vehicle):
    print("确保 Q_GUIDED_MODE=1, Q_RTL_MODE=1")
    try:
        # 读取当前 ARMING_CHECK 参数（有些固件上可能不存在或读取失败）
        orig_arm_check = vehicle.parameters.get('ARMING_CHECK', None)
    except Exception:
        orig_arm_check = None

    try:
        # 把 Q_GUIDED_MODE 和 Q_RTL_MODE 设置为 1，
        vehicle.parameters['Q_GUIDED_MODE'] = 1
        vehicle.parameters['Q_RTL_MODE'] = 1
        time.sleep(0.5)
    except Exception as e:
        # 如果设置失败，打印错误但继续执行（SITL 中偶尔会有短暂失败）
        print("设置 VTOL 参数失败（继续，但可能出现行为异常）:", e)
    return orig_arm_check

# 安全切换模式并等待切换完成
def safe_set_mode(vehicle, mode_name, timeout=5):
    try:
        vehicle.mode = VehicleMode(mode_name)
        return wait_for(lambda: vehicle.mode.name == mode_name, timeout, desc=f"mode {mode_name}")
    except Exception as e:
        print("切换模式出错:", e)
        return False

# 解锁（arm）流程，包含必要的前置检查
def arm_with_checks(vehicle, arm_timeout=30, allow_disable_checks=True):
    # 如果当前在 QRTL/RTL 模式，先回到 STABILIZE，避免 PreArm 校验拒绝解锁
    if vehicle.mode and vehicle.mode.name in ("QRTL", "RTL"):
        print("当前处于 RTL/QRTL，先切成 STABILIZE 以允许解锁")
        safe_set_mode(vehicle, "STABILIZE", timeout=3)

    # 等待飞控 reports is_armable 为 True（表示自检通过）
    print("等待 vehicle.is_armable（飞控自检）...")
    ok = wait_for(lambda: getattr(vehicle, 'is_armable', False), 10, desc="is_armable")
    if not ok:
        print("飞控仍未准备好（is_armable=False）。")
        if allow_disable_checks:
            try:
                orig = vehicle.parameters.get('ARMING_CHECK', None)
                print("尝试临时将 ARMING_CHECK 置 0 以允许解锁（仅调试/ SITL）")
                if orig is not None:
                    vehicle.parameters['ARMING_CHECK'] = 0
                    time.sleep(0.5)  # 等待参数传播
                    ok2 = wait_for(lambda: getattr(vehicle, 'is_armable', False), 6, desc="is_armable after disabling checks")
                    if ok2:
                        # 返回原始值以便后续恢复
                        return orig
                else:
                    print("无法读取 ARMING_CHECK 参数，不能临时禁用检查")
            except Exception as e:
                print("设置 ARMING_CHECK 失败:", e)
        # 如果不能变为 armable，就报错退出
        raise RuntimeError("飞控未准备好且无法通过临时禁用检查解决（请检查 GPS/传感器/模拟环境）")

    # 切到 GUIDED 模式准备解锁
    print("切换到 GUIDED 并尝试解锁")
    safe_set_mode(vehicle, "GUIDED", timeout=5)
    vehicle.armed = True  # 发送解锁命令
    t0 = time.time()
    while not vehicle.armed:
        if time.time() - t0 > arm_timeout:
            raise RuntimeError("解锁超时")
        print(" 等待解锁...")
        time.sleep(0.5)
    print("已解锁")
    return None

# 使用 DroneKit 的 simple_takeoff 功能并等待达到目标高度
def simple_takeoff_wait(vehicle, target_alt, timeout=60):
    print("调用 simple_takeoff")
    try:
        vehicle.simple_takeoff(target_alt)  # 让飞控开始起飞到目标高度
    except Exception as e:
        print("simple_takeoff 调用异常:", e)
    # 等待到达目标高度的条件，使用 wait_for 轮询高度
    return wait_for(lambda: (vehicle.location.global_relative_frame.alt is not None and vehicle.location.global_relative_frame.alt >= target_alt*0.9),
                    timeout, poll=1, desc=f"reach alt {target_alt}")

# 使用simple_goto导航到目标经纬
def goto_wait(vehicle, lat, lon, radius=ARRIVAL_RADIUS, timeout=180):
    loc = LocationGlobalRelative(lat, lon, TAKEOFF_ALT)  # 构造目标位置对象
    try:
        vehicle.simple_goto(loc)  # 发送导航命令
    except Exception as e:
        print("simple_goto 调用失败:", e)
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

# 触发 QRTL(QuadPlane的返回策略)
def do_qrtl_return_and_wait(vehicle, land_timeout=300):
    print("切换到QRTL")
    safe_set_mode(vehicle, "QRTL", timeout=3)  
    # landed 函数判断是否已经着陆并断开油门（armed == False）
    def landed():
        try:
            alt = vehicle.location.global_relative_frame.alt
            if alt is None:
                return False
            return alt <= 0.9 and not vehicle.armed
        except Exception:
            return False
    # 等待着陆，超时则返回 False
    return wait_for(landed, land_timeout, poll=2, desc="landing via QRTL")

# 任务结束后的清理工作
def cleanup_after_mission(vehicle, orig_arm_check):
    try:
        if vehicle.armed:
            print("尝试 disarm...")
            vehicle.armed = False
            wait_for(lambda: not vehicle.armed, 8, desc="disarm")
    except Exception as e:
        print("disarm 出错:", e)
    if orig_arm_check is not None:
        try:
            vehicle.parameters['ARMING_CHECK'] = orig_arm_check
            print("恢复 ARMING_CHECK =", orig_arm_check)
        except Exception as e:
            print("恢复 ARMING_CHECK 失败:", e)
    # 把模式切回 STABILIZE 作为安全默认状态
    try:
        safe_set_mode(vehicle, "STABILIZE", timeout=3)
    except Exception:
        pass

# 主函数
def main():
    print("连接到", CONNECTION_STRING)
    try:
        vehicle = connect(CONNECTION_STRING, wait_ready=True, timeout=60)
    except Exception as e:
        print("连接失败:", e); sys.exit(1)

    # 在开始之前把Q_GUIDED_MODE和Q_RTL_MODE设置好
    orig_arm_check = ensure_params_for_vtol(vehicle)

    # 等待飞控提供 home（位置）信息，最多等待 15 秒（SITL 有时会短暂延迟）
    t0 = time.time()
    while not vehicle.location.global_frame.lat:
        if time.time() - t0 > 15:
            print("等待 home 超时")
            break
        print("等待 home...")
        time.sleep(0.5)
    # 读取 home 的经纬度
    home = vehicle.location.global_frame
    print("Home:", home.lat, home.lon)

    # 根据 home 计算航点 A 与航点 B 的经纬度
    wpA_lat, wpA_lon = WP_A_LAT, WP_A_LON
    wpB_lat, wpB_lon = WP_B_LAT, WP_B_LON
    print("航点A:", wpA_lat, wpA_lon)
    print("航点B:", wpB_lat, wpB_lon)

    # 如果当前模式处于 QRTL 或 RTL，先切回 STABILIZE 再继续，以免 PreArm 阻止解锁
    if vehicle.mode and vehicle.mode.name in ("QRTL", "RTL"):
        print("检测到当前模式为RTL/QRTL,切换为STABILIZE")
        safe_set_mode(vehicle, "STABILIZE", timeout=3)
        time.sleep(0.5)

    # 执行解锁流程（可能会临时修改 ARMING_CHECK，返回值为原始 ARMING_CHECK）
    modified_arm_check = None
    try:
        modified_arm_check = arm_with_checks(vehicle, arm_timeout=30, allow_disable_checks=True)
    except Exception as e:
        print("arm 失败:", e)
        cleanup_after_mission(vehicle, modified_arm_check if modified_arm_check is not None else orig_arm_check)
        vehicle.close()
        return

    # 起飞到 TAKEOFF_ALT（等待爬升完成或超时）
    ok = simple_takeoff_wait(vehicle, TAKEOFF_ALT, timeout=60)
    if not ok:
        print("爬升到目标高度失败或超时,进入RTL保护")
        do_qrtl_return_and_wait(vehicle)
        cleanup_after_mission(vehicle, modified_arm_check if modified_arm_check is not None else orig_arm_check)
        vehicle.close()
        return

    # 飞向航点A
    ok_wp = goto_wait(vehicle, wpA_lat, wpA_lon, radius=ARRIVAL_RADIUS, timeout=180)
    if not ok_wp:
        print("到达航点A超时,直接返航")
        do_qrtl_return_and_wait(vehicle)
        cleanup_after_mission(vehicle, modified_arm_check if modified_arm_check is not None else orig_arm_check)
        vehicle.close()
        return

    # 到达航点A后在该点停留10秒
    print("到达航点A,悬停10s")
    time.sleep(10)

    # 飞向航点B
    ok_wp2 = goto_wait(vehicle, wpB_lat, wpB_lon, radius=ARRIVAL_RADIUS, timeout=180)
    if not ok_wp2:
        print("到达航点B超时,直接返航")
        do_qrtl_return_and_wait(vehicle)
        cleanup_after_mission(vehicle, modified_arm_check if modified_arm_check is not None else orig_arm_check)
        vehicle.close()
        return

    # 到达航点B后停留10秒
    print("到达航点B,悬停10s")
    time.sleep(10)

    # 发起QRTL返回并等待降落完成
    landed = do_qrtl_return_and_wait(vehicle, land_timeout=300)
    if landed:
        print("检测到降落并上锁")
    else:
        print("QRTL未在超时内完成")

    cleanup_after_mission(vehicle, modified_arm_check if modified_arm_check is not None else orig_arm_check)

    print("关闭连接")
    vehicle.close()


if __name__ == "__main__":
    main()
