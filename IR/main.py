import threading
import time
import os
import platform
import logging

# Add libusb DLL to PATH on Windows
if platform.system() == 'Windows':
    _libusb_dir = os.path.join(
        os.path.expanduser("~"),
        r"AppData\Roaming\Python\Python310\site-packages\libusb\_platform\windows\x86_64"
    )
    if os.path.isdir(_libusb_dir):
        os.environ['PATH'] = _libusb_dir + ';' + os.environ.get('PATH', '')

import P2Pro.video
import P2Pro.P2Pro_cmd as P2Pro_CMD

logging.basicConfig()
logging.getLogger('P2Pro').setLevel(logging.INFO)
logging.getLogger('P2Pro.video').setLevel(logging.INFO)
logging.getLogger('P2Pro.P2Pro_cmd').setLevel(logging.INFO)

try:
    # Try to initialize USB command interface (requires Zadig driver)
    cam_cmd = None
    try:
        cam_cmd = P2Pro_CMD.P2Pro()
        print("USB command interface: OK")
    except Exception as e:
        print(f"USB command interface: UNAVAILABLE ({e})")
        print("  -> Gain/NUC commands disabled. Install Zadig filter driver to enable.")
        print("  -> Video stream and VTK export will still work.\n")

    print("Hotkeys:")
    print("  [q]     - Quit")
    print("  [ENTER] - Save VTK + screenshot to ~/P2Pro_VTK/")
    if cam_cmd:
        print("  [l]     - Low gain (high temp, ~600°C)")
        print("  [h]     - High gain (normal, ~120°C)")
        print("  [s]     - NUC shutter calibration")
        print("  [b]     - NUC background")
        print("  [d]     - Read shutter state")

    vid = P2Pro.video.Video()
    video_thread = threading.Thread(target=vid.open, args=(cam_cmd, 1,), daemon=True)
    video_thread.start()

    while not vid.video_running:
        time.sleep(0.01)

    # Set Low Gain if USB commands are available
    if cam_cmd:
        try:
            cam_cmd.gain_set_low()
            print("\nGain mode: LOW (high temperature, up to ~600°C)")
            gain = cam_cmd.get_prop_tpd_params(P2Pro_CMD.PropTpdParams.TPD_PROP_GAIN_SEL)
            print(f"Current gain: {gain} (0=low, 1=high)")
        except Exception as e:
            print(f"Failed to set gain: {e}")

    video_thread.join()

except KeyboardInterrupt:
    print("\nExiting...")
    time.sleep(1)

os._exit(0)
