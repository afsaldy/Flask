import os
import time
import requests
import subprocess
import signal

NGROK_CONFIG = 'ngrok.yml'
NGROK_API_URL = 'tunnels'
LARAVEL_UPDATE_URL = 'update_ngrok'
TIMEOUT_TUNNEL = 30
REFRESH_INTERVAL = 1020

def start_ngrok():
    print("[AUTO-NGROK] Memulai ngrok...")
    # Jalankan ngrok sebagai background process, bukan di cmd terpisah!
    return subprocess.Popen(
        ['ngrok', 'start', '--all', '--config', NGROK_CONFIG],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP  # Penting untuk Windows, agar mudah kill
    )

def kill_ngrok(proc):
    if proc and proc.poll() is None:
        print("[AUTO-NGROK] Kill ngrok (background process)...")
        proc.send_signal(signal.CTRL_BREAK_EVENT)  # Khusus Windows
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    # Untuk berjaga-jaga, matikan juga via nama proses (redundan, aman)
    os.system("taskkill /f /im ngrok.exe >nul 2>&1")

def get_ngrok_public_urls():
    try:
        resp = requests.get(NGROK_API_URL, timeout=3)
        tunnels = resp.json().get('tunnels', [])
        urls = {}
        for t in tunnels:
            if t.get('proto') == 'https':
                name = t.get('name')
                urls[name] = t.get('public_url')
        return urls
    except Exception as e:
        print("Belum dapat URL ngrok:", e)
    return {}

def send_urls_to_laravel(urls):
    try:
        payload = {'ngrok_urls': urls}
        resp = requests.post(LARAVEL_UPDATE_URL, json=payload, timeout=5)
        print(f"[AUTO-NGROK] Kirim semua URL ke Laravel ({resp.status_code}): {urls}")
    except Exception as e:
        print("[AUTO-NGROK] Gagal kirim URL ke Laravel:", e)

def wait_for_tunnels():
    print("[AUTO-NGROK] Menunggu tunnel ngrok siap...")
    start_time = time.time()
    while True:
        urls = get_ngrok_public_urls()
        if urls and 'flask1' in urls and 'flask2' in urls:
            print(f"[AUTO-NGROK] Tunnel siap: flask1={urls['flask1']}, flask2={urls['flask2']}")
            return urls
        if time.time() - start_time > TIMEOUT_TUNNEL:
            print("[AUTO-NGROK] Tunnel ngrok TIDAK siap dalam waktu maksimal. Akan restart ulang ngrok.")
            return None
        time.sleep(2)

if __name__ == "__main__":
    ngrok_proc = None
    while True:
        kill_ngrok(ngrok_proc)
        ngrok_proc = start_ngrok()
        time.sleep(3)

        urls = wait_for_tunnels()
        if urls:
            send_urls_to_laravel(urls)
            print(f"[AUTO-NGROK] Selesai, akan restart lagi dalam {REFRESH_INTERVAL//60} menit.")
            time.sleep(REFRESH_INTERVAL)
        else:
            print("[AUTO-NGROK] Restart ngrok lebih cepat karena tunnel tidak siap.")
            time.sleep(5)
