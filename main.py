import traceback
from app import MapStitcherApp


if __name__ == "__main__":
    try:
        app = MapStitcherApp()
        app.run()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        traceback.print_exc()
        input("Нажмите Enter...")