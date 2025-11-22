from trackmania_rl.tmi_client import TMIClient, TMIConfig

OUTPUT_PATH = "coords.txt"  # сюда запишутся координаты
RACE_TIME_THRESHOLD = 0.0  # начинаем писать, когда race_time > 0.0


def wait_for_first_state(client: TMIClient, timeout: float = 5.0) -> None:
    """Ждём первый валидный state от TMInterface."""
    if not client.wait_for_state(timeout):
        raise RuntimeError("Не дождался первого состояния от TMInterface")


def wait_for_race_start(client: TMIClient) -> None:
    """Ждём, пока race_time станет > 0 (после отсчёта −3…−2…−1…0)."""
    print("Ожидаю старт гонки (race_time > 0.0)...")
    while True:
        client.wait_for_state(0.1)
        state = client.get_state()
        t = float(state.get("race_time", 0.0))
        # как только время стало положительным — считаем, что старт был
        if t > RACE_TIME_THRESHOLD:
            print(f"Гонка началась, race_time = {t:.3f} c — стартуем запись.")
            return


def record_coords(client: TMIClient, path: str) -> None:
    """Основной цикл записи координат x;y;z до конца заезда."""
    last_time = 0.0
    print(f"Пишу координаты в {path} ...")

    with open(path, "w", encoding="utf-8") as f:
        while True:
            # ждём новый тик физики
            if not client.wait_for_state(0.5):
                continue

            state = client.get_state()

            # время и позиция
            t = float(state.get("race_time", 0.0))
            pos = state.get("position")
            finished = bool(state.get("finished", False))

            if pos is None:
                continue

            x, y, z = pos

            # пишем строку в формате x;y;z
            f.write(f"{x:.3f};{y:.3f};{z:.3f}\n")
            f.flush()

            # можно по желанию выводить прогресс
            # print(f"{t:.3f}s -> {x:.3f};{y:.3f};{z:.3f}")

            # условие остановки:
            # 1) игра сообщила, что финишировали
            # 2) время резко "откатилось" назад (респавн/рестарт)
            if finished:
                print("Финиш (finished = True), останавливаю запись.")
                break

            if t < last_time - 0.1:  # небольшой допуск на шум
                print("Обнаружен сброс таймера (респавн/рестарт), останавливаю запись.")
                break

            last_time = t


def main():
    # настройки подключения к плагину (порт/хост бери тот же, что и в AngelScript-плагине)
    cfg = TMIConfig(
        host="127.0.0.1",
        port=54540,
        game_speed=0.6,
        prevent_finish=False,  # нам нужно увидеть нормальный финиш
        connect_timeout_s=5.0,
    )

    client = TMIClient(cfg=cfg)

    try:
        print("Подключаюсь к TMInterface...")
        client.start()
        print("Подключение установлено.")

        # ждём первый state
        wait_for_first_state(client)

        # ждём, пока закончится отсчёт и время станет > 0
        wait_for_race_start(client)

        # записываем координаты до конца заезда/рестарта
        record_coords(client, OUTPUT_PATH)

    except KeyboardInterrupt:
        print("Остановлено пользователем (Ctrl+C).")
    finally:
        client.close()
        print("Соединение с TMInterface закрыто.")


if __name__ == "__main__":
    main()
