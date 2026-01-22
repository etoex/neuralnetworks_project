from itertools import product

# перевод code -> level
letters = "GBRY"
nums = "1234"
levels = [ch + num for num, ch in product(nums, letters)]
level_codes = [2**i for i in range(len(levels))]

code_to_level = {i: j for i, j in zip(level_codes, levels)}
level_to_code = {j: i for i, j in zip(level_codes, levels)}


def read_seg(file_name: str) -> tuple[dict[str, int], list[dict]]:
    """
    Функция считывает параметры и метки из исходного файла  и возвращает их в виде словаря и списка.

    Параметры:
    file_name (str): путь к файлу для чтения.

    Возвращает:
    - parameters: словарь с параметрами
    - labels: список словарей с ключами position, level, name.

    Исключения:
    ValueError: если файл невалиден (не содержит секции [PARAMETERS] или [LABELS],
                или если строки не соответствуют формату, или значение параметров должно быть целым числом)
    """
    parameters = {}
    raw_labels = []

    with open(file_name, "r", encoding="utf-8-sig") as file:
        lines = file.readlines()

    in_parameters_section = False
    in_labels_section = False

    for line in lines:
        line = line.strip()

        if line == "[PARAMETERS]":
            in_parameters_section = True
            continue

        if in_parameters_section and "=" in line:
            parts = line.split("=")

            if len(parts) != 2:
                raise ValueError(f"Необходим формат ключ-значение")

            key, value = parts[0].strip(), parts[1].strip()

            try:
                parameters[key] = int(value)
            except ValueError:
                raise ValueError(f"Значение для {key} должно быть целым числом")

        if in_parameters_section and line.startswith("["):
            in_parameters_section = False

        # переходим к секции [LABELS]
        if line == "[LABELS]":
            in_labels_section = True
            continue

        if in_labels_section and line:
            raw_labels.append(line.strip())

        if in_labels_section and line.startswith("["):
            in_labels_section = False

    if not parameters or not raw_labels:
        raise ValueError(f"В файле неправильная структура")

    labels = []
    for line in raw_labels:
        pos, level, name = line.split(",", maxsplit=2)
        pos_samples = int(pos)
        level_name = code_to_level.get(int(level), f"code_{level}")
        labels.append(
            {
                "position": pos_samples
                // parameters["BYTE_PER_SAMPLE"]
                // parameters["N_CHANNEL"],
                "level": level_name,
                "name": name,
            }
        )

    return parameters, labels


def write_seg(parameters: dict[str, int], labels: list[dict], filename: str) -> None:
    """
    Функция записывает файл .seg в формате.

    parameters: dict с целочисленными параметрами:
        - SAMPLING_FREQ
        - BYTE_PER_SAMPLE
        - N_CHANNEL

    labels: список словарей:
        {"position": <int>, "level": "G1", "name": "V"}

    filename: путь к выходному файлу.
    """
    required = ["SAMPLING_FREQ", "BYTE_PER_SAMPLE", "N_CHANNEL"]
    for k in required:
        if k not in parameters:
            raise ValueError(f"parameters must contain {k}")

    bps = int(parameters["BYTE_PER_SAMPLE"])
    nch = int(parameters["N_CHANNEL"])
    if bps <= 0 or nch <= 0:
        raise ValueError("BYTE_PER_SAMPLE and N_CHANNEL must be positive integers")

    if not isinstance(labels, list) or len(labels) == 0:
        raise ValueError("labels must be a non-empty list")

    for i, lab in enumerate(labels):
        if "position" not in lab or "level" not in lab or "name" not in lab:
            raise ValueError(f"label #{i} must have keys: position, level, name")
        try:
            lab["position"] = int(lab["position"])
        except Exception:
            raise ValueError(f"label #{i} position must be int-like")
        if lab["position"] < 0:
            raise ValueError(f"label #{i} position must be >= 0")
        if not isinstance(lab["level"], str):
            raise ValueError(f"label #{i} level must be str")
        if not isinstance(lab["name"], str):
            raise ValueError(f"label #{i} name must be str")

    labels_sorted = sorted(labels, key=lambda d: d["position"])

    with open(filename, "w", encoding="utf-8-sig") as f:
        f.write("[PARAMETERS]\n")
        base_order = ["SAMPLING_FREQ", "BYTE_PER_SAMPLE", "N_CHANNEL"]
        written = set()

        for k in base_order:
            v = int(parameters[k])
            f.write(f"{k}={v}\n")
            written.add(k)

        for k in sorted(parameters.keys()):
            if k in written:
                continue
            f.write(f"{k}={int(parameters[k])}\n")

        f.write("\n[LABELS]\n")

        for lab in labels_sorted:
            pos_samples = int(lab["position"])

            pos_raw = pos_samples * bps * nch

            level_name = lab["level"]
            level_code = level_to_code.get(level_name)
            if level_code is None:
                raise ValueError(f"Unknown level '{level_name}'")

            name = lab["name"]
            f.write(f"{pos_raw},{level_code},{name}\n")
