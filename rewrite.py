import re

import cn2an

unit_dict = {
    # 长度单位
    "M": "米",
    "NM": "纳米",
    "UM": "微米",
    "MM": "毫米",
    "CM": "厘米",
    "DM": "分米",
    "KM": "千米",

    # 重量单位
    "G": "克",
    "KG": "千克",

    # 面积单位
    "MM²": "平方毫米",
    "CM²": "平方厘米",
    "M²": "平方米",
    "KM²": "平方千米",

    # 体积单位
    "MM³": "立方毫米",
    "CM³": "立方厘米",
    "M³": "立方米",
    "ML": "毫升",

    # 温度单位
    "°C": "摄氏度",
    "°F": "华氏度",
    "°": "度",

    # 电力单位
    "MAH": "毫安时",
    "W": "瓦",
    "KW": "千瓦",

    # 速度单位
    "KM/H": "千米每小时",
    "M/S": "米每秒",
    "MPH": "英里每小时",

    # 频率单位
    "HZ": "赫兹",
    "KHZ": "千赫兹",
    "MHZ": "兆赫兹",
    "GHZ": "吉赫兹",

    # 时间单位
    "MS": "毫秒",

    # 计算机相关单位
    "FPS": "帧每秒",

    # 其他单位
    "+": "加",
}

ignore_dict = [
    r"[12]\d{3}\s*[年款]",
    r"\d+%"
]

replace_dict = {
    "+": "加",
    # 温度单位
    "℃": "摄氏度",
    "℉": "华氏度",

    # 面积单位
    "㎡": "平方米",
    "㎢": "平方公里",
    "㎠": "平方厘米",
    "㎟": "平方毫米",

    # 体积单位
    "㎥": "立方米",
    "㎣": "立方厘米",
    "㎤": "立方分米",
    "㎦": "立方毫米",

    # 长度单位
    "㎜": "毫米",
    "㎝": "厘米",
    "㎞": "公里",
    "㎎": "毫克",
    "㎏": "千克",
    "㎍": "微克",

    # 小写转大写
    "led": "LED",
}


def number_to_chinese(number: str):
    chinese_str = cn2an.an2cn(number)
    if len(number) >= 3 and chinese_str.startswith("二"):
        chinese_str = "两" + chinese_str[1:]
    return chinese_str


def time_to_chinese(match):
    hour, minute = match.groups()
    if minute == "00":
        hour_chinese = f"{int(hour)}点" if hour != "00" else "零点"
        return hour_chinese
    else:
        hour_chinese = f"{int(hour)}点" if hour != "00" else "零点"
        minute_chinese = f"{minute}分"
        return f"{hour_chinese}{minute_chinese}"


def year_to_chinese(year: str):
    chinese_year = ""
    for i in range(len(year)):
        chinese_year += number_to_chinese(year[i])
    return chinese_year


def transcribe(text):
    # 小写转大写
    regex_words = r"\b[a-zA-Z]+\b"
    text = re.sub(regex_words, lambda m: m.group(0).lower(), text)

    # 字典替换
    for k, v in replace_dict.items():
        text = re.sub(re.escape(k), v, text, flags=re.IGNORECASE)

    # 区间替换
    regex_from_to = r"(\d+)\s*-\s*(\d+)"
    text = re.sub(regex_from_to, r"\1至\2", text)

    # 时间替换
    regex_time = r"(\d{2}):([0-5][0-9])"
    text = re.sub(regex_time, time_to_chinese, text)

    # 数字与单位替换
    regex_num_unit = r"(\d+\.?\d*)\s*([A-Za-z/°²³%\+]+|[\u4e00-\u9fa5]|[,.，。、])"
    result = []

    last_end = 0
    for match in re.finditer(regex_num_unit, text):
        number = match.group(1)
        unit = match.group(2)

        match_str = number + unit
        if any(re.match(regex, match_str) for regex in ignore_dict):
            continue

        # transform number to chinese
        chinese_number = number_to_chinese(number)
        # transform unit to chinese
        unit = unit_dict.get(unit.upper(), unit)

        # append chinese number and unit to result list
        result.append(text[last_end:match.start()] + chinese_number + unit)

        # update last position
        last_end = match.end()

    # add the rest of the text to result list
    result.append(text[last_end:])

    return "".join(result)


if __name__ == "__main__":
    test_text = "Nvidia rtx 3060显卡是一个性价比很高的显卡，LED配备了usb阵列卡升级功能，再加上H330、H750这样的高端raid控制器加持，简直就是数据界的“保险箱”。就算你的业务量突然暴涨，cpu也能稳如泰山，绝不掉链子。"
    print(transcribe(test_text))
