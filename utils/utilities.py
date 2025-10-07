# Created by aaronkueh on 9/19/2025
import os
import yaml
from LSTM_Autoencoder.definitions import CONFIG_DIR

DEFAULT_CONFIG = f'{CONFIG_DIR}/api_config.yaml'


def config_param(config_file=DEFAULT_CONFIG, field=None):
    """Configures the script using a config file

    Args:
        field:
        config_file: the path of the config file, relative or absolute

    """
    with open(config_file) as config:
        try:
            configs = yaml.safe_load(config)
            param = configs.get(field)
            return param

        except yaml.YAMLError as e:
            print(e)


def percent_round(stat_cnt, total_cnt, decimal) -> float:
    """
    Calculates the percentage of the actual vs. total

    Args:
        stat_cnt: actual count
        total_cnt: total count
        decimal: number of decimal points to use
    Returns:
        float: rounded percentage
    """
    percent_rd = round((stat_cnt / total_cnt) * 100, decimal)

    return percent_rd


def keyword_filter(input_list: list, keyword: str) -> list:
    """
    Filter string list: check if an input list contains given keyword
    Args:
        input_list: a list of string (exp: thing name)
        keyword: keyword for filtering
    Returns:
        list: filtered list
    """
    return [word for word in input_list if keyword in word]


def odbc_escape_pwd(pw: str) -> str:
    """
    Wrap password for ODBC string to handle ";" and "}" safely
    Args:
        pw: password
    Returns:
        pw: wrapped password
    """
    return "{" + pw.replace("}", "}}") + "}"


def get_profile(cfg: dict, name: str | None = None) -> dict:
    profiles = cfg.get("profiles", {})
    if not profiles:
        raise KeyError("missing mssql.profiles")
    if name is None:
        name = cfg.get("default_profile")
        if not name:
            raise KeyError("missing mssql.default_profile")
    try:
        return name, profiles[name]
    except KeyError:
        raise KeyError(f"profile '{name}' not found; have {list(profiles)}")


def resolve_auth(profile: dict) -> dict:
    auth = profile.get("auth", {})
    user = auth.get("user") or os.environ.get(auth.get("user_env", ""))
    pwd = auth.get("pwd") or os.environ.get(auth.get("pwd_env", ""))
    if not user or not pwd:
        raise RuntimeError("Missing credentials: check user_env/pwd_env and .env")
    return {"user": user, "pwd": odbc_escape_pwd(pwd)}


