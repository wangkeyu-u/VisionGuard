from visionguard.environment import collect_environment_info


def test_environment_info_selects_supported_device() -> None:
    info = collect_environment_info()

    assert info.python_version
    assert info.selected_device in {"mps", "cpu"}
    assert info.selected_device == ("mps" if info.mps_available else "cpu")
