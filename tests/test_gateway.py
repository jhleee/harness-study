from harness.nodes.gateway import gateway


def test_gateway_defaults_channel_to_cli() -> None:
    result = gateway({})  # type: ignore[arg-type]
    assert result == {"channel": "cli"}


def test_gateway_passes_through_explicit_channel() -> None:
    result = gateway({"channel": "telegram"})  # type: ignore[arg-type]
    assert result == {}
