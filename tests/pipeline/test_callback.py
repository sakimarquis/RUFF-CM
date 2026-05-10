from ruff_cm.pipeline.callback import Callback


def test_default_callback_methods_are_no_ops():
    cb = Callback()
    state = {}
    assert cb.augment(state) == ""
    assert cb.on_response(state, "response") is None
    assert cb.on_finish(state) is None
    assert state == {}


def test_subclass_overrides_only_what_it_needs():
    class RecordingCallback(Callback):
        name = "recorder"

        def on_response(self, state, response):
            state.setdefault("seen", []).append(response)

    cb = RecordingCallback()
    state: dict = {}
    cb.augment(state)
    cb.on_response(state, "hi")
    cb.on_response(state, "there")
    cb.on_finish(state)
    assert state == {"seen": ["hi", "there"]}
    assert cb.name == "recorder"


def test_callback_name_defaults_to_empty_string():
    assert Callback().name == ""
