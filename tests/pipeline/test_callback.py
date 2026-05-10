from ruff_cm.pipeline.callback import Callback, CallbackChain


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


def test_chain_augment_returns_only_nonempty_strings():
    class Aug(Callback):
        def augment(self, state):
            return state["msg"]

    class Empty(Callback):
        pass

    chain = CallbackChain([Aug(), Empty(), Aug()])
    state = {"msg": "context"}
    assert chain.augment(state) == ["context", "context"]


def test_chain_on_response_dispatches_in_order():
    class Recorder(Callback):
        def __init__(self, tag):
            self.tag = tag

        def on_response(self, state, response):
            state.setdefault("trace", []).append((self.tag, response))

    chain = CallbackChain([Recorder("a"), Recorder("b"), Recorder("c")])
    state: dict = {}
    chain.on_response(state, "X")
    assert state["trace"] == [("a", "X"), ("b", "X"), ("c", "X")]


def test_chain_on_finish_dispatches_in_order():
    class Finisher(Callback):
        def __init__(self, tag):
            self.tag = tag

        def on_finish(self, state):
            state.setdefault("done", []).append(self.tag)

    chain = CallbackChain([Finisher("a"), Finisher("b")])
    state: dict = {}
    chain.on_finish(state)
    assert state["done"] == ["a", "b"]


def test_chain_propagates_exceptions_from_callback():
    import pytest

    class Boom(Callback):
        def on_response(self, state, response):
            raise RuntimeError("boom")

    chain = CallbackChain([Callback(), Boom(), Callback()])
    with pytest.raises(RuntimeError, match="boom"):
        chain.on_response({}, "x")


def test_chain_with_no_callbacks_is_a_silent_no_op():
    chain = CallbackChain([])
    assert chain.augment({}) == []
    chain.on_response({}, "x")
    chain.on_finish({})


def test_chain_preserves_callback_order_via_iteration():
    cbs = [Callback(), Callback(), Callback()]
    chain = CallbackChain(cbs)
    assert tuple(chain) == tuple(cbs)
