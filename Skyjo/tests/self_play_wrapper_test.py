from Skyjo.src.rl import self_play_wrapper as wrapper


def test_self_play_worker_uses_one_torch_thread(monkeypatch):
    calls = []
    monkeypatch.setattr(wrapper.torch, "set_num_threads", calls.append)

    env = wrapper.make_env("unused")()
    env.close()

    assert calls == [1]
