import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from Skyjo.src.rl import self_play_wrapper as wrapper


def test_direct_opponent_probabilities_match_maskable_policy():
    torch.manual_seed(123)
    policy = MaskableActorCriticPolicy(
        spaces.Box(-1, 1, shape=(4,)),
        spaces.Discrete(5),
        lr_schedule=lambda _: 1e-3,
        net_arch=dict(pi=[8], vf=[8]),
        activation_fn=torch.nn.ReLU,
    )
    observations = torch.randn(3, 4)
    masks = torch.tensor(
        [
            [True, False, True, False, True],
            [False, True, True, False, False],
            [True] * 5,
        ]
    )

    expected = policy.get_distribution(observations)
    expected.apply_masking(masks)
    actual = wrapper._opponent_action_probabilities(policy, observations, masks)

    assert torch.allclose(actual, expected.distribution.probs, atol=1e-7)


def test_self_play_worker_uses_one_torch_thread(monkeypatch):
    calls = []
    monkeypatch.setattr(wrapper.torch, "set_num_threads", calls.append)

    env = wrapper.make_env("unused")()
    env.close()

    assert calls == [1]
