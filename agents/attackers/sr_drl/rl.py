import torch


def a2c(
    r: torch.Tensor,
    v: torch.Tensor,
    v_next: torch.Tensor,
    pi: torch.Tensor,
    gamma: float,
    alpha_v: float,
    alpha_h: float,
    q_range: tuple[float, float] | None = None,
    log_num_actions: torch.Tensor | None = None,
):
    """
    Advantage Actor-Critic loss, adapted from jaromiru/sr-drl.

    Args:
        r: rewards for taken actions, shape (batch,)
        v: value estimates for states, shape (batch,)
        v_next: bootstrap value estimates for next states, shape (batch,)
        pi: probabilities of taken actions under current policy, shape (batch,)
        gamma: discount factor
        alpha_v: value loss scaling
        alpha_h: entropy regularisation scaling
        q_range: optional clamp range for target Q
        log_num_actions: optional log(|A(s)|) per sample, to normalise entropy
    """
    # Flatten inputs
    r = r.flatten()
    v = v.flatten()
    v_next = v_next.flatten()
    pi = pi.flatten()

    # Policy log-prob
    log_pi = torch.log(pi + 1e-9)

    # One-step return
    q = r + gamma * v_next.detach()
    if q_range is not None:
        v_target = q.clamp(*q_range)
    else:
        v_target = q

    adv = q - v
    v_err = v_target - v

    loss_pi = -adv.detach() * log_pi
    loss_v = v_err ** 2

    if log_num_actions is not None:
        log_num_actions = log_num_actions.flatten()
        loss_h = (log_pi.detach() * log_pi) / log_num_actions
        ent = log_pi / log_num_actions
        entropy = -torch.mean(ent)
    else:
        loss_h = log_pi.detach() * log_pi
        entropy = -torch.mean(log_pi)

    loss_pi = torch.mean(loss_pi)
    loss_v = alpha_v * torch.mean(loss_v)
    loss_h = alpha_h * torch.mean(loss_h)

    loss = loss_pi + loss_v + loss_h
    return loss, loss_pi, loss_v, loss_h, entropy

