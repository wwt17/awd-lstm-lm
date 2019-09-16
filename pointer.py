import torch
import torch.nn.functional as F

def init_history(batch_size, vocab_size, state_size, device):
    next_word_history = torch.zeros([0, batch_size, vocab_size], dtype=torch.float, device=device)
    pointer_history = torch.zeros([0, batch_size, state_size], dtype=torch.float, device=device)
    return next_word_history, pointer_history

def get_p(targets, states, window, theta, history):
    next_word_history, pointer_history = history
    batch_size = next_word_history.size(1)
    vocab_size = next_word_history.size(-1)
    start_t = next_word_history.size(0)
    next_word_history = torch.cat([next_word_history, F.one_hot(targets, vocab_size).float()], dim=0).detach()
    pointer_history = torch.cat([pointer_history, states], dim=0).detach()
    ptr_p = []
    for t in range(targets.size(0)):
        en_t = start_t + t
        st_t = max(0, en_t - window)
        if st_t == en_t:
            ptr_dist = torch.zeros(batch_size, vocab_size, dtype=torch.float, device=next_word_history.device)
        else:
            valid_next_word_history = next_word_history[st_t:en_t]
            valid_pointer_history = pointer_history[st_t:en_t]
            copy_scores = torch.einsum('tbi,bi->tb', valid_pointer_history, states[t])
            ptr_attn = F.softmax(theta * copy_scores, 0)
            ptr_dist = (ptr_attn.unsqueeze(-1) * valid_next_word_history).sum(0)
        ptr_p.append(ptr_dist.detach())
    ptr_p = torch.stack(ptr_p, dim=0)
    next_word_history = next_word_history[-window:].detach()
    pointer_history = pointer_history[-window:].detach()
    return ptr_p, (next_word_history, pointer_history)
