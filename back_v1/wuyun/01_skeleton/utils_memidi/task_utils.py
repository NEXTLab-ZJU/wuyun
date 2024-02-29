import torch


def collate_1d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None, shift_id=1):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
            dst[0] = shift_id
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def collate_2d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None):
    """Convert a list of 2d tensors into a padded 3d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size, values[0].shape[1]).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def batchify(items, max_len=None):
    items = [torch.LongTensor(item) if not isinstance(item, torch.Tensor) else item for item in items]
    is_1d = len(items[0].shape) == 1
    res = collate_1d(items, max_len=max_len) if is_1d else collate_2d(items, max_len=max_len)
    return res


def prepare_to_events(tokens, vels, durs, id2token):
    tokens = tokens.detach().cpu().numpy()
    vels = vels.detach().cpu().numpy()
    durs = durs.detach().cpu().numpy()
    tokens = [id2token[t] for t in tokens]
    assert len(tokens) == len(vels) == len(durs)
    return list(zip(tokens, vels, durs))

