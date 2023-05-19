import torch


def cuda_info():
    '''
    Show CUDA device information.

    Args:
        None.
    '''

    if torch.cuda.is_available():
        print("# CUDA devices: ", torch.cuda.device_count())
        for e in range(torch.cuda.device_count()):
            print("# device number ", e, ": ", torch.cuda.get_device_name(e))
    else:
        print('torch.cuda.is unavailable')


if __name__ == '__main__':
    cuda_info()

