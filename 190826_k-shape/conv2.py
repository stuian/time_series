from scipy import signal
import h5py

def conv2_MST(a,b):
    conv2 = signal.fftconvolve(a,b)
    return conv2

def main():
    file = h5py.File('E:\\Jade\\time_series\\190808_MTS-clustering\\cricket_data.h5', 'r')
    X = file['train_x'][:] # (180, 1197, 6)
    # y = file['train_y'][:]
    dist = conv2_MST(X[0],X[-1])
    print(dist.shape)
    print(dist)

if __name__ == '__main__':
    main()